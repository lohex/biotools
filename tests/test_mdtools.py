"""Tests for molecular-dynamics preparation helpers."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

OPENMM_AVAILABLE = True

try:
    from biotools.mdtools import (
        MinimizationResult,
        _IterationReporter,
        _classify_minimization_termination,
        _get_raw_state_diagnostics,
        _should_restart_optimizer,
        fix_pdb,
        minimize,
        model_solvent,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"openmm", "pdbfixer"}:
        raise

    OPENMM_AVAILABLE = False
    openmm_module = ModuleType("openmm")
    openmm_app_module = ModuleType("openmm.app")
    openmm_unit_module = ModuleType("openmm.unit")
    pdbfixer_module = ModuleType("pdbfixer")
    openmm_module.MinimizationReporter = type("MinimizationReporter", (), {})
    openmm_module.Platform = MagicMock()
    openmm_module.VerletIntegrator = MagicMock()
    openmm_app_module.ForceField = MagicMock()
    openmm_app_module.HBonds = object()
    openmm_app_module.Modeller = MagicMock()
    openmm_app_module.NoCutoff = object()
    openmm_app_module.PDBFile = MagicMock()
    openmm_app_module.PME = object()
    openmm_app_module.Simulation = MagicMock()
    openmm_unit_module.kilojoule_per_mole = 1.0
    openmm_unit_module.molar = 1.0
    openmm_unit_module.nanometer = 1.0
    openmm_unit_module.picoseconds = 1.0
    pdbfixer_module.PDBFixer = MagicMock()
    openmm_module.app = openmm_app_module
    openmm_module.unit = openmm_unit_module
    sys.modules.update(
        {
            "openmm": openmm_module,
            "openmm.app": openmm_app_module,
            "openmm.unit": openmm_unit_module,
            "pdbfixer": pdbfixer_module,
        }
    )
    from biotools.mdtools import (
        MinimizationResult,
        _IterationReporter,
        _classify_minimization_termination,
        _get_raw_state_diagnostics,
        _should_restart_optimizer,
        fix_pdb,
        minimize,
        model_solvent,
    )


class FixPDBTests(unittest.TestCase):
    """Verify PDBFixer orchestration without requiring an OpenMM install."""

    def test_fix_pdb_runs_default_repair_pipeline(self) -> None:
        """The default pipeline should run all conservative repair steps."""
        fixer = MagicMock()
        fixer.missingResidues = {("A", 0): ["ALA"]}
        fixer.topology = object()
        fixer.positions = object()
        fixer_constructor = MagicMock(return_value=fixer)
        write_file = MagicMock()

        pdb_file = SimpleNamespace(writeFile=write_file)

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "fixed.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFixer", fixer_constructor),
                patch("biotools.mdtools.PDBFile", pdb_file),
            ):
                result = fix_pdb(input_path, output_path)

        self.assertEqual(result, output_path)
        fixer_constructor.assert_called_once_with(filename=str(input_path))
        fixer.findMissingResidues.assert_called_once_with()
        self.assertEqual(fixer.missingResidues, {})
        fixer.findNonstandardResidues.assert_called_once_with()
        fixer.replaceNonstandardResidues.assert_called_once_with()
        fixer.removeHeterogens.assert_called_once_with(keepWater=True)
        fixer.findMissingAtoms.assert_called_once_with()
        fixer.addMissingAtoms.assert_called_once_with()
        fixer.addMissingHydrogens.assert_called_once_with(7.0)
        write_file.assert_called_once()
        self.assertTrue(write_file.call_args.kwargs["keepIds"])

    def test_fix_pdb_rejects_overwriting_input(self) -> None:
        """The source PDB should not be overwritten in place."""
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            input_path.write_text("END\n")

            with self.assertRaisesRegex(ValueError, "must be different"):
                fix_pdb(input_path, input_path)


class ModelAndMinimizeTests(unittest.TestCase):
    """Verify OpenMM modelling and minimization orchestration."""

    def test_minimization_diagnostics_calculates_force_statistics(self) -> None:
        """RMS should span components while max force should span particles."""

        class Quantity:
            def __init__(self, value) -> None:
                self.value = value

            def value_in_unit(self, unit):
                return self.value

        state = MagicMock()
        state.getPotentialEnergy.return_value = Quantity(42.5)
        state.getForces.return_value = Quantity(
            np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
        )
        context = MagicMock()
        context.getState.return_value = state

        energy, rms_force, max_force = _get_raw_state_diagnostics(context)

        self.assertEqual(energy, 42.5)
        self.assertAlmostEqual(rms_force, np.sqrt(169.0 / 6.0))
        self.assertEqual(max_force, 12.0)
        context.getState.assert_called_once_with(getEnergy=True, getForces=True)
        state.getForces.assert_called_once_with(asNumpy=True)

    def test_reporter_uses_openmm_objective_gradient_norm(self) -> None:
        """Reporter RMS gradient should use OpenMM's per-particle norm."""
        reporter = _IterationReporter(num_particles=2)
        args = {
            "system energy": 42.0,
            "restraint energy": 0.5,
            "restraint strength": 1000.0,
            "max constraint error": 5e-5,
        }

        reporter.report(
            0,
            None,
            [3.0, 4.0, 0.0, 0.0, 0.0, 12.0],
            args,
        )

        self.assertAlmostEqual(
            reporter.history[0]["objective_rms_gradient_kj_mol_nm"],
            np.sqrt(169.0 / 2.0),
        )
        self.assertEqual(reporter.history[0]["max_constraint_error"], 5e-5)

    def test_termination_reason_identifies_iteration_limit(self) -> None:
        """A high final objective gradient at the phase limit is not converged."""
        reporter = _IterationReporter(num_particles=1)
        args = {
            "system energy": 42.0,
            "restraint energy": 0.0,
            "restraint strength": 1000.0,
            "max constraint error": 5e-5,
        }
        reporter.report(0, None, [20.0, 0.0, 0.0], args)
        reporter.report(1, None, [15.0, 0.0, 0.0], args)

        converged, reason = _classify_minimization_termination(
            reporter,
            tolerance_kj_mol_nm=10.0,
            constraint_tolerance=1e-4,
            max_iterations=2,
        )

        self.assertFalse(converged)
        self.assertEqual(reason, "max_iterations")

    def test_termination_reason_identifies_constraint_error(self) -> None:
        """A small gradient alone must not hide unsatisfied constraints."""
        reporter = _IterationReporter(num_particles=1)
        reporter.report(
            0,
            None,
            [1.0, 0.0, 0.0],
            {
                "system energy": 42.0,
                "restraint energy": 0.5,
                "restraint strength": 1000.0,
                "max constraint error": 2e-4,
            },
        )

        converged, reason = _classify_minimization_termination(
            reporter,
            tolerance_kj_mol_nm=10.0,
            constraint_tolerance=1e-4,
            max_iterations=0,
        )

        self.assertFalse(converged)
        self.assertEqual(reason, "constraint_error")

    def test_optimizer_restart_requires_exact_retry_conditions(self) -> None:
        """Only optimizer stops with valid constraints and budget may restart."""
        reporter = _IterationReporter(num_particles=1)
        report_args = {
            "system energy": 42.0,
            "restraint energy": 0.0,
            "restraint strength": 1000.0,
            "max constraint error": 5e-5,
        }
        reporter.report(0, None, [20.0, 0.0, 0.0], report_args)
        retry_options = {
            "tolerance_kj_mol_nm": 10.0,
            "constraint_tolerance": 1e-4,
            "optimizer_restarts": 0,
        }

        self.assertFalse(
            _should_restart_optimizer(
                reporter,
                termination_reason="optimizer_stopped",
                max_optimizer_restarts=0,
                **retry_options,
            )
        )
        self.assertTrue(
            _should_restart_optimizer(
                reporter,
                termination_reason="optimizer_stopped",
                max_optimizer_restarts=1,
                **retry_options,
            )
        )
        for reason in ("max_iterations", "constraint_error", "converged"):
            with self.subTest(reason=reason):
                self.assertFalse(
                    _should_restart_optimizer(
                        reporter,
                        termination_reason=reason,
                        max_optimizer_restarts=1,
                        **retry_options,
                    )
                )

        report_args["max constraint error"] = 2e-4
        constrained_reporter = _IterationReporter(num_particles=1)
        constrained_reporter.report(
            0,
            None,
            [20.0, 0.0, 0.0],
            report_args,
        )
        self.assertFalse(
            _should_restart_optimizer(
                constrained_reporter,
                termination_reason="optimizer_stopped",
                max_optimizer_restarts=1,
                **retry_options,
            )
        )

    @unittest.skipUnless(OPENMM_AVAILABLE, "OpenMM is not installed")
    def test_minimize_small_structure_with_openmm(self) -> None:
        """A local water fixture should produce finite diagnostics and a PDB."""
        input_path = Path(__file__).with_name("data") / "water.pdb"

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "minimized.pdb"
            result = minimize(
                input_path,
                output_path,
                forcefield_files=("amber14/tip3pfb.xml",),
                verbose=False,
                return_diagnostics=True,
            )

            self.assertIsInstance(result, MinimizationResult)
            self.assertTrue(output_path.is_file())
            self.assertIn("END", output_path.read_text())
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for value in (
                        result.initial_energy_kj_mol,
                        result.final_energy_kj_mol,
                        result.initial_raw_rms_force_kj_mol_nm,
                        result.final_raw_rms_force_kj_mol_nm,
                        result.final_raw_max_force_kj_mol_nm,
                    )
                )
            )
            self.assertIsNone(result.final_objective_rms_gradient_kj_mol_nm)
            self.assertIsNone(result.final_max_constraint_error)
            self.assertTrue(result.converged)
            self.assertEqual(result.termination_reason, "already_converged")
            self.assertEqual(result.optimizer_restarts, 0)

    @unittest.skipUnless(OPENMM_AVAILABLE, "OpenMM is not installed")
    def test_minimize_constrained_solvated_system_with_openmm(self) -> None:
        """Reporter convergence should work for a periodic rigid-water box."""
        from openmm import Vec3
        from openmm.app import (
            ForceField as OpenMMForceField,
            HBonds as OpenMMHBonds,
            Modeller as OpenMMModeller,
            PDBFile as OpenMMPDBFile,
            Topology,
        )
        from openmm.unit import nanometer as openmm_nanometer

        forcefield = OpenMMForceField("amber14/tip3pfb.xml")
        modeller = OpenMMModeller(Topology(), [])
        modeller.addSolvent(
            forcefield,
            boxSize=Vec3(1.0, 1.0, 1.0) * openmm_nanometer,
        )
        constrained_system = forcefield.createSystem(
            modeller.topology,
            constraints=OpenMMHBonds,
        )
        self.assertGreater(constrained_system.getNumConstraints(), 0)

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "solvated.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            with input_path.open("w") as output:
                OpenMMPDBFile.writeFile(
                    modeller.topology,
                    modeller.positions,
                    output,
                )

            result = minimize(
                input_path,
                output_path,
                forcefield_files=("amber14/tip3pfb.xml",),
                tolerance_kj_mol_nm=500.0,
                nonbonded_cutoff_nm=0.4,
                verbose=False,
                return_diagnostics=True,
            )

        self.assertIsInstance(result, MinimizationResult)
        self.assertGreater(result.iterations, 0)
        self.assertIsNotNone(result.final_objective_rms_gradient_kj_mol_nm)
        self.assertIsNotNone(result.final_max_constraint_error)
        self.assertLessEqual(
            result.final_objective_rms_gradient_kj_mol_nm,
            result.tolerance_kj_mol_nm,
        )
        self.assertLessEqual(
            result.final_max_constraint_error,
            result.constraint_tolerance,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.termination_reason, "converged")
        self.assertEqual(result.optimizer_restarts, 0)

    def test_model_solvent_adds_ph_hydrogens_and_solvent(self) -> None:
        """Model preparation should protonate and solvate with chosen values."""
        topology = object()
        positions = object()
        pdb = SimpleNamespace(topology=topology, positions=positions)
        pdb_file = MagicMock(return_value=pdb)
        pdb_file.writeFile = MagicMock()
        forcefield = MagicMock()
        forcefield_constructor = MagicMock(return_value=forcefield)
        modeller = MagicMock()
        modeller.topology = object()
        modeller.positions = object()
        modeller_constructor = MagicMock(return_value=modeller)

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "modelled.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", pdb_file),
                patch("biotools.mdtools.ForceField", forcefield_constructor),
                patch("biotools.mdtools.Modeller", modeller_constructor),
                patch("biotools.mdtools.nanometer", 1.0),
                patch("biotools.mdtools.molar", 1.0),
            ):
                result = model_solvent(
                    input_path,
                    output_path,
                    ph=6.5,
                    padding_nm=1.2,
                    ionic_strength_molar=0.15,
                )

        self.assertEqual(result, output_path)
        modeller_constructor.assert_called_once_with(topology, positions)
        modeller.addHydrogens.assert_called_once_with(forcefield, pH=6.5)
        modeller.addSolvent.assert_called_once_with(
            forcefield,
            model="tip3p",
            padding=1.2,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=0.15,
            neutralize=True,
        )
        pdb_file.writeFile.assert_called_once()

    def test_minimize_uses_pme_for_periodic_structure(self) -> None:
        """A solvated periodic structure should be minimized with PME."""
        topology = MagicMock()
        topology.getPeriodicBoxVectors.return_value = object()
        positions = object()
        pdb = SimpleNamespace(topology=topology, positions=positions)
        pdb_file = MagicMock(return_value=pdb)
        pdb_file.writeFile = MagicMock()
        forcefield = MagicMock()
        system = MagicMock()
        system.getNumParticles.return_value = 2
        forcefield.createSystem.return_value = system
        forcefield_constructor = MagicMock(return_value=forcefield)
        integrator = MagicMock()
        integrator.getConstraintTolerance.return_value = 1e-5
        integrator_constructor = MagicMock(return_value=integrator)
        minimized_positions = object()
        state = MagicMock()
        state.getPositions.return_value = minimized_positions
        simulation = MagicMock()
        simulation.topology = topology
        simulation.context.getState.return_value = state
        simulation_constructor = MagicMock(return_value=simulation)

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "modelled.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", pdb_file),
                patch("biotools.mdtools.ForceField", forcefield_constructor),
                patch("biotools.mdtools.VerletIntegrator", integrator_constructor),
                patch("biotools.mdtools.Simulation", simulation_constructor),
                patch("biotools.mdtools.HBonds", "HBonds"),
                patch("biotools.mdtools.NoCutoff", "NoCutoff"),
                patch("biotools.mdtools.PME", "PME"),
                patch("biotools.mdtools.kilojoule_per_mole", 1.0),
                patch("biotools.mdtools.nanometer", 1.0),
                patch("biotools.mdtools.picoseconds", 1.0),
                patch(
                    "biotools.mdtools._get_raw_state_diagnostics",
                    side_effect=[(100.0, 50.0, 75.0), (10.0, 5.0, 8.0)],
                ),
            ):
                result = minimize(input_path, output_path)

        self.assertEqual(result, output_path)
        forcefield.createSystem.assert_called_once_with(
            topology,
            constraints="HBonds",
            nonbondedMethod="PME",
            nonbondedCutoff=1.0,
        )
        simulation_constructor.assert_called_once_with(
            topology,
            system,
            integrator,
        )
        simulation.context.setPositions.assert_called_once_with(positions)
        simulation.minimizeEnergy.assert_called_once_with(
            tolerance=10.0,
            maxIterations=1000,
            reporter=simulation.minimizeEnergy.call_args.kwargs["reporter"],
        )
        reporter = simulation.minimizeEnergy.call_args.kwargs["reporter"]
        self.assertEqual(reporter.history, [])
        pdb_file.writeFile.assert_called_once()

    def test_minimize_selects_requested_gpu_platform(self) -> None:
        """GPU platform and device properties should reach the Context."""
        topology = MagicMock()
        topology.getPeriodicBoxVectors.return_value = None
        pdb = SimpleNamespace(topology=topology, positions=object())
        pdb_file = MagicMock(return_value=pdb)
        pdb_file.writeFile = MagicMock()
        forcefield = MagicMock()
        system = MagicMock()
        system.getNumParticles.return_value = 2
        forcefield.createSystem.return_value = system
        integrator = MagicMock()
        integrator.getConstraintTolerance.return_value = 1e-5
        selected_platform = object()
        platform = MagicMock()
        platform.getPlatformByName.return_value = selected_platform
        position_state = MagicMock()
        position_state.getPositions.return_value = object()
        simulation = MagicMock()
        simulation.topology = topology
        simulation.context.getState.return_value = position_state

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", pdb_file),
                patch("biotools.mdtools.ForceField", return_value=forcefield),
                patch(
                    "biotools.mdtools.VerletIntegrator",
                    return_value=integrator,
                ),
                patch("biotools.mdtools.Platform", platform),
                patch(
                    "biotools.mdtools.Simulation",
                    return_value=simulation,
                ) as constructor,
                patch("biotools.mdtools.NoCutoff", "NoCutoff"),
                patch("biotools.mdtools.kilojoule_per_mole", 1.0),
                patch("biotools.mdtools.nanometer", 1.0),
                patch("biotools.mdtools.picoseconds", 1.0),
                patch(
                    "biotools.mdtools._get_raw_state_diagnostics",
                    side_effect=[(100.0, 50.0, 75.0), (10.0, 5.0, 8.0)],
                ),
            ):
                result = minimize(
                    input_path,
                    output_path,
                    platform_name="CUDA",
                    platform_properties={
                        "DeviceIndex": "1",
                        "Precision": "mixed",
                    },
                    verbose=False,
                )

        self.assertEqual(result, output_path)
        platform.getPlatformByName.assert_called_once_with("CUDA")
        constructor.assert_called_once_with(
            topology,
            system,
            integrator,
            platform=selected_platform,
            platformProperties={
                "DeviceIndex": "1",
                "Precision": "mixed",
            },
        )

    def test_minimize_converges_after_optimizer_restart(self) -> None:
        """A fresh reporter should converge in the unchanged Context."""
        topology = MagicMock()
        topology.getPeriodicBoxVectors.return_value = None
        pdb = SimpleNamespace(topology=topology, positions=object())
        pdb_file = MagicMock(return_value=pdb)
        pdb_file.writeFile = MagicMock(
            side_effect=lambda topology, positions, output, keepIds: output.write(
                "END\n"
            )
        )
        forcefield = MagicMock()
        system = MagicMock()
        system.getNumParticles.return_value = 2
        forcefield.createSystem.return_value = system
        simulation = MagicMock()
        simulation.topology = topology
        position_state = MagicMock()
        position_state.getPositions.return_value = object()
        simulation.context.getState.return_value = position_state

        def run_minimization(**kwargs) -> None:
            reporter = kwargs["reporter"]
            report_args = {
                "system energy": 20.0,
                "restraint energy": 0.0,
                "restraint strength": 0.0,
                "max constraint error": 5e-5,
            }
            if simulation.minimizeEnergy.call_count == 1:
                gradient = [30.0, 0.0, 0.0] * 2
            else:
                gradient = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
            reporter.report(0, None, gradient, report_args)

        simulation.minimizeEnergy.side_effect = run_minimization

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", pdb_file),
                patch("biotools.mdtools.ForceField", return_value=forcefield),
                patch("biotools.mdtools.VerletIntegrator") as integrator,
                patch("biotools.mdtools.Simulation", return_value=simulation),
                patch("biotools.mdtools.NoCutoff", "NoCutoff"),
                patch("biotools.mdtools.kilojoule_per_mole", 1.0),
                patch("biotools.mdtools.nanometer", 1.0),
                patch("biotools.mdtools.picoseconds", 1.0),
                patch(
                    "biotools.mdtools._get_raw_state_diagnostics",
                    side_effect=[(100.0, 25.0, 40.0), (30.0, 12.0, 18.0)],
                ),
            ):
                integrator.return_value.getConstraintTolerance.return_value = 1e-5
                result = minimize(
                    input_path,
                    output_path,
                    tolerance_kj_mol_nm=10.0,
                    max_iterations=2,
                    max_optimizer_restarts=1,
                    return_diagnostics=True,
                )
            self.assertEqual(output_path.read_text(), "END\n")

        self.assertIsInstance(result, MinimizationResult)
        self.assertEqual(result.output_path, output_path)
        self.assertEqual(result.delta_energy_kj_mol, -70.0)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.optimizer_restarts, 1)
        self.assertEqual(result.final_raw_rms_force_kj_mol_nm, 12.0)
        self.assertEqual(result.final_raw_max_force_kj_mol_nm, 18.0)
        self.assertEqual(result.final_rms_force_kj_mol_nm, 12.0)
        self.assertEqual(result.final_max_force_kj_mol_nm, 18.0)
        self.assertAlmostEqual(
            result.final_objective_rms_gradient_kj_mol_nm,
            np.sqrt(25.0 / 2.0),
        )
        self.assertEqual(result.final_max_constraint_error, 5e-5)
        self.assertTrue(result.converged)
        self.assertEqual(result.termination_reason, "converged")
        self.assertEqual(result.max_iterations, 2)
        self.assertEqual(simulation.minimizeEnergy.call_count, 2)
        reporters = [
            call.kwargs["reporter"]
            for call in simulation.minimizeEnergy.call_args_list
        ]
        self.assertIsNot(reporters[0], reporters[1])
        simulation.context.setPositions.assert_called_once_with(pdb.positions)
        pdb_file.writeFile.assert_called_once()

    def test_minimize_preserves_numeric_and_path_validation(self) -> None:
        """Diagnostics must not weaken the existing argument validation."""
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "output.pdb"
            input_path.write_text("END\n")

            invalid_arguments = (
                {"tolerance_kj_mol_nm": 0.0},
                {"tolerance_kj_mol_nm": -1.0},
                {"max_iterations": -1},
                {"max_optimizer_restarts": -1},
                {"nonbonded_cutoff_nm": 0.0},
                {"nonbonded_cutoff_nm": -1.0},
            )
            for arguments in invalid_arguments:
                with (
                    self.subTest(arguments=arguments),
                    self.assertRaises(ValueError),
                ):
                    minimize(input_path, output_path, **arguments)

            with self.assertRaisesRegex(ValueError, "must be different"):
                minimize(input_path, input_path)

            with self.assertRaisesRegex(ValueError, "requires.*platform_name"):
                minimize(
                    input_path,
                    output_path,
                    platform_properties={"DeviceIndex": "0"},
                )

    def test_minimize_rejects_nonfinite_diagnostics(self) -> None:
        """An invalid OpenMM energy or force should fail the preparation."""
        topology = MagicMock()
        topology.getPeriodicBoxVectors.return_value = None
        pdb = SimpleNamespace(topology=topology, positions=object())
        simulation = MagicMock()
        simulation.topology = topology
        forcefield = MagicMock()
        system = MagicMock()
        system.getNumParticles.return_value = 2
        forcefield.createSystem.return_value = system

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", return_value=pdb),
                patch("biotools.mdtools.ForceField", return_value=forcefield),
                patch("biotools.mdtools.VerletIntegrator"),
                patch("biotools.mdtools.Simulation", return_value=simulation),
                patch(
                    "biotools.mdtools._get_raw_state_diagnostics",
                    side_effect=[(100.0, 25.0, 40.0), (float("nan"), 5.0, 8.0)],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    minimize(input_path, output_path)

        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
