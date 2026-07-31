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
        _get_minimization_diagnostics,
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
        _get_minimization_diagnostics,
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

        energy, rms_force, max_force = _get_minimization_diagnostics(context)

        self.assertEqual(energy, 42.5)
        self.assertAlmostEqual(rms_force, np.sqrt(169.0 / 6.0))
        self.assertEqual(max_force, 12.0)
        context.getState.assert_called_once_with(getEnergy=True, getForces=True)
        state.getForces.assert_called_once_with(asNumpy=True)

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
                        result.initial_rms_force_kj_mol_nm,
                        result.final_rms_force_kj_mol_nm,
                        result.final_max_force_kj_mol_nm,
                    )
                )
            )

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
        system = object()
        forcefield.createSystem.return_value = system
        forcefield_constructor = MagicMock(return_value=forcefield)
        integrator = object()
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
                    "biotools.mdtools._get_minimization_diagnostics",
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

    def test_minimize_returns_diagnostics_when_requested(self) -> None:
        """Diagnostics should describe energy change and force convergence."""
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
        forcefield.createSystem.return_value = object()
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
                "max constraint error": 0.0,
            }
            reporter.report(0, None, None, report_args)
            reporter.report(1, None, None, report_args)

        simulation.minimizeEnergy.side_effect = run_minimization

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", pdb_file),
                patch("biotools.mdtools.ForceField", return_value=forcefield),
                patch("biotools.mdtools.VerletIntegrator"),
                patch("biotools.mdtools.Simulation", return_value=simulation),
                patch("biotools.mdtools.NoCutoff", "NoCutoff"),
                patch("biotools.mdtools.kilojoule_per_mole", 1.0),
                patch("biotools.mdtools.nanometer", 1.0),
                patch("biotools.mdtools.picoseconds", 1.0),
                patch(
                    "biotools.mdtools._get_minimization_diagnostics",
                    side_effect=[(100.0, 25.0, 40.0), (30.0, 12.0, 18.0)],
                ),
            ):
                result = minimize(
                    input_path,
                    output_path,
                    tolerance_kj_mol_nm=10.0,
                    max_iterations=2,
                    return_diagnostics=True,
                )
            self.assertEqual(output_path.read_text(), "END\n")

        self.assertIsInstance(result, MinimizationResult)
        self.assertEqual(result.output_path, output_path)
        self.assertEqual(result.delta_energy_kj_mol, -70.0)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.final_rms_force_kj_mol_nm, 12.0)
        self.assertEqual(result.final_max_force_kj_mol_nm, 18.0)
        self.assertFalse(result.converged)
        self.assertEqual(result.max_iterations, 2)
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

    def test_minimize_rejects_nonfinite_diagnostics(self) -> None:
        """An invalid OpenMM energy or force should fail the preparation."""
        topology = MagicMock()
        topology.getPeriodicBoxVectors.return_value = None
        pdb = SimpleNamespace(topology=topology, positions=object())
        simulation = MagicMock()
        simulation.topology = topology

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "minimized.pdb"
            input_path.write_text("END\n")
            with (
                patch("biotools.mdtools.PDBFile", return_value=pdb),
                patch("biotools.mdtools.ForceField"),
                patch("biotools.mdtools.VerletIntegrator"),
                patch("biotools.mdtools.Simulation", return_value=simulation),
                patch(
                    "biotools.mdtools._get_minimization_diagnostics",
                    side_effect=[(100.0, 25.0, 40.0), (float("nan"), 5.0, 8.0)],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    minimize(input_path, output_path)

        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
