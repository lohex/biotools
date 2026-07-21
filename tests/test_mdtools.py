"""Tests for molecular-dynamics preparation helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

try:
    from biotools.mdtools import fix_pdb, minimize, model_solvent
except ModuleNotFoundError as exc:
    if exc.name not in {"openmm", "pdbfixer"}:
        raise

    openmm_module = ModuleType("openmm")
    openmm_app_module = ModuleType("openmm.app")
    openmm_unit_module = ModuleType("openmm.unit")
    pdbfixer_module = ModuleType("pdbfixer")
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
    from biotools.mdtools import fix_pdb, minimize, model_solvent


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
        )
        pdb_file.writeFile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
