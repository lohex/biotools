"""Tests for molecular-dynamics preparation helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from biotools.mdtools import fix_pdb


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

        pdbfixer_module = ModuleType("pdbfixer")
        pdbfixer_module.PDBFixer = fixer_constructor
        openmm_module = ModuleType("openmm")
        openmm_app_module = ModuleType("openmm.app")
        openmm_app_module.PDBFile = SimpleNamespace(writeFile=write_file)
        openmm_module.app = openmm_app_module

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.pdb"
            output_path = Path(temp_dir) / "fixed.pdb"
            input_path.write_text("END\n")
            with patch.dict(
                sys.modules,
                {
                    "openmm": openmm_module,
                    "openmm.app": openmm_app_module,
                    "pdbfixer": pdbfixer_module,
                },
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


if __name__ == "__main__":
    unittest.main()
