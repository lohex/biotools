"""Tests for DSSP secondary-structure assignment."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from biotools import pdbtools
from biotools.structure import (
    DSSPResidue,
    assign_secondary_structure,
)


class DSSPAssignmentTests(unittest.TestCase):
    def test_returns_typed_residue_assignments(self) -> None:
        residue_id_a = (" ", 10, " ")
        residue_id_b = (" ", 11, "A")
        dssp_output = {
            ("A", residue_id_a): (
                1,
                "A",
                "H",
                0.4,
                -60.0,
                -40.0,
                2,
                -0.5,
                -2,
                -0.4,
                3,
                -0.3,
                -3,
                -0.2,
            ),
            ("A", residue_id_b): (
                2,
                "G",
                "-",
                "NA",
                70.0,
                20.0,
                0,
                0.0,
                0,
                0.0,
                0,
                0.0,
                0,
                0.0,
            ),
        }
        model = object()
        structure = MagicMock()
        structure.get_models.return_value = iter((model,))
        parser = MagicMock()
        parser.get_structure.return_value = structure

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with (
                patch(
                    "biotools.structure.secondary_structure.PDBParser",
                    return_value=parser,
                ) as parser_type,
                patch(
                    "biotools.structure.secondary_structure.DSSP",
                    return_value=dssp_output,
                ) as dssp_type,
            ):
                result = assign_secondary_structure(
                    source_path,
                    executable="custom-dssp",
                    accessibility_scale="Wilke",
                )

        parser_type.assert_called_once_with(QUIET=True)
        parser.get_structure.assert_called_once_with(
            "protein",
            str(source_path),
        )
        dssp_type.assert_called_once_with(
            model,
            str(source_path),
            dssp="custom-dssp",
            acc_array="Wilke",
            file_type="PDB",
        )
        self.assertEqual(result.source_path, source_path)
        self.assertEqual(result.accessibility_scale, "Wilke")
        self.assertEqual(len(result.residues), 2)
        self.assertIsInstance(result.residues[0], DSSPResidue)
        self.assertEqual(result.residues[0].chain_id, "A")
        self.assertEqual(result.residues[0].residue_id, residue_id_a)
        self.assertEqual(result.residues[0].secondary_structure, "H")
        self.assertEqual(result.residues[0].relative_accessibility, 0.4)
        self.assertEqual(result.residues[0].nh_o_1_energy, -0.5)
        self.assertIsNone(result.residues[1].relative_accessibility)

    def test_uses_mmcif_parser_and_file_type(self) -> None:
        model = object()
        structure = MagicMock()
        structure.get_models.return_value = iter((model,))
        parser = MagicMock()
        parser.get_structure.return_value = structure

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.mmcif"
            source_path.write_text("data_protein\n")
            with (
                patch(
                    "biotools.structure.secondary_structure.MMCIFParser",
                    return_value=parser,
                ) as parser_type,
                patch(
                    "biotools.structure.secondary_structure.DSSP",
                    return_value={},
                ) as dssp_type,
            ):
                result = assign_secondary_structure(source_path)

        parser_type.assert_called_once_with(QUIET=True)
        dssp_type.assert_called_once_with(
            model,
            str(source_path),
            dssp="dssp",
            acc_array="Sander",
            file_type="MMCIF",
        )
        self.assertEqual(result.residues, ())

    def test_reports_missing_dssp_executable(self) -> None:
        structure = MagicMock()
        structure.get_models.return_value = iter((object(),))
        parser = MagicMock()
        parser.get_structure.return_value = structure

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with (
                patch(
                    "biotools.structure.secondary_structure.PDBParser",
                    return_value=parser,
                ),
                patch(
                    "biotools.structure.secondary_structure.DSSP",
                    side_effect=FileNotFoundError("dssp"),
                ),
                self.assertRaisesRegex(FileNotFoundError, "DSSP is not installed"),
            ):
                assign_secondary_structure(source_path)

    def test_validates_input_before_running_dssp(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.pdb"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                assign_secondary_structure(missing_path)

            unsupported_path = Path(temp_dir) / "protein.txt"
            unsupported_path.write_text("structure\n")
            with self.assertRaisesRegex(ValueError, "PDB.*mmCIF"):
                assign_secondary_structure(unsupported_path)

            pdb_path = Path(temp_dir) / "protein.pdb"
            pdb_path.write_text("END\n")
            with self.assertRaisesRegex(ValueError, "accessibility scale"):
                assign_secondary_structure(
                    pdb_path,
                    accessibility_scale="invalid",
                )
            with self.assertRaisesRegex(ValueError, "executable"):
                assign_secondary_structure(pdb_path, executable="")

    def test_pdbtools_facade_exports_dssp_api(self) -> None:
        self.assertIs(
            pdbtools.assign_secondary_structure,
            assign_secondary_structure,
        )
        self.assertIs(pdbtools.DSSPResidue, DSSPResidue)


if __name__ == "__main__":
    unittest.main()
