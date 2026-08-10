"""Tests for direct DSSP secondary-structure assignment."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from biotools import pdbtools
from biotools.structure import DSSPResidue, assign_secondary_structure


def _dssp_line(
    dssp_index: int,
    residue_number: int,
    amino_acid: str,
    secondary_structure: str,
    accessibility: int,
    *,
    chain_id: str = "A",
    insertion_code: str = " ",
    phi: float = -60.0,
    psi: float = -40.0,
) -> str:
    """Create one classic fixed-width DSSP residue line."""
    line = [" "] * 120

    def put(start: int, end: int, value: str) -> None:
        line[start:end] = list(value)

    put(0, 5, f"{dssp_index:5d}")
    put(5, 10, f"{residue_number:5d}")
    line[10] = insertion_code
    line[11] = chain_id
    line[13] = amino_acid
    line[16] = " " if secondary_structure == "-" else secondary_structure
    put(34, 38, f"{accessibility:4d}")
    put(38, 45, f"{2:7d}")
    put(46, 50, f"{-0.5:4.1f}")
    put(50, 56, f"{-2:6d}")
    put(57, 61, f"{-0.4:4.1f}")
    put(61, 67, f"{3:6d}")
    put(68, 72, f"{-0.3:4.1f}")
    put(72, 78, f"{-3:6d}")
    put(79, 83, f"{-0.2:4.1f}")
    put(103, 109, f"{phi:6.1f}")
    put(109, 115, f"{psi:6.1f}")
    return "".join(line)


def _dssp_output() -> str:
    return "\n".join(
        (
            "==== DSSP test output ====",
            "  #  RESIDUE AA STRUCTURE",
            _dssp_line(1, 10, "A", "H", 200),
            _dssp_line(
                2,
                11,
                "G",
                "-",
                20,
                insertion_code="A",
                phi=70.0,
                psi=20.0,
            ),
            "",
        )
    )


def _completed(
    command: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class DSSPAssignmentTests(unittest.TestCase):
    def test_calls_dssp_directly_and_returns_sasa_summaries(self) -> None:
        dssp_input = {}

        def run(command, **kwargs):
            self.assertEqual(
                kwargs,
                {"capture_output": True, "text": True, "check": False},
            )
            if command[-1] == "--version":
                return _completed(command, stdout="mkdssp version 4.2.2\n")
            dssp_input["command"] = command
            dssp_input["path"] = Path(command[-1])
            dssp_input["content"] = Path(command[-1]).read_text()
            return _completed(command, stdout=_dssp_output())

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with patch(
                "biotools.structure.secondary_structure.subprocess.run",
                side_effect=run,
            ) as run_process:
                result = assign_secondary_structure(
                    source_path,
                    executable="custom-dssp",
                    accessibility_scale="Wilke",
                )
            self.assertEqual(source_path.read_text(), "END\n")

        self.assertEqual(run_process.call_count, 2)
        self.assertEqual(
            dssp_input["command"][:2],
            ["custom-dssp", "--output-format=dssp"],
        )
        self.assertNotEqual(dssp_input["path"], source_path)
        self.assertTrue(dssp_input["content"].startswith("HEADER"))
        self.assertIn("\nCRYST1", dssp_input["content"])
        self.assertTrue(dssp_input["content"].endswith("END\n"))

        self.assertEqual(result.source_path, source_path)
        self.assertEqual(result.accessibility_scale, "Wilke")
        self.assertEqual(len(result.residues), 2)
        self.assertIsInstance(result.residues[0], DSSPResidue)
        self.assertEqual(result.residues[0].chain_id, "A")
        self.assertEqual(result.residues[0].residue_id, (" ", 10, " "))
        self.assertEqual(result.residues[0].secondary_structure, "H")
        self.assertEqual(result.residues[0].absolute_accessibility, 200.0)
        self.assertEqual(result.residues[0].relative_accessibility, 1.0)
        self.assertEqual(result.residues[0].nh_o_1_energy, -0.5)
        self.assertEqual(result.residues[1].residue_id, (" ", 11, "A"))
        self.assertEqual(result.secondary_structure, "H-")
        self.assertEqual(
            result.relative_sasa,
            [1.0, 20.0 / 104.0],
        )
        self.assertEqual(result.absolute_sasa, [200.0, 20.0])

    def test_mmcif_is_passed_without_pdb_compatibility_copy(self) -> None:
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[-1] == "--version":
                return _completed(command, stdout="mkdssp version 4.2.2\n")
            return _completed(command, stdout="  #  RESIDUE AA STRUCTURE\n")

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.mmcif"
            source_path.write_text("data_protein\n")
            with patch(
                "biotools.structure.secondary_structure.subprocess.run",
                side_effect=run,
            ):
                result = assign_secondary_structure(source_path)

        self.assertEqual(commands[-1][-1], str(source_path))
        self.assertEqual(result.residues, ())
        self.assertEqual(result.secondary_structure, "")
        self.assertEqual(result.relative_sasa, [])
        self.assertEqual(result.absolute_sasa, [])

    def test_preserves_compatible_pdb_input(self) -> None:
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[-1] == "--version":
                return _completed(command, stdout="mkdssp version 4.2.2\n")
            return _completed(command, stdout="  #  RESIDUE AA STRUCTURE\n")

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text(
                "HEADER    TEST STRUCTURE\n"
                "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 "
                "P 1           1\nEND\n"
            )
            with patch(
                "biotools.structure.secondary_structure.subprocess.run",
                side_effect=run,
            ):
                assign_secondary_structure(source_path)

        self.assertEqual(commands[-1][-1], str(source_path))

    def test_uses_legacy_command_for_dssp_before_version_four(self) -> None:
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[-1] == "--version":
                return _completed(command, stdout="DSSP 3.9.9\n")
            return _completed(command, stdout="  #  RESIDUE AA STRUCTURE\n")

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with patch(
                "biotools.structure.secondary_structure.subprocess.run",
                side_effect=run,
            ):
                assign_secondary_structure(source_path, executable="legacy-dssp")

        self.assertEqual(commands[-1][0], "legacy-dssp")
        self.assertNotIn("--output-format=dssp", commands[-1])

    def test_falls_back_between_dssp_executable_names(self) -> None:
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[0] == "dssp":
                raise FileNotFoundError("dssp")
            if command[-1] == "--version":
                return _completed(command, stdout="mkdssp version 4.2.2\n")
            return _completed(command, stdout="  #  RESIDUE AA STRUCTURE\n")

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with patch(
                "biotools.structure.secondary_structure.subprocess.run",
                side_effect=run,
            ):
                assign_secondary_structure(source_path)

        self.assertEqual(commands[0], ["dssp", "--version"])
        self.assertEqual(commands[1], ["mkdssp", "--version"])
        self.assertEqual(commands[2][0], "mkdssp")

    def test_reports_missing_dssp_executable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with (
                patch(
                    "biotools.structure.secondary_structure.subprocess.run",
                    side_effect=FileNotFoundError("dssp"),
                ),
                self.assertRaisesRegex(FileNotFoundError, "DSSP is not installed"),
            ):
                assign_secondary_structure(source_path)

    def test_reports_dssp_process_failure(self) -> None:
        def run(command, **kwargs):
            if command[-1] == "--version":
                return _completed(command, stdout="mkdssp version 4.2.2\n")
            return _completed(command, stderr="invalid structure", returncode=1)

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with (
                patch(
                    "biotools.structure.secondary_structure.subprocess.run",
                    side_effect=run,
                ),
                self.assertRaisesRegex(OSError, "invalid structure"),
            ):
                assign_secondary_structure(source_path)

    def test_rejects_malformed_dssp_output(self) -> None:
        def run(command, **kwargs):
            if command[-1] == "--version":
                return _completed(command, stdout="mkdssp version 4.2.2\n")
            return _completed(command, stdout="not a DSSP residue table\n")

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "protein.pdb"
            source_path.write_text("END\n")
            with (
                patch(
                    "biotools.structure.secondary_structure.subprocess.run",
                    side_effect=run,
                ),
                self.assertRaisesRegex(ValueError, "residue table header"),
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
