"""Secondary-structure assignment with DSSP."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

from Bio.Data.PDBData import protein_letters_1to3, residue_sasa_scales

AccessibilityScale: TypeAlias = Literal["Sander", "Wilke", "Miller", "Ahmad"]
ResidueID: TypeAlias = tuple[str, int, str]

_ACCESSIBILITY_SCALES = {"Sander", "Wilke", "Miller", "Ahmad"}
_FILE_TYPES = {
    ".cif": "MMCIF",
    ".ent": "PDB",
    ".mmcif": "MMCIF",
    ".pdb": "PDB",
}
_DUMMY_CRYST1 = (
    b"CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 "
    b"P 1           1\n"
)
_DUMMY_HEADER = b"HEADER    BIOTOOLS DSSP COMPATIBILITY INPUT\n"


@dataclass(frozen=True)
class DSSPResidue:
    """DSSP assignment and geometry for one protein residue."""

    chain_id: str
    residue_id: ResidueID
    dssp_index: int
    amino_acid: str
    secondary_structure: str
    relative_accessibility: float | None
    absolute_accessibility: float | None
    phi: float
    psi: float
    nh_o_1_relative_index: int
    nh_o_1_energy: float
    o_nh_1_relative_index: int
    o_nh_1_energy: float
    nh_o_2_relative_index: int
    nh_o_2_energy: float
    o_nh_2_relative_index: int
    o_nh_2_energy: float


@dataclass(frozen=True)
class DSSPResult:
    """Complete DSSP assignment for one structure file."""

    source_path: Path
    accessibility_scale: AccessibilityScale
    residues: tuple[DSSPResidue, ...]
    secondary_structure: str
    relative_sasa: list[float | None]
    absolute_sasa: list[float | None]


def _pdb_compatibility_content(source_path: Path) -> bytes | None:
    """Return patched PDB bytes, or None when the file is already compatible."""
    content = source_path.read_bytes()
    lines = content.splitlines(keepends=True)
    has_header = bool(lines) and lines[0].startswith(b"HEADER")
    has_cryst1 = any(line.startswith(b"CRYST1") for line in lines)
    if has_header and has_cryst1:
        return None

    if has_header:
        return lines[0] + _DUMMY_CRYST1 + b"".join(lines[1:])
    if has_cryst1:
        return _DUMMY_HEADER + content
    return _DUMMY_HEADER + _DUMMY_CRYST1 + content


@contextmanager
def _dssp_input_path(source_path: Path, file_type: str) -> Iterator[Path]:
    """Yield a DSSP-compatible input path without modifying the source file."""
    compatibility_content = (
        _pdb_compatibility_content(source_path) if file_type == "PDB" else None
    )
    if file_type != "PDB" or compatibility_content is None:
        yield source_path
        return

    with TemporaryDirectory(prefix="biotools-dssp-") as temp_dir:
        temporary_path = Path(temp_dir) / source_path.name
        temporary_path.write_bytes(compatibility_content)
        yield temporary_path


def _parse_version(output: str) -> tuple[int, ...]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)", output)
    if match is None:
        raise OSError(f"Could not determine DSSP version from: {output.strip()!r}")
    return tuple(int(component) for component in match.group(1).split("."))


def _resolve_executable(executable: str) -> tuple[str, tuple[int, ...]]:
    candidates = [executable]
    if executable == "dssp":
        candidates.append("mkdssp")
    elif executable == "mkdssp":
        candidates.append("dssp")

    for candidate in candidates:
        try:
            completed = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise OSError(
                f"DSSP version check failed for {candidate!r}: {detail}"
            )
        version = _parse_version(f"{completed.stdout}\n{completed.stderr}")
        return candidate, version

    raise FileNotFoundError(
        "DSSP is not installed or its executable is not available on PATH "
        "(expected 'dssp' or 'mkdssp')"
    )


def _run_dssp(input_path: Path, executable: str) -> str:
    resolved_executable, version = _resolve_executable(executable)
    command = [resolved_executable]
    if version >= (4, 0, 0):
        command.append("--output-format=dssp")
    command.append(str(input_path))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = (completed.stderr or completed.stdout).strip()
        raise OSError(
            f"DSSP failed for {input_path} with exit code "
            f"{completed.returncode}: {detail}"
        )
    return completed.stdout


def _relative_accessibility(
    amino_acid: str,
    absolute_accessibility: float,
    accessibility_scale: AccessibilityScale,
) -> float | None:
    residue_name = protein_letters_1to3.get(amino_acid.upper())
    if residue_name is None:
        return None
    maximum = residue_sasa_scales[accessibility_scale].get(residue_name)
    if maximum is None:
        return None
    return min(absolute_accessibility / maximum, 1.0)


def _parse_dssp_output(
    output: str,
    accessibility_scale: AccessibilityScale,
) -> tuple[DSSPResidue, ...]:
    """Parse classic fixed-width DSSP output into typed residue records."""
    records = []
    started = False
    for line_number, line in enumerate(output.splitlines(), start=1):
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "RESIDUE":
            started = True
            continue
        if not started or len(line) < 17:
            continue
        if len(line) <= 9 or line[9] == " ":
            continue

        try:
            dssp_index = int(line[:5])
            residue_number = int(line[5:10])
            insertion_code = line[10]
            chain_id = line[11]
            amino_acid = line[13]
            if amino_acid.islower():
                amino_acid = "C"
            secondary_structure = line[16] if line[16] != " " else "-"

            def parse_numeric_fields(shift: int) -> tuple[float | int, ...]:
                return (
                    float(int(line[34 + shift : 38 + shift])),
                    int(line[38 + shift : 45 + shift]),
                    float(line[46 + shift : 50 + shift]),
                    int(line[50 + shift : 56 + shift]),
                    float(line[57 + shift : 61 + shift]),
                    int(line[61 + shift : 67 + shift]),
                    float(line[68 + shift : 72 + shift]),
                    int(line[72 + shift : 78 + shift]),
                    float(line[79 + shift : 83 + shift]),
                    float(line[103 + shift : 109 + shift]),
                    float(line[109 + shift : 115 + shift]),
                )

            try:
                numeric_fields = parse_numeric_fields(0)
            except ValueError:
                shift = line[34:].find(" ") if len(line) > 34 else -1
                if len(line) <= 34 or line[34] == " " or shift < 0:
                    raise
                numeric_fields = parse_numeric_fields(shift)
            (
                absolute_accessibility,
                nh_o_1_relative_index,
                nh_o_1_energy,
                o_nh_1_relative_index,
                o_nh_1_energy,
                nh_o_2_relative_index,
                nh_o_2_energy,
                o_nh_2_relative_index,
                o_nh_2_energy,
                phi,
                psi,
            ) = numeric_fields
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Could not parse DSSP output line {line_number}: {line!r}"
            ) from exc

        records.append(
            DSSPResidue(
                chain_id=chain_id,
                residue_id=(" ", residue_number, insertion_code),
                dssp_index=dssp_index,
                amino_acid=amino_acid,
                secondary_structure=secondary_structure,
                relative_accessibility=_relative_accessibility(
                    amino_acid,
                    absolute_accessibility,
                    accessibility_scale,
                ),
                absolute_accessibility=absolute_accessibility,
                phi=phi,
                psi=psi,
                nh_o_1_relative_index=nh_o_1_relative_index,
                nh_o_1_energy=nh_o_1_energy,
                o_nh_1_relative_index=o_nh_1_relative_index,
                o_nh_1_energy=o_nh_1_energy,
                nh_o_2_relative_index=nh_o_2_relative_index,
                nh_o_2_energy=nh_o_2_energy,
                o_nh_2_relative_index=o_nh_2_relative_index,
                o_nh_2_energy=o_nh_2_energy,
            )
        )
    if not started:
        raise ValueError("DSSP output does not contain a residue table header")
    return tuple(records)


def assign_secondary_structure(
    structure_file: str | PathLike[str],
    *,
    executable: str = "dssp",
    accessibility_scale: AccessibilityScale = "Sander",
) -> DSSPResult:
    """Assign secondary structure to a PDB or mmCIF file using DSSP.

    The DSSP executable is called directly and its classic fixed-width output
    is parsed without the Biopython DSSP wrapper. The names ``dssp`` and
    ``mkdssp`` are tried as mutual fallbacks. For PDB inputs without a
    ``CRYST1`` record, a temporary copy containing a dummy record is passed to
    DSSP; a compatibility ``HEADER`` is also added when necessary. The source
    file is never modified. Absolute solvent accessibility is retained in
    square angstroms and relative accessibility is calculated with the chosen
    Biopython reference scale.

    Args:
        structure_file: Existing PDB, ``.ent``, mmCIF, or ``.cif`` file.
        executable: DSSP executable name or path.
        accessibility_scale: Reference scale used for relative solvent
            accessibility.

    Returns:
        Source metadata and one DSSP record per assigned residue.

    Raises:
        FileNotFoundError: If the structure file or DSSP executable is absent.
        ValueError: If the file format, executable, accessibility scale, or
            DSSP output is invalid.
        OSError: If DSSP cannot process the structure file or report a version.
    """
    source_path = Path(structure_file)
    if not source_path.is_file():
        raise FileNotFoundError(f"Structure file does not exist: {source_path}")
    if not executable:
        raise ValueError("executable must not be empty")
    if accessibility_scale not in _ACCESSIBILITY_SCALES:
        supported = ", ".join(sorted(_ACCESSIBILITY_SCALES))
        raise ValueError(
            f"Unsupported accessibility scale {accessibility_scale!r}; "
            f"choose one of: {supported}"
        )

    try:
        file_type = _FILE_TYPES[source_path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(
            "DSSP input must be a PDB (.pdb/.ent) or mmCIF (.cif/.mmcif) file"
        ) from exc

    with _dssp_input_path(source_path, file_type) as dssp_input_path:
        output = _run_dssp(dssp_input_path, executable)
    residue_records = _parse_dssp_output(output, accessibility_scale)
    return DSSPResult(
        source_path=source_path,
        accessibility_scale=accessibility_scale,
        residues=residue_records,
        secondary_structure="".join(
            residue.secondary_structure for residue in residue_records
        ),
        relative_sasa=[
            residue.relative_accessibility for residue in residue_records
        ],
        absolute_sasa=[
            residue.absolute_accessibility for residue in residue_records
        ],
    )
