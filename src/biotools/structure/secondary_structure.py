"""Secondary-structure assignment with DSSP."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

from Bio.PDB import DSSP, MMCIFParser, PDBParser

AccessibilityScale: TypeAlias = Literal["Sander", "Wilke", "Miller", "Ahmad"]
ResidueID: TypeAlias = tuple[str, int, str]

_ACCESSIBILITY_SCALES = {"Sander", "Wilke", "Miller", "Ahmad"}
_FILE_TYPES = {
    ".cif": "MMCIF",
    ".ent": "PDB",
    ".mmcif": "MMCIF",
    ".pdb": "PDB",
}


@dataclass(frozen=True)
class DSSPResidue:
    """DSSP assignment and geometry for one protein residue."""

    chain_id: str
    residue_id: ResidueID
    dssp_index: int
    amino_acid: str
    secondary_structure: str
    relative_accessibility: float | None
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


def assign_secondary_structure(
    structure_file: str | PathLike[str],
    *,
    executable: str = "dssp",
    accessibility_scale: AccessibilityScale = "Sander",
) -> DSSPResult:
    """Assign secondary structure to a PDB or mmCIF file using DSSP.

    Biopython automatically tries ``mkdssp`` when the default ``dssp``
    executable is unavailable, and vice versa. DSSP annotates the parsed
    Biopython residues internally; this function returns those assignments as
    immutable, file-independent records.

    Args:
        structure_file: Existing PDB, ``.ent``, mmCIF, or ``.cif`` file.
        executable: DSSP executable name or path.
        accessibility_scale: Reference scale used for relative solvent
            accessibility.

    Returns:
        Source metadata and one DSSP record per assigned residue.

    Raises:
        FileNotFoundError: If the structure file or DSSP executable is absent.
        ValueError: If the file format, executable, or accessibility scale is
            invalid, or if the structure contains no model.
        OSError: If DSSP cannot process the structure file.
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

    parser_type = PDBParser if file_type == "PDB" else MMCIFParser
    structure = parser_type(QUIET=True).get_structure(
        source_path.stem,
        str(source_path),
    )
    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise ValueError("Structure contains no model for DSSP analysis") from exc

    try:
        dssp = DSSP(
            model,
            str(source_path),
            dssp=executable,
            acc_array=accessibility_scale,
            file_type=file_type,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "DSSP is not installed or its executable is not available on "
            "PATH (expected 'dssp' or 'mkdssp')"
        ) from exc

    residues = []
    for chain_id, residue_id in dssp.keys():
        (
            dssp_index,
            amino_acid,
            secondary_structure,
            relative_accessibility,
            phi,
            psi,
            nh_o_1_relative_index,
            nh_o_1_energy,
            o_nh_1_relative_index,
            o_nh_1_energy,
            nh_o_2_relative_index,
            nh_o_2_energy,
            o_nh_2_relative_index,
            o_nh_2_energy,
        ) = dssp[(chain_id, residue_id)]
        residues.append(
            DSSPResidue(
                chain_id=chain_id,
                residue_id=residue_id,
                dssp_index=int(dssp_index),
                amino_acid=amino_acid,
                secondary_structure=secondary_structure,
                relative_accessibility=(
                    None
                    if relative_accessibility == "NA"
                    else float(relative_accessibility)
                ),
                phi=float(phi),
                psi=float(psi),
                nh_o_1_relative_index=int(nh_o_1_relative_index),
                nh_o_1_energy=float(nh_o_1_energy),
                o_nh_1_relative_index=int(o_nh_1_relative_index),
                o_nh_1_energy=float(o_nh_1_energy),
                nh_o_2_relative_index=int(nh_o_2_relative_index),
                nh_o_2_energy=float(nh_o_2_energy),
                o_nh_2_relative_index=int(o_nh_2_relative_index),
                o_nh_2_energy=float(o_nh_2_energy),
            )
        )

    return DSSPResult(
        source_path=source_path,
        accessibility_scale=accessibility_scale,
        residues=tuple(residues),
    )
