"""OpenMM-based molecular-dynamics preparation utilities."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_pdb(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *,
    add_missing_residues: bool = False,
    replace_nonstandard_residues: bool = True,
    remove_heterogens: bool = True,
    keep_water: bool = True,
    add_missing_atoms: bool = True,
    add_hydrogens: bool = True,
    ph: float = 7.0,
    keep_ids: bool = True,
) -> Path:
    """Repair a PDB file with PDBFixer and write it through OpenMM.

    The repair steps follow PDBFixer's required order: identify missing
    residues, optionally replace nonstandard residues, optionally remove
    heterogens, add missing atoms, and finally add hydrogens. Missing complete
    residues are not built by default because their coordinates are modeled
    rather than experimentally observed.

    Args:
        input_file: Existing PDB file to repair.
        output_file: Destination for the repaired PDB file. It must differ from
            ``input_file`` to prevent accidental loss of the source structure.
        add_missing_residues: Build residues reported as missing by SEQRES.
        replace_nonstandard_residues: Convert recognized nonstandard residues
            to their standard equivalents.
        remove_heterogens: Remove ligands, ions, and other heterogens.
        keep_water: Preserve water when heterogens are removed.
        add_missing_atoms: Add missing heavy atoms and, when enabled, missing
            residues.
        add_hydrogens: Add hydrogens appropriate for ``ph``.
        ph: pH used to select protonation states when adding hydrogens.
        keep_ids: Preserve valid chain and residue IDs in the written PDB.

    Returns:
        Path to the repaired PDB file.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
        ImportError: If OpenMM or PDBFixer is not installed.
        ValueError: If input and output refer to the same path, or missing
            residues are requested without enabling missing atoms.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")
    if add_missing_residues and not add_missing_atoms:
        raise ValueError(
            "add_missing_residues=True requires add_missing_atoms=True"
        )

    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError(
            "fix_pdb() requires OpenMM and PDBFixer. Install both from "
            "conda-forge with: conda install -c conda-forge openmm pdbfixer"
        ) from exc

    logger.info("Repairing PDB file %s", input_path)
    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    if not add_missing_residues:
        fixer.missingResidues = {}

    if replace_nonstandard_residues:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_water)

    if add_missing_atoms:
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

    if add_hydrogens:
        fixer.addMissingHydrogens(ph)

    with output_path.open("w") as output:
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            output,
            keepIds=keep_ids,
        )

    logger.info("Saved repaired PDB file to %s", output_path)
    return output_path
