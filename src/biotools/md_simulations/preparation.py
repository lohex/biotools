"""PDB repair and explicit-solvent simulation preparation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from openmm.app import ForceField, Modeller, PDBFile
from openmm.unit import molar, nanometer
from pdbfixer import PDBFixer

from .common import validate_io_paths

logger = logging.getLogger(__name__)


def model_solvent(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *,
    ph: float = 7.0,
    padding_nm: float = 1.0,
    water_model: str = "tip3p",
    ionic_strength_molar: float = 0.0,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    neutralize: bool = True,
    forcefield_files: Sequence[str] = (
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ),
    keep_ids: bool = False,
    verbose: bool = True,
) -> Path:
    """Add pH-dependent hydrogens and explicit solvent to a PDB model."""
    input_path, output_path = validate_io_paths(input_file, output_file)
    if padding_nm <= 0:
        raise ValueError("padding_nm must be greater than zero")
    if ionic_strength_molar < 0:
        raise ValueError("ionic_strength_molar must not be negative")

    if verbose:
        logger.info("Adding hydrogens and solvent to %s", input_path)
    pdb = PDBFile(str(input_path))
    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=ph)
    modeller.addSolvent(
        forcefield,
        model=water_model,
        padding=padding_nm * nanometer,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
        ionicStrength=ionic_strength_molar * molar,
        neutralize=neutralize,
    )

    with output_path.open("w") as output:
        PDBFile.writeFile(
            modeller.topology,
            modeller.positions,
            output,
            keepIds=keep_ids,
        )

    if verbose:
        logger.info("Saved modeled PDB file to %s", output_path)
    return output_path


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
    verbose: bool = True,
) -> Path:
    """Repair a PDB file with PDBFixer and write it through OpenMM."""
    input_path, output_path = validate_io_paths(input_file, output_file)
    if add_missing_residues and not add_missing_atoms:
        raise ValueError(
            "add_missing_residues=True requires add_missing_atoms=True"
        )

    if verbose:
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

    if verbose:
        logger.info("Saved repaired PDB file to %s", output_path)
    return output_path
