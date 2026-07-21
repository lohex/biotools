"""OpenMM-based molecular-dynamics preparation utilities."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from openmm import VerletIntegrator
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    NoCutoff,
    PDBFile,
    PME,
    Simulation,
)
from openmm.unit import kilojoule_per_mole, molar, nanometer, picoseconds
from pdbfixer import PDBFixer

logger = logging.getLogger(__name__)

__all__ = ["fix_pdb", "minimize", "model_solvent"]


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
) -> Path:
    """Add pH-dependent hydrogens and explicit solvent to a PDB model.

    OpenMM selects protonation states for the requested pH, adds missing
    hydrogens, creates a periodic solvent box, and optionally adds ions to
    neutralize the solute or reach the requested bulk ionic strength.

    Args:
        input_file: Existing PDB structure to model.
        output_file: Destination for the solvated PDB structure.
        ph: pH used by OpenMM to select protonation states.
        padding_nm: Minimum solvent padding around the solute in nanometers.
        water_model: Water geometry passed to ``Modeller.addSolvent()``.
        ionic_strength_molar: Added bulk ion-pair concentration in molar.
        positive_ion: Positive ion species used for neutralization and salt.
        negative_ion: Negative ion species used for neutralization and salt.
        neutralize: Add counterions to neutralize the solute charge.
        forcefield_files: OpenMM force-field XML files for solute and solvent.
        keep_ids: Preserve existing chain and residue IDs when writing. Newly
            added solvent makes generated IDs safer by default.

    Returns:
        Path to the modeled and solvated PDB file.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
        ValueError: If paths are identical or numeric parameters are invalid.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")
    if padding_nm <= 0:
        raise ValueError("padding_nm must be greater than zero")
    if ionic_strength_molar < 0:
        raise ValueError("ionic_strength_molar must not be negative")

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

    logger.info("Saved modeled PDB file to %s", output_path)
    return output_path


def minimize(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *,
    forcefield_files: Sequence[str] = (
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ),
    tolerance_kj_mol_nm: float = 10.0,
    max_iterations: int = 1000,
    nonbonded_cutoff_nm: float = 1.0,
    keep_ids: bool = False,
) -> Path:
    """Energy-minimize a PDB structure with an OpenMM force field.

    Periodic structures are minimized with PME electrostatics. Structures
    without periodic box vectors use ``NoCutoff`` instead.

    Args:
        input_file: Existing PDB structure to minimize.
        output_file: Destination for the minimized PDB structure.
        forcefield_files: OpenMM force-field XML files matching the model.
        tolerance_kj_mol_nm: RMS force tolerance in kJ/(mol nm).
        max_iterations: Maximum minimization iterations; zero means unlimited.
        nonbonded_cutoff_nm: Nonbonded cutoff for periodic systems in nanometers.
        keep_ids: Preserve valid chain and residue IDs in the output PDB.

    Returns:
        Path to the minimized PDB file.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
        ValueError: If paths are identical or numeric parameters are invalid.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")
    if tolerance_kj_mol_nm <= 0:
        raise ValueError("tolerance_kj_mol_nm must be greater than zero")
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")
    if nonbonded_cutoff_nm <= 0:
        raise ValueError("nonbonded_cutoff_nm must be greater than zero")

    logger.info("Energy-minimizing PDB file %s", input_path)
    pdb = PDBFile(str(input_path))
    forcefield = ForceField(*forcefield_files)
    system_options = {"constraints": HBonds}
    if pdb.topology.getPeriodicBoxVectors() is None:
        system_options["nonbondedMethod"] = NoCutoff
    else:
        system_options["nonbondedMethod"] = PME
        system_options["nonbondedCutoff"] = nonbonded_cutoff_nm * nanometer

    system = forcefield.createSystem(pdb.topology, **system_options)
    integrator = VerletIntegrator(0.001 * picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(
        tolerance=tolerance_kj_mol_nm * kilojoule_per_mole / nanometer,
        maxIterations=max_iterations,
    )
    positions = simulation.context.getState(positions=True).getPositions()

    with output_path.open("w") as output:
        PDBFile.writeFile(
            simulation.topology,
            positions,
            output,
            keepIds=keep_ids,
        )

    logger.info("Saved minimized PDB file to %s", output_path)
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
