"""RMSD calculation and structural superposition helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from Bio.PDB import Superimposer

from ..sequence.alignment import global_alignment_seqs
from .chains import (
    _get_chain,
    _validate_chain_for_protein_alignment,
    extract_chain,
    get_aa_sequence,
)

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure


def get_rmsd(
    structure1: Structure,
    structure2: Structure,
    ca_only: bool = True,
) -> float:
    """Compute RMSD between two structures using Biopython superposition.

    Args:
        structure1: Fixed reference structure.
        structure2: Moving structure to compare with the reference.
        ca_only: Use only C-alpha atoms instead of every atom.

    Returns:
        Root-mean-square deviation after optimal superposition.

    Raises:
        PDBException: If the structures contain different atom counts.
    """
    if ca_only:
        atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
        atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]
    else:
        atoms1 = structure1.get_atoms()
        atoms2 = structure2.get_atoms()
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    return superimposer.rms


def align_structure(
    structure1: Structure,
    structure2: Structure,
) -> Structure:
    """Align one structure onto another using C-alpha atoms.

    Args:
        structure1: Fixed reference structure.
        structure2: Moving structure to transform.

    Returns:
        Transformed copy of ``structure2``; both inputs remain unchanged.

    Raises:
        PDBException: If the structures contain different C-alpha atom counts.
    """
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    target = structure2.copy()
    superimposer.apply(target.get_atoms())
    return target


def _identify_homo_aa(
    gapped_a: str,
    gapped_b: str,
) -> tuple[list[int], list[int]]:
    """Identify corresponding ungapped residue indices in two alignments.

    Args:
        gapped_a: First aligned sequence containing optional gap characters.
        gapped_b: Second aligned sequence containing optional gap characters.

    Returns:
        Parallel lists of corresponding zero-based ungapped residue indices.
    """
    a_pos = 0
    b_pos = 0
    a_select = []
    b_select = []
    for a, b in zip(gapped_a, gapped_b):
        if a != "-" and b != "-":
            a_select.append(a_pos)
            b_select.append(b_pos)
            a_pos += 1
            b_pos += 1
        elif a == "-":
            b_pos += 1
        elif b == "-":
            a_pos += 1
    return a_select, b_select


def align_homologs(
    structure1: Structure,
    structure2: Structure,
    chain1: str,
    chain2: str,
) -> Structure:
    """Align a complete structure from corresponding residues in two chains.

    Args:
        structure1: Fixed reference structure.
        structure2: Complete moving structure to transform.
        chain1: Reference chain used to calculate residue correspondence.
        chain2: Moving chain used to calculate residue correspondence.

    Returns:
        Transformed copy of the complete ``structure2``.

    Raises:
        ValueError: If a chain is absent or unsuitable for protein alignment.
        PDBException: If corresponding atom selections cannot be superimposed.
    """
    chain_obj_a = _get_chain(structure1, chain1)
    chain_obj_b = _get_chain(structure2, chain2)
    _validate_chain_for_protein_alignment(chain_obj_a)
    _validate_chain_for_protein_alignment(chain_obj_b)

    seqs_a = get_aa_sequence(structure1)
    seqs_b = get_aa_sequence(structure2)
    gapped_seq_a, gapped_seq_b = global_alignment_seqs(
        seqs_a[chain1],
        seqs_b[chain2],
    )
    selected_a, selected_b = _identify_homo_aa(gapped_seq_a, gapped_seq_b)
    atoms1 = [atom for atom in chain_obj_a.get_atoms() if atom.get_id() == "CA"]
    atoms2 = [atom for atom in chain_obj_b.get_atoms() if atom.get_id() == "CA"]
    atoms1_aligned = [atom for index, atom in enumerate(atoms1) if index in selected_a]
    atoms2_aligned = [atom for index, atom in enumerate(atoms2) if index in selected_b]

    superimposer = Superimposer()
    superimposer.set_atoms(atoms1_aligned, atoms2_aligned)
    target = structure2.copy()
    superimposer.apply(target.get_atoms())
    return target


def get_alignment(
    structure1: Structure,
    structure2: Structure,
    chain1: str | None = None,
    chain2: str | None = None,
) -> Superimposer:
    """Create a fitted Biopython superimposer for structures or chains.

    Args:
        structure1: Fixed reference structure.
        structure2: Moving structure.
        chain1: Optional reference chain selection.
        chain2: Optional moving chain selection.

    Returns:
        Superimposer fitted to the selected C-alpha atoms.

    Raises:
        Exception: If a requested chain does not exist.
        PDBException: If the selected C-alpha atom counts differ.
    """
    if chain1 is not None:
        structure1 = extract_chain(structure1, chain1)
    if chain2 is not None:
        structure2 = extract_chain(structure2, chain2)
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    return superimposer


def apply_transformation(
    superimposer: Superimposer,
    structure: Structure,
) -> Structure:
    """Apply a fitted Biopython superposition to a structure in place.

    Args:
        superimposer: Previously fitted transformation to apply.
        structure: Structure whose atom coordinates are modified.

    Returns:
        The same transformed structure object.
    """
    superimposer.apply(structure.get_atoms())
    return structure
