"""RMSD calculation and structural superposition helpers."""

from Bio.PDB import Superimposer

from ..sequence.alignment import global_alignment_seqs
from .chains import (
    _get_chain,
    _validate_chain_for_protein_alignment,
    extract_chain,
    get_aa_sequence,
)


def get_rmsd(structure1, structure2, ca_only=True):
    """Compute RMSD between two structures using Biopython superposition."""
    if ca_only:
        atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
        atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]
    else:
        atoms1 = structure1.get_atoms()
        atoms2 = structure2.get_atoms()
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    return superimposer.rms


def align_structure(structure1, structure2):
    """Align one structure onto another using C-alpha atoms."""
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    target = structure2.copy()
    superimposer.apply(target.get_atoms())
    return target


def _identify_homo_aa(gapped_a, gapped_b):
    """Identify corresponding ungapped residue indices in two alignments."""
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


def align_homologs(structure1, structure2, chain1, chain2):
    """Align a complete structure from corresponding residues in two chains."""
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


def get_alignment(structure1, structure2, chain1=None, chain2=None):
    """Create a Biopython superimposer for two structures or chains."""
    if chain1 is not None:
        structure1 = extract_chain(structure1, chain1)
    if chain2 is not None:
        structure2 = extract_chain(structure2, chain2)
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    return superimposer


def apply_transformation(superimposer, structure):
    """Apply a fitted Biopython superposition to a structure in place."""
    superimposer.apply(structure.get_atoms())
    return structure
