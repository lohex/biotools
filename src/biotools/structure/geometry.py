"""Coordinate, contact, centering, and orientation utilities."""

import numpy as np
from Bio.PDB.Polypeptide import is_aa

from .chains import extract_chain


def get_residue_coords(structure, c_alpha: bool = False):
    """Collect residue-wise coordinate vectors from a structure."""
    centers = []
    for residue in structure.get_residues():
        if not is_aa(residue):
            continue
        atoms = list(residue.get_atoms())
        if c_alpha:
            atoms = [atom for atom in atoms if atom.id == "CA"]
        centers.append(
            (
                residue.id[1],
                residue.get_resname(),
                [atom.get_vector() for atom in atoms],
            )
        )
    return centers


def get_min_dist(atom_list_a, atom_list_b, cutoff=30.0):
    """Compute the minimal Euclidean distance between two atom lists."""
    min_dist = np.inf
    for atom_a in atom_list_a:
        for atom_b in atom_list_b:
            distance = np.linalg.norm(atom_a - atom_b)
            if distance < min_dist:
                min_dist = distance
                if min_dist == 0.0:
                    return 0.0
    return float(min_dist)


def get_interaction_residues(struc, chain_a, chain_b, cutoff=5.0):
    """Find residue pairs across two chains within a distance cutoff."""
    atoms_a = get_residue_coords(extract_chain(struc, chain_a))
    atoms_b = get_residue_coords(extract_chain(struc, chain_b))
    interactions = []
    for res_a, type_a, vectors_a in atoms_a:
        for res_b, type_b, vectors_b in atoms_b:
            distance = get_min_dist(vectors_a, vectors_b)
            if distance <= cutoff:
                interactions.append([res_a, type_a, res_b, type_b, distance])
    return interactions


def move_to_center(structure):
    """Translate a structure copy so its center of mass is at the origin."""
    center = structure.center_of_mass()
    centered = structure.copy()
    for residue in centered.get_residues():
        residue.transform(np.eye(3), -center)
    return centered


def superimpose_PCA(structure, apply_rot=True, apply_shift=True):
    """Reorient a structure along its principal component axes."""
    coords = get_residue_coords(structure, c_alpha=True)
    coordinate_matrix = np.array([vector.get_array() for _, _, (vector,) in coords])
    shift = -coordinate_matrix.mean(0)
    centered_matrix = coordinate_matrix + shift
    _, eigenvectors = np.linalg.eig(np.cov(centered_matrix.T))
    rotation = np.linalg.inv(eigenvectors)

    transformed = structure.copy()
    for residue in transformed.get_residues():
        if apply_shift:
            residue.transform(np.eye(3), shift)
        if apply_rot:
            residue.transform(rotation, np.zeros(3))
    return transformed, shift, rotation
