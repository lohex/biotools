"""Coordinate, contact, centering, and orientation utilities."""

import numpy as np
from Bio.PDB import NeighborSearch
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


def get_interaction_residues_full(struc, chain_a, chain_b, cutoff=5.0):
    """Find interacting residues using a full pairwise distance search.

    This is the original brute-force implementation. Prefer
    :func:`get_interaction_residues` for larger structures.
    """
    atoms_a = get_residue_coords(extract_chain(struc, chain_a))
    atoms_b = get_residue_coords(extract_chain(struc, chain_b))
    interactions = []
    for res_a, type_a, vectors_a in atoms_a:
        for res_b, type_b, vectors_b in atoms_b:
            distance = get_min_dist(vectors_a, vectors_b)
            if distance <= cutoff:
                interactions.append([res_a, type_a, res_b, type_b, distance])
    return interactions


def get_interaction_residues(struc, chain_a, chain_b, cutoff=5.0):
    """Find interacting residues using a KD-tree neighbor search.

    All amino-acid residues from chains with the requested IDs are considered.
    For each residue pair with at least one atom pair inside ``cutoff``, the
    smallest atom-to-atom distance is returned.
    """
    chains_a = [chain for chain in struc.get_chains() if chain.id == chain_a]
    chains_b = [chain for chain in struc.get_chains() if chain.id == chain_b]
    if not chains_a:
        raise ValueError(f"Chain {chain_a!r} not found in structure")
    if not chains_b:
        raise ValueError(f"Chain {chain_b!r} not found in structure")

    residues_a = [
        residue
        for chain in chains_a
        for residue in chain.get_residues()
        if is_aa(residue)
    ]
    residues_b = [
        residue
        for chain in chains_b
        for residue in chain.get_residues()
        if is_aa(residue)
    ]

    if not residues_a or not residues_b:
        return []

    atoms_b = [atom for residue in residues_b for atom in residue.get_atoms()]
    neighbor_search = NeighborSearch(atoms_b)
    min_distances = {}

    for residue_a in residues_a:
        for atom_a in residue_a.get_atoms():
            for atom_b in neighbor_search.search(atom_a.coord, cutoff, level="A"):
                residue_b = atom_b.get_parent()
                key = (residue_a, residue_b)
                distance = atom_a - atom_b
                if key not in min_distances or distance < min_distances[key]:
                    min_distances[key] = distance

    residue_order_a = {residue: index for index, residue in enumerate(residues_a)}
    residue_order_b = {residue: index for index, residue in enumerate(residues_b)}
    residue_pairs = sorted(
        min_distances,
        key=lambda pair: (residue_order_a[pair[0]], residue_order_b[pair[1]]),
    )
    return [
        [
            residue_a.id[1],
            residue_a.get_resname(),
            residue_b.id[1],
            residue_b.get_resname(),
            min_distances[(residue_a, residue_b)],
        ]
        for residue_a, residue_b in residue_pairs
    ]


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
