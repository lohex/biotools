"""Coordinate, contact, centering, and orientation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast, Literal, TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import Vector

if TYPE_CHECKING:
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

ResidueCoordinates: TypeAlias = tuple[int, str, list[Vector]]
InteractionRecord: TypeAlias = list[int | str | float]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
DistanceMetric: TypeAlias = Literal[
    "c_alpha",
    "c_beta",
    "min_heavy_atom",
    "min_atom",
]

_DISTANCE_METRICS = {
    "c_alpha",
    "c_beta",
    "min_heavy_atom",
    "min_atom",
}


def get_residue_coords(
    structure: Structure,
    c_alpha: bool = False,
) -> list[ResidueCoordinates]:
    """Collect residue-wise coordinate vectors from a structure.

    Args:
        structure: Structure containing residues and atoms.
        c_alpha: Include only C-alpha atoms for each residue.

    Returns:
        Tuples of residue number, residue name, and coordinate vectors for each
        amino-acid residue.
    """
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


def get_min_dist(
    atom_list_a: Iterable[Vector],
    atom_list_b: Iterable[Vector],
    cutoff: float = 30.0,
) -> float:
    """Compute the minimal Euclidean distance between two coordinate lists.

    Args:
        atom_list_a: Coordinate vectors in the first group.
        atom_list_b: Coordinate vectors in the second group.
        cutoff: Retained for API compatibility; it does not restrict the full
            distance calculation.

    Returns:
        Minimum pairwise distance, or positive infinity if a group is empty.
    """
    min_dist = np.inf
    for atom_a in atom_list_a:
        for atom_b in atom_list_b:
            distance = np.linalg.norm(atom_a - atom_b)
            if distance < min_dist:
                min_dist = distance
                if min_dist == 0.0:
                    return 0.0
    return float(min_dist)


def _validate_distance_metric(distance_metric: str) -> DistanceMetric:
    if distance_metric not in _DISTANCE_METRICS:
        choices = ", ".join(sorted(_DISTANCE_METRICS))
        raise ValueError(
            f"Unsupported distance_metric {distance_metric!r}; choose from "
            f"{choices}"
        )
    return cast(DistanceMetric, distance_metric)


def _is_heavy_atom(atom: Any) -> bool:
    element = (getattr(atom, "element", None) or "").strip().upper()
    if element:
        return element not in {"H", "D"}
    atom_name = atom.get_name().strip().upper()
    return not atom_name.startswith(("H", "D"))


def _distance_atoms(
    residue: Residue,
    distance_metric: DistanceMetric,
) -> list[Any]:
    if distance_metric == "c_alpha":
        return [residue["CA"]] if residue.has_id("CA") else []
    if distance_metric == "c_beta":
        atom_name = "CA" if residue.get_resname().upper() == "GLY" else "CB"
        return [residue[atom_name]] if residue.has_id(atom_name) else []

    atoms = list(residue.get_atoms())
    if distance_metric == "min_heavy_atom":
        return [atom for atom in atoms if _is_heavy_atom(atom)]
    return atoms


def _chain_residues(
    structure: Structure,
    chain_id: str,
) -> list[Residue]:
    chains = [chain for chain in structure.get_chains() if chain.id == chain_id]
    if not chains:
        raise ValueError(f"Chain {chain_id!r} not found in structure")
    return [
        residue
        for chain in chains
        for residue in chain.get_residues()
        if is_aa(residue)
    ]


def get_interaction_residues_full(
    struc: Structure,
    chain_a: str,
    chain_b: str,
    cutoff: float = 5.0,
    distance_metric: DistanceMetric = "min_heavy_atom",
) -> list[InteractionRecord]:
    """Find interacting residues using a full pairwise distance search.

    This is the original brute-force implementation. Prefer
    :func:`get_interaction_residues` for larger structures.

    Args:
        struc: Structure containing both requested chains.
        chain_a: ID of the first interacting chain.
        chain_b: ID of the second interacting chain.
        cutoff: Maximum distance in angstroms. C-alpha and C-beta metrics
            commonly use a larger cutoff, such as 8 angstroms.
        distance_metric: Residue distance definition. ``"c_alpha"`` uses
            C-alpha atoms, ``"c_beta"`` uses C-beta atoms with a C-alpha
            fallback for glycine, ``"min_heavy_atom"`` uses the minimum
            non-hydrogen atom distance, and ``"min_atom"`` retains the former
            all-atom behavior.

    Returns:
        Contact records containing residue numbers, residue names, and minimum
        atom-to-atom distances.

    Raises:
        ValueError: If a requested chain or distance metric is invalid.
    """
    distance_metric = _validate_distance_metric(distance_metric)
    residues_a = _chain_residues(struc, chain_a)
    residues_b = _chain_residues(struc, chain_b)
    interactions = []
    for residue_a in residues_a:
        vectors_a = [
            atom.get_vector()
            for atom in _distance_atoms(residue_a, distance_metric)
        ]
        if not vectors_a:
            continue
        for residue_b in residues_b:
            vectors_b = [
                atom.get_vector()
                for atom in _distance_atoms(residue_b, distance_metric)
            ]
            if not vectors_b:
                continue
            distance = get_min_dist(vectors_a, vectors_b)
            if distance <= cutoff:
                interactions.append(
                    [
                        residue_a.id[1],
                        residue_a.get_resname(),
                        residue_b.id[1],
                        residue_b.get_resname(),
                        distance,
                    ]
                )
    return interactions


def get_interaction_residues(
    struc: Structure,
    chain_a: str,
    chain_b: str,
    cutoff: float = 5.0,
    distance_metric: DistanceMetric = "min_heavy_atom",
) -> list[InteractionRecord]:
    """Find interacting residues using a KD-tree neighbor search.

    All amino-acid residues from chains with the requested IDs are considered.
    For each residue pair with a selected atom pair inside ``cutoff``, the
    smallest distance according to ``distance_metric`` is returned.

    Args:
        struc: Structure containing both requested chains.
        chain_a: ID of the first interacting chain.
        chain_b: ID of the second interacting chain.
        cutoff: Maximum distance in angstroms. C-alpha and C-beta metrics
            commonly use a larger cutoff, such as 8 angstroms.
        distance_metric: Residue distance definition. ``"c_alpha"`` uses
            C-alpha atoms, ``"c_beta"`` uses C-beta atoms with a C-alpha
            fallback for glycine, ``"min_heavy_atom"`` uses the minimum
            non-hydrogen atom distance, and ``"min_atom"`` retains the former
            all-atom behavior.

    Returns:
        Contact records containing residue numbers, residue names, and minimum
        atom-to-atom distances.

    Raises:
        ValueError: If a requested chain or distance metric is invalid.
    """
    distance_metric = _validate_distance_metric(distance_metric)
    residues_a = _chain_residues(struc, chain_a)
    residues_b = _chain_residues(struc, chain_b)

    if not residues_a or not residues_b:
        return []

    atoms_by_residue_b = {
        residue: _distance_atoms(residue, distance_metric)
        for residue in residues_b
    }
    atoms_b = [
        atom
        for residue in residues_b
        for atom in atoms_by_residue_b[residue]
    ]
    if not atoms_b:
        return []
    atom_residues_b = {
        id(atom): residue
        for residue, atoms in atoms_by_residue_b.items()
        for atom in atoms
    }
    neighbor_search = NeighborSearch(atoms_b)
    min_distances: dict[tuple[Residue, Residue], float] = {}

    for residue_a in residues_a:
        for atom_a in _distance_atoms(residue_a, distance_metric):
            for atom_b in neighbor_search.search(atom_a.coord, cutoff, level="A"):
                residue_b = atom_residues_b[id(atom_b)]
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


def move_to_center(structure: Structure) -> Structure:
    """Translate a structure copy so its center of mass is at the origin.

    Args:
        structure: Structure to center without modifying the input.

    Returns:
        Centered copy of ``structure``.
    """
    center = structure.center_of_mass()
    centered = structure.copy()
    for residue in centered.get_residues():
        residue.transform(np.eye(3), -center)
    return centered


def superimpose_PCA(
    structure: Structure,
    apply_rot: bool = True,
    apply_shift: bool = True,
) -> tuple[Structure, FloatArray, FloatArray]:
    """Reorient a structure along its principal component axes.

    Principal components are calculated from C-alpha coordinates.

    Args:
        structure: Structure to copy and transform.
        apply_rot: Apply the principal-axis rotation to the copy.
        apply_shift: Center the C-alpha coordinates before rotation.

    Returns:
        Transformed structure copy, calculated translation vector, and
        principal-axis rotation matrix.

    Raises:
        ValueError: If no usable C-alpha coordinates are present.
    """
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


def characterize_chain_contacts(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    atomic: bool = False,
    hydrogen_bond: bool = True,
    salt_bridge: bool = True,
    hydrophobic_contact: bool = True,
    van_der_waals_contact: bool = True,
    pi_stacking_parallel: bool = True,
    cation_pi_candidate: bool = True,
    water_bridge: bool = True,
    pi_stacking_t_shaped: bool = True,
    topology_backend: str = "templates",
) -> list[dict[str, Any]]:
    """Characterize noncovalent interactions between two protein chains.

    The function applies transparent, snapshot-based geometric rules for
    hydrogen bonds, salt bridges, hydrophobic and van der Waals contacts,
    aromatic interactions, and single-water bridges. A detected contact is a
    geometric candidate, not a binding energy or evidence of temporal
    stability.

    Args:
        structure: Biopython structure containing the requested chains.
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        atomic: Return every atom/group observation when True. Otherwise
            return one representative observation per residue pair and type.
        hydrogen_bond: Include direct hydrogen bonds.
        salt_bridge: Include salt bridges.
        hydrophobic_contact: Include conservative hydrophobic contacts.
        van_der_waals_contact: Include heavy-atom van der Waals contacts.
        pi_stacking_parallel: Include parallel pi-stacking candidates.
        cation_pi_candidate: Include cation-pi candidates.
        water_bridge: Include bridges mediated by the same explicit water.
        pi_stacking_t_shaped: Include T-shaped pi-stacking candidates.
        topology_backend: Bond-topology provider. Use ``"templates"`` for the
            built-in protein templates or ``"openmm"`` with the optional
            ``contacts`` dependencies.

    Returns:
        Normalized interaction dictionaries whose A/B orientation always
        matches chain_a and chain_b.

    Raises:
        ValueError: If the chains are absent, ambiguous, identical, or occur
            in different models.
    """
    from ._contacts import characterize_chain_contacts_impl

    return characterize_chain_contacts_impl(
        structure,
        chain_a,
        chain_b,
        atomic=atomic,
        hydrogen_bond=hydrogen_bond,
        salt_bridge=salt_bridge,
        hydrophobic_contact=hydrophobic_contact,
        van_der_waals_contact=van_der_waals_contact,
        pi_stacking_parallel=pi_stacking_parallel,
        cation_pi_candidate=cation_pi_candidate,
        water_bridge=water_bridge,
        pi_stacking_t_shaped=pi_stacking_t_shaped,
        topology_backend=topology_backend,
    )


def characterize_intrachain_contacts(
    structure: Structure,
    chain: str,
    *,
    min_sequence_separation: int = 2,
    hydrogen_bond: bool = True,
    salt_bridge: bool = True,
    hydrophobic_contact: bool = True,
    van_der_waals_contact: bool = True,
    pi_stacking_parallel: bool = True,
    cation_pi_candidate: bool = True,
    water_bridge: bool = True,
    pi_stacking_t_shaped: bool = True,
    topology_backend: str = "templates",
) -> list[dict[str, Any]]:
    """Characterize noncovalent interactions within one protein chain.

    Residue pairs are unique and oriented in chain order. Self-pairs and pairs
    closer than ``min_sequence_separation`` positions in the chain are omitted.
    The default value of two therefore excludes directly adjacent residues.

    Args:
        structure: Biopython structure containing the requested chain.
        chain: Chain identifier.
        min_sequence_separation: Minimum difference between residue indices.
        hydrogen_bond: Include direct hydrogen bonds.
        salt_bridge: Include salt bridges.
        hydrophobic_contact: Include conservative hydrophobic contacts.
        van_der_waals_contact: Include heavy-atom van der Waals contacts.
        pi_stacking_parallel: Include parallel pi-stacking candidates.
        cation_pi_candidate: Include cation-pi candidates.
        water_bridge: Include bridges mediated by the same explicit water.
        pi_stacking_t_shaped: Include T-shaped pi-stacking candidates.
        topology_backend: Built-in ``"templates"`` or optional ``"openmm"``.

    Returns:
        One representative record per residue pair and interaction type.

    Raises:
        ValueError: If the chain, model, separation, or backend is invalid.
    """
    from ._contacts import characterize_intrachain_contacts_impl

    return characterize_intrachain_contacts_impl(
        structure,
        chain,
        min_sequence_separation=min_sequence_separation,
        hydrogen_bond=hydrogen_bond,
        salt_bridge=salt_bridge,
        hydrophobic_contact=hydrophobic_contact,
        van_der_waals_contact=van_der_waals_contact,
        pi_stacking_parallel=pi_stacking_parallel,
        cation_pi_candidate=cation_pi_candidate,
        water_bridge=water_bridge,
        pi_stacking_t_shaped=pi_stacking_t_shaped,
        topology_backend=topology_backend,
    )
