"""Coordinate, contact, centering, and orientation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import Vector

from .chains import extract_chain

if TYPE_CHECKING:
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

ResidueCoordinates: TypeAlias = tuple[int, str, list[Vector]]
InteractionRecord: TypeAlias = list[int | str | float]
FloatArray: TypeAlias = NDArray[np.floating[Any]]


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


def get_interaction_residues_full(
    struc: Structure,
    chain_a: str,
    chain_b: str,
    cutoff: float = 5.0,
) -> list[InteractionRecord]:
    """Find interacting residues using a full pairwise distance search.

    This is the original brute-force implementation. Prefer
    :func:`get_interaction_residues` for larger structures.

    Args:
        struc: Structure containing both requested chains.
        chain_a: ID of the first interacting chain.
        chain_b: ID of the second interacting chain.
        cutoff: Maximum atom-to-atom contact distance in angstroms.

    Returns:
        Contact records containing residue numbers, residue names, and minimum
        atom-to-atom distances.

    Raises:
        Exception: If either requested chain does not exist.
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


def get_interaction_residues(
    struc: Structure,
    chain_a: str,
    chain_b: str,
    cutoff: float = 5.0,
) -> list[InteractionRecord]:
    """Find interacting residues using a KD-tree neighbor search.

    All amino-acid residues from chains with the requested IDs are considered.
    For each residue pair with at least one atom pair inside ``cutoff``, the
    smallest atom-to-atom distance is returned.

    Args:
        struc: Structure containing both requested chains.
        chain_a: ID of the first interacting chain.
        chain_b: ID of the second interacting chain.
        cutoff: Maximum atom-to-atom contact distance in angstroms.

    Returns:
        Contact records containing residue numbers, residue names, and minimum
        atom-to-atom distances.

    Raises:
        ValueError: If either requested chain does not exist.
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
    min_distances: dict[tuple[Residue, Residue], float] = {}

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
) -> list[dict[str, Any]]:
    """Characterize noncovalent interactions between two chains.

    Args:
        structure: Biopython structure containing the requested chains.
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        atomic: Return atom-level interactions when True, otherwise residue-level.
        hydrogen_bond: Include hydrogen bonds.
        salt_bridge: Include salt bridges.
        hydrophobic_contact: Include hydrophobic contacts.
        van_der_waals_contact: Include van der Waals contacts.
        pi_stacking_parallel: Include parallel pi-stacking candidates.
        cation_pi_candidate: Include cation-pi candidates.
        water_bridge: Include water bridges mediated by the same water molecule.

    Returns:
        A list of normalized interaction records.

    Raises:
        ValueError: If either requested chain does not exist.
    """
    from Bio.PDB import Atom

    _, chain_obj_a = next(
        ((model, chain) for model in structure for chain in model if chain.id == chain_a),
        (None, None),
    )
    _, chain_obj_b = next(
        ((model, chain) for model in structure for chain in model if chain.id == chain_b),
        (None, None),
    )
    if chain_obj_a is None:
        raise ValueError(f"Chain {chain_a!r} not found in structure")
    if chain_obj_b is None:
        raise ValueError(f"Chain {chain_b!r} not found in structure")

    def atom_element(atom: Atom.Atom) -> str:
        element = getattr(atom, "element", None)
        if element:
            element = str(element).strip().upper()
        if not element:
            element = atom.get_id().strip()[0].upper()
        return element[0] if element else ""

    def coord(atom: Atom.Atom) -> np.ndarray:
        return np.asarray(atom.get_vector().get_array(), dtype=float)

    def distance(atom_a: Atom.Atom, atom_b: Atom.Atom) -> float:
        return float(np.linalg.norm(coord(atom_a) - coord(atom_b)))

    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        cos_value = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_value)))

    def is_water(residue: "Residue") -> bool:
        return residue.get_resname().strip() in {"HOH", "WAT", "H2O"}

    def residue_atoms(residue: "Residue") -> list[Atom.Atom]:
        return [atom for atom in residue.get_atoms() if atom_element(atom) != ""]

    def build_covalent_map(residue: "Residue") -> dict[Atom.Atom, list[Atom.Atom]]:
        atoms = residue_atoms(residue)
        neighbors: dict[Atom.Atom, list[Atom.Atom]] = {atom: [] for atom in atoms}
        for i, atom_i in enumerate(atoms):
            elem_i = atom_element(atom_i)
            pos_i = coord(atom_i)
            for atom_j in atoms[i + 1 :]:
                elem_j = atom_element(atom_j)
                pos_j = coord(atom_j)
                dist = float(np.linalg.norm(pos_i - pos_j))
                if dist <= 1.9:
                    if {elem_i, elem_j} <= {"H", "C", "N", "O", "S", "P"}:
                        neighbors[atom_i].append(atom_j)
                        neighbors[atom_j].append(atom_i)
        return neighbors

    def has_hydrogen_neighbor(atom: Atom.Atom, cov_map: dict[Atom.Atom, list[Atom.Atom]]) -> bool:
        return any(atom_element(neighbor) == "H" for neighbor in cov_map.get(atom, []))

    def is_hydrophobic_atom(atom: Atom.Atom, cov_map: dict[Atom.Atom, list[Atom.Atom]]) -> bool:
        element = atom_element(atom)
        if element == "C":
            neighbors = cov_map.get(atom, [])
            return bool(neighbors) and all(atom_element(neighbor) in {"C", "H"} for neighbor in neighbors)
        if element == "S":
            residue_name = atom.get_parent().get_resname().strip()
            return residue_name in {"MET", "CYS"}
        return False

    VDW_RADII = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}

    def vdw_contact(atom_a: Atom.Atom, atom_b: Atom.Atom) -> bool:
        if atom_element(atom_a) == "H" or atom_element(atom_b) == "H":
            return False
        radius_a = VDW_RADII.get(atom_element(atom_a))
        radius_b = VDW_RADII.get(atom_element(atom_b))
        if radius_a is None or radius_b is None:
            return False
        return distance(atom_a, atom_b) <= radius_a + radius_b + 0.5

    RING_DEFINITIONS = {
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "HID": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "HIE": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "HIP": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "TRP": ["CD2", "CE2", "CE3", "CZ3", "CH2", "CZ2"],
    }

    def ring_centers(chain: Any) -> list[dict[str, Any]]:
        rings = []
        for residue in chain.get_residues():
            residue_name = residue.get_resname().strip()
            atom_names = RING_DEFINITIONS.get(residue_name)
            if not atom_names:
                continue
            ring_atoms = [residue[atom_name] for atom_name in atom_names if atom_name in residue]
            if len(ring_atoms) < 3:
                continue
            coords = np.array([coord(atom) for atom in ring_atoms])
            center = coords.mean(axis=0)
            _, _, vh = np.linalg.svd(coords - center)
            normal = vh[-1]
            rings.append({
                "residue": residue,
                "center": center,
                "normal": normal / np.linalg.norm(normal),
                "atoms": ring_atoms,
            })
        return rings

    def cation_groups(chain: Any) -> list[dict[str, Any]]:
        groups = []
        residues = [residue for residue in chain.get_residues() if is_aa(residue)]
        if residues:
            first = residues[0]
            if "N" in first:
                groups.append(
                    {
                        "residue": first,
                        "center": coord(first["N"]),
                        "charge": 1,
                        "type": "terminal_n",
                    }
                )
            last = residues[-1]
            if "O" in last:
                groups.append(
                    {
                        "residue": last,
                        "center": coord(last["O"]),
                        "charge": -1,
                        "type": "terminal_c",
                    }
                )
        for residue in residues:
            name = residue.get_resname().strip()
            if name == "LYS" and "NZ" in residue:
                groups.append(
                    {
                        "residue": residue,
                        "center": coord(residue["NZ"]),
                        "charge": 1,
                        "type": "lysine",
                    }
                )
            elif name == "ARG" and all(atom_name in residue for atom_name in ("NH1", "NH2")):
                groups.append(
                    {
                        "residue": residue,
                        "center": (
                            coord(residue["NH1"]) + coord(residue["NH2"])) / 2.0,
                        "charge": 1,
                        "type": "arginine",
                    }
                )
            elif name in {"HIS", "HID", "HIE", "HIP"} and all(atom_name in residue for atom_name in ("ND1", "NE2")):
                groups.append(
                    {
                        "residue": residue,
                        "center": (
                            coord(residue["ND1"]) + coord(residue["NE2"])) / 2.0,
                        "charge": 1,
                        "type": "histidine",
                    }
                )
        return groups

    def anion_groups(chain: Any) -> list[dict[str, Any]]:
        groups = []
        residues = [residue for residue in chain.get_residues() if is_aa(residue)]
        if residues:
            last = residues[-1]
            if "O" in last:
                oxygen_atoms = [atom for atom in last if atom.get_id().strip().startswith("O")]
                if oxygen_atoms:
                    center = np.mean([coord(atom) for atom in oxygen_atoms], axis=0)
                    groups.append(
                        {
                            "residue": last,
                            "center": center,
                            "charge": -1,
                            "type": "terminal_carboxylate",
                        }
                    )
        for residue in residues:
            name = residue.get_resname().strip()
            if name == "ASP" and all(atom_name in residue for atom_name in ("OD1", "OD2")):
                groups.append(
                    {
                        "residue": residue,
                        "center": (
                            coord(residue["OD1"]) + coord(residue["OD2"])) / 2.0,
                        "charge": -1,
                        "type": "aspartate",
                    }
                )
            elif name == "GLU" and all(atom_name in residue for atom_name in ("OE1", "OE2")):
                groups.append(
                    {
                        "residue": residue,
                        "center": (
                            coord(residue["OE1"]) + coord(residue["OE2"])) / 2.0,
                        "charge": -1,
                        "type": "glutamate",
                    }
                )
        return groups

    def representative_record(
        interaction_type: str,
        residue_a: "Residue",
        residue_b: "Residue",
        atom_a: Atom.Atom | None,
        atom_b: Atom.Atom | None,
        distance_value: float,
        angle_value: float | None = None,
        geometry: str | None = None,
        mediator: str | None = None,
    ) -> dict[str, Any]:
        return {
            "structure_id": getattr(structure, "id", None),
            "interaction_type": interaction_type,
            "chain_a": chain_a,
            "chain_b": chain_b,
            "residue_a_num": residue_a.id[1],
            "residue_a_name": residue_a.get_resname(),
            "atom_a": atom_a.get_id() if atom_a is not None else None,
            "residue_b_num": residue_b.id[1],
            "residue_b_name": residue_b.get_resname(),
            "atom_b": atom_b.get_id() if atom_b is not None else None,
            "distance": float(distance_value),
            "angle": float(angle_value) if angle_value is not None else None,
            "geometry": geometry,
            "mediator": mediator,
        }

    covalent_maps = {
        residue: build_covalent_map(residue)
        for residue in list(chain_obj_a.get_residues()) + list(chain_obj_b.get_residues())
    }

    interactions: list[dict[str, Any]] = []

    if hydrogen_bond:
        donors_a = []
        donors_b = []
        acceptors_a = []
        acceptors_b = []
        for residue in chain_obj_a.get_residues():
            for atom in residue_atoms(residue):
                if atom_element(atom) in {"N", "O"} and has_hydrogen_neighbor(atom, covalent_maps[residue]):
                    donors_a.append((residue, atom))
                if atom_element(atom) in {"O", "N"}:
                    acceptors_a.append((residue, atom))
        for residue in chain_obj_b.get_residues():
            for atom in residue_atoms(residue):
                if atom_element(atom) in {"N", "O"} and has_hydrogen_neighbor(atom, covalent_maps[residue]):
                    donors_b.append((residue, atom))
                if atom_element(atom) in {"O", "N"}:
                    acceptors_b.append((residue, atom))

        def add_hydrogen_bonds(donors, acceptors, direction: str) -> None:
            for donor_residue, donor_atom in donors:
                donor_hydrogens = [
                    hydrogen
                    for hydrogen in covalent_maps[donor_residue].get(donor_atom, [])
                    if atom_element(hydrogen) == "H"
                ]
                if not donor_hydrogens:
                    continue
                for acceptor_residue, acceptor_atom in acceptors:
                    if donor_residue is acceptor_residue:
                        continue
                    pair_distance = distance(donor_atom, acceptor_atom)
                    if pair_distance > 3.0:
                        continue
                    best_angle = 0.0
                    best_hydrogen = None
                    for hydrogen in donor_hydrogens:
                        angle_value = angle_between(
                            coord(hydrogen) - coord(donor_atom),
                            coord(acceptor_atom) - coord(donor_atom),
                        )
                        if angle_value > best_angle:
                            best_angle = angle_value
                            best_hydrogen = hydrogen
                    if best_angle >= 150.0 and best_hydrogen is not None:
                        interactions.append(
                            representative_record(
                                "hydrogen_bond",
                                donor_residue if direction == "a->b" else acceptor_residue,
                                acceptor_residue if direction == "a->b" else donor_residue,
                                donor_atom if direction == "a->b" else acceptor_atom,
                                acceptor_atom if direction == "a->b" else donor_atom,
                                pair_distance,
                                best_angle,
                                "D-H···A",
                                None,
                            )
                        )

        add_hydrogen_bonds(donors_a, acceptors_b, "a->b")
        add_hydrogen_bonds(donors_b, acceptors_a, "b->a")

    if salt_bridge:
        pos_groups_a = cation_groups(chain_obj_a)
        pos_groups_b = cation_groups(chain_obj_b)
        neg_groups_a = anion_groups(chain_obj_a)
        neg_groups_b = anion_groups(chain_obj_b)
        for pos_group, neg_group in (
            *((pa, nb) for pa in pos_groups_a for nb in neg_groups_b),
            *((pb, na) for pb in pos_groups_b for na in neg_groups_a),
        ):
            if np.linalg.norm(pos_group["center"] - neg_group["center"]) <= 5.5:
                interactions.append(
                    representative_record(
                        "salt_bridge",
                        pos_group["residue"],
                        neg_group["residue"],
                        None,
                        None,
                        float(np.linalg.norm(pos_group["center"] - neg_group["center"])),
                        None,
                        f"{pos_group['type']}/{neg_group['type']}",
                        None,
                    )
                )

    if hydrophobic_contact:
        for residue_a in chain_obj_a.get_residues():
            for residue_b in chain_obj_b.get_residues():
                for atom_a in residue_atoms(residue_a):
                    if not is_hydrophobic_atom(atom_a, covalent_maps[residue_a]):
                        continue
                    for atom_b in residue_atoms(residue_b):
                        if not is_hydrophobic_atom(atom_b, covalent_maps[residue_b]):
                            continue
                        if distance(atom_a, atom_b) <= 4.0:
                            interactions.append(
                                representative_record(
                                    "hydrophobic_contact",
                                    residue_a,
                                    residue_b,
                                    atom_a,
                                    atom_b,
                                    distance(atom_a, atom_b),
                                    None,
                                    None,
                                    None,
                                )
                            )

    if van_der_waals_contact:
        for residue_a in chain_obj_a.get_residues():
            for residue_b in chain_obj_b.get_residues():
                for atom_a in residue_atoms(residue_a):
                    for atom_b in residue_atoms(residue_b):
                        if vdw_contact(atom_a, atom_b):
                            interactions.append(
                                representative_record(
                                    "van_der_waals_contact",
                                    residue_a,
                                    residue_b,
                                    atom_a,
                                    atom_b,
                                    distance(atom_a, atom_b),
                                    None,
                                    None,
                                    None,
                                )
                            )

    if pi_stacking_parallel or cation_pi_candidate:
        rings_a = ring_centers(chain_obj_a)
        rings_b = ring_centers(chain_obj_b)
        for ring_a in rings_a:
            for ring_b in rings_b:
                center_dist = float(np.linalg.norm(ring_a["center"] - ring_b["center"]))
                plane_angle = angle_between(ring_a["normal"], ring_b["normal"])
                lateral_offset = float(
                    np.linalg.norm(
                        (ring_a["center"] - ring_b["center"]) -
                        np.dot(ring_a["center"] - ring_b["center"], ring_a["normal"]) * ring_a["normal"]
                    )
                )
                if pi_stacking_parallel and center_dist <= 5.5 and plane_angle <= 30.0 and lateral_offset <= 2.0:
                    interactions.append(
                        representative_record(
                            "pi_stacking_parallel",
                            ring_a["residue"],
                            ring_b["residue"],
                            ring_a["atoms"][0],
                            ring_b["atoms"][0],
                            center_dist,
                            plane_angle,
                            "parallel",
                            None,
                        )
                    )
                if center_dist <= 5.5 and plane_angle >= 60.0:
                    interactions.append(
                        representative_record(
                            "pi_stacking_t_shaped",
                            ring_a["residue"],
                            ring_b["residue"],
                            ring_a["atoms"][0],
                            ring_b["atoms"][0],
                            center_dist,
                            plane_angle,
                            "t-shaped",
                            None,
                        )
                    )
                if cation_pi_candidate:
                    for cation_group in cation_groups(chain_obj_a):
                        if np.linalg.norm(cation_group["center"] - ring_b["center"]) <= 6.0:
                            interactions.append(
                                representative_record(
                                    "cation_pi_candidate",
                                    cation_group["residue"],
                                    ring_b["residue"],
                                    None,
                                    ring_b["atoms"][0],
                                    float(np.linalg.norm(cation_group["center"] - ring_b["center"])),
                                    None,
                                    "cation-pi",
                                    None,
                                )
                            )
                    for cation_group in cation_groups(chain_obj_b):
                        if np.linalg.norm(cation_group["center"] - ring_a["center"]) <= 6.0:
                            interactions.append(
                                representative_record(
                                    "cation_pi_candidate",
                                    cation_group["residue"],
                                    ring_a["residue"],
                                    None,
                                    ring_a["atoms"][0],
                                    float(np.linalg.norm(cation_group["center"] - ring_a["center"])),
                                    None,
                                    "cation-pi",
                                    None,
                                )
                            )

    if water_bridge:
        waters = [
            residue
            for model in structure
            for chain in model
            for residue in chain
            if is_water(residue)
        ]
        water_contacts: dict[tuple[int, int, str], list[str]] = {}
        for water in waters:
            o_atoms = [atom for atom in residue_atoms(water) if atom_element(atom) == "O"]
            h_atoms = [atom for atom in residue_atoms(water) if atom_element(atom) == "H"]
            if not o_atoms or not h_atoms:
                continue
            water_oxygen = o_atoms[0]
            for residue, atom in [
                (res, atm)
                for res in chain_obj_a.get_residues()
                for atm in residue_atoms(res)
                if atom_element(atm) in {"O", "N"}
            ]:
                if distance(water_oxygen, atom) <= 3.0:
                    for hydrogen in h_atoms:
                        if angle_between(coord(water_oxygen) - coord(hydrogen), coord(atom) - coord(water_oxygen)) >= 150.0:
                            water_contacts[(residue.id[1], atom.get_id(), chain_a)] = [water.get_resname()]
            for residue, atom in [
                (res, atm)
                for res in chain_obj_b.get_residues()
                for atm in residue_atoms(res)
                if atom_element(atm) in {"O", "N"}
            ]:
                if distance(water_oxygen, atom) <= 3.0:
                    for hydrogen in h_atoms:
                        if angle_between(coord(water_oxygen) - coord(hydrogen), coord(atom) - coord(water_oxygen)) >= 150.0:
                            water_contacts[(residue.id[1], atom.get_id(), chain_b)] = [water.get_resname()]
        if water_contacts:
            a_keys = [k for k in water_contacts if k[2] == chain_a]
            b_keys = [k for k in water_contacts if k[2] == chain_b]
            for a_key in a_keys:
                for b_key in b_keys:
                    interactions.append(
                        {
                            "structure_id": getattr(structure, "id", None),
                            "interaction_type": "water_bridge",
                            "chain_a": chain_a,
                            "chain_b": chain_b,
                            "residue_a_num": a_key[0],
                            "residue_a_name": None,
                            "atom_a": a_key[1],
                            "residue_b_num": b_key[0],
                            "residue_b_name": None,
                            "atom_b": b_key[1],
                            "distance": None,
                            "angle": None,
                            "geometry": "water-mediated",
                            "mediator": ",".join(water_contacts[a_key] + water_contacts[b_key]),
                        }
                    )

    if not atomic:
        unique: dict[tuple[int, str, int, str, str], dict[str, Any]] = {}
        for record in interactions:
            key = (
                record["residue_a_num"],
                record["residue_a_name"],
                record["residue_b_num"],
                record["residue_b_name"],
                record["interaction_type"],
            )
            existing = unique.get(key)
            if existing is None or record["distance"] is not None and (
                existing["distance"] is None or record["distance"] < existing["distance"]
            ):
                unique[key] = record
        interactions = list(unique.values())

    return interactions
