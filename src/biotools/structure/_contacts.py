"""Protein-specific geometric contact characterization.

The public entry point lives in :mod:`biotools.structure.geometry`.  This
module contains the chemistry and geometry primitives so that the individual
contact detectors can be tested and maintained independently.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import is_aa

if TYPE_CHECKING:
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure


HBOND_DISTANCE_A = 3.0
HBOND_ANGLE_DEG = 150.0
SALT_BRIDGE_DISTANCE_A = 5.5
HYDROPHOBIC_DISTANCE_A = 4.0
VDW_TOLERANCE_A = 0.5
PI_STACKING_DISTANCE_A = 5.5
PI_PARALLEL_ANGLE_DEG = 30.0
PI_TSHAPED_ANGLE_DEG = 60.0
PI_PARALLEL_OFFSET_A = 2.0
CATION_PI_DISTANCE_A = 6.0

WATER_RESIDUE_NAMES = {"HOH", "WAT", "H2O", "SOL", "TIP3", "TIP3P"}
VDW_RADII_A = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}

ACCEPTOR_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "MET": {"SD"},
}

AROMATIC_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"))
SIDECHAIN_BONDS = {
    "ALA": (("CA", "CB"),),
    "ARG": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
            ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")),
    "ASN": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "ASP": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "CYS": (("CA", "CB"), ("CB", "SG")),
    "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"),
            ("CD", "NE2")),
    "GLU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"),
            ("CD", "OE2")),
    "GLY": (),
    "HIS": (("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"),
            ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")),
    "ILE": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    "LEU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"),
            ("CE", "NZ")),
    "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "PHE": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")),
    "PRO": (("N", "CD"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
    "SER": (("CA", "CB"), ("CB", "OG")),
    "THR": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
    "TRP": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"), ("CD2", "CE3"),
            ("CE3", "CZ3"), ("CZ3", "CH2"), ("CH2", "CZ2"), ("CZ2", "CE2")),
    "TYR": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
            ("CZ", "OH")),
    "VAL": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
}

RESIDUE_BOND_ALIASES = {
    "ASH": "ASP", "GLH": "GLU", "CYM": "CYS", "CYX": "CYS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "LYN": "LYS",
}


@dataclass(frozen=True)
class _Observation:
    interaction_type: str
    residue_a: Any
    residue_b: Any
    atom_a: str | None
    atom_b: str | None
    distance: float | None
    angle: float | None = None
    geometry: str | None = None
    mediator: str | None = None


@dataclass(frozen=True)
class _ChargedGroup:
    residue: Any
    sign: int
    atoms: tuple[Any, ...]
    atom_label: str
    group_label: str
    center: np.ndarray


@dataclass(frozen=True)
class _AromaticRing:
    residue: Any
    atoms: tuple[Any, ...]
    center: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class _WaterLeg:
    water: Any
    protein_atom: Any
    distance: float
    angle: float
    direction: str


def _element(atom: Atom) -> str:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element == "D":
        return "H"
    if element:
        return element
    name = str(atom.get_name()).strip().upper()
    return name[0] if name else ""


def _coord(atom: Atom) -> np.ndarray:
    return np.asarray(atom.get_coord(), dtype=float)


def _distance(atom_a: Atom, atom_b: Atom) -> float:
    return float(np.linalg.norm(_coord(atom_a) - _coord(atom_b)))


def _angle_degrees(point1: np.ndarray, vertex: np.ndarray, point3: np.ndarray) -> float:
    vector1 = np.asarray(point1) - np.asarray(vertex)
    vector2 = np.asarray(point3) - np.asarray(vertex)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if denominator == 0.0:
        return float("nan")
    cosine = np.clip(np.dot(vector1, vector2) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _residue_name(residue: Residue) -> str:
    return residue.get_resname().strip().upper()


def _atoms_by_name(residue: Residue) -> dict[str, Atom]:
    return {str(atom.get_name()).strip(): atom for atom in residue.get_atoms()}


def _add_bond(adjacency: dict[Any, list[Any]], atom1: Any, atom2: Any) -> None:
    if atom2 not in adjacency[atom1]:
        adjacency[atom1].append(atom2)
    if atom1 not in adjacency[atom2]:
        adjacency[atom2].append(atom1)


def _build_template_bond_adjacency(
    residues: list[Residue],
) -> dict[Any, list[Any]]:
    """Build a conservative protein bond graph from standard residue templates.

    Biopython structures do not retain PDB topology bonds.  Heavy-atom bonds
    are therefore assigned by residue templates; explicit hydrogens are bound
    only to their nearest plausible heavy atom at a covalent H-X distance.
    This avoids the previous generic 1.9 A rule, which created false heavy-atom
    bonds in close contacts.
    """
    adjacency: dict[Any, list[Any]] = defaultdict(list)
    for residue in residues:
        atoms = list(residue.get_atoms())
        for atom in atoms:
            adjacency[atom]
        named = _atoms_by_name(residue)
        if _residue_name(residue) in WATER_RESIDUE_NAMES:
            oxygen = next((atom for atom in atoms if _element(atom) == "O"), None)
            if oxygen is not None:
                for hydrogen in atoms:
                    if _element(hydrogen) == "H":
                        _add_bond(adjacency, oxygen, hydrogen)
            continue

        residue_name = RESIDUE_BOND_ALIASES.get(_residue_name(residue), _residue_name(residue))
        for name1, name2 in BACKBONE_BONDS + SIDECHAIN_BONDS.get(residue_name, ()):
            if name1 in named and name2 in named:
                _add_bond(adjacency, named[name1], named[name2])

        heavy_atoms = [atom for atom in atoms if _element(atom) != "H"]
        for hydrogen in (atom for atom in atoms if _element(atom) == "H"):
            candidates = [
                (float(np.linalg.norm(_coord(hydrogen) - _coord(heavy))), heavy)
                for heavy in heavy_atoms
                if _element(heavy) in {"C", "N", "O", "S"}
            ]
            if not candidates:
                continue
            bond_distance, heavy = min(candidates, key=lambda item: item[0])
            if bond_distance <= 1.40:
                _add_bond(adjacency, heavy, hydrogen)
    return adjacency


def _build_openmm_bond_adjacency(
    model: Model,
    validated_residues: list[Residue],
) -> dict[Any, list[Any]]:
    """Build an adjacency graph through an in-memory OpenMM topology.

    The OpenMM topology mirrors the selected Biopython model in atom order, so
    every OpenMM bond can be translated back to the original Biopython atom
    objects without a lossy PDB serialization round trip.
    """
    try:
        from openmm import Vec3, unit
        from openmm.app import Topology, element
    except ModuleNotFoundError as exc:
        if exc.name == "openmm" or (exc.name or "").startswith("openmm."):
            raise ModuleNotFoundError(
                "The OpenMM topology backend requires the optional contacts "
                "dependencies. Install them with: pip install "
                "'biotools[contacts]'"
            ) from exc
        raise

    topology = Topology()
    openmm_to_biopython: dict[Any, Any] = {}
    position_vectors = []

    for chain in model:
        openmm_chain = topology.addChain(str(chain.id))
        for residue in chain:
            _, residue_number, insertion_code = residue.id
            openmm_residue = topology.addResidue(
                _residue_name(residue),
                openmm_chain,
                id=str(residue_number),
                insertionCode=str(insertion_code).strip(),
            )
            for atom in residue.get_atoms():
                symbol = _element(atom)
                try:
                    openmm_element = element.get_by_symbol(symbol)
                except (KeyError, ValueError):
                    openmm_element = None
                serial_number = atom.get_serial_number()
                openmm_atom = topology.addAtom(
                    str(atom.get_name()).strip(),
                    openmm_element,
                    openmm_residue,
                    id=str(serial_number) if serial_number is not None else None,
                )
                openmm_to_biopython[openmm_atom] = atom
                x, y, z = _coord(atom)
                position_vectors.append(Vec3(float(x), float(y), float(z)))

    positions = unit.Quantity(position_vectors, unit.angstrom)
    topology.createStandardBonds()
    topology.createDisulfideBonds(positions)

    adjacency: dict[Any, list[Any]] = defaultdict(list)
    for atom in openmm_to_biopython.values():
        adjacency[atom]
    for openmm_atom1, openmm_atom2 in topology.bonds():
        atom1 = openmm_to_biopython[openmm_atom1]
        atom2 = openmm_to_biopython[openmm_atom2]
        _add_bond(adjacency, atom1, atom2)

    for residue in validated_residues:
        atoms = list(residue.get_atoms())
        if len(atoms) > 1 and not any(
            neighbor.get_parent() is residue
            for atom in atoms
            for neighbor in adjacency.get(atom, [])
        ):
            raise ValueError(
                "OpenMM could not assign internal bonds for residue "
                f"{_residue_name(residue)} {residue.id[1]}. Use the template "
                "backend or provide a structure with standard atom and "
                "residue names."
            )
        for hydrogen in (atom for atom in atoms if _element(atom) == "H"):
            if not adjacency.get(hydrogen):
                raise ValueError(
                    "OpenMM could not assign a bond for hydrogen "
                    f"{hydrogen.get_name()} in residue "
                    f"{_residue_name(residue)} {residue.id[1]}."
                )
    return adjacency


def _build_bond_adjacency(
    model: Model,
    residues: list[Residue],
    topology_backend: str,
) -> dict[Any, list[Any]]:
    if topology_backend == "templates":
        return _build_template_bond_adjacency(residues)
    if topology_backend == "openmm":
        return _build_openmm_bond_adjacency(model, residues)
    raise ValueError(
        "topology_backend must be either 'templates' or 'openmm', "
        f"not {topology_backend!r}"
    )


def _bonded_hydrogens(atom: Atom, adjacency: dict[Any, list[Any]]) -> list[Atom]:
    return [neighbor for neighbor in adjacency.get(atom, []) if _element(neighbor) == "H"]


def _donor_pairs(atoms: list[Atom], adjacency: dict[Any, list[Any]]) -> list[tuple[Atom, Atom]]:
    return [
        (atom, hydrogen)
        for atom in atoms
        if _element(atom) in {"N", "O", "S"}
        for hydrogen in _bonded_hydrogens(atom, adjacency)
    ]


def _is_acceptor(atom: Atom, adjacency: dict[Any, list[Any]]) -> bool:
    residue_name = _residue_name(atom.get_parent())
    atom_name = str(atom.get_name()).strip()
    if atom_name in {"O", "OXT"}:
        return True
    if atom_name in ACCEPTOR_ATOMS.get(residue_name, set()):
        return True
    if residue_name == "ASH" and atom_name in {"OD1", "OD2"}:
        return not _bonded_hydrogens(atom, adjacency)
    if residue_name == "GLH" and atom_name in {"OE1", "OE2"}:
        return not _bonded_hydrogens(atom, adjacency)
    if residue_name in {"HIS", "HID", "HIE", "HIP"} and atom_name in {"ND1", "NE2"}:
        return not _bonded_hydrogens(atom, adjacency)
    return False


def _hydrophobic_atoms(atoms: list[Atom], adjacency: dict[Any, list[Any]]) -> list[Atom]:
    selected = []
    for atom in atoms:
        symbol = _element(atom)
        atom_name = str(atom.get_name()).strip()
        if symbol == "C" and atom_name not in {"C", "CA"}:
            neighbors = adjacency.get(atom, [])
            if neighbors and {_element(neighbor) for neighbor in neighbors} <= {"C", "H"}:
                selected.append(atom)
        elif symbol == "S" and _residue_name(atom.get_parent()) in {"CYS", "CYM", "CYX", "MET"}:
            selected.append(atom)
    return selected


def _pair_hits(
    atoms_a: list[Atom], atoms_b: list[Atom], cutoff: float
) -> list[tuple[Atom, Atom, float]]:
    if not atoms_a or not atoms_b:
        return []
    search = NeighborSearch(atoms_b)
    order_b = {atom: index for index, atom in enumerate(atoms_b)}
    hits = [
        (index_a, order_b[atom_b], atom_a, atom_b, _distance(atom_a, atom_b))
        for index_a, atom_a in enumerate(atoms_a)
        for atom_b in search.search(_coord(atom_a), cutoff, level="A")
    ]
    hits.sort(key=lambda item: (item[0], item[1]))
    return [(atom_a, atom_b, distance) for _, _, atom_a, atom_b, distance in hits]


def _charged_groups(
    residues: list[Residue], adjacency: dict[Any, list[Any]]
) -> list[_ChargedGroup]:
    groups: list[_ChargedGroup] = []

    def add_group(residue: Residue, sign: int, names: tuple[str, ...], label: str) -> None:
        named = _atoms_by_name(residue)
        if not all(name in named for name in names):
            return
        atoms = tuple(named[name] for name in names)
        groups.append(
            _ChargedGroup(
                residue=residue,
                sign=sign,
                atoms=atoms,
                atom_label="/".join(names),
                group_label=label,
                center=np.mean([_coord(atom) for atom in atoms], axis=0),
            )
        )

    for residue in residues:
        name = _residue_name(residue)
        if name == "ARG":
            add_group(residue, 1, ("CZ", "NH1", "NH2"), "guanidinium")
        elif name == "LYS":
            add_group(residue, 1, ("NZ",), "ammonium")
        elif name == "ASP":
            add_group(residue, -1, ("OD1", "OD2"), "carboxylate")
        elif name == "GLU":
            add_group(residue, -1, ("OE1", "OE2"), "carboxylate")
        elif name in {"HIS", "HIP"}:
            named = _atoms_by_name(residue)
            ring_nitrogens = (named.get("ND1"), named.get("NE2"))
            if all(
                atom is not None and _bonded_hydrogens(atom, adjacency)
                for atom in ring_nitrogens
            ):
                add_group(residue, 1, AROMATIC_ATOMS["HIS"], "protonated imidazolium")

    if residues:
        first_named = _atoms_by_name(residues[0])
        terminal_n = first_named.get("N")
        if terminal_n is not None and len(_bonded_hydrogens(terminal_n, adjacency)) >= 2:
            add_group(residues[0], 1, ("N",), "N-terminus")
        last_named = _atoms_by_name(residues[-1])
        if "O" in last_named and "OXT" in last_named:
            add_group(residues[-1], -1, ("O", "OXT"), "C-terminus")
    return groups


def _aromatic_rings(residues: list[Residue]) -> list[_AromaticRing]:
    rings = []
    for residue in residues:
        expected_names = AROMATIC_ATOMS.get(_residue_name(residue))
        if expected_names is None:
            continue
        named = _atoms_by_name(residue)
        if not all(name in named for name in expected_names):
            continue
        atoms = tuple(named[name] for name in expected_names)
        coordinates = np.asarray([_coord(atom) for atom in atoms])
        center = coordinates.mean(axis=0)
        _, _, right_singular_vectors = np.linalg.svd(coordinates - center)
        normal = right_singular_vectors[-1]
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0.0:
            continue
        rings.append(_AromaticRing(residue, atoms, center, normal / normal_norm))
    return rings


def _hydrogen_bonds(
    atoms_a: list[Atom], atoms_b: list[Atom], adjacency: dict[Any, list[Any]]
) -> list[_Observation]:
    records = []
    directions = ((atoms_a, atoms_b, True), (atoms_b, atoms_a, False))
    for donor_atoms, candidate_acceptors, a_is_donor in directions:
        acceptors = [atom for atom in candidate_acceptors if _is_acceptor(atom, adjacency)]
        if not acceptors:
            continue
        search = NeighborSearch(acceptors)
        for donor, hydrogen in _donor_pairs(donor_atoms, adjacency):
            for acceptor in search.search(_coord(donor), HBOND_DISTANCE_A, level="A"):
                angle = _angle_degrees(_coord(donor), _coord(hydrogen), _coord(acceptor))
                if not np.isfinite(angle) or angle < HBOND_ANGLE_DEG:
                    continue
                if a_is_donor:
                    residue_a, residue_b = donor.get_parent(), acceptor.get_parent()
                    atom_a, atom_b = donor.get_name(), acceptor.get_name()
                    direction = "chain A donor"
                else:
                    residue_a, residue_b = acceptor.get_parent(), donor.get_parent()
                    atom_a, atom_b = acceptor.get_name(), donor.get_name()
                    direction = "chain B donor"
                records.append(
                    _Observation(
                        "hydrogen_bond", residue_a, residue_b, atom_a, atom_b,
                        _distance(donor, acceptor), angle,
                        f"{direction}; H={hydrogen.get_name()}",
                    )
                )
    return records


def _salt_bridges(
    groups_a: list[_ChargedGroup], groups_b: list[_ChargedGroup]
) -> list[_Observation]:
    records = []
    for group_a in groups_a:
        for group_b in groups_b:
            if group_a.sign * group_b.sign != -1:
                continue
            distance = float(np.linalg.norm(group_a.center - group_b.center))
            if distance <= SALT_BRIDGE_DISTANCE_A:
                records.append(
                    _Observation(
                        "salt_bridge", group_a.residue, group_b.residue,
                        group_a.atom_label, group_b.atom_label, distance,
                        geometry=f"{group_a.group_label}/{group_b.group_label}",
                    )
                )
    return records


def _hydrophobic_contacts(
    atoms_a: list[Atom], atoms_b: list[Atom], adjacency: dict[Any, list[Any]]
) -> list[_Observation]:
    return [
        _Observation(
            "hydrophobic_contact", atom_a.get_parent(), atom_b.get_parent(),
            atom_a.get_name(), atom_b.get_name(), distance,
        )
        for atom_a, atom_b, distance in _pair_hits(
            _hydrophobic_atoms(atoms_a, adjacency),
            _hydrophobic_atoms(atoms_b, adjacency),
            HYDROPHOBIC_DISTANCE_A,
        )
    ]


def _vdw_contacts(atoms_a: list[Atom], atoms_b: list[Atom]) -> list[_Observation]:
    heavy_a = [atom for atom in atoms_a if _element(atom) in VDW_RADII_A]
    heavy_b = [atom for atom in atoms_b if _element(atom) in VDW_RADII_A]
    maximum_cutoff = 2 * max(VDW_RADII_A.values()) + VDW_TOLERANCE_A
    records = []
    for atom_a, atom_b, distance in _pair_hits(heavy_a, heavy_b, maximum_cutoff):
        cutoff = VDW_RADII_A[_element(atom_a)] + VDW_RADII_A[_element(atom_b)] + VDW_TOLERANCE_A
        if distance <= cutoff:
            records.append(
                _Observation(
                    "van_der_waals_contact", atom_a.get_parent(), atom_b.get_parent(),
                    atom_a.get_name(), atom_b.get_name(), distance,
                )
            )
    return records


def _aromatic_interactions(
    rings_a: list[_AromaticRing],
    rings_b: list[_AromaticRing],
    groups_a: list[_ChargedGroup],
    groups_b: list[_ChargedGroup],
    include_parallel: bool,
    include_t_shaped: bool,
    include_cation_pi: bool,
) -> list[_Observation]:
    records = []
    if include_parallel or include_t_shaped:
        for ring_a in rings_a:
            for ring_b in rings_b:
                displacement = ring_b.center - ring_a.center
                distance = float(np.linalg.norm(displacement))
                if distance > PI_STACKING_DISTANCE_A:
                    continue
                normal_cosine = np.clip(abs(np.dot(ring_a.normal, ring_b.normal)), 0.0, 1.0)
                plane_angle = float(np.degrees(np.arccos(normal_cosine)))
                offset_a = float(
                    np.linalg.norm(
                        displacement
                        - np.dot(displacement, ring_a.normal) * ring_a.normal
                    )
                )
                offset_b = float(
                    np.linalg.norm(
                        displacement
                        - np.dot(displacement, ring_b.normal) * ring_b.normal
                    )
                )
                lateral_offset = max(offset_a, offset_b)
                if (
                    include_parallel
                    and plane_angle <= PI_PARALLEL_ANGLE_DEG
                    and lateral_offset <= PI_PARALLEL_OFFSET_A
                ):
                    records.append(
                        _Observation(
                            "pi_stacking_parallel", ring_a.residue, ring_b.residue,
                            "aromatic ring", "aromatic ring", distance, plane_angle,
                            f"parallel; offset={lateral_offset:.2f} A",
                        )
                    )
                if include_t_shaped and plane_angle >= PI_TSHAPED_ANGLE_DEG:
                    records.append(
                        _Observation(
                            "pi_stacking_t_shaped", ring_a.residue, ring_b.residue,
                            "aromatic ring", "aromatic ring", distance, plane_angle,
                            "t-shaped candidate",
                        )
                    )

    if include_cation_pi:
        for group in (group for group in groups_a if group.sign == 1):
            for ring in rings_b:
                distance = float(np.linalg.norm(group.center - ring.center))
                if distance <= CATION_PI_DISTANCE_A:
                    records.append(
                        _Observation(
                            "cation_pi_candidate", group.residue, ring.residue,
                            group.atom_label, "aromatic ring", distance,
                            geometry="chain A cation",
                        )
                    )
        for group in (group for group in groups_b if group.sign == 1):
            for ring in rings_a:
                distance = float(np.linalg.norm(group.center - ring.center))
                if distance <= CATION_PI_DISTANCE_A:
                    records.append(
                        _Observation(
                            "cation_pi_candidate", ring.residue, group.residue,
                            "aromatic ring", group.atom_label, distance,
                            geometry="chain B cation",
                        )
                    )
    return records


def _protein_water_legs(
    protein_atoms: list[Atom],
    water_residues: list[Residue],
    adjacency: dict[Any, list[Any]],
) -> dict[Residue, list[_WaterLeg]]:
    water_sites = []
    for water in water_residues:
        atoms = list(water.get_atoms())
        oxygen = next((atom for atom in atoms if _element(atom) == "O"), None)
        if oxygen is None:
            continue
        hydrogens = _bonded_hydrogens(oxygen, adjacency)
        if hydrogens:
            water_sites.append((water, oxygen, hydrogens))
    if not water_sites:
        return {}

    site_by_oxygen = {oxygen: (water, hydrogens) for water, oxygen, hydrogens in water_sites}
    water_search = NeighborSearch(list(site_by_oxygen))
    legs: dict[Any, list[_WaterLeg]] = defaultdict(list)

    for donor, hydrogen in _donor_pairs(protein_atoms, adjacency):
        for oxygen in water_search.search(_coord(donor), HBOND_DISTANCE_A, level="A"):
            water, _ = site_by_oxygen[oxygen]
            angle = _angle_degrees(_coord(donor), _coord(hydrogen), _coord(oxygen))
            if np.isfinite(angle) and angle >= HBOND_ANGLE_DEG:
                legs[water].append(
                    _WaterLeg(water, donor, _distance(donor, oxygen), angle, "protein donor")
                )

    for acceptor in (atom for atom in protein_atoms if _is_acceptor(atom, adjacency)):
        for oxygen in water_search.search(_coord(acceptor), HBOND_DISTANCE_A, level="A"):
            water, hydrogens = site_by_oxygen[oxygen]
            angles = [
                _angle_degrees(_coord(oxygen), _coord(hydrogen), _coord(acceptor))
                for hydrogen in hydrogens
            ]
            valid_angles = [angle for angle in angles if np.isfinite(angle)]
            if valid_angles and max(valid_angles) >= HBOND_ANGLE_DEG:
                legs[water].append(
                    _WaterLeg(
                        water, acceptor, _distance(oxygen, acceptor),
                        max(valid_angles), "water donor",
                    )
                )
    return legs


def _water_label(water: Residue) -> str:
    chain_id = getattr(water.get_parent(), "id", "")
    _, number, insertion_code = water.id
    position = f"{number}{str(insertion_code).strip()}"
    return f"{chain_id}:{position}:{_residue_name(water)}"


def _water_bridges(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    water_residues: list[Residue],
    adjacency: dict[Any, list[Any]],
) -> list[_Observation]:
    legs_a = _protein_water_legs(atoms_a, water_residues, adjacency)
    legs_b = _protein_water_legs(atoms_b, water_residues, adjacency)
    water_order = {water: index for index, water in enumerate(water_residues)}
    records = []
    for water in sorted(set(legs_a) & set(legs_b), key=water_order.__getitem__):
        for leg_a in legs_a[water]:
            for leg_b in legs_b[water]:
                records.append(
                    _Observation(
                        "water_bridge",
                        leg_a.protein_atom.get_parent(),
                        leg_b.protein_atom.get_parent(),
                        leg_a.protein_atom.get_name(), leg_b.protein_atom.get_name(),
                        max(leg_a.distance, leg_b.distance), min(leg_a.angle, leg_b.angle),
                        f"chain A: {leg_a.direction}; chain B: {leg_b.direction}",
                        _water_label(water),
                    )
                )
    return records


def _select_model_and_chains(
    structure: Structure, chain_a: str, chain_b: str
) -> tuple[Model, Chain, Chain]:
    if chain_a == chain_b:
        raise ValueError("chain_a and chain_b must identify different chains")
    models = list(structure)
    has_a = any(any(chain.id == chain_a for chain in model) for model in models)
    has_b = any(any(chain.id == chain_b for chain in model) for model in models)
    if not has_a:
        raise ValueError(f"Chain {chain_a!r} not found in structure")
    if not has_b:
        raise ValueError(f"Chain {chain_b!r} not found in structure")
    matches = [model for model in models if chain_a in model and chain_b in model]
    if not matches:
        raise ValueError("The requested chains do not occur in the same model")
    if len(matches) > 1:
        raise ValueError("The requested chains occur in multiple models; select one model first")
    model = matches[0]
    return model, model[chain_a], model[chain_b]


def _is_protein_residue(residue: Residue) -> bool:
    """Recognize standard amino acids and common force-field variants."""
    name = _residue_name(residue)
    return is_aa(residue) or name in SIDECHAIN_BONDS or name in RESIDUE_BOND_ALIASES


def _record(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    observation: _Observation,
) -> dict[str, Any]:
    return {
        "structure_id": getattr(structure, "id", None),
        "interaction_type": observation.interaction_type,
        "chain_a": chain_a,
        "chain_b": chain_b,
        "residue_a_id": observation.residue_a.id,
        "residue_a_num": observation.residue_a.id[1],
        "residue_a_name": observation.residue_a.get_resname(),
        "atom_a": observation.atom_a,
        "residue_b_id": observation.residue_b.id,
        "residue_b_num": observation.residue_b.id[1],
        "residue_b_name": observation.residue_b.get_resname(),
        "atom_b": observation.atom_b,
        "distance": float(observation.distance) if observation.distance is not None else None,
        "angle": float(observation.angle) if observation.angle is not None else None,
        "geometry": observation.geometry,
        "mediator": observation.mediator,
    }


def _aggregate(observations: list[_Observation]) -> list[_Observation]:
    grouped: dict[tuple[Any, Any, str], list[_Observation]] = defaultdict(list)
    order: list[tuple[Any, Any, str]] = []
    for observation in observations:
        key = (observation.residue_a, observation.residue_b, observation.interaction_type)
        if key not in grouped:
            order.append(key)
        grouped[key].append(observation)

    result = []
    for key in order:
        candidates = grouped[key]
        representative = min(
            candidates,
            key=lambda item: (
                item.distance if item.distance is not None else float("inf"),
                -(item.angle if item.angle is not None else float("-inf")),
            ),
        )
        mediators = sorted({item.mediator for item in candidates if item.mediator})
        if mediators:
            representative = _Observation(
                representative.interaction_type,
                representative.residue_a,
                representative.residue_b,
                representative.atom_a,
                representative.atom_b,
                representative.distance,
                representative.angle,
                representative.geometry,
                ",".join(mediators),
            )
        result.append(representative)
    return result


def _detect_contact_observations(
    model: Model,
    residues_a: list[Residue],
    residues_b: list[Residue],
    *,
    hydrogen_bond: bool,
    salt_bridge: bool,
    hydrophobic_contact: bool,
    van_der_waals_contact: bool,
    pi_stacking_parallel: bool,
    cation_pi_candidate: bool,
    water_bridge: bool,
    pi_stacking_t_shaped: bool,
    topology_backend: str,
) -> list[_Observation]:
    atoms_a = [atom for residue in residues_a for atom in residue.get_atoms()]
    atoms_b = [atom for residue in residues_b for atom in residue.get_atoms()]
    water_residues = [
        residue
        for chain in model
        for residue in chain
        if _residue_name(residue) in WATER_RESIDUE_NAMES
    ]
    topology_residues = list(
        dict.fromkeys(
            residues_a
            + residues_b
            + (water_residues if water_bridge else [])
        )
    )
    adjacency = _build_bond_adjacency(
        model,
        topology_residues,
        topology_backend,
    )

    include_charged = salt_bridge or cation_pi_candidate
    groups_a = _charged_groups(residues_a, adjacency) if include_charged else []
    groups_b = _charged_groups(residues_b, adjacency) if include_charged else []
    include_aromatic = (
        pi_stacking_parallel or pi_stacking_t_shaped or cation_pi_candidate
    )
    rings_a = _aromatic_rings(residues_a) if include_aromatic else []
    rings_b = _aromatic_rings(residues_b) if include_aromatic else []

    observations: list[_Observation] = []
    if hydrogen_bond:
        observations.extend(_hydrogen_bonds(atoms_a, atoms_b, adjacency))
    if salt_bridge:
        observations.extend(_salt_bridges(groups_a, groups_b))
    if hydrophobic_contact:
        observations.extend(_hydrophobic_contacts(atoms_a, atoms_b, adjacency))
    if van_der_waals_contact:
        observations.extend(_vdw_contacts(atoms_a, atoms_b))
    if include_aromatic:
        observations.extend(
            _aromatic_interactions(
                rings_a,
                rings_b,
                groups_a,
                groups_b,
                pi_stacking_parallel,
                pi_stacking_t_shaped,
                cation_pi_candidate,
            )
        )
    if water_bridge:
        observations.extend(
            _water_bridges(
                atoms_a,
                atoms_b,
                water_residues,
                adjacency,
            )
        )
    return observations


def _select_model_and_chain(
    structure: Structure,
    chain_id: str,
) -> tuple[Model, Chain]:
    matches = [model for model in structure if chain_id in model]
    if not matches:
        raise ValueError(f"Chain {chain_id!r} not found in structure")
    if len(matches) > 1:
        raise ValueError(
            f"Chain {chain_id!r} occurs in multiple models; select one model "
            "before characterizing intrachain contacts"
        )
    model = matches[0]
    return model, model[chain_id]


def characterize_intrachain_contacts_impl(
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
    """Characterize unique noncovalent residue pairs within one chain."""
    if (
        isinstance(min_sequence_separation, bool)
        or not isinstance(min_sequence_separation, int)
        or min_sequence_separation < 1
    ):
        raise ValueError("min_sequence_separation must be an integer >= 1")

    model, chain_object = _select_model_and_chain(structure, chain)
    residues = [
        residue
        for residue in chain_object.get_residues()
        if _is_protein_residue(residue)
    ]
    residue_order = {residue: index for index, residue in enumerate(residues)}
    observations = _detect_contact_observations(
        model,
        residues,
        residues,
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

    normalized = []
    for observation in observations:
        index_a = residue_order.get(observation.residue_a)
        index_b = residue_order.get(observation.residue_b)
        if index_a is None or index_b is None:
            continue
        if abs(index_a - index_b) < min_sequence_separation:
            continue
        if index_a > index_b:
            observation = _Observation(
                observation.interaction_type,
                observation.residue_b,
                observation.residue_a,
                observation.atom_b,
                observation.atom_a,
                observation.distance,
                observation.angle,
                observation.geometry,
                observation.mediator,
            )
        normalized.append(observation)

    return [
        _record(structure, chain, chain, observation)
        for observation in _aggregate(normalized)
    ]


def characterize_chain_contacts_impl(
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
    """Implement :func:`biotools.structure.geometry.characterize_chain_contacts`."""
    model, chain_object_a, chain_object_b = _select_model_and_chains(structure, chain_a, chain_b)
    residues_a = [
        residue
        for residue in chain_object_a.get_residues()
        if _is_protein_residue(residue)
    ]
    residues_b = [
        residue
        for residue in chain_object_b.get_residues()
        if _is_protein_residue(residue)
    ]
    observations = _detect_contact_observations(
        model,
        residues_a,
        residues_b,
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

    if not atomic:
        observations = _aggregate(observations)
    return [_record(structure, chain_a, chain_b, observation) for observation in observations]
