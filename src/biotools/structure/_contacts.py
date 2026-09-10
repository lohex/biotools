"""Protein-specific geometric contact characterization.

The public entry point lives in :mod:`biotools.structure.geometry`.  This
module contains the chemistry and geometry primitives so that the individual
contact detectors can be tested and maintained independently.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import is_aa

from .contacts.config import ContactConfig, get_contact_config

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

AROMATIC_RINGS = {
    "PHE": (("phenyl", ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")),),
    "TYR": (("phenyl", ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")),),
    "HIS": (("imidazole", ("CG", "ND1", "CE1", "NE2", "CD2")),),
    "HID": (("imidazole", ("CG", "ND1", "CE1", "NE2", "CD2")),),
    "HIE": (("imidazole", ("CG", "ND1", "CE1", "NE2", "CD2")),),
    "HIP": (("imidazolium", ("CG", "ND1", "CE1", "NE2", "CD2")),),
    "TRP": (
        ("pyrrole", ("CG", "CD1", "NE1", "CE2", "CD2")),
        ("benzene", ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3")),
    ),
}
LEGACY_AROMATIC_RINGS = {
    name: (("aromatic ring", atoms),)
    for name, atoms in {
        "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
        "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
        "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
        "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
        "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
        "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
        "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    }.items()
}
# Backwards-compatible internal alias used for the complete imidazole group.
AROMATIC_ATOMS = {
    name: rings[0][1] for name, rings in AROMATIC_RINGS.items()
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
    measurements: tuple[tuple[str, float, str], ...] = ()
    criteria: tuple[tuple[str, bool, str], ...] = ()
    role_a: str = "participant"
    role_b: str = "participant"
    evidence_level: str = "explicit_geometry"
    quality_flags: tuple[str, ...] = ()
    water: Any | None = None
    member_count: int = 1
    independent_group_count: int = 1
    distance_min: float | None = None
    distance_median: float | None = None
    distance_max: float | None = None


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
    ring_id: str
    planarity_error: float
    radius: float


@dataclass(frozen=True)
class _WaterLeg:
    water: Any
    protein_atom: Any
    distance: float
    angle: float
    direction: str
    evidence_level: str = "explicit_geometry"


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

    # Add peptide bonds only for consecutive residues in the same chain whose
    # C--N geometry is covalent.  A selection boundary therefore does not turn
    # an internal residue into an artificial terminus, while real chain breaks
    # remain breaks.
    residues_by_chain: dict[Any, list[Any]] = defaultdict(list)
    for residue in residues:
        residues_by_chain[residue.get_parent()].append(residue)
    for chain_residues in residues_by_chain.values():
        for previous, following in zip(chain_residues, chain_residues[1:]):
            previous_atoms = _atoms_by_name(previous)
            following_atoms = _atoms_by_name(following)
            carbon = previous_atoms.get("C")
            nitrogen = following_atoms.get("N")
            if carbon is not None and nitrogen is not None and _distance(carbon, nitrogen) <= 1.8:
                _add_bond(adjacency, carbon, nitrogen)

    # A conservative geometry fallback supplies disulfide exclusions to the
    # dependency-free backend, including links between different chains.
    sulfurs = [
        _atoms_by_name(residue).get("SG")
        for residue in residues
        if _residue_name(residue) in {"CYS", "CYX"}
    ]
    sulfurs = [atom for atom in sulfurs if atom is not None]
    for index, sulfur_a in enumerate(sulfurs):
        for sulfur_b in sulfurs[index + 1 :]:
            if 1.8 <= _distance(sulfur_a, sulfur_b) <= 2.3:
                _add_bond(adjacency, sulfur_a, sulfur_b)
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
        if residue_name in {"ASP", "GLU"} and _bonded_hydrogens(atom, adjacency):
            return False
        return True
    if residue_name == "ASH" and atom_name in {"OD1", "OD2"}:
        return not _bonded_hydrogens(atom, adjacency)
    if residue_name == "GLH" and atom_name in {"OE1", "OE2"}:
        return not _bonded_hydrogens(atom, adjacency)
    if residue_name in {"HIS", "HID", "HIE", "HIP"} and atom_name in {"ND1", "NE2"}:
        return not _bonded_hydrogens(atom, adjacency)
    return False


def _hydrophobic_atoms(
    atoms: list[Atom],
    adjacency: dict[Any, list[Any]],
    config: ContactConfig,
) -> list[Atom]:
    selected = []
    for atom in atoms:
        symbol = _element(atom)
        atom_name = str(atom.get_name()).strip()
        if symbol == "C" and atom_name not in {"C", "CA"}:
            neighbors = adjacency.get(atom, [])
            if neighbors and {_element(neighbor) for neighbor in neighbors} <= {"C", "H"}:
                selected.append(atom)
        elif (
            symbol == "S"
            and not config.refined_geometry
            and _residue_name(atom.get_parent()) in {"CYS", "CYM", "CYX", "MET"}
        ):
            selected.append(atom)
        elif symbol == "S" and _residue_name(atom.get_parent()) in {"CYS", "MET"}:
            # Charged thiolate (CYM) and disulfide sulfur (CYX or an explicit
            # interresidue S--S bond) are not ordinary nonpolar sulfur sites.
            if not any(
                _element(neighbor) == "S" and neighbor.get_parent() is not atom.get_parent()
                for neighbor in adjacency.get(atom, [])
            ):
                selected.append(atom)
    return selected


def _bond_separation(
    atom_a: Atom,
    atom_b: Atom,
    adjacency: dict[Any, list[Any]] | Any,
    maximum: int,
) -> int | None:
    """Return a shortest bond separation up to ``maximum``."""
    if atom_a is atom_b:
        return 0
    frontier = {atom_a}
    visited = {atom_a}
    for separation in range(1, maximum + 1):
        frontier = {
            neighbor
            for atom in frontier
            for neighbor in adjacency.get(atom, ())
            if neighbor not in visited
        }
        if atom_b in frontier:
            return separation
        visited.update(frontier)
        if not frontier:
            break
    return None


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
    residues: list[Residue],
    adjacency: dict[Any, list[Any]],
    config: ContactConfig,
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
            if (name == "HIP" and config.refined_geometry) or all(
                atom is not None and _bonded_hydrogens(atom, adjacency)
                for atom in ring_nitrogens
            ):
                add_group(residue, 1, AROMATIC_ATOMS["HIS"], "protonated imidazolium")

    terminal_residues = (
        residues
        if config.refined_geometry
        else list(dict.fromkeys(residues[:1] + residues[-1:]))
    )
    for residue in terminal_residues:
        named = _atoms_by_name(residue)
        terminal_n = named.get("N")
        has_previous_peptide_bond = terminal_n is not None and any(
            neighbor.get_parent() is not residue
            and str(neighbor.get_name()).strip() == "C"
            for neighbor in adjacency.get(terminal_n, ())
        )
        if (
            terminal_n is not None
            and (not has_previous_peptide_bond or not config.refined_geometry)
            and len(_bonded_hydrogens(terminal_n, adjacency)) >= 2
        ):
            add_group(residue, 1, ("N",), "N-terminus")
        if "O" in named and "OXT" in named:
            add_group(residue, -1, ("O", "OXT"), "C-terminus")
    return groups


def _aromatic_rings(
    residues: list[Residue], config: ContactConfig
) -> list[_AromaticRing]:
    rings = []
    for residue in residues:
        ring_table = AROMATIC_RINGS if config.refined_geometry else LEGACY_AROMATIC_RINGS
        definitions = ring_table.get(_residue_name(residue))
        if definitions is None:
            continue
        named = _atoms_by_name(residue)
        for ring_id, expected_names in definitions:
            if not all(name in named for name in expected_names):
                continue
            atoms = tuple(named[name] for name in expected_names)
            coordinates = np.asarray([_coord(atom) for atom in atoms])
            center = coordinates.mean(axis=0)
            centered = coordinates - center
            if np.linalg.matrix_rank(centered) < 2:
                continue
            _, _, right_singular_vectors = np.linalg.svd(centered)
            normal = right_singular_vectors[-1]
            normal_norm = np.linalg.norm(normal)
            if normal_norm == 0.0:
                continue
            normal = normal / normal_norm
            planarity = float(np.max(np.abs(centered @ normal)))
            if planarity > config.ring_planarity_tolerance:
                continue
            radius = float(np.max(np.linalg.norm(centered, axis=1)))
            rings.append(
                _AromaticRing(
                    residue, atoms, center, normal, ring_id, planarity, radius
                )
            )
    return rings


def _hydrogen_bonds(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    adjacency: dict[Any, list[Any]],
    config: ContactConfig,
) -> list[_Observation]:
    records = []
    directions = ((atoms_a, atoms_b, True), (atoms_b, atoms_a, False))
    for donor_atoms, candidate_acceptors, a_is_donor in directions:
        acceptors = [atom for atom in candidate_acceptors if _is_acceptor(atom, adjacency)]
        if not acceptors:
            continue
        search = NeighborSearch(acceptors)
        for donor, hydrogen in _donor_pairs(donor_atoms, adjacency):
            for acceptor in search.search(
                _coord(donor), config.hydrogen_bond_distance, level="A"
            ):
                angle = _angle_degrees(_coord(donor), _coord(hydrogen), _coord(acceptor))
                if not np.isfinite(angle) or angle < config.hydrogen_bond_angle:
                    continue
                donor_acceptor = _distance(donor, acceptor)
                hydrogen_acceptor = _distance(hydrogen, acceptor)
                if a_is_donor:
                    residue_a, residue_b = donor.get_parent(), acceptor.get_parent()
                    atom_a, atom_b = donor.get_name(), acceptor.get_name()
                    direction = "chain A donor"
                    roles = ("donor", "acceptor")
                else:
                    residue_a, residue_b = acceptor.get_parent(), donor.get_parent()
                    atom_a, atom_b = acceptor.get_name(), donor.get_name()
                    direction = "chain B donor"
                    roles = ("acceptor", "donor")
                records.append(
                    _Observation(
                        "hydrogen_bond", residue_a, residue_b, atom_a, atom_b,
                        donor_acceptor, angle,
                        f"{direction}; H={hydrogen.get_name()}",
                        measurements=(
                            ("donor_acceptor_distance", donor_acceptor, "angstrom"),
                            ("hydrogen_acceptor_distance", hydrogen_acceptor, "angstrom"),
                            ("donor_hydrogen_acceptor_angle", angle, "degree"),
                        ),
                        criteria=(
                            (
                                "donor_acceptor_distance",
                                donor_acceptor <= config.hydrogen_bond_distance,
                                f"<= {config.hydrogen_bond_distance} angstrom",
                            ),
                            (
                                "donor_hydrogen_acceptor_angle",
                                angle >= config.hydrogen_bond_angle,
                                f">= {config.hydrogen_bond_angle} degree",
                            ),
                        ),
                        role_a=roles[0],
                        role_b=roles[1],
                    )
                )
    return records


def _salt_bridges(
    groups_a: list[_ChargedGroup],
    groups_b: list[_ChargedGroup],
    config: ContactConfig,
) -> list[_Observation]:
    records = []
    for group_a in groups_a:
        for group_b in groups_b:
            if group_a.sign * group_b.sign != -1:
                continue
            distance = float(np.linalg.norm(group_a.center - group_b.center))
            if distance <= config.salt_bridge_distance:
                nearest = min(
                    _distance(atom_a, atom_b)
                    for atom_a in group_a.atoms
                    for atom_b in group_b.atoms
                )
                roles = ("cation", "anion") if group_a.sign > 0 else ("anion", "cation")
                records.append(
                    _Observation(
                        "salt_bridge", group_a.residue, group_b.residue,
                        group_a.atom_label, group_b.atom_label, distance,
                        geometry=f"{group_a.group_label}/{group_b.group_label}",
                        measurements=(
                            ("charge_center_distance", distance, "angstrom"),
                            ("nearest_group_atom_distance", nearest, "angstrom"),
                        ),
                        criteria=((
                            "charge_center_distance",
                            distance <= config.salt_bridge_distance,
                            f"<= {config.salt_bridge_distance} angstrom",
                        ),),
                        role_a=roles[0],
                        role_b=roles[1],
                    )
                )
    return records


def _hydrophobic_contacts(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    adjacency: dict[Any, list[Any]],
    config: ContactConfig,
) -> list[_Observation]:
    records = []
    for atom_a, atom_b, distance in _pair_hits(
        _hydrophobic_atoms(atoms_a, adjacency, config),
        _hydrophobic_atoms(atoms_b, adjacency, config),
        config.hydrophobic_distance,
    ):
        radii = VDW_RADII_A.get(_element(atom_a), 0.0) + VDW_RADII_A.get(_element(atom_b), 0.0)
        surface_gap = distance - radii
        records.append(
            _Observation(
                "hydrophobic_contact", atom_a.get_parent(), atom_b.get_parent(),
                atom_a.get_name(), atom_b.get_name(), distance,
                measurements=(
                    ("atom_distance", distance, "angstrom"),
                    ("surface_gap", surface_gap, "angstrom"),
                ),
                criteria=((
                    "atom_distance", distance <= config.hydrophobic_distance,
                    f"<= {config.hydrophobic_distance} angstrom",
                ),),
                role_a="nonpolar_atom",
                role_b="nonpolar_atom",
            )
        )
    return records


def _vdw_contacts(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    adjacency: dict[Any, list[Any]],
    config: ContactConfig,
) -> list[_Observation]:
    heavy_a = [atom for atom in atoms_a if _element(atom) in VDW_RADII_A]
    heavy_b = [atom for atom in atoms_b if _element(atom) in VDW_RADII_A]
    maximum_cutoff = 2 * max(VDW_RADII_A.values()) + config.vdw_tolerance
    records = []
    for atom_a, atom_b, distance in _pair_hits(heavy_a, heavy_b, maximum_cutoff):
        maximum_separation = 3 if config.exclude_one_four else 2
        if (
            config.refined_geometry
            and _bond_separation(atom_a, atom_b, adjacency, maximum_separation) is not None
        ):
            continue
        radii_sum = VDW_RADII_A[_element(atom_a)] + VDW_RADII_A[_element(atom_b)]
        cutoff = radii_sum + config.vdw_tolerance
        if distance <= cutoff:
            surface_gap = distance - radii_sum
            overlap_depth = max(0.0, -surface_gap)
            steric_clash = overlap_depth > config.steric_clash_overlap
            records.append(
                _Observation(
                    "van_der_waals_contact", atom_a.get_parent(), atom_b.get_parent(),
                    atom_a.get_name(), atom_b.get_name(), distance,
                    geometry="steric clash" if steric_clash else "vdW proximity",
                    measurements=(
                        ("atom_distance", distance, "angstrom"),
                        ("surface_gap", surface_gap, "angstrom"),
                        ("overlap_depth", overlap_depth, "angstrom"),
                    ),
                    criteria=((
                        "vdw_proximity", distance <= cutoff,
                        f"<= radii_sum + {config.vdw_tolerance} angstrom",
                    ),),
                    role_a="atom",
                    role_b="atom",
                    quality_flags=("steric_clash",) if steric_clash else (),
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
    config: ContactConfig,
) -> list[_Observation]:
    records = []
    if include_parallel or include_t_shaped:
        for ring_a in rings_a:
            for ring_b in rings_b:
                displacement = ring_b.center - ring_a.center
                distance = float(np.linalg.norm(displacement))
                if distance > config.pi_distance:
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
                height_a = abs(float(np.dot(displacement, ring_a.normal)))
                height_b = abs(float(np.dot(displacement, ring_b.normal)))
                nearest_atom = min(
                    _distance(atom_a, atom_b)
                    for atom_a in ring_a.atoms
                    for atom_b in ring_b.atoms
                )
                common_measurements = (
                    ("centroid_distance", distance, "angstrom"),
                    ("interplane_angle", plane_angle, "degree"),
                    ("height_from_ring_a", height_a, "angstrom"),
                    ("height_from_ring_b", height_b, "angstrom"),
                    ("lateral_offset_a", offset_a, "angstrom"),
                    ("lateral_offset_b", offset_b, "angstrom"),
                    ("nearest_ring_atom_distance", nearest_atom, "angstrom"),
                    ("planarity_error_a", ring_a.planarity_error, "angstrom"),
                    ("planarity_error_b", ring_b.planarity_error, "angstrom"),
                )
                if (
                    include_parallel
                    and plane_angle <= config.pi_parallel_angle
                    and lateral_offset <= config.pi_parallel_offset
                    and nearest_atom >= config.pi_min_nearest_atom_distance
                ):
                    records.append(
                        _Observation(
                            "pi_stacking_parallel", ring_a.residue, ring_b.residue,
                            ring_a.ring_id, ring_b.ring_id, distance, plane_angle,
                            f"parallel; offset={lateral_offset:.2f} A",
                            measurements=common_measurements,
                            criteria=(
                                (
                                    "centroid_distance",
                                    distance <= config.pi_distance,
                                    f"<= {config.pi_distance} angstrom",
                                ),
                                (
                                    "interplane_angle",
                                    plane_angle <= config.pi_parallel_angle,
                                    f"<= {config.pi_parallel_angle} degree",
                                ),
                                (
                                    "lateral_offset",
                                    lateral_offset <= config.pi_parallel_offset,
                                    f"<= {config.pi_parallel_offset} angstrom",
                                ),
                                (
                                    "nearest_atom_distance",
                                    nearest_atom >= config.pi_min_nearest_atom_distance,
                                    f">= {config.pi_min_nearest_atom_distance} angstrom",
                                ),
                            ),
                            role_a="ring",
                            role_b="ring",
                        )
                    )
                # For an edge-to-face contact the edge-ring centre must lie
                # above the face polygon (plus tolerance), and the face centre
                # must be near the edge-ring plane.  Evaluate both assignments.
                assignments = (
                    (ring_a, ring_b, offset_a, height_a, height_b),
                    (ring_b, ring_a, offset_b, height_b, height_a),
                )
                valid_assignment = next(
                    (
                        (face, edge)
                        for face, edge, face_offset, face_height, edge_plane_offset in assignments
                        if face_offset <= face.radius + config.pi_projection_tolerance
                        and face_height >= config.cation_pi_min_height
                        and edge_plane_offset <= edge.radius + config.pi_projection_tolerance
                    ),
                    None,
                )
                if (
                    include_t_shaped
                    and plane_angle >= config.pi_t_shaped_angle
                    and (valid_assignment is not None or not config.refined_geometry)
                    and nearest_atom >= config.pi_min_nearest_atom_distance
                ):
                    face, edge = valid_assignment or (ring_a, ring_b)
                    roles = (
                        ("face_ring", "edge_ring")
                        if face is ring_a
                        else ("edge_ring", "face_ring")
                    )
                    records.append(
                        _Observation(
                            "pi_stacking_t_shaped", ring_a.residue, ring_b.residue,
                            ring_a.ring_id, ring_b.ring_id, distance, plane_angle,
                            f"edge-to-face candidate; face={face.ring_id}; edge={edge.ring_id}",
                            measurements=common_measurements,
                            criteria=(
                                (
                                    "centroid_distance",
                                    distance <= config.pi_distance,
                                    f"<= {config.pi_distance} angstrom",
                                ),
                                (
                                    "interplane_angle",
                                    plane_angle >= config.pi_t_shaped_angle,
                                    f">= {config.pi_t_shaped_angle} degree",
                                ),
                                (
                                    "edge_to_face_projection",
                                    valid_assignment is not None,
                                    "projection within ring radius plus tolerance",
                                ),
                                (
                                    "nearest_atom_distance",
                                    nearest_atom >= config.pi_min_nearest_atom_distance,
                                    f">= {config.pi_min_nearest_atom_distance} angstrom",
                                ),
                            ),
                            role_a=roles[0],
                            role_b=roles[1],
                            evidence_level=(
                                "explicit_geometry"
                                if valid_assignment is not None
                                else "distance_candidate"
                            ),
                        )
                    )

    if include_cation_pi:
        for group in (group for group in groups_a if group.sign == 1):
            for ring in rings_b:
                displacement = group.center - ring.center
                distance = float(np.linalg.norm(displacement))
                height = abs(float(np.dot(displacement, ring.normal)))
                lateral = float(
                    np.linalg.norm(
                        displacement
                        - np.dot(displacement, ring.normal) * ring.normal
                    )
                )
                projection_valid = lateral <= ring.radius + config.pi_projection_tolerance
                if (
                    distance <= config.cation_pi_distance
                    and (
                        not config.refined_geometry
                        or (height >= config.cation_pi_min_height and projection_valid)
                    )
                ):
                    records.append(
                        _Observation(
                            "cation_pi_candidate", group.residue, ring.residue,
                            group.atom_label, ring.ring_id, distance,
                            geometry="chain A cation",
                            measurements=(
                                ("cation_centroid_distance", distance, "angstrom"),
                                ("ring_plane_height", height, "angstrom"),
                                ("lateral_offset", lateral, "angstrom"),
                                ("ring_planarity_error", ring.planarity_error, "angstrom"),
                            ),
                            criteria=(
                                (
                                    "cation_centroid_distance",
                                    distance <= config.cation_pi_distance,
                                    f"<= {config.cation_pi_distance} angstrom",
                                ),
                                (
                                    "ring_plane_height",
                                    height >= config.cation_pi_min_height,
                                    f">= {config.cation_pi_min_height} angstrom",
                                ),
                                (
                                    "projection",
                                    projection_valid,
                                    "within ring radius plus tolerance",
                                ),
                            ),
                            role_a="cation",
                            role_b="ring",
                        )
                    )
        for group in (group for group in groups_b if group.sign == 1):
            for ring in rings_a:
                displacement = group.center - ring.center
                distance = float(np.linalg.norm(displacement))
                height = abs(float(np.dot(displacement, ring.normal)))
                lateral = float(
                    np.linalg.norm(
                        displacement
                        - np.dot(displacement, ring.normal) * ring.normal
                    )
                )
                projection_valid = lateral <= ring.radius + config.pi_projection_tolerance
                if (
                    distance <= config.cation_pi_distance
                    and (
                        not config.refined_geometry
                        or (height >= config.cation_pi_min_height and projection_valid)
                    )
                ):
                    records.append(
                        _Observation(
                            "cation_pi_candidate", ring.residue, group.residue,
                            ring.ring_id, group.atom_label, distance,
                            geometry="chain B cation",
                            measurements=(
                                ("cation_centroid_distance", distance, "angstrom"),
                                ("ring_plane_height", height, "angstrom"),
                                ("lateral_offset", lateral, "angstrom"),
                                ("ring_planarity_error", ring.planarity_error, "angstrom"),
                            ),
                            criteria=(
                                (
                                    "cation_centroid_distance",
                                    distance <= config.cation_pi_distance,
                                    f"<= {config.cation_pi_distance} angstrom",
                                ),
                                (
                                    "ring_plane_height",
                                    height >= config.cation_pi_min_height,
                                    f">= {config.cation_pi_min_height} angstrom",
                                ),
                                (
                                    "projection",
                                    projection_valid,
                                    "within ring radius plus tolerance",
                                ),
                            ),
                            role_a="ring",
                            role_b="cation",
                        )
                    )
    return records


def _protein_water_legs(
    protein_atoms: list[Atom],
    water_residues: list[Residue],
    adjacency: dict[Any, list[Any]],
    config: ContactConfig,
) -> dict[Residue, list[_WaterLeg]]:
    water_sites = []
    for water in water_residues:
        atoms = list(water.get_atoms())
        oxygen = next((atom for atom in atoms if _element(atom) == "O"), None)
        if oxygen is None:
            continue
        hydrogens = _bonded_hydrogens(oxygen, adjacency)
        if hydrogens or config.refined_geometry:
            water_sites.append((water, oxygen, hydrogens))
    if not water_sites:
        return {}

    site_by_oxygen = {oxygen: (water, hydrogens) for water, oxygen, hydrogens in water_sites}
    water_search = NeighborSearch(list(site_by_oxygen))
    legs: dict[Any, list[_WaterLeg]] = defaultdict(list)

    for donor, hydrogen in _donor_pairs(protein_atoms, adjacency):
        for oxygen in water_search.search(
            _coord(donor), config.hydrogen_bond_distance, level="A"
        ):
            water, _ = site_by_oxygen[oxygen]
            angle = _angle_degrees(_coord(donor), _coord(hydrogen), _coord(oxygen))
            if np.isfinite(angle) and angle >= config.hydrogen_bond_angle:
                legs[water].append(
                    _WaterLeg(water, donor, _distance(donor, oxygen), angle, "protein donor")
                )

    for acceptor in (atom for atom in protein_atoms if _is_acceptor(atom, adjacency)):
        for oxygen in water_search.search(
            _coord(acceptor), config.hydrogen_bond_distance, level="A"
        ):
            water, hydrogens = site_by_oxygen[oxygen]
            angles = [
                _angle_degrees(_coord(oxygen), _coord(hydrogen), _coord(acceptor))
                for hydrogen in hydrogens
            ]
            valid_angles = [angle for angle in angles if np.isfinite(angle)]
            if valid_angles and max(valid_angles) >= config.hydrogen_bond_angle:
                legs[water].append(
                    _WaterLeg(
                        water, acceptor, _distance(oxygen, acceptor),
                        max(valid_angles), "water donor",
                    )
                )
            elif not hydrogens and config.refined_geometry:
                distance = _distance(oxygen, acceptor)
                legs[water].append(
                    _WaterLeg(
                        water,
                        acceptor,
                        distance,
                        float("nan"),
                        "water donor orientation unknown",
                        "distance_candidate",
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
    config: ContactConfig,
) -> list[_Observation]:
    legs_a = _protein_water_legs(atoms_a, water_residues, adjacency, config)
    legs_b = _protein_water_legs(atoms_b, water_residues, adjacency, config)
    water_order = {water: index for index, water in enumerate(water_residues)}
    records = []
    for water in sorted(set(legs_a) & set(legs_b), key=water_order.__getitem__):
        for leg_a in legs_a[water]:
            for leg_b in legs_b[water]:
                angles = [angle for angle in (leg_a.angle, leg_b.angle) if np.isfinite(angle)]
                minimum_angle = min(angles) if angles else None
                evidence = (
                    "distance_candidate"
                    if "distance_candidate" in {leg_a.evidence_level, leg_b.evidence_level}
                    else "explicit_geometry"
                )
                measurements = [
                    ("leg_a_distance", leg_a.distance, "angstrom"),
                    ("leg_b_distance", leg_b.distance, "angstrom"),
                ]
                if np.isfinite(leg_a.angle):
                    measurements.append(("leg_a_angle", leg_a.angle, "degree"))
                if np.isfinite(leg_b.angle):
                    measurements.append(("leg_b_angle", leg_b.angle, "degree"))
                records.append(
                    _Observation(
                        "water_bridge",
                        leg_a.protein_atom.get_parent(),
                        leg_b.protein_atom.get_parent(),
                        leg_a.protein_atom.get_name(), leg_b.protein_atom.get_name(),
                        max(leg_a.distance, leg_b.distance), minimum_angle,
                        f"chain A: {leg_a.direction}; chain B: {leg_b.direction}",
                        _water_label(water),
                        measurements=tuple(measurements),
                        criteria=(
                            (
                                "leg_a_distance",
                                leg_a.distance <= config.hydrogen_bond_distance,
                                f"<= {config.hydrogen_bond_distance} angstrom",
                            ),
                            (
                                "leg_b_distance",
                                leg_b.distance <= config.hydrogen_bond_distance,
                                f"<= {config.hydrogen_bond_distance} angstrom",
                            ),
                            (
                                "joint_orientation",
                                evidence == "explicit_geometry",
                                "both bridge legs orientation-supported",
                            ),
                        ),
                        role_a=(
                            "donor"
                            if leg_a.direction.startswith("protein donor")
                            else "acceptor"
                        ),
                        role_b=(
                            "donor"
                            if leg_b.direction.startswith("protein donor")
                            else "acceptor"
                        ),
                        evidence_level=evidence,
                        quality_flags=(
                            ("unoriented_water",)
                            if evidence == "distance_candidate"
                            else ()
                        ),
                        water=water,
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
    measurement_values = {name: float(value) for name, value, _ in observation.measurements}
    measurement_units = {name: unit for name, _, unit in observation.measurements}
    satisfied = [name for name, status, _ in observation.criteria if status]
    violated = [name for name, status, _ in observation.criteria if not status]
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
        "geometry_metrics": measurement_values,
        "geometry_units": measurement_units,
        "roles": {"a": observation.role_a, "b": observation.role_b},
        "satisfied_criteria": satisfied,
        "violated_criteria": violated,
        "evidence_level": observation.evidence_level,
        "quality_flags": list(observation.quality_flags),
        "observation_count": observation.member_count,
        "independent_group_count": observation.independent_group_count,
        "distance_statistics": {
            "min": (
                observation.distance_min
                if observation.distance_min is not None
                else observation.distance
            ),
            "median": (
                observation.distance_median
                if observation.distance_median is not None
                else observation.distance
            ),
            "max": (
                observation.distance_max
                if observation.distance_max is not None
                else observation.distance
            ),
        },
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
        distances = [item.distance for item in candidates if item.distance is not None]
        physical_groups = {
            (item.atom_a, item.atom_b, item.mediator) for item in candidates
        }
        representative = replace(
            representative,
            mediator=",".join(mediators) if mediators else representative.mediator,
            member_count=len(candidates),
            independent_group_count=len(physical_groups),
            distance_min=min(distances) if distances else None,
            distance_median=float(np.median(distances)) if distances else None,
            distance_max=max(distances) if distances else None,
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
    config: ContactConfig | None = None,
    adjacency_override: Any | None = None,
) -> list[_Observation]:
    config = config or get_contact_config()
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
    adjacency = adjacency_override or _build_bond_adjacency(
        model, topology_residues, topology_backend
    )

    include_charged = salt_bridge or cation_pi_candidate
    groups_a = _charged_groups(residues_a, adjacency, config) if include_charged else []
    groups_b = _charged_groups(residues_b, adjacency, config) if include_charged else []
    include_aromatic = (
        pi_stacking_parallel or pi_stacking_t_shaped or cation_pi_candidate
    )
    rings_a = _aromatic_rings(residues_a, config) if include_aromatic else []
    rings_b = _aromatic_rings(residues_b, config) if include_aromatic else []

    observations: list[_Observation] = []
    if hydrogen_bond:
        observations.extend(_hydrogen_bonds(atoms_a, atoms_b, adjacency, config))
    if salt_bridge:
        observations.extend(_salt_bridges(groups_a, groups_b, config))
    if hydrophobic_contact:
        observations.extend(_hydrophobic_contacts(atoms_a, atoms_b, adjacency, config))
    if van_der_waals_contact:
        observations.extend(_vdw_contacts(atoms_a, atoms_b, adjacency, config))
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
                config,
            )
        )
    if water_bridge:
        observations.extend(
            _water_bridges(
                atoms_a,
                atoms_b,
                water_residues,
                adjacency,
                config,
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
    profile: str | ContactConfig = "refined",
) -> list[dict[str, Any]]:
    """Characterize unique noncovalent residue pairs within one chain."""
    if (
        isinstance(min_sequence_separation, bool)
        or not isinstance(min_sequence_separation, int)
        or min_sequence_separation < 1
    ):
        raise ValueError("min_sequence_separation must be an integer >= 1")

    model, chain_object = _select_model_and_chain(structure, chain)
    config = get_contact_config(profile)
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
        config=config,
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
            geometry = observation.geometry
            if geometry:
                geometry = (
                    geometry.replace("chain A", "__partner__")
                    .replace("chain B", "chain A")
                    .replace("__partner__", "chain B")
                )
            observation = replace(
                observation,
                residue_a=observation.residue_b,
                residue_b=observation.residue_a,
                atom_a=observation.atom_b,
                atom_b=observation.atom_a,
                role_a=observation.role_b,
                role_b=observation.role_a,
                geometry=geometry,
            )
        normalized.append(observation)

    records = [
        _record(structure, chain, chain, observation)
        for observation in _aggregate(normalized)
    ]
    for record in records:
        record["rule_profile"] = config.name
    return records


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
    profile: str | ContactConfig = "refined",
) -> list[dict[str, Any]]:
    """Implement :func:`biotools.structure.geometry.characterize_chain_contacts`."""
    config = get_contact_config(profile)
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
        config=config,
    )

    if not atomic:
        observations = _aggregate(observations)
    records = [_record(structure, chain_a, chain_b, observation) for observation in observations]
    for record in records:
        record["rule_profile"] = config.name
    return records
