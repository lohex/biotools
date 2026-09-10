"""Deterministic local modeling of single-water bridges.

This module deliberately implements a dependency-light geometric backend.
It works on :class:`~biotools.structure.contacts.ContactSystem`, never mutates
the source structure, preserves both bridge legs, and independently evaluates
the final geometry after proposal or optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from . import _contacts
from .contacts.config import ContactConfig, get_contact_config
from .contacts.models import AtomReference, ContactDiagnostic, ContactSystem


@dataclass(frozen=True)
class WaterModelConfig:
    """Settings for local rigid-water proposal and refinement."""

    oxygen_hydrogen_distance: float = 0.9572
    hydrogen_oxygen_hydrogen_angle: float = 104.52
    target_anchor_distance: float = 2.8
    cluster_distance: float = 0.75
    clash_overlap_tolerance: float = 0.6
    orientations_per_pair: int = 8
    max_candidates: int = 256
    optimization_step: float = 0.20
    optimization_min_step: float = 0.0125
    optimization_iterations: int = 80


@dataclass(frozen=True)
class WaterBridgeLeg:
    partner: str
    anchor: AtomReference
    protein_role: str
    water_role: str
    distance: float
    angle: float | None
    distance_satisfied: bool
    angle_satisfied: bool | None


@dataclass(frozen=True)
class WaterOrientation:
    hydrogen_positions: tuple[tuple[float, float, float], tuple[float, float, float]]
    legs: tuple[WaterBridgeLeg, ...]
    score_components: Mapping[str, float]
    accepted: bool


@dataclass(frozen=True)
class WaterSite:
    site_id: str
    oxygen_position: tuple[float, float, float]
    provenance: str
    orientations: tuple[WaterOrientation, ...]
    anchor_pairs: tuple[tuple[AtomReference, AtomReference], ...]
    status: str
    source_water: AtomReference | None = None
    crystal_occupancy: float | None = None
    b_factor: float | None = None
    displacement: float = 0.0


@dataclass(frozen=True)
class WaterAnalysisResult:
    sites: tuple[WaterSite, ...]
    diagnostics: tuple[ContactDiagnostic, ...] = ()
    backend: str = "geometric"
    config: Mapping[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Anchor:
    partner: str
    atom: Any
    reference: AtomReference
    role: str
    donor_hydrogens: tuple[Any, ...] = ()


class _OpenMMWaterEnergy:
    """Reusable nonperiodic full-environment energy evaluator."""

    def __init__(self, system: ContactSystem) -> None:
        try:
            from openmm import Context, Platform, Vec3, VerletIntegrator, unit
            from openmm.app import ForceField, NoCutoff, Topology, element
        except ModuleNotFoundError as exc:
            if exc.name == "openmm" or (exc.name or "").startswith("openmm."):
                raise ModuleNotFoundError(
                    "The openmm_rigid_water backend requires the optional "
                    "contacts dependencies. Install them with: pip install "
                    "'biotools[contacts]'"
                ) from exc
            raise

        topology = Topology()
        positions = []
        for source_chain in system.model:
            target_chain = topology.addChain(str(source_chain.id))
            for source_residue in source_chain:
                hetero, number, insertion = source_residue.id
                del hetero
                target_residue = topology.addResidue(
                    _contacts._residue_name(source_residue),
                    target_chain,
                    id=str(number),
                    insertionCode=str(insertion).strip(),
                )
                for source_atom in source_residue.get_atoms():
                    symbol = _contacts._element(source_atom)
                    try:
                        atom_element = element.get_by_symbol(symbol)
                    except (KeyError, ValueError):
                        atom_element = None
                    serial = source_atom.get_serial_number()
                    topology.addAtom(
                        str(source_atom.get_name()).strip(),
                        atom_element,
                        target_residue,
                        id=str(serial) if serial is not None else None,
                    )
                    positions.append(Vec3(*map(float, _contacts._coord(source_atom))))
        topology.createStandardBonds()
        topology.createDisulfideBonds(unit.Quantity(positions, unit.angstrom))

        water_chain = topology.addChain("_MODELED_WATER")
        water_residue = topology.addResidue("HOH", water_chain, id="1")
        oxygen = topology.addAtom("O", element.oxygen, water_residue)
        hydrogen1 = topology.addAtom("H1", element.hydrogen, water_residue)
        hydrogen2 = topology.addAtom("H2", element.hydrogen, water_residue)
        topology.addBond(oxygen, hydrogen1)
        topology.addBond(oxygen, hydrogen2)
        positions.extend((Vec3(0, 0, 0), Vec3(0.9572, 0, 0), Vec3(-0.2399872, 0.9266272, 0)))

        forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        try:
            openmm_system = forcefield.createSystem(
                topology, nonbondedMethod=NoCutoff, constraints=None, rigidWater=True
            )
        except Exception as exc:
            raise ValueError(
                "OpenMM could not parameterize the complete selected model. "
                "Prepare all residues and explicit waters before local energy refinement."
            ) from exc
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        try:
            platform = Platform.getPlatformByName("CPU")
        except Exception:
            platform = Platform.getPlatformByName("Reference")
        self._context = Context(openmm_system, integrator, platform)
        self._integrator = integrator
        self._positions = positions
        self._unit = unit
        self._vec3 = Vec3

    def energy(
        self,
        oxygen: np.ndarray,
        hydrogens: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> float:
        positions = list(self._positions)
        positions[-3:] = [
            self._vec3(*map(float, oxygen)),
            self._vec3(*map(float, hydrogens[0])),
            self._vec3(*map(float, hydrogens[1])),
        ]
        self._context.setPositions(self._unit.Quantity(positions, self._unit.angstrom))
        state = self._context.getState(getEnergy=True)
        energy = float(
            state.getPotentialEnergy().value_in_unit(self._unit.kilojoule_per_mole)
        )
        if not math.isfinite(energy):
            raise ValueError("OpenMM returned a nonfinite potential energy")
        return energy


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, trial))) > 0.85:
        trial = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(axis, trial))
    assert first is not None
    second = np.cross(axis, first)
    return first, second


def _anchors(system: ContactSystem, residues: Iterable[Any], partner: str) -> tuple[_Anchor, ...]:
    atoms = [atom for residue in residues for atom in residue.get_atoms()]
    anchors: list[_Anchor] = []
    donor_atoms: dict[Any, list[Any]] = {}
    for donor, hydrogen in _contacts._donor_pairs(atoms, system.adjacency):
        donor_atoms.setdefault(donor, []).append(hydrogen)
    for atom, hydrogens in donor_atoms.items():
        anchors.append(
            _Anchor(
                partner,
                atom,
                system.atom_references[atom],
                "donor",
                tuple(hydrogens),
            )
        )
    for atom in atoms:
        if _contacts._is_acceptor(atom, system.adjacency):
            anchors.append(
                _Anchor(partner, atom, system.atom_references[atom], "acceptor")
            )
    return tuple(
        sorted(
            anchors,
            key=lambda item: (item.reference, item.role),
        )
    )


def _all_anchors(system: ContactSystem) -> tuple[tuple[_Anchor, ...], tuple[_Anchor, ...]]:
    return (
        _anchors(system, system.residues_a, "a"),
        _anchors(system, system.residues_b, "b"),
    )


def _orientation_from_directions(
    oxygen: np.ndarray,
    directions: list[np.ndarray],
    model: WaterModelConfig,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(model.hydrogen_oxygen_hydrogen_angle)
    if directions:
        first = _unit(directions[0])
    else:
        first = np.array([1.0, 0.0, 0.0])
    if first is None:
        first = np.array([1.0, 0.0, 0.0])
    if len(directions) >= 2:
        second_target = _unit(directions[1])
        if second_target is not None:
            perpendicular = second_target - np.dot(second_target, first) * first
            perpendicular = _unit(perpendicular)
        else:
            perpendicular = None
    else:
        perpendicular = None
    basis1, basis2 = _perpendicular_basis(first)
    if perpendicular is None:
        perpendicular = math.cos(phase) * basis1 + math.sin(phase) * basis2
    second = math.cos(angle) * first + math.sin(angle) * perpendicular
    bond = model.oxygen_hydrogen_distance
    return oxygen + bond * first, oxygen + bond * second


def _angle(point1: np.ndarray, vertex: np.ndarray, point3: np.ndarray) -> float | None:
    value = _contacts._angle_degrees(point1, vertex, point3)
    return float(value) if np.isfinite(value) else None


def _evaluate_leg(
    anchor: _Anchor,
    oxygen: np.ndarray,
    hydrogens: tuple[np.ndarray, np.ndarray],
    contact: ContactConfig,
) -> WaterBridgeLeg:
    anchor_position = _contacts._coord(anchor.atom)
    distance = float(np.linalg.norm(anchor_position - oxygen))
    if anchor.role == "donor":
        angles = [
            _angle(anchor_position, _contacts._coord(hydrogen), oxygen)
            for hydrogen in anchor.donor_hydrogens
        ]
        valid = [value for value in angles if value is not None]
        angle = max(valid) if valid else None
        water_role = "acceptor"
    else:
        angles = [_angle(oxygen, hydrogen, anchor_position) for hydrogen in hydrogens]
        valid = [value for value in angles if value is not None]
        angle = max(valid) if valid else None
        water_role = "donor"
    return WaterBridgeLeg(
        partner=anchor.partner,
        anchor=anchor.reference,
        protein_role=anchor.role,
        water_role=water_role,
        distance=distance,
        angle=angle,
        distance_satisfied=distance <= contact.hydrogen_bond_distance,
        angle_satisfied=(angle >= contact.hydrogen_bond_angle) if angle is not None else None,
    )


def _clash_penalty(
    system: ContactSystem,
    oxygen: np.ndarray,
    excluded: set[Any],
    model: WaterModelConfig,
) -> tuple[float, bool]:
    penalty = 0.0
    massive = False
    oxygen_radius = _contacts.VDW_RADII_A["O"]
    for atom, reference in system.atom_references.items():
        del reference
        if atom in excluded or _contacts._element(atom) == "H":
            continue
        radius = _contacts.VDW_RADII_A.get(_contacts._element(atom))
        if radius is None:
            continue
        distance = float(np.linalg.norm(oxygen - _contacts._coord(atom)))
        if (
            distance <= 1e-8
            and _contacts._residue_name(atom.get_parent())
            in _contacts.WATER_RESIDUE_NAMES
        ):
            # The oxygen of an observed site is the object being evaluated,
            # not an environmental clash.
            continue
        overlap = oxygen_radius + radius - distance
        allowed = model.clash_overlap_tolerance
        if overlap > allowed:
            penalty += (overlap - allowed) ** 2 * 25.0
        if overlap > 1.2:
            massive = True
    return penalty, massive


def _evaluate_orientation(
    system: ContactSystem,
    oxygen: np.ndarray,
    hydrogens: tuple[np.ndarray, np.ndarray],
    anchors: tuple[_Anchor, _Anchor],
    contact: ContactConfig,
    model: WaterModelConfig,
) -> WaterOrientation:
    legs = tuple(_evaluate_leg(anchor, oxygen, hydrogens, contact) for anchor in anchors)
    distance_penalty = sum(
        (leg.distance - model.target_anchor_distance) ** 2 for leg in legs
    )
    direction_penalty = sum(
        4.0
        if leg.angle is None
        else (max(0.0, contact.hydrogen_bond_angle - leg.angle) / 30.0) ** 2
        for leg in legs
    )
    clash_penalty, massive_clash = _clash_penalty(
        system, oxygen, {anchor.atom for anchor in anchors}, model
    )
    capacity_penalty = 0.0
    accepted = (
        all(leg.distance_satisfied and leg.angle_satisfied is True for leg in legs)
        and not massive_clash
    )
    score = distance_penalty + direction_penalty + clash_penalty + capacity_penalty
    return WaterOrientation(
        hydrogen_positions=(tuple(map(float, hydrogens[0])), tuple(map(float, hydrogens[1]))),
        legs=legs,
        score_components=MappingProxyType(
            {
                "distance_penalty": distance_penalty,
                "direction_penalty": direction_penalty,
                "clash_penalty": clash_penalty,
                "capacity_penalty": capacity_penalty,
                "total": score,
            }
        ),
        accepted=accepted,
    )


def _candidate_orientations(
    system: ContactSystem,
    oxygen: np.ndarray,
    anchors: tuple[_Anchor, _Anchor],
    contact: ContactConfig,
    model: WaterModelConfig,
) -> tuple[WaterOrientation, ...]:
    acceptor_directions = [
        _contacts._coord(anchor.atom) - oxygen
        for anchor in anchors
        if anchor.role == "acceptor"
    ]
    orientations = []
    count = max(1, model.orientations_per_pair)
    for index in range(count):
        phase = 2.0 * math.pi * index / count
        hydrogens = _orientation_from_directions(
            oxygen, acceptor_directions, model, phase
        )
        orientations.append(
            _evaluate_orientation(system, oxygen, hydrogens, anchors, contact, model)
        )
    unique: dict[tuple[tuple[float, ...], tuple[float, ...]], WaterOrientation] = {}
    for orientation in orientations:
        key = tuple(
            tuple(round(value, 8) for value in position)
            for position in orientation.hydrogen_positions
        )
        unique[key] = orientation
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.score_components["total"],
                item.hydrogen_positions,
            ),
        )
    )


def _site_id(prefix: str, oxygen: np.ndarray, anchors: tuple[_Anchor, _Anchor]) -> str:
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(np.round(oxygen, 6).astype("<f8").tobytes())
    for anchor in anchors:
        digest.update(repr(anchor.reference).encode("utf-8"))
    return f"{prefix}-{digest.hexdigest()[:16]}"


def _anchor_pairs_at_oxygen(
    anchors_a: tuple[_Anchor, ...],
    anchors_b: tuple[_Anchor, ...],
    oxygen: np.ndarray,
    contact: ContactConfig,
) -> list[tuple[_Anchor, _Anchor]]:
    return [
        (anchor_a, anchor_b)
        for anchor_a in anchors_a
        for anchor_b in anchors_b
        if np.linalg.norm(_contacts._coord(anchor_a.atom) - oxygen)
        <= contact.hydrogen_bond_distance
        and np.linalg.norm(_contacts._coord(anchor_b.atom) - oxygen)
        <= contact.hydrogen_bond_distance
    ]


def orient_existing_waters(
    prepared: ContactSystem,
    *,
    profile: str | ContactConfig = "refined",
    config: WaterModelConfig = WaterModelConfig(),
) -> WaterAnalysisResult:
    """Orient rigid H2O geometry at each observed oxygen position."""
    contact = get_contact_config(profile)
    anchors_a, anchors_b = _all_anchors(prepared)
    sites: list[WaterSite] = []
    for water in prepared.water_residues:
        oxygen_atom = next(
            (atom for atom in water.get_atoms() if _contacts._element(atom) == "O"),
            None,
        )
        if oxygen_atom is None:
            continue
        oxygen = _contacts._coord(oxygen_atom)
        pairs = _anchor_pairs_at_oxygen(anchors_a, anchors_b, oxygen, contact)
        if not pairs:
            continue
        orientations = tuple(
            orientation
            for pair in pairs
            for orientation in _candidate_orientations(
                prepared, oxygen, pair, contact, config
            )
        )
        orientations = tuple(
            sorted(orientations, key=lambda item: item.score_components["total"])
        )
        occupancy = getattr(oxygen_atom, "get_occupancy", lambda: None)()
        b_factor = getattr(oxygen_atom, "get_bfactor", lambda: None)()
        source = prepared.atom_references[oxygen_atom]
        sites.append(
            WaterSite(
                site_id=_site_id("observed", oxygen, pairs[0]),
                oxygen_position=tuple(map(float, oxygen)),
                provenance="modeled_orientation_at_observed_oxygen",
                orientations=orientations,
                anchor_pairs=tuple(
                    (pair[0].reference, pair[1].reference) for pair in pairs
                ),
                status="accepted" if any(item.accepted for item in orientations) else "candidate",
                source_water=source,
                crystal_occupancy=float(occupancy) if occupancy is not None else None,
                b_factor=float(b_factor) if b_factor is not None else None,
            )
        )
    diagnostics = ()
    if not sites:
        diagnostics = (
            ContactDiagnostic(
                "no_observed_bridge_candidates",
                "No observed water oxygen lies within the configured distance of both partners",
                "info",
            ),
        )
    return WaterAnalysisResult(tuple(sites), diagnostics, "geometric", vars(config))


def _shell_candidates(
    position_a: np.ndarray,
    position_b: np.ndarray,
    model: WaterModelConfig,
) -> tuple[np.ndarray, ...]:
    displacement = position_b - position_a
    separation = float(np.linalg.norm(displacement))
    radius = model.target_anchor_distance
    if separation <= 1e-12 or separation > 2.0 * radius:
        return ()
    axis = displacement / separation
    midpoint = (position_a + position_b) / 2.0
    circle_radius = math.sqrt(max(0.0, radius * radius - (separation / 2.0) ** 2))
    if circle_radius <= 1e-8:
        return (midpoint,)
    basis1, basis2 = _perpendicular_basis(axis)
    count = max(4, model.orientations_per_pair)
    return tuple(
        midpoint
        + circle_radius
        * (
            math.cos(2 * math.pi * index / count) * basis1
            + math.sin(2 * math.pi * index / count) * basis2
        )
        for index in range(count)
    )


def propose_bridging_waters(
    prepared: ContactSystem,
    *,
    profile: str | ContactConfig = "refined",
    config: WaterModelConfig = WaterModelConfig(),
) -> WaterAnalysisResult:
    """Propose deterministic single-water sites in a dry interface."""
    contact = get_contact_config(profile)
    anchors_a, anchors_b = _all_anchors(prepared)
    raw: list[tuple[np.ndarray, tuple[_Anchor, _Anchor]]] = []
    for anchor_a in anchors_a:
        for anchor_b in anchors_b:
            for oxygen in _shell_candidates(
                _contacts._coord(anchor_a.atom), _contacts._coord(anchor_b.atom), config
            ):
                distances = (
                    float(np.linalg.norm(oxygen - _contacts._coord(anchor_a.atom))),
                    float(np.linalg.norm(oxygen - _contacts._coord(anchor_b.atom))),
                )
                if not all(
                    contact.water_anchor_min_distance <= value <= contact.water_anchor_max_distance
                    for value in distances
                ):
                    continue
                _, massive = _clash_penalty(
                    prepared, oxygen, {anchor_a.atom, anchor_b.atom}, config
                )
                if not massive:
                    raw.append((oxygen, (anchor_a, anchor_b)))
                if len(raw) >= config.max_candidates:
                    break
            if len(raw) >= config.max_candidates:
                break
        if len(raw) >= config.max_candidates:
            break

    clusters: list[list[tuple[np.ndarray, tuple[_Anchor, _Anchor]]]] = []
    for candidate in raw:
        cluster = next(
            (
                members
                for members in clusters
                if np.linalg.norm(candidate[0] - np.mean([item[0] for item in members], axis=0))
                <= config.cluster_distance
            ),
            None,
        )
        if cluster is None:
            clusters.append([candidate])
        else:
            cluster.append(candidate)

    sites = []
    for members in clusters:
        oxygen = np.mean([item[0] for item in members], axis=0)
        pairs = []
        for _, pair in members:
            if pair not in pairs:
                pairs.append(pair)
        orientations = tuple(
            orientation
            for pair in pairs
            for orientation in _candidate_orientations(
                prepared, oxygen, pair, contact, config
            )
        )
        orientations = tuple(
            sorted(orientations, key=lambda item: item.score_components["total"])
        )
        sites.append(
            WaterSite(
                site_id=_site_id("proposed", oxygen, pairs[0]),
                oxygen_position=tuple(map(float, oxygen)),
                provenance="geometric_shell_intersection",
                orientations=orientations,
                anchor_pairs=tuple((a.reference, b.reference) for a, b in pairs),
                status="accepted" if any(item.accepted for item in orientations) else "candidate",
            )
        )
    sites.sort(key=lambda site: (site.orientations[0].score_components["total"], site.site_id))
    diagnostics = ()
    if not sites:
        diagnostics = (
            ContactDiagnostic(
                "no_water_proposals",
                "No clash-free cross-partner anchor geometry produced a water proposal",
                "info",
            ),
        )
    return WaterAnalysisResult(tuple(sites), diagnostics, "geometric", vars(config))


def _anchor_lookup(system: ContactSystem) -> dict[AtomReference, _Anchor]:
    anchors_a, anchors_b = _all_anchors(system)
    return {anchor.reference: anchor for anchor in anchors_a + anchors_b}


def optimize_bridging_waters(
    prepared: ContactSystem,
    proposals: WaterAnalysisResult,
    *,
    profile: str | ContactConfig = "refined",
    config: WaterModelConfig = WaterModelConfig(),
    backend: str = "geometric",
) -> WaterAnalysisResult:
    """Locally refine proposed oxygen positions by deterministic coordinate search.

    The geometric backend is a plausibility refinement, not an energy or
    occupancy estimate. ``openmm_rigid_water`` compares the same full,
    nonperiodic fixed-solute environment for every hypothesis and still uses
    the independent geometric evaluator for final acceptance.
    """
    if backend not in {"geometric", "openmm_rigid_water"}:
        raise ValueError("backend must be 'geometric' or 'openmm_rigid_water'")
    contact = get_contact_config(profile)
    lookup = _anchor_lookup(prepared)
    energy_backend = _OpenMMWaterEnergy(prepared) if backend == "openmm_rigid_water" else None
    optimized_sites = []
    directions = np.vstack((np.eye(3), -np.eye(3)))
    for site in proposals.sites:
        if not site.anchor_pairs:
            optimized_sites.append(replace(site, status="failed_no_anchors"))
            continue
        references = site.anchor_pairs[0]
        try:
            anchors = (lookup[references[0]], lookup[references[1]])
        except KeyError:
            optimized_sites.append(replace(site, status="failed_anchor_mapping"))
            continue
        initial = np.asarray(site.oxygen_position, dtype=float)
        oxygen = initial.copy()

        def orientations_at(position: np.ndarray) -> tuple[WaterOrientation, ...]:
            orientations = _candidate_orientations(
                prepared, position, anchors, contact, config
            )
            if energy_backend is None:
                return orientations
            with_energies = []
            for orientation in orientations:
                energy = energy_backend.energy(
                    position, orientation.hydrogen_positions
                )
                components = dict(orientation.score_components)
                components["potential_energy_kj_mol"] = energy
                # The geometry term prevents an energy-only optimum from
                # escaping its bridge hypothesis. Total energies are only
                # comparable within this unchanged parameterized environment.
                components["total"] = energy + 100.0 * (
                    components["distance_penalty"]
                    + components["direction_penalty"]
                    + components["clash_penalty"]
                    + components["capacity_penalty"]
                )
                with_energies.append(
                    replace(
                        orientation,
                        score_components=MappingProxyType(components),
                    )
                )
            return tuple(
                sorted(
                    with_energies,
                    key=lambda item: item.score_components["total"],
                )
            )

        best_orientations = orientations_at(oxygen)
        best_score = best_orientations[0].score_components["total"]
        step = config.optimization_step
        iterations = 0
        while step >= config.optimization_min_step and iterations < config.optimization_iterations:
            iterations += 1
            choices = []
            for direction in directions:
                candidate = oxygen + step * direction
                candidate_orientations = orientations_at(candidate)
                choices.append(
                    (
                        candidate_orientations[0].score_components["total"],
                        tuple(candidate),
                        candidate,
                        candidate_orientations,
                    )
                )
            score, _, candidate, candidate_orientations = min(choices)
            if score + 1e-12 < best_score:
                oxygen = candidate
                best_score = score
                best_orientations = candidate_orientations
            else:
                step /= 2.0
        optimized_sites.append(
            replace(
                site,
                site_id=_site_id("optimized", oxygen, anchors),
                oxygen_position=tuple(map(float, oxygen)),
                provenance=f"{site.provenance}+{backend}_coordinate_search",
                orientations=best_orientations,
                status=(
                    "accepted"
                    if any(item.accepted for item in best_orientations)
                    else "rejected_after_optimization"
                ),
                displacement=float(np.linalg.norm(oxygen - initial)),
            )
        )
    return WaterAnalysisResult(
        tuple(optimized_sites), proposals.diagnostics, backend, vars(config)
    )


def evaluate_water_bridges(
    prepared: ContactSystem,
    sites: WaterAnalysisResult,
    *,
    profile: str | ContactConfig = "refined",
    config: WaterModelConfig = WaterModelConfig(),
) -> WaterAnalysisResult:
    """Independently recompute final bridge legs and clash acceptance."""
    contact = get_contact_config(profile)
    lookup = _anchor_lookup(prepared)
    evaluated = []
    diagnostics = list(sites.diagnostics)
    for site in sites.sites:
        oxygen = np.asarray(site.oxygen_position, dtype=float)
        orientations = []
        for pair in site.anchor_pairs:
            if pair[0] not in lookup or pair[1] not in lookup:
                continue
            anchors = (lookup[pair[0]], lookup[pair[1]])
            for old in site.orientations:
                hydrogens = tuple(
                    np.asarray(value, dtype=float)
                    for value in old.hydrogen_positions
                )
                orientations.append(
                    _evaluate_orientation(
                        prepared, oxygen, hydrogens, anchors, contact, config
                    )
                )
        orientations.sort(key=lambda item: item.score_components["total"])
        if not orientations:
            diagnostics.append(
                ContactDiagnostic(
                    "water_site_not_evaluable",
                    f"Site {site.site_id} has no mappable anchor pair or orientation",
                )
            )
            evaluated.append(replace(site, status="not_evaluable", orientations=()))
            continue
        evaluated.append(
            replace(
                site,
                orientations=tuple(orientations),
                status="accepted" if any(item.accepted for item in orientations) else "rejected",
            )
        )
    return WaterAnalysisResult(tuple(evaluated), tuple(diagnostics), sites.backend, vars(config))
