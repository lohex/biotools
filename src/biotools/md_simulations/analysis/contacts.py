"""Apply the eight snapshot contact detectors to every selected MD frame.

Topology and atom identities are prepared once.  The reusable detector state
owns a private structure copy; each frame only replaces its coordinates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBParser

from ...structure import _contacts
from ...structure.contacts import (
    ContactConfig, ContactDiagnostic, get_contact_config, prepare_contact_system,
)

_CONTACT_TYPES = (
    "hydrogen_bond",
    "salt_bridge",
    "hydrophobic_contact",
    "van_der_waals_contact",
    "pi_stacking_parallel",
    "cation_pi_candidate",
    "water_bridge",
    "pi_stacking_t_shaped",
)


@dataclass(frozen=True)
class TrajectoryFrame:
    """Coordinates and optional row-vector periodic box, both in angstroms."""

    coordinates: Any
    box_vectors: Any | None = None
    frame_index: int | None = None


@dataclass(frozen=True, order=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str


@dataclass(frozen=True)
class ContactFrequency:
    partner_a: ResiduePosition
    partner_b: ResiduePosition
    interaction_type: str
    positive_frame_count: int
    valid_frame_count: int
    frequency: float
    first_observed_frame: int
    last_observed_frame: int


@dataclass(frozen=True)
class TrajectoryContactResult:
    frequencies: tuple[ContactFrequency, ...]
    valid_frame_count: int
    first_analyzed_frame: int
    last_analyzed_frame: int
    partners: tuple[str, str]
    contact_types: tuple[str, ...]
    rule_profile: str
    config_hash: str
    reference_structure_hash: str
    periodic: bool
    selection_start: int
    selection_stop: int | None
    selection_step: int
    diagnostics: tuple[ContactDiagnostic, ...]

    def to_records(self) -> list[dict[str, Any]]:
        """Return flat, CSV-ready residue-pair frequency records."""
        return [
            {
                "partner_a_chain": item.partner_a.chain_id,
                "partner_a_position": item.partner_a.residue_number,
                "partner_a_insertion_code": item.partner_a.insertion_code,
                "partner_a_residue": item.partner_a.residue_name,
                "partner_b_chain": item.partner_b.chain_id,
                "partner_b_position": item.partner_b.residue_number,
                "partner_b_insertion_code": item.partner_b.insertion_code,
                "partner_b_residue": item.partner_b.residue_name,
                "interaction_type": item.interaction_type,
                "positive_frame_count": item.positive_frame_count,
                "valid_frame_count": item.valid_frame_count,
                "frequency": item.frequency,
                "first_observed_frame": item.first_observed_frame,
                "last_observed_frame": item.last_observed_frame,
            }
            for item in self.frequencies
        ]


def _position(residue: Any) -> ResiduePosition:
    _, number, insertion = residue.id
    return ResiduePosition(
        str(residue.get_parent().id), int(number), str(insertion).strip(),
        str(residue.resname).strip(),
    )


def _minimum_image(delta: np.ndarray, box: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    fractional = delta @ inverse
    return (fractional - np.rint(fractional)) @ box


class PreparedTrajectoryContacts:
    """Reusable contact chemistry and stable atom order for one topology."""

    def __init__(
        self, topology: Any, partners: tuple[str, str], *,
        profile: str | ContactConfig = "refined",
        contact_types: Iterable[str] = _CONTACT_TYPES,
    ) -> None:
        self.structure = deepcopy(topology)
        if len(tuple(self.structure.get_models())) != 1:
            raise ValueError("Trajectory topology must contain exactly one model")
        self.system = prepare_contact_system(self.structure, partners)
        self.config = get_contact_config(profile)
        self.partners = tuple(partners)
        if len(self.partners) != 2 or self.partners[0] == self.partners[1]:
            raise ValueError("partners must contain two distinct chain IDs")
        self.contact_types = frozenset(contact_types)
        unknown = self.contact_types - set(_CONTACT_TYPES)
        if unknown:
            raise ValueError(f"Unknown contact types: {sorted(unknown)}")
        self._residues_a = list(self.system.residues_a)
        self._residues_b = list(self.system.residues_b)
        self._positions = {
            residue: _position(residue)
            for residue in self._residues_a + self._residues_b
        }
        self._detector_options = {
            name: name in self.contact_types for name in _CONTACT_TYPES
        }
        self.atoms = tuple(self.structure.get_atoms())
        self.atom_count = len(self.atoms)
        self._indices = {atom: i for i, atom in enumerate(self.atoms)}
        self._bond_neighbors = {
            self._indices[atom]: tuple(self._indices[other] for other in neighbors)
            for atom, neighbors in self.system.adjacency.items()
        }
        # The spanning forest depends only on topology, not frame coordinates.
        seen: set[int] = set()
        spanning_edges: list[tuple[int, int]] = []
        for root in self._bond_neighbors:
            if root in seen:
                continue
            seen.add(root)
            frontier = [root]
            while frontier:
                source = frontier.pop()
                for target in self._bond_neighbors.get(source, ()):
                    if target not in seen:
                        seen.add(target)
                        spanning_edges.append((source, target))
                        frontier.append(target)
        self._spanning_edges = tuple(spanning_edges)
        self._chain_indices = {
            chain: np.array([
                self._indices[atom]
                for residue in residues for atom in residue.get_atoms()
            ], dtype=int)
            for chain, residues in (
                (self.partners[0], self.system.residues_a),
                (self.partners[1], self.system.residues_b),
            )
        }
        self._water_indices = tuple(
            np.array([self._indices[atom] for atom in water.get_atoms()], dtype=int)
            for water in self.system.water_residues
        )

    def _periodic_coordinates(self, xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
        inverse = np.linalg.inv(box)
        result = xyz.copy()
        # Unwrap along the once-prepared covalent spanning forest.
        for source, target in self._spanning_edges:
            result[target] = result[source] + _minimum_image(
                xyz[target] - xyz[source], box, inverse
            )
        # Align the two partners by whole-molecule translation.  No atom is
        # independently wrapped after the covalent geometry is reconstructed.
        a = self._chain_indices[self.partners[0]]
        b = self._chain_indices[self.partners[1]]
        center_a = result[a].mean(axis=0)
        center_b = result[b].mean(axis=0)
        displacement = center_b - center_a
        result[b] += _minimum_image(displacement, box, inverse) - displacement
        interface_center = (center_a + result[b].mean(axis=0)) / 2
        for water in self._water_indices:
            center = result[water].mean(axis=0)
            displacement = center - interface_center
            result[water] += _minimum_image(displacement, box, inverse) - displacement
        return result

    def analyze_frame(
        self, frame: TrajectoryFrame | Any, *, periodic: bool = False
    ) -> frozenset[tuple[ResiduePosition, ResiduePosition, str]]:
        """Return one hit per residue pair and type; fail on invalid frames."""
        if not isinstance(frame, TrajectoryFrame):
            frame = TrajectoryFrame(frame)
        xyz = np.asarray(frame.coordinates, dtype=float)
        if xyz.shape != (self.atom_count, 3) or not np.isfinite(xyz).all():
            raise ValueError(
                f"Frame coordinates must be finite with shape ({self.atom_count}, 3)"
            )
        if periodic:
            if frame.box_vectors is None:
                raise ValueError("Periodic analysis requires box vectors in every frame")
            box = np.asarray(frame.box_vectors, dtype=float)
            if box.shape != (3, 3) or not np.isfinite(box).all() or abs(np.linalg.det(box)) < 1e-8:
                raise ValueError("Periodic box must be a finite, nonsingular 3x3 matrix")
            xyz = self._periodic_coordinates(xyz, box)
        for atom, position in zip(self.atoms, xyz):
            atom.set_coord(position)
        raw = _contacts._detect_contact_observations(
            self.system.model, self._residues_a,
            self._residues_b,
            topology_backend=self.system.topology_backend,
            adjacency_override=self.system.adjacency,
            config=self.config, **self._detector_options,
        )
        return frozenset(
            (self._positions[item.residue_a], self._positions[item.residue_b], item.interaction_type)
            for item in raw
        )


def prepare_trajectory_contacts(
    topology: Any, partners: tuple[str, str], *,
    profile: str | ContactConfig = "refined",
    contact_types: Iterable[str] = _CONTACT_TYPES,
) -> PreparedTrajectoryContacts:
    """Prepare a structure copy and contact chemistry once for many frames."""
    if isinstance(topology, (str, PathLike)):
        topology = PDBParser(QUIET=True).get_structure("trajectory", str(topology))
    return PreparedTrajectoryContacts(
        topology, partners, profile=profile, contact_types=contact_types
    )


def _file_frames(path: str | PathLike[str]) -> Iterator[TrajectoryFrame]:
    """Read DCD/XTC one frame at a time using Biotite's trajectory backend."""
    from biotite.structure.io import dcd, xtc

    suffix = Path(path).suffix.lower()
    reader = {".dcd": dcd.DCDFile, ".xtc": xtc.XTCFile}.get(suffix)
    if reader is None:
        raise ValueError("trajectory must be a DCD or XTC file")
    with reader.traj_type()(str(path), "r") as stream:
        index = 0
        while True:
            values = stream.read(n_frames=1)
            if len(values[0]) == 0:
                return
            coordinates, boxes, _ = reader.process_read_values(values)
            box = None if boxes is None else boxes[0]
            yield TrajectoryFrame(coordinates[0], box, index)
            index += 1


def analyze_trajectory_contacts(
    topology: Any,
    trajectory: Iterable[TrajectoryFrame | Any] | str | PathLike[str] | None = None,
    *,
    partners: tuple[str, str],
    profile: str | ContactConfig = "refined",
    contact_types: Iterable[str] = _CONTACT_TYPES,
    periodic: bool = False,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> TrajectoryContactResult:
    """Count each residue-pair interaction at most once per selected frame.

    Invalid frames raise immediately.  Frequency denominators are identical
    for every reported pair and equal the number of selected analyzed frames.
    """
    if trajectory is None:
        if not (hasattr(topology, "output_path") and hasattr(topology, "trajectory_path")):
            raise ValueError("trajectory is required unless topology is a production result")
        trajectory = topology.trajectory_path
        topology = topology.output_path
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if stop is not None and (isinstance(stop, bool) or not isinstance(stop, int) or stop <= start):
        raise ValueError("stop must be an integer greater than start")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    prepared = prepare_trajectory_contacts(
        topology, partners, profile=profile, contact_types=contact_types
    )
    frames = _file_frames(trajectory) if isinstance(trajectory, (str, PathLike)) else iter(trajectory)
    counts: Counter[tuple[ResiduePosition, ResiduePosition, str]] = Counter()
    first: dict[tuple[ResiduePosition, ResiduePosition, str], int] = {}
    last: dict[tuple[ResiduePosition, ResiduePosition, str], int] = {}
    valid_frame_count = 0
    first_analyzed_frame: int | None = None
    last_analyzed_frame: int | None = None
    for ordinal, frame in enumerate(frames):
        if ordinal < start:
            continue
        if stop is not None and ordinal >= stop:
            break
        if (ordinal - start) % step:
            continue
        index = frame.frame_index if isinstance(frame, TrajectoryFrame) and frame.frame_index is not None else ordinal
        try:
            hits = prepared.analyze_frame(frame, periodic=periodic)
        except (ValueError, IndexError, KeyError) as exc:
            raise ValueError(f"Frame {index} is not evaluable: {exc}") from exc
        counts.update(hits)
        for hit in hits:
            first.setdefault(hit, index)
            last[hit] = index
        valid_frame_count += 1
        if first_analyzed_frame is None:
            first_analyzed_frame = index
        last_analyzed_frame = index
    if valid_frame_count == 0:
        raise ValueError("No trajectory frames were selected")
    frequency_rows = tuple(
        ContactFrequency(
            *key, count, valid_frame_count, count / valid_frame_count,
            first[key], last[key],
        )
        for key, count in sorted(counts.items())
    )
    return TrajectoryContactResult(
        frequency_rows, valid_frame_count,
        first_analyzed_frame, last_analyzed_frame, prepared.partners,
        tuple(kind for kind in _CONTACT_TYPES if kind in prepared.contact_types),
        prepared.config.name, prepared.config.config_hash,
        prepared.system.input_hash, periodic, start, stop, step,
        prepared.system.diagnostics,
    )
