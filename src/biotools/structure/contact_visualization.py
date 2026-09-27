"""Interactive py3Dmol views of typed molecular contacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
import json
import math
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from Bio.PDB.Polypeptide import is_aa

from ._contacts import AROMATIC_RINGS
from .contacts import AtomReference, ContactAnalysisResult, ContactObservation
from .visualization import (
    _CONTACT_TYPE_COLORS,
    _CONTACT_TYPE_LABELS,
    _CONTACT_TYPES,
    _structure_pdb_text,
)

LabelMode = Literal["off", "all", "active"]
ResidueKey = tuple[str, str, int, str]
Point = tuple[float, float, float]


@dataclass(frozen=True)
class ContactStyle:
    """Rendering and sidebar style for one interaction type."""

    label: str
    color: str
    radius: float
    dash_length: float = 0.52
    gap_length: float = 0.28
    opacity: float = 0.90

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ContactStyle.label must not be empty")
        if not isinstance(self.color, str) or not self.color:
            raise ValueError("ContactStyle.color must be a nonempty string")
        for name in ("radius", "dash_length", "gap_length"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"ContactStyle.{name} must be finite and positive")
        if not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError("ContactStyle.opacity must be between 0 and 1")


DEFAULT_CONTACT_STYLES: Mapping[str, ContactStyle] = {
    name: ContactStyle(
        label=_CONTACT_TYPE_LABELS[name].capitalize(),
        color=_CONTACT_TYPE_COLORS[name],
        radius=0.075 if name == "van_der_waals_contact" else 0.11,
    )
    for name in _CONTACT_TYPES
}


@dataclass(frozen=True)
class RingPlane:
    center: Point
    normal: Point
    radius: float


@dataclass(frozen=True)
class DisplayedContact:
    """Resolved geometry for one observation and a deterministic pair ID."""

    pair_id: str
    interaction_type: str
    start: Point
    end: Point
    anchor: Point
    distance: float
    residues: tuple[ResidueKey, ...]
    ring_planes: tuple[RingPlane, ...] = ()
    water: Point | None = None
    label: str = ""


def _point(value: Any) -> Point:
    xyz = np.asarray(value, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("Contact coordinates must be finite 3D points")
    return tuple(float(item) for item in xyz)


def _xyz(value: Point) -> dict[str, float]:
    return dict(zip(("x", "y", "z"), value))


def _residue_key(ref: AtomReference) -> ResidueKey:
    return (ref.chain_id, ref.hetero_flag, ref.residue_number, ref.insertion_code)


def _atom_key(ref: AtomReference) -> tuple[str, str, str, int, str, str, str]:
    return (
        ref.model_id,
        ref.chain_id,
        ref.hetero_flag,
        ref.residue_number,
        ref.insertion_code,
        ref.atom_name,
        ref.altloc,
    )


def _atom_coordinates(structure: Any) -> dict[tuple[str, str, str, int, str, str, str], Point]:
    coordinates = {}
    for atom in structure.get_atoms():
        ref = AtomReference.from_atom(atom)
        coordinates[_atom_key(ref)] = _point(atom.get_coord())
    return coordinates


def _group_center(
    references: tuple[AtomReference, ...],
    coordinates: Mapping[tuple[str, str, str, int, str, str, str], Point],
) -> Point:
    if not references:
        raise ValueError("A contact partner has no atoms")
    points = []
    for ref in references:
        key = _atom_key(ref)
        if key not in coordinates:
            raise ValueError(f"Contact atom is absent from structure: {ref}")
        points.append(coordinates[key])
    return _point(np.mean(points, axis=0))


def _ring_plane(
    references: tuple[AtomReference, ...],
    coordinates: Mapping[tuple[str, str, str, int, str, str, str], Point],
) -> RingPlane | None:
    if not references:
        return None
    name = references[0].residue_name
    definitions = AROMATIC_RINGS.get(name, ())
    if not definitions:
        return None
    available = {ref.atom_name: ref for ref in references}
    candidates = [
        atom_names
        for _, atom_names in definitions
        if len(set(atom_names) & available.keys()) >= 3
    ]
    if not candidates:
        return None
    names = max(candidates, key=lambda items: len(set(items) & available.keys()))
    points = np.asarray(
        [coordinates[_atom_key(available[item])] for item in names if item in available],
        dtype=float,
    )
    center = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    if np.linalg.norm(normal) < 1e-12:
        return None
    radius = float(np.max(np.linalg.norm(points - center, axis=1)) + 0.15)
    return RingPlane(_point(center), _point(normal), radius)


def _distance(observation: ContactObservation, start: Point, end: Point) -> float:
    values = {item.name: item.value for item in observation.geometry}
    for name in (
        "donor_acceptor_distance",
        "charge_center_distance",
        "centroid_distance",
        "cation_centroid_distance",
        "atom_distance",
        "nearest_atom_distance",
    ):
        if name in values:
            return float(values[name])
    return float(np.linalg.norm(np.asarray(end) - start))


def _normalized_contacts(
    structure: Any,
    observations: Iterable[ContactObservation],
    styles: Mapping[str, ContactStyle],
) -> tuple[DisplayedContact, ...]:
    coordinates = _atom_coordinates(structure)
    ordered = sorted(
        observations,
        key=lambda item: json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")),
    )
    occurrence: dict[str, int] = {}
    result = []
    for observation in ordered:
        if not isinstance(observation, ContactObservation):
            raise TypeError("contacts must contain ContactObservation objects")
        kind = observation.interaction_type
        if kind not in styles:
            raise ValueError(f"No ContactStyle is defined for {kind!r}")
        canonical = json.dumps(observation.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = sha256(canonical.encode("utf-8")).hexdigest()[:12]
        occurrence[digest] = occurrence.get(digest, 0) + 1
        pair_id = f"pair-{digest}-{occurrence[digest]}"
        start = _group_center(observation.partner_a, coordinates)
        end = _group_center(observation.partner_b, coordinates)
        water = (
            _group_center(observation.mediator_waters, coordinates)
            if observation.mediator_waters else None
        )
        anchor = water if water is not None else _point((np.asarray(start) + end) / 2)
        residues = tuple(dict.fromkeys(
            _residue_key(ref)
            for ref in observation.partner_a + observation.partner_b
        ))
        ring_planes = ()
        if kind in {"pi_stacking_parallel", "pi_stacking_t_shaped", "cation_pi_candidate"}:
            planes = []
            for refs, role in (
                (observation.partner_a, observation.role_a),
                (observation.partner_b, observation.role_b),
            ):
                if kind != "cation_pi_candidate" or role != "cation":
                    plane = _ring_plane(refs, coordinates)
                    if plane is not None:
                        planes.append(plane)
            ring_planes = tuple(planes)
        distance = _distance(observation, start, end)
        result.append(DisplayedContact(
            pair_id=pair_id,
            interaction_type=kind,
            start=start,
            end=end,
            anchor=anchor,
            distance=distance,
            residues=residues,
            ring_planes=ring_planes,
            water=water,
            label=f"{styles[kind].label} · {distance:.2f} Å",
        ))
    return tuple(result)


def _copy_and_transform(
    structure: Any,
    transform: Callable[[Any], Any] | np.ndarray | None,
) -> Any:
    copied = deepcopy(structure)
    if transform is None:
        return copied
    if callable(transform):
        transformed = transform(copied)
        if transformed is None:
            raise ValueError("coordinate_transform must return a structure")
        return transformed
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("coordinate_transform must be callable or a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("coordinate_transform must contain a rigid rotation")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError("coordinate_transform must be a homogeneous transform")
    for atom in copied.get_atoms():
        atom.set_coord(rotation @ atom.get_coord() + matrix[:3, 3])
    return copied


def _residue_universe(structure: Any) -> dict[ResidueKey, dict[str, Any]]:
    result = {}
    for residue in structure.get_residues():
        if not is_aa(residue, standard=True):
            continue
        atom = residue["CA"] if "CA" in residue else next(residue.get_atoms(), None)
        if atom is None:
            continue
        chain = residue.get_parent()
        hetero, number, insertion = residue.id
        key = (str(chain.id), str(hetero).strip(), int(number), str(insertion).strip())
        result[key] = {
            "key": list(key),
            "label": f"{residue.resname} {number}{str(insertion).strip()}",
            "position": _xyz(_point(atom.get_coord())),
        }
    return result


_DEFAULT_PRESETS: Mapping[str, tuple[float, ...]] = {
    "xy": (0.0, 0.0, 0.0, 1.0),
    "xz": (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
    "yz": (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
}


class StructureContactView:
    """Contact state, py3Dmol viewer, controls, and standalone HTML."""

    def __init__(
        self,
        viewer: Any,
        contacts: tuple[DisplayedContact, ...],
        *,
        styles: Mapping[str, ContactStyle],
        chain_styles: Mapping[str, Mapping[str, Any]],
        chain_labels: Mapping[str, str],
        residue_universe: Mapping[ResidueKey, Mapping[str, Any]],
        residue_label_universe: str,
        residue_labels: Mapping[str, LabelMode],
        view_presets: Mapping[str, tuple[float, ...]],
        initial_view: str | None,
        show_controls: bool,
        enable_on_select: bool,
        active_contact_sticks: bool,
        ring_opacity: float,
        width: int,
        height: int,
        contact_types: set[str] | None,
        enabled_contact_types: set[str] | None,
    ) -> None:
        self.viewer = viewer
        self._source = contacts
        self._type_filter = contact_types
        self._pair_filter: set[str] | None = None
        self._enabled_ids = {
            contact.pair_id for contact in contacts
            if enabled_contact_types is None or contact.interaction_type in enabled_contact_types
        }
        self._highlighted_ids: set[str] = set()
        self._last_highlighted_id: str | None = None
        self._styles = dict(styles)
        self._chain_styles = {key: dict(value) for key, value in chain_styles.items()}
        self._chain_labels = dict(chain_labels)
        self._residue_universe = dict(residue_universe)
        self._residue_label_universe = residue_label_universe
        self._label_modes = dict(residue_labels)
        self._label_enabled = {
            chain: mode != "off" for chain, mode in residue_labels.items()
        }
        self._view_presets = dict(view_presets)
        self._view_name = initial_view
        self._show_controls = show_controls
        self._enable_on_select = enable_on_select
        self._active_contact_sticks = active_contact_sticks
        self._ring_opacity = ring_opacity
        self._width = width
        self._height = height
        self._id = f"biotools-contacts-{uuid4().hex}"

    @property
    def contacts(self) -> tuple[DisplayedContact, ...]:
        return tuple(
            contact for contact in self._source
            if (self._type_filter is None or contact.interaction_type in self._type_filter)
            and (self._pair_filter is None or contact.pair_id in self._pair_filter)
        )

    @property
    def enabled_pair_ids(self) -> frozenset[str]:
        return frozenset(self._enabled_ids & {item.pair_id for item in self.contacts})

    @property
    def active_residues(self) -> frozenset[ResidueKey]:
        return frozenset(
            residue
            for item in self.contacts if item.pair_id in self._enabled_ids
            for residue in item.residues
        )

    @property
    def highlighted_pair_id(self) -> str | None:
        return self._last_highlighted_id

    @property
    def highlighted_pair_ids(self) -> frozenset[str]:
        return frozenset(self._highlighted_ids & self.enabled_pair_ids)

    def checkbox_states(self) -> dict[str, Any]:
        pairs = self.contacts
        enabled = self.enabled_pair_ids
        def state(ids: set[str]) -> str:
            if not ids or not ids & enabled:
                return "off"
            if ids <= enabled:
                return "on"
            return "indeterminate"
        return {
            "master": state({item.pair_id for item in pairs}),
            "types": {
                kind: state({item.pair_id for item in pairs if item.interaction_type == kind})
                for kind in dict.fromkeys(item.interaction_type for item in pairs)
            },
            "pairs": {item.pair_id: item.pair_id in enabled for item in pairs},
        }

    def _known_ids(self) -> set[str]:
        return {item.pair_id for item in self._source}

    def _included_ids(self) -> set[str]:
        return {item.pair_id for item in self.contacts}

    def _refresh_state(self) -> None:
        self._highlighted_ids.intersection_update(self.enabled_pair_ids)
        if self._last_highlighted_id not in self._highlighted_ids:
            self._last_highlighted_id = None
        self._push()

    def _push(self) -> None:
        # A displayed notebook output can be updated from its Python wrapper.
        try:
            from IPython import get_ipython
            from IPython.display import Javascript, display
            shell = get_ipython()
            if shell is not None and getattr(shell, "kernel", None) is not None:
                payload = json.dumps(self._payload(), ensure_ascii=True).replace("<", "\\u003c")
                display(Javascript(
                    f"window.__biotoolsContactViews?.[{json.dumps(self._id)}]?.load({payload});"
                ))
        except ImportError:
            pass

    def show_contact_types(self, contact_types: Iterable[str]) -> None:
        kinds = set(contact_types)
        unknown = kinds - set(self._styles)
        if unknown:
            raise ValueError(f"Unknown contact types: {sorted(unknown)}")
        self._type_filter = kinds
        self._refresh_state()

    def set_contact_type_enabled(self, contact_type: str, enabled: bool) -> None:
        ids = {
            item.pair_id for item in self.contacts
            if item.interaction_type == contact_type
        }
        if not ids:
            raise KeyError(contact_type)
        self._enabled_ids.difference_update(ids)
        if enabled:
            self._enabled_ids.update(ids)
        self._refresh_state()

    def set_all_contacts_enabled(self, enabled: bool) -> None:
        ids = self._included_ids()
        self._enabled_ids.difference_update(ids)
        if enabled:
            self._enabled_ids.update(ids)
        self._refresh_state()

    def show_contact_pairs(self, pair_ids: Iterable[str]) -> None:
        ids = set(pair_ids)
        unknown = ids - self._known_ids()
        if unknown:
            raise ValueError(f"Unknown pair IDs: {sorted(unknown)}")
        self._pair_filter = ids
        self._refresh_state()

    def set_contact_pair_enabled(self, pair_id: str, enabled: bool) -> None:
        if pair_id not in self._included_ids():
            raise KeyError(pair_id)
        if enabled:
            self._enabled_ids.add(pair_id)
        else:
            self._enabled_ids.discard(pair_id)
        self._refresh_state()

    def highlight_contact_pair(self, pair_id: str) -> None:
        """Highlight a pair without clearing other highlighted pairs."""
        self.set_contact_pair_highlighted(pair_id, True)

    def set_contact_pair_highlighted(self, pair_id: str, highlighted: bool) -> None:
        """Toggle one highlight, optionally enabling the pair when selected."""
        if pair_id not in self._included_ids():
            raise KeyError(pair_id)
        if highlighted and pair_id not in self._enabled_ids:
            if self._enable_on_select:
                self._enabled_ids.add(pair_id)
            else:
                raise ValueError("Cannot highlight a disabled pair")
        if highlighted:
            self._highlighted_ids.add(pair_id)
            self._last_highlighted_id = pair_id
        else:
            self._highlighted_ids.discard(pair_id)
        self._refresh_state()

    def clear_highlight(self) -> None:
        """Clear all floating contact highlights without hiding contacts."""
        self._highlighted_ids.clear()
        self._refresh_state()

    def set_residue_label_mode(self, chain_id: str, mode: LabelMode) -> None:
        if chain_id not in self._label_modes:
            raise KeyError(chain_id)
        if mode not in {"off", "all", "active"}:
            raise ValueError("mode must be 'off', 'all', or 'active'")
        self._label_modes[chain_id] = mode
        self._label_enabled[chain_id] = mode != "off"
        self._refresh_state()

    def set_residue_labels_enabled(self, chain_id: str, enabled: bool) -> None:
        if chain_id not in self._label_modes:
            raise KeyError(chain_id)
        if enabled and self._label_modes[chain_id] == "off":
            self._label_modes[chain_id] = "all"
        self._label_enabled[chain_id] = bool(enabled)
        self._refresh_state()

    def set_view(self, preset: str) -> None:
        if preset not in self._view_presets:
            raise KeyError(preset)
        self._view_name = preset
        self._refresh_state()

    def _payload(self) -> dict[str, Any]:
        contacts = self.contacts
        kinds = list(dict.fromkeys(
            list(_CONTACT_TYPES) + [item.interaction_type for item in contacts]
        ))
        kinds = [kind for kind in kinds if any(item.interaction_type == kind for item in contacts)]
        contact_residues = {key for item in contacts for key in item.residues}
        stick_residues = contact_residues & self._residue_universe.keys()

        def pair_label(item: DisplayedContact) -> str:
            partners = []
            for key in item.residues:
                residue = self._residue_universe.get(key, {})
                label = residue.get("label", f"Residue {key[2]}{key[3]}")
                partners.append(f"{key[0]} {label}")
            return f"{' – '.join(partners)} · {item.distance:.2f} Å"

        return {
            "contacts": [
                {
                    "id": item.pair_id,
                    "type": item.interaction_type,
                    "start": _xyz(item.start),
                    "end": _xyz(item.end),
                    "anchor": _xyz(item.anchor),
                    "water": _xyz(item.water) if item.water else None,
                    "rings": [
                        {
                            "center": _xyz(plane.center),
                            "normal": _xyz(plane.normal),
                            "radius": plane.radius,
                        }
                        for plane in item.ring_planes
                    ],
                    "residues": [list(key) for key in item.residues],
                    "label": item.label,
                    "pairLabel": pair_label(item),
                    "distance": item.distance,
                }
                for item in contacts
            ],
            "types": kinds,
            "enabled": sorted(self.enabled_pair_ids),
            "selected": self._last_highlighted_id,
            "highlighted": sorted(self.highlighted_pair_ids),
            "styles": {kind: asdict(self._styles[kind]) for kind in kinds},
            "chainStyles": self._chain_styles,
            "chainLabels": self._chain_labels,
            "residueUniverse": [
                value for key, value in self._residue_universe.items()
                if self._residue_label_universe == "all" or key in contact_residues
            ],
            "stickResidues": [list(key) for key in sorted(stick_residues)],
            "labelModes": self._label_modes,
            "labelEnabled": self._label_enabled,
            "presets": self._view_presets,
            "viewName": self._view_name,
            "enableOnSelect": self._enable_on_select,
            "activeSticks": self._active_contact_sticks,
            "ringOpacity": self._ring_opacity,
        }

    def _body_html(self) -> str:
        viewer_html = self.viewer._make_html()
        viewer_name = f"viewer_{self.viewer.uniqueid}"
        payload = json.dumps(self._payload(), ensure_ascii=True, separators=(",", ":"))
        payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        ident = escape(self._id, quote=True)
        sidebar = (
            f'<aside class="biotools-contact-controls" aria-label="Interaction controls">'
            '<div class="bc-toolbar"><button type="button" data-controls-toggle '
            f'aria-expanded="true" aria-controls="{ident}-controls-body">'
            'Hide controls ▴</button></div>'
            f'<div data-controls-body id="{ident}-controls-body">'
            '<div class="bc-head">Views</div><div data-views></div>'
            '<div class="bc-head">Residue labels</div><div data-chains></div>'
            '<section class="bc-interaction-section">'
            '<div class="bc-head"><label><input type="checkbox" data-master> Interaction types</label></div>'
            '<div data-types></div></section></div></aside>'
            if self._show_controls else ""
        )
        return (
            f'<div class="biotools-contact-root" id="{ident}" '
            f'style="width:{self._width}px;height:{self._height}px">'
            f"{viewer_html}{sidebar}</div>"
            f"<style>{_CONTACT_CSS}</style>"
            f"<script>Promise.resolve($3Dmolpromise).then(function(){{"
            f"const root=document.getElementById({json.dumps(self._id)});"
            f"const viewer=window[{json.dumps(viewer_name)}];"
            f"if(root&&viewer){{window.__biotoolsContactViews=window.__biotoolsContactViews||{{}};"
            f"window.__biotoolsContactViews[{json.dumps(self._id)}]="
            f"createBiotoolsContactController(root,viewer,{payload});}}"
            f"}});{_CONTACT_JS}</script>"
        )

    def _repr_html_(self) -> str:
        return self._body_html()

    def to_html(self) -> str:
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>Structure contacts</title></head><body>"
            + self._body_html() + "</body></html>"
        )

    def write_html(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.to_html(), encoding="utf-8")
        return destination


def plot_structure_contacts(
    structure: Any,
    contacts: ContactAnalysisResult | Iterable[ContactObservation],
    *,
    chain_styles: Mapping[str, Mapping[str, Any]] | None = None,
    chain_labels: Mapping[str, str] | None = None,
    contact_styles: Mapping[str, ContactStyle | Mapping[str, Any]] | None = None,
    contact_types: Iterable[str] | None = None,
    enabled_contact_types: Iterable[str] | None = None,
    residue_labels: Mapping[str, LabelMode] | None = None,
    residue_label_universe: Literal["all", "contact_residues"] = "all",
    show_controls: bool = True,
    active_contact_sticks: bool = True,
    enable_on_select: bool = True,
    ring_opacity: float = 0.45,
    coordinate_transform: Callable[[Any], Any] | np.ndarray | None = None,
    view_presets: Mapping[str, Iterable[float]] | None = None,
    initial_view: str | None = None,
    width: int = 1100,
    height: int = 720,
) -> StructureContactView:
    """Render typed contacts with synchronized notebook and HTML controls.

    Contact detection is not performed. The input structure is copied before
    optional coordinate transformation or rendering. Amino acids participating
    in visible contacts are shown as sticks by default, independently of
    highlighting. Set active_contact_sticks=False to disable these sticks.
    """
    import py3Dmol

    if isinstance(contacts, ContactAnalysisResult):
        observations = contacts.observations
    else:
        observations = tuple(contacts)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise ValueError("height must be a positive integer")
    if residue_label_universe not in {"all", "contact_residues"}:
        raise ValueError("residue_label_universe must be 'all' or 'contact_residues'")
    if not math.isfinite(ring_opacity) or not 0 <= ring_opacity <= 1:
        raise ValueError("ring_opacity must be between 0 and 1")

    styles = dict(DEFAULT_CONTACT_STYLES)
    for kind, override in (contact_styles or {}).items():
        if isinstance(override, ContactStyle):
            styles[kind] = override
        else:
            baseline = styles.get(kind)
            values = asdict(baseline) if baseline is not None else {}
            values.update(override)
            styles[kind] = ContactStyle(**values)
    type_filter = set(contact_types) if contact_types is not None else None
    enabled_types = set(enabled_contact_types) if enabled_contact_types is not None else None
    unknown = (type_filter or set()) | (enabled_types or set())
    unknown -= styles.keys()
    if unknown:
        raise ValueError(f"Unknown contact types: {sorted(unknown)}")

    copied = _copy_and_transform(structure, coordinate_transform)
    normalized = _normalized_contacts(copied, observations, styles)
    chains = {str(chain.id) for chain in copied.get_chains()}
    styled = {chain: {"cartoon": {"color": "#bfc4c9", "opacity": 0.68}} for chain in chains}
    for chain, style in (chain_styles or {}).items():
        if chain not in chains:
            raise KeyError(f"Unknown chain: {chain}")
        styled[chain] = dict(style)
    universe = _residue_universe(copied)
    label_chains = {key[0] for key in universe}
    labels = {
        chain: str((chain_labels or {}).get(chain, chain))
        for chain in sorted(label_chains)
    }
    modes: dict[str, LabelMode] = {chain: "off" for chain in label_chains}
    for chain, mode in (residue_labels or {}).items():
        if chain not in label_chains:
            raise KeyError(f"Chain has no standard residues: {chain}")
        if mode not in {"off", "all", "active"}:
            raise ValueError("residue label mode must be 'off', 'all', or 'active'")
        modes[chain] = mode
    presets = dict(_DEFAULT_PRESETS)
    for name, values in (view_presets or {}).items():
        sequence = tuple(float(value) for value in values)
        if len(sequence) not in {4, 8} or not all(math.isfinite(item) for item in sequence):
            raise ValueError("view presets must contain four quaternion or eight view numbers")
        presets[name] = sequence
    if initial_view is not None and initial_view not in presets:
        raise KeyError(initial_view)

    # Keep the molecule clear of the controls on desktop-sized views.
    canvas_width = width - 360 if show_controls and width >= 720 else width
    viewer = py3Dmol.view(width=canvas_width, height=height)
    viewer.addModel(_structure_pdb_text(copied), "pdb")
    for chain, style in styled.items():
        viewer.setStyle({"chain": chain}, style)
    viewer.zoomTo()
    viewer.render()
    return StructureContactView(
        viewer, normalized, styles=styles, chain_styles=styled, chain_labels=labels,
        residue_universe=universe, residue_label_universe=residue_label_universe,
        residue_labels=modes, view_presets=presets,
        initial_view=initial_view, show_controls=show_controls,
        enable_on_select=enable_on_select, active_contact_sticks=active_contact_sticks,
        ring_opacity=ring_opacity, width=width, height=height,
        contact_types=type_filter, enabled_contact_types=enabled_types,
    )

_CONTACT_CSS = """
.biotools-contact-root{position:relative;font:13px/1.4 system-ui,sans-serif;color:#202124}
.biotools-contact-controls{position:absolute;right:10px;top:10px;z-index:5;
  width:340px;max-width:calc(100% - 20px);max-height:calc(100% - 20px);
  overflow:auto;box-sizing:border-box;background:rgba(255,255,255,.96);
  border:1px solid #cbd1d6;border-radius:8px;padding:12px;box-shadow:0 2px 12px #0002}
.biotools-contact-controls .bc-toolbar{display:flex;justify-content:flex-end;margin-bottom:12px}
.biotools-contact-controls.bc-collapsed{width:auto;padding:8px}
.biotools-contact-controls.bc-collapsed .bc-toolbar{margin-bottom:0}
.biotools-contact-controls .bc-head{font-weight:650;margin:3px 0 7px}
.biotools-contact-controls [data-views]{display:flex;gap:5px;margin-bottom:16px}
.biotools-contact-controls .bc-interaction-section{margin-top:18px;padding-top:14px;
  border-top:1px solid #cbd1d6}
.biotools-contact-controls .bc-type{border-top:1px solid #e5e7eb;padding:8px 0}
.biotools-contact-controls .bc-type-head{display:flex;align-items:center;gap:5px}
.biotools-contact-controls .bc-type-name{flex:1}
.biotools-contact-controls input[type=checkbox]{margin:0 5px 0 0;vertical-align:middle}
.biotools-contact-controls .bc-swatch{width:9px;height:9px;border-radius:50%;
  flex:none;display:inline-block}
.biotools-contact-controls .bc-pairs{margin:10px 0 2px 8px;padding:8px;
  border-left:3px solid #d7dfe8;border-radius:4px;background:#f4f6f8}
.biotools-contact-controls .bc-pair{display:flex;align-items:center;gap:6px;
  border-radius:4px;padding:6px 3px}
.biotools-contact-controls .bc-pair + .bc-pair{border-top:1px solid #e1e5eb}
.biotools-contact-controls .bc-pair-label{flex:1;min-width:0;font-size:12px}
.biotools-contact-controls .bc-pair.bc-selected{background:#dbeafe}
.biotools-contact-controls button{cursor:pointer;border:1px solid #ccd3db;background:#fff;
  border-radius:4px;color:#202124;font:inherit;padding:3px 6px}
.biotools-contact-controls button:focus-visible{outline:2px solid #2563eb;outline-offset:2px}
.biotools-contact-controls [data-expand]{font-size:11px;white-space:nowrap}
.biotools-contact-controls [data-pair-select]{font-size:11px;flex:none}
.biotools-contact-controls [data-pair-select]::before{content:"";display:inline-block;
  width:8px;height:8px;border:1px solid #94a3b8;border-radius:50%;margin-right:4px}
.biotools-contact-controls [data-pair-select][aria-pressed=true]::before{background:#2563eb;border-color:#2563eb}
.biotools-contact-controls .bc-chain{margin:5px 0}
.biotools-contact-controls .bc-chain button{margin-left:4px}
.biotools-contact-controls button[aria-pressed=true]{background:#dbeafe;border-color:#93b4e8}
.biotools-contact-controls .bc-empty{color:#69717b;font-style:italic}
"""

_CONTACT_JS = r"""
function createBiotoolsContactController(root, viewer, initial) {
  let data = initial;
  let enabled = new Set(data.enabled);
  let highlighted = new Set(data.highlighted || []);
  let typeShapes = {};
  let planeShapes = {};
  let residueLabels = [];
  let highlightLabels = [];
  let lastViewName = null;
  const controls = root.querySelector('.biotools-contact-controls');
  const byId = () => new Map(data.contacts.map(c => [c.id, c]));
  const residueKey = r => JSON.stringify(r);
  const activeResidues = () => {
    const result = new Set();
    data.contacts.forEach(c => {
      if(enabled.has(c.id)) c.residues.forEach(r => result.add(residueKey(r)));
    });
    return result;
  };
  function clearShapes() {
    Object.values(typeShapes).forEach(s => viewer.removeShape(s));
    Object.values(planeShapes).forEach(s => viewer.removeShape(s));
    typeShapes = {};
    planeShapes = {};
  }
  function rebuildShapes() {
    clearShapes();
    data.types.forEach(kind => {
      const contacts = data.contacts.filter(c => c.type === kind && enabled.has(c.id));
      if(!contacts.length) return;
      const style = data.styles[kind];
      const shape = viewer.addShape({color:style.color,opacity:style.opacity});
      let hasPlanes = false;
      let planes = null;
      contacts.forEach(c => {
        const leg = (start,end) => shape.addDashedCylinder({
          start:start,end:end,color:style.color,radius:style.radius,
          dashLength:style.dash_length,gapLength:style.gap_length,
          fromCap:'flat',toCap:'flat'
        });
        if(c.water) {
          leg(c.start,c.water);
          leg(c.water,c.end);
          shape.addSphere({center:c.water,radius:Math.max(0.18,style.radius*2),
            color:style.color});
        } else {
          leg(c.start,c.end);
        }
        c.rings.forEach(ring => {
          if(!planes) planes = viewer.addShape({color:style.color,
            opacity:data.ringOpacity});
          const h = 0.025;
          const center = ring.center, n = ring.normal;
          const start = {x:center.x-n.x*h,y:center.y-n.y*h,z:center.z-n.z*h};
          const end = {x:center.x+n.x*h,y:center.y+n.y*h,z:center.z+n.z*h};
          planes.addCylinder({start:start,end:end,radius:ring.radius,
            color:style.color,fromCap:'flat',toCap:'flat'});
          hasPlanes = true;
        });
      });
      typeShapes[kind] = shape;
      if(hasPlanes) planeShapes[kind] = planes;
    });
  }
  function clearLabels() {
    residueLabels.forEach(label => viewer.removeLabel(label));
    residueLabels = [];
  }
  function rebuildSticksAndLabels() {
    const active = activeResidues();
    Object.entries(data.chainStyles).forEach(([chain,style]) => {
      viewer.setStyle({chain:chain},style);
    });
    if(data.activeSticks) {
      data.stickResidues.forEach(r => {
        if(active.has(residueKey(r))) {
          const [chain,het,num,icode] = r;
          viewer.addStyle({chain:chain,resi:num,icode:icode||' ',hetflag:!!het},
            {stick:{radius:0.18}});
        }
      });
    }
    clearLabels();
    data.residueUniverse.forEach(r => {
      const chain = r.key[0], mode = data.labelModes[chain];
      if(!data.labelEnabled[chain] || mode === 'off') return;
      if(mode === 'active' && !active.has(residueKey(r.key))) return;
      residueLabels.push(viewer.addLabel(r.label,{
        position:r.position,fontColor:'#202124',fontSize:11,
        backgroundColor:'#ffffff',backgroundOpacity:0.7,
        inFront:true,screenOffset:{x:0,y:-9}
      }));
    });
  }
  function refreshHighlight() {
    highlightLabels.forEach(label => viewer.removeLabel(label));
    highlightLabels = [];
    highlighted = new Set([...highlighted].filter(id => enabled.has(id) && byId().has(id)));
    highlighted.forEach(id => {
      const contact = byId().get(id);
      highlightLabels.push(viewer.addLabel(contact.pairLabel,{
        position:contact.anchor,backgroundColor:'#ffffff',
        backgroundOpacity:0.90,borderThickness:2,
        borderColor:data.styles[contact.type].color,
        fontColor:'#202124',fontSize:12,inFront:true,
        screenOffset:{x:0,y:-12}
      }));
    });
  }
  function applyView(name) {
    const preset = data.presets[name];
    if(!preset) return;
    const current = viewer.getView();
    viewer.setView(preset.length === 8
      ? preset : current.slice(0,4).concat(preset));
    lastViewName = name;
  }
  function checkboxState(ids) {
    const count = ids.filter(id => enabled.has(id)).length;
    return {checked:ids.length > 0 && count === ids.length,
      indeterminate:count > 0 && count < ids.length};
  }
  function syncControls() {
    if(!controls) return;
    const all = data.contacts.map(c => c.id);
    const master = controls.querySelector('[data-master]');
    const state = checkboxState(all);
    master.checked = state.checked;
    master.indeterminate = state.indeterminate;
    master.disabled = !all.length;
    controls.querySelectorAll('[data-type-check]').forEach(input => {
      const ids = data.contacts.filter(c => c.type === input.dataset.typeCheck).map(c => c.id);
      const s = checkboxState(ids);
      input.checked = s.checked;
      input.indeterminate = s.indeterminate;
    });
    controls.querySelectorAll('[data-pair-check]').forEach(input => {
      input.checked = enabled.has(input.dataset.pairCheck);
    });
    controls.querySelectorAll('[data-pair-row]').forEach(row => {
      row.classList.toggle('bc-selected',highlighted.has(row.dataset.pairRow));
    });
    controls.querySelectorAll('[data-pair-select]').forEach(button => {
      button.setAttribute('aria-pressed',String(highlighted.has(button.dataset.pairSelect)));
    });
    controls.querySelectorAll('[data-label-check]').forEach(input => {
      input.checked = !!data.labelEnabled[input.dataset.labelCheck] &&
        data.labelModes[input.dataset.labelCheck] !== 'off';
    });
    controls.querySelectorAll('[data-label-mode]').forEach(button => {
      const [chain,mode] = button.dataset.labelMode.split('|');
      button.setAttribute('aria-pressed',
        String(data.labelEnabled[chain] && data.labelModes[chain] === mode));
    });
    controls.querySelectorAll('[data-view]').forEach(button => {
      button.setAttribute('aria-pressed',String(button.dataset.view === lastViewName));
    });
  }
  function renderScene() {
    enabled = new Set([...enabled].filter(id => byId().has(id)));
    rebuildShapes();
    rebuildSticksAndLabels();
    refreshHighlight();
    syncControls();
    viewer.render();
  }
  const el = (tag,className,text) => {
    const node = document.createElement(tag);
    if(className) node.className = className;
    if(text !== undefined) node.textContent = text;
    return node;
  };
  function buildControls() {
    if(!controls) return;
    const typesBox = controls.querySelector('[data-types]');
    const chainsBox = controls.querySelector('[data-chains]');
    const viewsBox = controls.querySelector('[data-views]');
    typesBox.replaceChildren();
    chainsBox.replaceChildren();
    viewsBox.replaceChildren();
    if(!data.contacts.length) typesBox.append(el('div','bc-empty','No contacts'));
    data.types.forEach(kind => {
      const group = el('div','bc-type');
      group.dataset.contactType = kind;
      const header = el('div','bc-type-head');
      const input = el('input');
      input.type = 'checkbox';
      input.dataset.typeCheck = kind;
      input.setAttribute('aria-label','Show '+data.styles[kind].label);
      const swatch = el('span','bc-swatch');
      swatch.style.backgroundColor = data.styles[kind].color;
      const contacts = data.contacts.filter(c => c.type === kind);
      const expand = el('button','','Show '+contacts.length+
        (contacts.length === 1 ? ' contact ▾' : ' contacts ▾'));
      expand.type = 'button';
      expand.dataset.expand = kind;
      expand.setAttribute('aria-expanded','false');
      const list = el('div','bc-pairs');
      list.dataset.pairs = kind;
      list.id = root.id+'-'+kind+'-pairs';
      list.hidden = true;
      expand.setAttribute('aria-controls',list.id);
      header.append(input,swatch,el('span','bc-type-name',data.styles[kind].label),expand);
      group.append(header,list);
      contacts.forEach(c => {
        const row = el('div','bc-pair');
        row.dataset.pairRow = c.id;
        const check = el('input');
        check.type = 'checkbox';
        check.dataset.pairCheck = c.id;
        check.setAttribute('aria-label','Show '+c.pairLabel);
        check.title = 'Show contact';
        const button = el('button','','Highlight');
        button.type = 'button';
        button.dataset.pairSelect = c.id;
        button.setAttribute('aria-label','Highlight '+c.pairLabel);
        button.title = 'Toggle highlighting independently of other contacts';
        row.append(check,el('span','bc-pair-label',c.pairLabel),button);
        list.append(row);
      });
      typesBox.append(group);
    });
    Object.entries(data.chainLabels).forEach(([chain,label]) => {
      const row = el('div','bc-chain');
      const box = el('input');
      box.type = 'checkbox';
      box.dataset.labelCheck = chain;
      row.append(box,el('span','',label));
      ['all','active'].forEach(mode => {
        const button = el('button','',mode === 'all' ? 'All' : 'Active');
        button.type = 'button';
        button.dataset.labelMode = chain+'|'+mode;
        row.append(button);
      });
      chainsBox.append(row);
    });
    Object.keys(data.presets).forEach(name => {
      const button = el('button','',name);
      button.type = 'button';
      button.dataset.view = name;
      viewsBox.append(button);
    });
    syncControls();
  }
  function setAll(value) {
    enabled = value ? new Set(data.contacts.map(c=>c.id)) : new Set();
    renderScene();
  }
  function setType(kind,value) {
    data.contacts.filter(c=>c.type===kind).forEach(c => {
      if(value) enabled.add(c.id); else enabled.delete(c.id);
    });
    renderScene();
  }
  function setPair(id,value) {
    if(!byId().has(id)) return;
    if(value) enabled.add(id); else enabled.delete(id);
    renderScene();
  }
  function highlight(id,value=true) {
    if(!byId().has(id)) return;
    if(value && !enabled.has(id)) {
      if(data.enableOnSelect) {
        enabled.add(id);
        renderScene();
      } else return;
    }
    if(value) highlighted.add(id); else highlighted.delete(id);
    refreshHighlight();
    syncControls();
    viewer.render();
  }
  if(controls) {
    controls.addEventListener('change',event => {
      const target = event.target;
      if(target.matches('[data-master]')) setAll(target.checked);
      else if(target.matches('[data-type-check]'))
        setType(target.dataset.typeCheck,target.checked);
      else if(target.matches('[data-pair-check]'))
        setPair(target.dataset.pairCheck,target.checked);
      else if(target.matches('[data-label-check]')) {
        const chain = target.dataset.labelCheck;
        if(target.checked && data.labelModes[chain] === 'off')
          data.labelModes[chain] = 'all';
        data.labelEnabled[chain] = target.checked;
        rebuildSticksAndLabels();
        syncControls();
        viewer.render();
      }
    });
    controls.addEventListener('click',event => {
      const target = event.target.closest('button');
      if(!target) return;
      if(target.matches('[data-controls-toggle]')) {
        const body = controls.querySelector('[data-controls-body]');
        body.hidden = !body.hidden;
        controls.classList.toggle('bc-collapsed',body.hidden);
        target.setAttribute('aria-expanded',String(!body.hidden));
        target.textContent = body.hidden ? 'Show controls ▾' : 'Hide controls ▴';
      } else if(target.dataset.expand) {
        const group = target.closest('[data-contact-type]');
        const list = group.querySelector('[data-pairs]');
        list.hidden = !list.hidden;
        target.setAttribute('aria-expanded',String(!list.hidden));
        target.textContent = (list.hidden ? 'Show ' : 'Hide ')+list.children.length+
          (list.children.length === 1 ? ' contact ' : ' contacts ')+(list.hidden ? '▾' : '▴');
      } else if(target.dataset.pairSelect) {
        const id = target.dataset.pairSelect;
        highlight(id,!highlighted.has(id));
      }
      else if(target.dataset.labelMode) {
        const [chain,mode] = target.dataset.labelMode.split('|');
        data.labelModes[chain] = mode;
        data.labelEnabled[chain] = true;
        rebuildSticksAndLabels();
        syncControls();
        viewer.render();
      } else if(target.dataset.view) {
        applyView(target.dataset.view);
        syncControls();
        viewer.render();
      }
    });
  }
  buildControls();
  renderScene();
  if(data.viewName) {
    applyView(data.viewName);
    syncControls();
    viewer.render();
  }
  return {
    load(next) {
      data = next;
      enabled = new Set(data.enabled);
      highlighted = new Set(data.highlighted || []);
      buildControls();
      renderScene();
      if(data.viewName) {applyView(data.viewName);viewer.render();}
    },
    setAll, setType, setPair, highlight,
    clearHighlight() {highlighted.clear();refreshHighlight();syncControls();viewer.render();},
    setLabelMode(chain,mode) {
      data.labelModes[chain]=mode;data.labelEnabled[chain]=mode!=='off';
      rebuildSticksAndLabels();syncControls();viewer.render();
    },
    setView(name) {applyView(name);syncControls();viewer.render();},
    state() {
      return {
        enabled:[...enabled],highlighted:[...highlighted],activeResidues:[...activeResidues()],
        master:checkboxState(data.contacts.map(c=>c.id)),
        types:Object.fromEntries(data.types.map(kind=>[
          kind,checkboxState(data.contacts.filter(c=>c.type===kind).map(c=>c.id))
        ])),
        shapeCount:Object.keys(typeShapes).length+Object.keys(planeShapes).length,
        residueLabelCount:residueLabels.length,
        highlightLabelCount:highlightLabels.length
      };
    }
  };
}
"""
