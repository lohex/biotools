"""Interactive protein structure visualization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from html import escape
from io import StringIO
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    Normalize,
    to_hex,
    to_rgba,
    TwoSlopeNorm,
)
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray

from .matrices import (
    _matrix_residues,
    _residue_label,
    get_distance_matrix,
    get_interchain_distance_matrix,
)
from .molecular_surface import (
    color_from_labels,
    MolecularSurfaceMesh,
    MolecularSurfaceResult,
    SurfacePatch,
    SurfacePatchSet,
    SurfaceSamples,
)

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

InteractionMeasure: TypeAlias = Literal[
    "c_alpha",
    "c_beta",
    "min_heavy_atom",
    "hydrogen_bond",
    "salt_bridge",
    "hydrophobic_contact",
    "van_der_waals_contact",
    "pi_stacking_parallel",
    "pi_stacking_t_shaped",
    "cation_pi_candidate",
    "water_bridge",
    "interaction_type",
]
StructureStyle: TypeAlias = Literal[
    "cartoon",
    "stick",
    "cartoon+stick",
    "line",
    "none",
]

_DISTANCE_MEASURES = {"c_alpha", "c_beta", "min_heavy_atom"}
_CONTACT_TYPES = (
    "hydrogen_bond",
    "salt_bridge",
    "hydrophobic_contact",
    "van_der_waals_contact",
    "pi_stacking_parallel",
    "pi_stacking_t_shaped",
    "cation_pi_candidate",
    "water_bridge",
)
_CONTACT_TYPE_ORDER = {
    interaction_type: index
    for index, interaction_type in enumerate(_CONTACT_TYPES)
}
_CONTACT_TYPE_COLORS = {
    "hydrogen_bond": "#1f77b4",
    "salt_bridge": "#d62728",
    "hydrophobic_contact": "#2ca02c",
    "van_der_waals_contact": "#7f7f7f",
    "pi_stacking_parallel": "#9467bd",
    "pi_stacking_t_shaped": "#8c564b",
    "cation_pi_candidate": "#e377c2",
    "water_bridge": "#17becf",
}
_CONTACT_TYPE_LABELS = {
    "hydrogen_bond": "hydrogen bond",
    "salt_bridge": "salt bridge",
    "hydrophobic_contact": "hydrophobic contact",
    "van_der_waals_contact": "van der Waals contact",
    "pi_stacking_parallel": "parallel π stacking",
    "pi_stacking_t_shaped": "T-shaped π stacking",
    "cation_pi_candidate": "cation–π candidate",
    "water_bridge": "water bridge",
}


def _structure_pdb_text(structure: Structure) -> str:
    from Bio.PDB import PDBIO

    buffer = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(buffer)
    return buffer.getvalue()


def plot_structure(structure: Structure) -> Any:
    """Create an interactive ``py3Dmol`` view for a structure.

    Args:
        structure: Biopython structure to render as a spectrum-colored cartoon.

    Returns:
        Configured ``py3Dmol.view`` instance.
    """
    import py3Dmol

    view = py3Dmol.view(width=800, height=400)
    view.addModel(_structure_pdb_text(structure), "pdb")
    view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view


def _surface_mesh(
    surface_obj: MolecularSurfaceResult | MolecularSurfaceMesh,
    component_id: int | None,
) -> MolecularSurfaceMesh:
    if isinstance(surface_obj, MolecularSurfaceMesh):
        if component_id is not None and component_id != surface_obj.component_id:
            raise ValueError(
                f"Mesh component {surface_obj.component_id} does not match "
                f"component_id={component_id}"
            )
        return surface_obj
    if isinstance(surface_obj, MolecularSurfaceResult):
        if component_id is None:
            return surface_obj.surface
        return surface_obj.get_component(component_id)
    raise TypeError(
        "surface_obj must be a MolecularSurfaceResult or MolecularSurfaceMesh"
    )


def _structure_style_spec(
    style: StructureStyle | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if isinstance(style, Mapping):
        return dict(style)
    styles: dict[str, Mapping[str, Any] | None] = {
        "cartoon": {"cartoon": {"color": "spectrum"}},
        "stick": {"stick": {"colorscheme": "Jmol", "radius": 0.18}},
        "cartoon+stick": {
            "cartoon": {"color": "spectrum"},
            "stick": {"colorscheme": "Jmol", "radius": 0.15},
        },
        "line": {"line": {"colorscheme": "Jmol"}},
        "none": None,
    }
    try:
        return styles[style]
    except KeyError as exc:
        choices = ", ".join(styles)
        raise ValueError(
            f"Unknown structure_style {style!r}; choose from {choices}"
        ) from exc


def _rgb_records(colors: NDArray[np.float64]) -> list[dict[str, float]]:
    return [
        {"r": float(color[0]), "g": float(color[1]), "b": float(color[2])}
        for color in colors
    ]


def _vector_records(vectors: NDArray[np.float64]) -> list[dict[str, float]]:
    return [
        {"x": float(vector[0]), "y": float(vector[1]), "z": float(vector[2])}
        for vector in vectors
    ]


def _field_colors(
    mesh: MolecularSurfaceMesh,
    field_name: str,
    *,
    cmap: str,
    field_range: tuple[float, float] | None,
    field_center: float | None,
    invalid_color: Any,
    face_indices: NDArray[np.int32] | None = None,
) -> tuple[NDArray[np.float64], str, Normalize, str | None]:
    field = mesh.get_field(field_name)
    try:
        values = np.asarray(field.values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Surface field {field_name!r} cannot be converted to numeric values"
        ) from exc
    finite = np.isfinite(values)
    if face_indices is None:
        displayed_values = values
    elif field.location == "face":
        displayed_values = values[face_indices]
    else:
        displayed_values = values[np.unique(mesh.faces[face_indices])]
    displayed_finite = displayed_values[np.isfinite(displayed_values)]
    if not len(displayed_finite):
        raise ValueError(f"Surface field {field_name!r} has no finite values")
    if field_range is None:
        lower = float(np.min(displayed_finite))
        upper = float(np.max(displayed_finite))
    else:
        if len(field_range) != 2:
            raise ValueError("field_range must contain (minimum, maximum)")
        lower, upper = (float(value) for value in field_range)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("field_range values must be finite and increasing")
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1.0)
        lower -= padding
        upper += padding
    if field_center is not None:
        center = float(field_center)
        if not np.isfinite(center) or not lower < center < upper:
            raise ValueError("field_center must lie strictly inside field_range")
        normalization = TwoSlopeNorm(vmin=lower, vcenter=center, vmax=upper)
    else:
        normalization = Normalize(vmin=lower, vmax=upper)
    colors = np.asarray(colormaps.get_cmap(cmap)(normalization(values)), dtype=float)
    colors[~finite] = to_rgba(invalid_color)
    return colors, field.location, normalization, field.units


def _surface_colorbar_html(
    *,
    field_name: str,
    units: str | None,
    cmap: str,
    normalization: Normalize,
) -> str:
    """Build a compact legend overlay matching the mesh's scalar colors."""
    palette = colormaps.get_cmap(cmap)
    values = np.linspace(normalization.vmin, normalization.vmax, 33)
    stops = ", ".join(
        f"{to_hex(palette(normalization(value)))} {position:.1f}%"
        for position, value in zip(np.linspace(0, 100, len(values)), values)
    )
    title = escape(field_name.replace("_", " "))
    if units:
        title += f" ({escape(units)})"
    lower = f"{normalization.vmin:.3g}"
    upper = f"{normalization.vmax:.3g}"
    return (
        '<div style="position:absolute;left:50%;bottom:12px;transform:translateX(-50%);'
        'z-index:10;pointer-events:none;background:rgba(255,255,255,.88);'
        'padding:6px 10px;border-radius:4px;color:#222;font:12px sans-serif;'
        'width:min(320px,80%);box-sizing:border-box">'
        f'<div style="text-align:center;margin-bottom:3px">{title}</div>'
        f'<div style="height:12px;background:linear-gradient(to right,{stops});'
        'border:1px solid #777"></div>'
        '<div style="display:flex;justify-content:space-between;margin-top:2px">'
        f"<span>{lower}</span><span>{upper}</span></div></div>"
    )


def _custom_surface_spec(
    mesh: MolecularSurfaceMesh,
    *,
    colors: NDArray[np.float64] | None,
    color_location: Literal["vertex", "face"] | None,
    surface_color: Any,
    opacity: float,
    wireframe: bool,
    face_indices: NDArray[np.int32] | None = None,
    normal_offset: float = 0.0,
) -> dict[str, Any]:
    selected_faces = mesh.faces if face_indices is None else mesh.faces[face_indices]
    if color_location == "face" or face_indices is not None:
        vertices = mesh.vertices[selected_faces].reshape(-1, 3)
        normals = mesh.normals[selected_faces].reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32)
        if colors is None:
            vertex_colors = None
        elif color_location == "face":
            selected_colors = colors if face_indices is None else colors[face_indices]
            vertex_colors = np.repeat(selected_colors, 3, axis=0)
        else:
            vertex_colors = colors[selected_faces].reshape(-1, 4)
    else:
        vertices = mesh.vertices
        normals = mesh.normals
        faces = mesh.faces.reshape(-1)
        vertex_colors = colors
    if normal_offset:
        vertices = vertices + normal_offset * normals
    spec: dict[str, Any] = {
        "vertexArr": _vector_records(vertices),
        "normalArr": _vector_records(normals),
        "faceArr": faces.tolist(),
        "opacity": float(opacity),
        "wireframe": bool(wireframe),
    }
    if vertex_colors is None:
        spec["color"] = surface_color
    else:
        spec["color"] = _rgb_records(vertex_colors)
    return spec


def plot_molecular_surface(
    structure: Structure,
    surface_obj: MolecularSurfaceResult | MolecularSurfaceMesh,
    *,
    component_id: int | None = None,
    structure_style: StructureStyle | Mapping[str, Any] = "cartoon+stick",
    field_name: str | None = None,
    patch: SurfacePatch | None = None,
    patches: SurfacePatchSet | None = None,
    samples: SurfaceSamples | None = None,
    surface_color: Any = "lightgray",
    surface_opacity: float = 0.7,
    cmap: str = "coolwarm",
    colorbar: bool = False,
    field_range: tuple[float, float] | None = None,
    field_center: float | None = None,
    invalid_color: Any = "gray",
    patch_background_color: Any = "lightgray",
    wireframe: bool = False,
    patch_wireframe: bool = False,
    patch_wireframe_color: Any = "#303030",
    normal_length: float = 1.0,
    normal_color: Any = "black",
    normal_radius: float = 0.04,
    sample_radius: float = 0.0,
    width: int = 900,
    height: int = 600,
    background_color: Any = "white",
) -> Any:
    """Combine a Biopython structure and an MSMS mesh in py3Dmol.

    The structure can be drawn as cartoon, sticks, both, lines, a custom
    py3Dmol style, or hidden. The surface can use one scalar field or a set of
    face patch labels. ``patch`` restricts the mesh to one patch and can be
    combined with ``field_name`` for continuous coloring. ``patch_wireframe``
    overlays triangle edges only on the selected patch or patch set.
    ``colorbar=True`` adds a legend to a scalar-field plot. Optional
    samples are shown as normal-vector arrows.
    """
    import py3Dmol

    mesh = _surface_mesh(surface_obj, component_id)
    if field_name is not None and patches is not None:
        raise ValueError("Specify either field_name or patches, not both")
    if patch is not None and patches is not None:
        raise ValueError("Specify either patch or patches, not both")
    if patch_wireframe and patch is None and patches is None:
        raise ValueError("patch_wireframe requires patch or patches")
    if colorbar and field_name is None:
        raise ValueError("colorbar=True requires field_name")
    face_indices = None
    if patch is not None:
        if patch.component_id != mesh.component_id:
            raise ValueError("patch belongs to a different surface component")
        face_indices = patch.face_indices
        if (
            not len(face_indices)
            or np.any(face_indices < 0)
            or np.any(face_indices >= len(mesh.faces))
        ):
            raise ValueError("patch face indices do not match the surface")
    if not np.isfinite(surface_opacity) or not 0.0 <= surface_opacity <= 1.0:
        raise ValueError("surface_opacity must be between 0 and 1")
    if not np.isfinite(normal_length) or normal_length <= 0.0:
        raise ValueError("normal_length must be finite and positive")
    if not np.isfinite(normal_radius) or normal_radius <= 0.0:
        raise ValueError("normal_radius must be finite and positive")
    if not np.isfinite(sample_radius) or sample_radius < 0.0:
        raise ValueError("sample_radius must be finite and non-negative")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise ValueError("height must be a positive integer")

    colors = None
    color_location = None
    normalization = None
    field_units = None
    if patches is not None:
        if patches.component_id != mesh.component_id:
            raise ValueError("patches belong to a different surface component")
        if len(patches.face_labels) != len(mesh.faces):
            raise ValueError("patch labels do not match the surface face count")
        colors = color_from_labels(
            patches.face_labels,
            cmap=cmap,
            background_color=patch_background_color,
        )
        color_location = "face"
    elif field_name is not None:
        colors, color_location, normalization, field_units = _field_colors(
            mesh,
            field_name,
            cmap=cmap,
            field_range=field_range,
            field_center=field_center,
            invalid_color=invalid_color,
            face_indices=face_indices,
        )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(background_color)
    view.addModel(_structure_pdb_text(structure), "pdb")
    style_spec = _structure_style_spec(structure_style)
    if style_spec is not None:
        view.setStyle({"model": -1}, style_spec)
    view.addCustom(
        _custom_surface_spec(
            mesh,
            colors=colors,
            color_location=color_location,
            surface_color=surface_color,
            opacity=surface_opacity,
            wireframe=wireframe,
            face_indices=face_indices,
        )
    )
    if patch_wireframe:
        wireframe_faces = (
            face_indices
            if patch is not None
            else np.flatnonzero(patches.face_labels >= 0).astype(np.int32)
        )
        if len(wireframe_faces):
            view.addCustom(
                _custom_surface_spec(
                    mesh,
                    colors=None,
                    color_location=None,
                    surface_color=patch_wireframe_color,
                    opacity=1.0,
                    wireframe=True,
                    face_indices=wireframe_faces,
                    normal_offset=0.02,
                )
            )

    if samples is not None:
        if samples.component_id != mesh.component_id:
            raise ValueError("samples belong to a different surface component")
        for position, normal in zip(samples.positions, samples.normals):
            end = position + normal_length * normal
            if sample_radius > 0.0:
                view.addSphere(
                    {
                        "center": _vector_records(position[np.newaxis, :])[0],
                        "radius": float(sample_radius),
                        "color": normal_color,
                    }
                )
            view.addArrow(
                {
                    "start": _vector_records(position[np.newaxis, :])[0],
                    "end": _vector_records(end[np.newaxis, :])[0],
                    "radius": float(normal_radius),
                    "color": normal_color,
                }
            )
    view.zoomTo()
    view.render()
    if colorbar:
        assert field_name is not None and normalization is not None
        legend = _surface_colorbar_html(
            field_name=field_name,
            units=field_units,
            cmap=cmap,
            normalization=normalization,
        )
        closing_tag = view.startjs.find("</div>")
        if closing_tag < 0:
            raise RuntimeError("Could not attach colorbar to the py3Dmol view")
        closing_tag += len("</div>")
        wrapper = (
            f'<div style="position:relative;width:{width}px;height:{height}px">'
        )
        view.startjs = (
            wrapper
            + view.startjs[:closing_tag]
            + legend
            + "</div>"
            + view.startjs[closing_tag:]
        )
    return view


def _validate_min_sequence_separation(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("min_sequence_separation must be an integer >= 1")
    return value


def _create_axes(
    rows: int,
    columns: int,
    ax: Axes | None,
) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    width = max(5.0, min(12.0, columns * 0.22 + 2.5))
    height = max(4.0, min(12.0, rows * 0.22 + 2.0))
    return plt.subplots(figsize=(width, height))


def _tick_positions(length: int, maximum_ticks: int = 25) -> NDArray[np.int_]:
    if length == 0:
        return np.asarray([], dtype=int)
    if length <= maximum_ticks:
        return np.arange(length, dtype=int)
    return np.unique(
        np.linspace(0, length - 1, maximum_ticks, dtype=int)
    )


def _configure_matrix_axes(
    ax: Axes,
    row_labels: list[str],
    column_labels: list[str],
    *,
    row_axis_label: str,
    column_axis_label: str,
    title: str,
) -> None:
    row_ticks = _tick_positions(len(row_labels))
    column_ticks = _tick_positions(len(column_labels))
    ax.set_yticks(row_ticks, [row_labels[index] for index in row_ticks])
    ax.set_xticks(
        column_ticks,
        [column_labels[index] for index in column_ticks],
        rotation=90,
    )
    ax.set_ylabel(row_axis_label)
    ax.set_xlabel(column_axis_label)
    ax.set_title(title)


def _plot_distance_data(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    *,
    row_axis_label: str,
    column_axis_label: str,
    title: str,
    ax: Axes | None,
    cmap: str,
) -> tuple[Figure, Axes]:
    if not row_labels or not column_labels:
        raise ValueError("Cannot plot a matrix without protein residues")
    figure, axes = _create_axes(len(row_labels), len(column_labels), ax)
    color_map = plt.get_cmap(cmap).with_extremes(bad="#d9d9d9")
    image = axes.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=0.0,
    )
    _configure_matrix_axes(
        axes,
        row_labels,
        column_labels,
        row_axis_label=row_axis_label,
        column_axis_label=column_axis_label,
        title=title,
    )
    figure.colorbar(image, ax=axes, label="Distance (Å)")
    return figure, axes


def plot_distance_matrix(
    structure: Structure,
    chain: str,
    distance_metric: str = "min_heavy_atom",
    *,
    ax: Axes | None = None,
    cmap: str = "viridis_r",
) -> tuple[Figure, Axes]:
    """Plot the residue-distance heatmap within one chain."""
    matrix = get_distance_matrix(structure, chain, distance_metric)
    labels = [
        _residue_label(chain, residue)
        for residue in _matrix_residues(structure, chain)
    ]
    return _plot_distance_data(
        matrix,
        labels,
        labels,
        row_axis_label=f"Chain {chain}",
        column_axis_label=f"Chain {chain}",
        title=f"Intrachain distance matrix: {chain} ({distance_metric})",
        ax=ax,
        cmap=cmap,
    )


def plot_interchain_distance_matrix(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    distance_metric: str = "min_heavy_atom",
    *,
    ax: Axes | None = None,
    cmap: str = "viridis_r",
) -> tuple[Figure, Axes]:
    """Plot the rectangular residue-distance heatmap between two chains."""
    matrix = get_interchain_distance_matrix(
        structure,
        chain_a,
        chain_b,
        distance_metric,
    )
    row_labels = [
        _residue_label(chain_a, residue)
        for residue in _matrix_residues(structure, chain_a)
    ]
    column_labels = [
        _residue_label(chain_b, residue)
        for residue in _matrix_residues(structure, chain_b)
    ]
    return _plot_distance_data(
        matrix,
        row_labels,
        column_labels,
        row_axis_label=f"Chain {chain_a}",
        column_axis_label=f"Chain {chain_b}",
        title=(
            f"Interchain distance matrix: {chain_a}–{chain_b} "
            f"({distance_metric})"
        ),
        ax=ax,
        cmap=cmap,
    )


def _contact_flags(interaction_measure: str) -> dict[str, bool]:
    if interaction_measure == "interaction_type":
        return {interaction_type: True for interaction_type in _CONTACT_TYPES}
    if interaction_measure not in _CONTACT_TYPE_ORDER:
        choices = sorted(
            _DISTANCE_MEASURES
            | set(_CONTACT_TYPES)
            | {"interaction_type"}
        )
        raise ValueError(
            f"Unsupported interaction_measure {interaction_measure!r}; choose "
            f"from {', '.join(choices)}"
        )
    return {
        interaction_type: interaction_type == interaction_measure
        for interaction_type in _CONTACT_TYPES
    }


def _record_residue_id(record: dict[str, Any], side: str) -> tuple[str, int, str]:
    residue_id = record.get(f"residue_{side}_id")
    if residue_id is not None:
        return tuple(residue_id)
    return (" ", int(record[f"residue_{side}_num"]), " ")


def _interaction_cells(
    records: list[dict[str, Any]],
    residues_a: list[Any],
    residues_b: list[Any],
    *,
    symmetric: bool,
) -> dict[tuple[int, int], frozenset[str]]:
    index_a = {residue.id: index for index, residue in enumerate(residues_a)}
    index_b = {residue.id: index for index, residue in enumerate(residues_b)}
    interactions: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for record in records:
        residue_a_id = _record_residue_id(record, "a")
        residue_b_id = _record_residue_id(record, "b")
        if residue_a_id not in index_a or residue_b_id not in index_b:
            continue
        cell = (index_a[residue_a_id], index_b[residue_b_id])
        interactions[cell].add(str(record["interaction_type"]))
        if symmetric:
            interactions[(cell[1], cell[0])].add(
                str(record["interaction_type"])
            )
    return {
        cell: frozenset(interaction_types)
        for cell, interaction_types in interactions.items()
    }


def _category_sort_key(category: frozenset[str]) -> tuple[Any, ...]:
    return (
        len(category),
        tuple(
            sorted(
                (_CONTACT_TYPE_ORDER.get(value, len(_CONTACT_TYPES)), value)
                for value in category
            )
        ),
    )


def _category_label(category: frozenset[str]) -> str:
    ordered = sorted(
        category,
        key=lambda value: _CONTACT_TYPE_ORDER.get(value, len(_CONTACT_TYPES)),
    )
    return " + ".join(_CONTACT_TYPE_LABELS[value] for value in ordered)


def _category_colors(
    categories: list[frozenset[str]],
) -> dict[frozenset[str], Any]:
    colors = {}
    combinations = [category for category in categories if len(category) > 1]
    combination_map = plt.get_cmap("turbo")
    combination_values = np.linspace(0.08, 0.92, max(len(combinations), 1))
    combination_colors = iter(combination_map(combination_values))
    for category in categories:
        if len(category) == 1:
            interaction_type = next(iter(category))
            colors[category] = _CONTACT_TYPE_COLORS[interaction_type]
        else:
            colors[category] = next(combination_colors)
    return colors


def _plot_contact_categories(
    cells: dict[tuple[int, int], frozenset[str]],
    row_labels: list[str],
    column_labels: list[str],
    *,
    row_axis_label: str,
    column_axis_label: str,
    title: str,
    ax: Axes | None,
) -> tuple[Figure, Axes]:
    if not row_labels or not column_labels:
        raise ValueError("Cannot plot a matrix without protein residues")
    categories = sorted(set(cells.values()), key=_category_sort_key)
    category_codes = {
        category: index for index, category in enumerate(categories, start=1)
    }
    matrix = np.zeros((len(row_labels), len(column_labels)), dtype=int)
    for (row, column), category in cells.items():
        matrix[row, column] = category_codes[category]

    colors_by_category = _category_colors(categories)
    colors = ["#ffffff"] + [colors_by_category[value] for value in categories]
    color_map = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5),
        color_map.N,
    )
    figure, axes = _create_axes(len(row_labels), len(column_labels), ax)
    axes.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=norm,
    )
    _configure_matrix_axes(
        axes,
        row_labels,
        column_labels,
        row_axis_label=row_axis_label,
        column_axis_label=column_axis_label,
        title=title,
    )

    handles = []
    single_categories = [category for category in categories if len(category) == 1]
    combination_categories = [category for category in categories if len(category) > 1]
    if single_categories:
        handles.append(
            Patch(
                facecolor="none",
                edgecolor="none",
                label="Single interactions",
            )
        )
        handles.extend(
            Patch(
                facecolor=colors_by_category[category],
                edgecolor="none",
                label=_category_label(category),
            )
            for category in single_categories
        )
    if combination_categories:
        handles.append(
            Patch(
                facecolor="none",
                edgecolor="none",
                label="Multiple interactions",
            )
        )
        handles.extend(
            Patch(
                facecolor=colors_by_category[category],
                edgecolor="#333333",
                linewidth=1.2,
                label=_category_label(category),
            )
            for category in combination_categories
        )
    if handles:
        axes.legend(
            handles=handles,
            title="Observed interaction categories",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
        )
    return figure, axes


def plot_interaction_matrix(
    structure: Structure,
    chain: str,
    interaction_measure: InteractionMeasure = "min_heavy_atom",
    *,
    min_sequence_separation: int = 2,
    topology_backend: str = "templates",
    ax: Axes | None = None,
    cmap: str = "viridis_r",
) -> tuple[Figure, Axes]:
    """Plot distance or characterized interactions within one chain."""
    separation = _validate_min_sequence_separation(min_sequence_separation)
    residues = _matrix_residues(structure, chain)
    labels = [_residue_label(chain, residue) for residue in residues]
    if interaction_measure in _DISTANCE_MEASURES:
        matrix = get_distance_matrix(structure, chain, interaction_measure)
        indices = np.arange(len(residues))
        mask = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :]) < separation
        matrix = matrix.copy()
        matrix[mask] = np.nan
        return _plot_distance_data(
            matrix,
            labels,
            labels,
            row_axis_label=f"Chain {chain}",
            column_axis_label=f"Chain {chain}",
            title=f"Intrachain interactions: {chain} ({interaction_measure})",
            ax=ax,
            cmap=cmap,
        )

    flags = _contact_flags(interaction_measure)
    from .geometry import characterize_intrachain_contacts

    records = characterize_intrachain_contacts(
        structure,
        chain,
        min_sequence_separation=separation,
        topology_backend=topology_backend,
        **flags,
    )
    cells = _interaction_cells(records, residues, residues, symmetric=True)
    return _plot_contact_categories(
        cells,
        labels,
        labels,
        row_axis_label=f"Chain {chain}",
        column_axis_label=f"Chain {chain}",
        title=f"Intrachain interactions: {chain} ({interaction_measure})",
        ax=ax,
    )


def plot_interchain_interaction_matrix(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    interaction_measure: InteractionMeasure = "min_heavy_atom",
    *,
    topology_backend: str = "templates",
    ax: Axes | None = None,
    cmap: str = "viridis_r",
) -> tuple[Figure, Axes]:
    """Plot distance or characterized interactions between two chains."""
    if interaction_measure in _DISTANCE_MEASURES:
        return plot_interchain_distance_matrix(
            structure,
            chain_a,
            chain_b,
            interaction_measure,
            ax=ax,
            cmap=cmap,
        )

    flags = _contact_flags(interaction_measure)
    from .geometry import characterize_chain_contacts

    records = characterize_chain_contacts(
        structure,
        chain_a,
        chain_b,
        atomic=False,
        topology_backend=topology_backend,
        **flags,
    )
    residues_a = _matrix_residues(structure, chain_a)
    residues_b = _matrix_residues(structure, chain_b)
    row_labels = [
        _residue_label(chain_a, residue) for residue in residues_a
    ]
    column_labels = [
        _residue_label(chain_b, residue) for residue in residues_b
    ]
    cells = _interaction_cells(
        records,
        residues_a,
        residues_b,
        symmetric=False,
    )
    return _plot_contact_categories(
        cells,
        row_labels,
        column_labels,
        row_axis_label=f"Chain {chain_a}",
        column_axis_label=f"Chain {chain_b}",
        title=(
            f"Interchain interactions: {chain_a}–{chain_b} "
            f"({interaction_measure})"
        ),
        ax=ax,
    )
