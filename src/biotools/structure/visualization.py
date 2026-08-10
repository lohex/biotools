"""Interactive protein structure visualization."""

from __future__ import annotations

from collections import defaultdict
from io import StringIO
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray

from .matrices import (
    _matrix_residues,
    _residue_label,
    get_distance_matrix,
    get_interchain_distance_matrix,
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


def plot_structure(structure: Structure) -> Any:
    """Create an interactive ``py3Dmol`` view for a structure.

    Args:
        structure: Biopython structure to render as a spectrum-colored cartoon.

    Returns:
        Configured ``py3Dmol.view`` instance.
    """
    import py3Dmol
    from Bio.PDB import PDBIO

    buffer = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(buffer)

    view = py3Dmol.view(width=800, height=400)
    view.addModel(buffer.getvalue(), "pdb")
    view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
    view.zoomTo()
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
