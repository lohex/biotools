"""Residue distance matrices for individual chains and chain interfaces."""

from __future__ import annotations

from typing import Any, cast, Literal, TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._contacts import _is_protein_residue
from .geometry import _distance_atoms

if TYPE_CHECKING:
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

MatrixDistanceMetric: TypeAlias = Literal[
    "c_alpha",
    "c_beta",
    "min_heavy_atom",
]
DistanceMatrix: TypeAlias = NDArray[np.float64]

_MATRIX_DISTANCE_METRICS = {
    "c_alpha",
    "c_beta",
    "min_heavy_atom",
}


def _validate_matrix_distance_metric(
    distance_metric: str,
) -> MatrixDistanceMetric:
    if distance_metric not in _MATRIX_DISTANCE_METRICS:
        choices = ", ".join(sorted(_MATRIX_DISTANCE_METRICS))
        raise ValueError(
            f"Unsupported distance_metric {distance_metric!r}; choose from "
            f"{choices}"
        )
    return cast(MatrixDistanceMetric, distance_metric)


def _matrix_residues(
    structure: Structure,
    chain_id: str,
) -> list[Residue]:
    models = list(structure)
    matches = [model for model in models if chain_id in model]
    if not matches:
        raise ValueError(f"Chain {chain_id!r} not found in structure")
    if len(matches) > 1:
        raise ValueError(
            f"Chain {chain_id!r} occurs in multiple models; select one model "
            "before calculating a matrix"
        )
    residues = [
        residue
        for residue in matches[0][chain_id].get_residues()
        if _is_protein_residue(residue)
    ]
    if not residues:
        raise ValueError(f"Chain {chain_id!r} contains no protein residues")
    return residues


def _atom_coordinates(
    residues: list[Residue],
    distance_metric: MatrixDistanceMetric,
) -> list[NDArray[np.float64]]:
    coordinates = []
    for residue in residues:
        atoms = _distance_atoms(residue, distance_metric)
        coordinates.append(
            np.asarray([atom.coord for atom in atoms], dtype=float).reshape(-1, 3)
        )
    return coordinates


def _minimum_distance(
    coordinates_a: NDArray[np.float64],
    coordinates_b: NDArray[np.float64],
) -> float:
    if coordinates_a.size == 0 or coordinates_b.size == 0:
        return float("nan")
    differences = coordinates_a[:, np.newaxis, :] - coordinates_b[np.newaxis, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    return float(np.sqrt(np.min(squared_distances)))


def get_distance_matrix(
    structure: Structure,
    chain: str,
    distance_metric: MatrixDistanceMetric = "min_heavy_atom",
) -> DistanceMatrix:
    """Return the symmetric residue-distance matrix of one protein chain.

    Matrix rows and columns follow residue order in the selected model. Missing
    atoms required by a metric produce ``NaN`` while retaining the residue's
    row and column. For ``"c_beta"``, glycine uses C-alpha; other residues
    without C-beta remain ``NaN``.

    Args:
        structure: Biopython structure containing exactly one matching model.
        chain: Chain identifier to analyze.
        distance_metric: ``"c_alpha"``, ``"c_beta"``, or
            ``"min_heavy_atom"``.

    Returns:
        Square floating-point matrix of distances in angstroms.

    Raises:
        ValueError: If the chain, model selection, or metric is invalid.
    """
    metric = _validate_matrix_distance_metric(distance_metric)
    residues = _matrix_residues(structure, chain)
    coordinates = _atom_coordinates(residues, metric)
    matrix = np.full((len(residues), len(residues)), np.nan, dtype=float)
    for index_a, coordinates_a in enumerate(coordinates):
        for index_b in range(index_a, len(coordinates)):
            distance = _minimum_distance(coordinates_a, coordinates[index_b])
            matrix[index_a, index_b] = distance
            matrix[index_b, index_a] = distance
    return matrix


def get_interchain_distance_matrix(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    distance_metric: MatrixDistanceMetric = "min_heavy_atom",
) -> DistanceMatrix:
    """Return the rectangular residue-distance matrix between two chains.

    Rows correspond to ``chain_a`` and columns to ``chain_b``. Missing atoms
    required by the selected metric produce ``NaN`` without removing residues.

    Args:
        structure: Biopython structure containing both chains in one model.
        chain_a: Chain represented by matrix rows.
        chain_b: Chain represented by matrix columns.
        distance_metric: ``"c_alpha"``, ``"c_beta"``, or
            ``"min_heavy_atom"``.

    Returns:
        Floating-point matrix of distances in angstroms.

    Raises:
        ValueError: If chains are identical, absent, ambiguous, or the metric
            is invalid.
    """
    if chain_a == chain_b:
        raise ValueError(
            "chain_a and chain_b must differ; use get_distance_matrix() for "
            "intrachain distances"
        )
    metric = _validate_matrix_distance_metric(distance_metric)
    residues_a = _matrix_residues(structure, chain_a)
    residues_b = _matrix_residues(structure, chain_b)
    model_ids_a = {residue.get_parent().get_parent().id for residue in residues_a}
    model_ids_b = {residue.get_parent().get_parent().id for residue in residues_b}
    if model_ids_a != model_ids_b:
        raise ValueError("The requested chains do not occur in the same model")

    coordinates_a = _atom_coordinates(residues_a, metric)
    coordinates_b = _atom_coordinates(residues_b, metric)
    matrix = np.full((len(residues_a), len(residues_b)), np.nan, dtype=float)
    for index_a, residue_coordinates_a in enumerate(coordinates_a):
        for index_b, residue_coordinates_b in enumerate(coordinates_b):
            matrix[index_a, index_b] = _minimum_distance(
                residue_coordinates_a,
                residue_coordinates_b,
            )
    return matrix


def _residue_label(chain_id: str, residue: Any) -> str:
    _, residue_number, insertion_code = residue.id
    return f"{chain_id}:{residue_number}{str(insertion_code).strip()}"
