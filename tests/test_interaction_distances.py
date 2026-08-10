"""Tests for residue-interaction distance metrics."""

from collections.abc import Sequence

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools.structure import (
    get_interaction_residues,
    get_interaction_residues_full,
)


def _atom(
    name: str,
    coordinate: Sequence[float],
    element: str,
    serial_number: int,
) -> Atom.Atom:
    return Atom.Atom(
        name,
        np.asarray(coordinate, dtype=float),
        0.0,
        1.0,
        " ",
        f"{name:>4}",
        serial_number,
        element=element,
    )


def _residue(
    number: int,
    name: str,
    atoms: Sequence[tuple[str, Sequence[float], str]],
) -> Residue.Residue:
    residue = Residue.Residue((" ", number, " "), name, " ")
    for serial_number, (atom_name, coordinate, element) in enumerate(
        atoms,
        start=1,
    ):
        residue.add(_atom(atom_name, coordinate, element, serial_number))
    return residue


def _structure(
    residues_a: Sequence[Residue.Residue],
    residues_b: Sequence[Residue.Residue],
) -> Structure.Structure:
    structure = Structure.Structure("distance-metrics")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residues in (("A", residues_a), ("B", residues_b)):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue in residues:
            chain.add(residue)
    return structure


def test_min_heavy_atom_is_default_and_min_atom_includes_hydrogens() -> None:
    structure = _structure(
        [_residue(1, "ALA", [("CA", (0, 0, 0), "C"), ("HA", (4, 0, 0), "H")])],
        [_residue(2, "ALA", [("CA", (10, 0, 0), "C"), ("HA", (6, 0, 0), "H")])],
    )

    assert get_interaction_residues(structure, "A", "B", cutoff=3.0) == []
    assert get_interaction_residues(
        structure,
        "A",
        "B",
        cutoff=3.0,
        distance_metric="min_atom",
    ) == [[1, "ALA", 2, "ALA", 2.0]]


def test_c_alpha_and_c_beta_use_the_requested_representative_atoms() -> None:
    structure = _structure(
        [
            _residue(
                1,
                "ALA",
                [("CA", (0, 0, 0), "C"), ("CB", (4, 0, 0), "C")],
            )
        ],
        [
            _residue(
                2,
                "ALA",
                [("CA", (10, 0, 0), "C"), ("CB", (6, 0, 0), "C")],
            )
        ],
    )

    assert get_interaction_residues(
        structure,
        "A",
        "B",
        cutoff=8.0,
        distance_metric="c_alpha",
    ) == []
    assert get_interaction_residues(
        structure,
        "A",
        "B",
        cutoff=8.0,
        distance_metric="c_beta",
    ) == [[1, "ALA", 2, "ALA", 2.0]]


def test_c_beta_uses_c_alpha_only_for_glycine() -> None:
    glycine_structure = _structure(
        [_residue(1, "GLY", [("CA", (0, 0, 0), "C")])],
        [
            _residue(
                2,
                "ALA",
                [("CA", (10, 0, 0), "C"), ("CB", (7, 0, 0), "C")],
            )
        ],
    )
    missing_cb_structure = _structure(
        [_residue(1, "ALA", [("CA", (0, 0, 0), "C")])],
        [
            _residue(
                2,
                "ALA",
                [("CA", (2, 0, 0), "C"), ("CB", (3, 0, 0), "C")],
            )
        ],
    )

    assert get_interaction_residues(
        glycine_structure,
        "A",
        "B",
        cutoff=8.0,
        distance_metric="c_beta",
    ) == [[1, "GLY", 2, "ALA", 7.0]]
    assert get_interaction_residues(
        missing_cb_structure,
        "A",
        "B",
        cutoff=8.0,
        distance_metric="c_beta",
    ) == []


@pytest.mark.parametrize(
    "distance_metric,cutoff",
    [
        ("min_heavy_atom", 5.0),
        ("min_atom", 5.0),
        ("c_alpha", 8.0),
        ("c_beta", 8.0),
    ],
)
def test_kd_tree_and_full_search_match(
    distance_metric: str,
    cutoff: float,
) -> None:
    structure = _structure(
        [
            _residue(
                1,
                "ALA",
                [
                    ("CA", (0, 0, 0), "C"),
                    ("CB", (1, 0, 0), "C"),
                    ("HA", (2, 0, 0), "H"),
                ],
            )
        ],
        [
            _residue(
                2,
                "VAL",
                [
                    ("CA", (7, 0, 0), "C"),
                    ("CB", (5, 0, 0), "C"),
                    ("HA", (3, 0, 0), "H"),
                ],
            )
        ],
    )

    optimized = get_interaction_residues(
        structure,
        "A",
        "B",
        cutoff=cutoff,
        distance_metric=distance_metric,
    )
    complete = get_interaction_residues_full(
        structure,
        "A",
        "B",
        cutoff=cutoff,
        distance_metric=distance_metric,
    )

    assert optimized == complete


@pytest.mark.parametrize(
    "function",
    [get_interaction_residues, get_interaction_residues_full],
)
def test_rejects_unknown_distance_metric(function) -> None:
    structure = _structure(
        [_residue(1, "ALA", [("CA", (0, 0, 0), "C")])],
        [_residue(2, "ALA", [("CA", (1, 0, 0), "C")])],
    )
    with pytest.raises(ValueError, match="Unsupported distance_metric"):
        function(structure, "A", "B", distance_metric="unknown")
