"""Tests for structural distance and interaction matrices."""

from collections.abc import Mapping, Sequence
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools import pdbtools
from biotools.structure import (
    characterize_intrachain_contacts,
    get_distance_matrix,
    get_interchain_distance_matrix,
    plot_distance_matrix,
    plot_interaction_matrix,
    plot_interchain_distance_matrix,
    plot_interchain_interaction_matrix,
)


def _residue(
    number: int,
    name: str,
    atoms: Sequence[tuple[str, Sequence[float], str]],
    *,
    insertion_code: str = " ",
) -> Residue.Residue:
    residue = Residue.Residue((" ", number, insertion_code), name, " ")
    for serial_number, (atom_name, coordinate, element) in enumerate(
        atoms,
        start=1,
    ):
        residue.add(
            Atom.Atom(
                atom_name,
                np.asarray(coordinate, dtype=float),
                0.0,
                1.0,
                " ",
                f"{atom_name:>4}",
                serial_number,
                element=element,
            )
        )
    return residue


def _structure(
    chains: Mapping[str, Sequence[Residue.Residue]],
) -> Structure.Structure:
    structure = Structure.Structure("matrices")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residues in chains.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue in residues:
            chain.add(residue)
    return structure


def _matrix_structure() -> Structure.Structure:
    return _structure(
        {
            "A": [
                _residue(
                    1,
                    "ALA",
                    [("CA", (0, 0, 0), "C"), ("CB", (1, 0, 0), "C")],
                ),
                _residue(2, "GLY", [("CA", (3, 0, 0), "C")]),
                _residue(3, "ALA", [("CA", (7, 0, 0), "C")]),
            ],
            "B": [
                _residue(
                    10,
                    "VAL",
                    [("CA", (0, 4, 0), "C"), ("CB", (1, 4, 0), "C")],
                    insertion_code="A",
                ),
                _residue(
                    11,
                    "ALA",
                    [("CA", (7, 4, 0), "C"), ("CB", (8, 4, 0), "C")],
                ),
            ],
        }
    )


def test_intrachain_distance_matrices_preserve_missing_residues() -> None:
    structure = _matrix_structure()

    c_alpha = get_distance_matrix(structure, "A", "c_alpha")
    c_beta = get_distance_matrix(structure, "A", "c_beta")

    np.testing.assert_allclose(
        c_alpha,
        np.asarray([[0.0, 3.0, 7.0], [3.0, 0.0, 4.0], [7.0, 4.0, 0.0]]),
    )
    assert c_beta.shape == (3, 3)
    np.testing.assert_allclose(c_beta[:2, :2], [[0.0, 2.0], [2.0, 0.0]])
    assert np.isnan(c_beta[2]).all()
    assert np.isnan(c_beta[:, 2]).all()


def test_interchain_distance_matrix_is_rectangular() -> None:
    matrix = get_interchain_distance_matrix(
        _matrix_structure(),
        "A",
        "B",
        "c_alpha",
    )

    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(
        matrix,
        [[4.0, np.sqrt(65.0)], [5.0, np.sqrt(32.0)], [np.sqrt(65.0), 4.0]],
    )


def test_matrix_validation_rejects_invalid_metric_and_same_chain_pair() -> None:
    structure = _matrix_structure()
    with pytest.raises(ValueError, match="Unsupported distance_metric"):
        get_distance_matrix(structure, "A", "min_atom")
    with pytest.raises(ValueError, match="must differ"):
        get_interchain_distance_matrix(structure, "A", "A")


def test_distance_plot_functions_return_labeled_heatmaps() -> None:
    structure = _matrix_structure()

    figure, axes = plot_distance_matrix(structure, "A", "c_alpha")
    inter_figure, inter_axes = plot_interchain_distance_matrix(
        structure,
        "A",
        "B",
        "c_alpha",
    )

    assert axes.figure is figure
    assert axes.images[0].get_array().shape == (3, 3)
    assert inter_axes.figure is inter_figure
    assert inter_axes.images[0].get_array().shape == (3, 2)
    assert "B:10A" in [label.get_text() for label in inter_axes.get_xticklabels()]
    plt.close(figure)
    plt.close(inter_figure)


def test_distance_interaction_plot_masks_sequence_neighbors() -> None:
    figure, axes = plot_interaction_matrix(
        _matrix_structure(),
        "A",
        interaction_measure="c_alpha",
        min_sequence_separation=2,
    )

    plotted = axes.images[0].get_array()
    assert np.ma.is_masked(plotted[0, 0])
    assert np.ma.is_masked(plotted[0, 1])
    assert plotted[0, 2] == pytest.approx(7.0)
    plt.close(figure)


def test_intrachain_contact_characterization_excludes_direct_neighbors() -> None:
    structure = _structure(
        {
            "A": [
                _residue(1, "ALA", [("CA", (0, 0, 0), "C")]),
                _residue(2, "ALA", [("CA", (20, 0, 0), "C")]),
                _residue(3, "ALA", [("CA", (3, 0, 0), "C")]),
            ]
        }
    )
    disabled = {
        "hydrogen_bond": False,
        "salt_bridge": False,
        "hydrophobic_contact": False,
        "pi_stacking_parallel": False,
        "pi_stacking_t_shaped": False,
        "cation_pi_candidate": False,
        "water_bridge": False,
    }

    records = characterize_intrachain_contacts(
        structure,
        "A",
        min_sequence_separation=2,
        van_der_waals_contact=True,
        **disabled,
    )

    assert len(records) == 1
    assert records[0]["residue_a_num"] == 1
    assert records[0]["residue_b_num"] == 3
    assert records[0]["interaction_type"] == "van_der_waals_contact"


def _contact_record(
    residue_a: int,
    residue_b: int,
    interaction_type: str,
    *,
    insertion_code_b: str = " ",
) -> dict[str, object]:
    return {
        "residue_a_id": (" ", residue_a, " "),
        "residue_a_num": residue_a,
        "residue_b_id": (" ", residue_b, insertion_code_b),
        "residue_b_num": residue_b,
        "interaction_type": interaction_type,
    }


def test_combined_interaction_plot_assigns_colors_to_observed_combinations() -> None:
    records = [
        _contact_record(1, 10, "hydrogen_bond", insertion_code_b="A"),
        _contact_record(1, 10, "salt_bridge", insertion_code_b="A"),
        _contact_record(2, 11, "van_der_waals_contact"),
    ]
    with patch(
        "biotools.structure.geometry.characterize_chain_contacts",
        return_value=records,
    ):
        figure, axes = plot_interchain_interaction_matrix(
            _matrix_structure(),
            "A",
            "B",
            interaction_measure="interaction_type",
        )

    plotted = np.asarray(axes.images[0].get_array())
    assert plotted[0, 0] != 0
    assert plotted[1, 1] != 0
    assert plotted[0, 0] != plotted[1, 1]
    legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert "van der Waals contact" in legend_labels
    assert "hydrogen bond + salt bridge" in legend_labels
    assert "hydrophobic contact" not in legend_labels
    plt.close(figure)


def test_intrachain_interaction_plot_is_symmetric_and_forwards_separation() -> None:
    records = [_contact_record(1, 3, "salt_bridge")]
    with patch(
        "biotools.structure.geometry.characterize_intrachain_contacts",
        return_value=records,
    ) as characterize:
        figure, axes = plot_interaction_matrix(
            _matrix_structure(),
            "A",
            interaction_measure="salt_bridge",
        )

    plotted = np.asarray(axes.images[0].get_array())
    assert plotted[0, 2] == plotted[2, 0] != 0
    characterize.assert_called_once()
    assert characterize.call_args.kwargs["min_sequence_separation"] == 2
    assert characterize.call_args.kwargs["salt_bridge"] is True
    assert characterize.call_args.kwargs["hydrogen_bond"] is False
    plt.close(figure)


def test_matrix_api_is_exported_from_legacy_facade() -> None:
    assert pdbtools.get_distance_matrix is get_distance_matrix
    assert pdbtools.plot_interaction_matrix is plot_interaction_matrix
