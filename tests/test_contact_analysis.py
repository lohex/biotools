"""Tests for typed contact results and refined chemical geometry."""

from __future__ import annotations

from itertools import count
import json

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools.structure import (
    aggregate_contact_features,
    ContactAnalysisResult,
    ContactConfig,
    analyze_contacts,
    characterize_chain_contacts,
    get_contact_config,
    prepare_contact_system,
)


_SERIALS = count(1)
_DISABLED = {
    "hydrogen_bond": False,
    "salt_bridge": False,
    "hydrophobic_contact": False,
    "van_der_waals_contact": False,
    "pi_stacking_parallel": False,
    "cation_pi_candidate": False,
    "water_bridge": False,
    "pi_stacking_t_shaped": False,
}


def _residue(number, name, atoms, hetero=" "):
    residue = Residue.Residue((hetero, number, " "), name, " ")
    for atom_name, coordinate, element in atoms:
        residue.add(
            Atom.Atom(
                atom_name,
                np.asarray(coordinate, dtype=float),
                0.0,
                1.0,
                " ",
                f"{atom_name:>4}",
                next(_SERIALS),
                element=element,
            )
        )
    return residue


def _structure(residues_a, residues_b):
    structure = Structure.Structure("typed-contacts")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residues in (("A", residues_a), ("B", residues_b)):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue in residues:
            chain.add(residue)
    return structure


def test_contact_config_round_trip_and_hash_are_stable() -> None:
    config = ContactConfig(hydrogen_bond_distance=3.25)

    restored = ContactConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.config_hash == config.config_hash
    with pytest.raises(ValueError, match="Unknown contact profile"):
        get_contact_config("missing")


def test_typed_result_round_trip_preserves_roles_geometry_and_diagnostics() -> None:
    structure = _structure(
        [_residue(1, "SER", [("OG", (0, 0, 0), "O"), ("HG", (1, 0, 0), "H")])],
        [_residue(2, "ASP", [("OD1", (3.2, 0, 0), "O")])],
    )
    prepared = prepare_contact_system(structure, ("A", "B"))

    result = analyze_contacts(prepared, **(_DISABLED | {"hydrogen_bond": True}))
    restored = ContactAnalysisResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored == result
    assert result.rule_profile == "refined"
    assert result.observations[0].role_a == "donor"
    assert result.observations[0].role_b == "acceptor"
    measurements = {item.name: item for item in result.observations[0].geometry}
    assert measurements["donor_acceptor_distance"].value == pytest.approx(3.2)
    assert measurements["donor_hydrogen_acceptor_angle"].value == pytest.approx(180.0)

    features = aggregate_contact_features(result)
    assert features.schema == "contact_features_v1"
    assert features.features["hydrogen_bond.observation_count"] == 1.0
    assert features.contact_config_hash == result.config_hash


def test_legacy_and_refined_hydrogen_bond_profiles_are_explicit() -> None:
    structure = _structure(
        [_residue(1, "SER", [("OG", (0, 0, 0), "O"), ("HG", (1, 0, 0), "H")])],
        [_residue(2, "ASP", [("OD1", (3.2, 0, 0), "O")])],
    )
    flags = _DISABLED | {"hydrogen_bond": True}

    refined = characterize_chain_contacts(
        structure, "A", "B", atomic=True, profile="refined", **flags
    )
    legacy = characterize_chain_contacts(
        structure, "A", "B", atomic=True, profile="legacy", **flags
    )

    assert len(refined) == 1
    assert refined[0]["rule_profile"] == "refined"
    assert legacy == []


def test_vdw_reports_surface_gap_and_clash_status() -> None:
    structure = _structure(
        [_residue(1, "ALA", [("CB", (0, 0, 0), "C")])],
        [_residue(2, "ALA", [("CB", (2.0, 0, 0), "C")])],
    )

    records = characterize_chain_contacts(
        structure,
        "A",
        "B",
        atomic=True,
        **(_DISABLED | {"van_der_waals_contact": True}),
    )

    assert records[0]["geometry_metrics"]["surface_gap"] == pytest.approx(-1.4)
    assert records[0]["geometry_metrics"]["overlap_depth"] == pytest.approx(1.4)
    assert records[0]["quality_flags"] == ["steric_clash"]


def test_charged_thiolate_is_not_classified_as_hydrophobic() -> None:
    structure = _structure(
        [_residue(1, "CYM", [("SG", (0, 0, 0), "S")])],
        [_residue(2, "MET", [("SD", (3.5, 0, 0), "S")])],
    )

    records = characterize_chain_contacts(
        structure,
        "A",
        "B",
        atomic=True,
        **(_DISABLED | {"hydrophobic_contact": True}),
    )

    assert records == []


def _phenylalanine(number, center, axes):
    names = ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")
    atoms = []
    for index, name in enumerate(names):
        angle = index * np.pi / 3
        coordinate = np.asarray(center) + 1.4 * (
            np.cos(angle) * np.asarray(axes[0])
            + np.sin(angle) * np.asarray(axes[1])
        )
        atoms.append((name, coordinate, "C"))
    return _residue(number, "PHE", atoms)


def test_refined_cation_pi_rejects_coplanar_and_laterally_displaced_groups() -> None:
    ring = _phenylalanine(2, (0, 0, 0), ((1, 0, 0), (0, 1, 0)))
    coplanar = _residue(1, "LYS", [("NZ", (0, 0, 0), "N")])
    displaced = _residue(1, "LYS", [("NZ", (4.5, 0, 3.0), "N")])
    flags = _DISABLED | {"cation_pi_candidate": True}

    assert characterize_chain_contacts(
        _structure([coplanar], [ring]), "A", "B", atomic=True, **flags
    ) == []
    assert characterize_chain_contacts(
        _structure([displaced], [ring.copy()]), "A", "B", atomic=True, **flags
    ) == []


def test_nonplanar_aromatic_group_is_rejected() -> None:
    axes = ((1, 0, 0), (0, 1, 0))
    planar = _phenylalanine(1, (0, 0, 0), axes)
    nonplanar = _phenylalanine(2, (0, 0, 4.5), axes)
    nonplanar["CZ"].set_coord(nonplanar["CZ"].get_coord() + np.array([0, 0, 1.0]))

    records = characterize_chain_contacts(
        _structure([planar], [nonplanar]),
        "A",
        "B",
        atomic=True,
        **(_DISABLED | {"pi_stacking_parallel": True}),
    )

    assert records == []
