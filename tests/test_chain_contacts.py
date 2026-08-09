"""Focused regression tests for geometric chain-contact characterization."""

from __future__ import annotations

import builtins
from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools.structure import characterize_chain_contacts
from biotools.structure._contacts import _build_openmm_bond_adjacency


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


def _atom(name: str, coordinate: tuple[float, float, float], element: str) -> Atom.Atom:
    return Atom.Atom(
        name,
        np.asarray(coordinate, dtype=float),
        0.0,
        1.0,
        " ",
        f"{name:>4}",
        next(_SERIALS),
        element=element,
    )


def _residue(
    number: int,
    name: str,
    atoms: list[tuple[str, tuple[float, float, float], str]],
    hetero: str = " ",
) -> Residue.Residue:
    residue = Residue.Residue((hetero, number, " "), name, " ")
    for atom_name, coordinate, element in atoms:
        residue.add(_atom(atom_name, coordinate, element))
    return residue


def _structure(
    residues_a: list[Residue.Residue],
    residues_b: list[Residue.Residue],
    waters: list[Residue.Residue] | None = None,
) -> Structure.Structure:
    structure = Structure.Structure("contacts")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residues in (("A", residues_a), ("B", residues_b)):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue in residues:
            chain.add(residue)
    if waters:
        water_chain = Chain.Chain("W")
        model.add(water_chain)
        for water in waters:
            water_chain.add(water)
    return structure


def _contacts(structure: Structure.Structure, **enabled: bool) -> list[dict]:
    flags = _DISABLED | enabled
    return characterize_chain_contacts(structure, "A", "B", atomic=True, **flags)


def _phenylalanine(
    number: int,
    center: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray],
) -> Residue.Residue:
    names = ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")
    coordinates = []
    for index, name in enumerate(names):
        angle = index * np.pi / 3
        coordinate = center + 1.4 * (np.cos(angle) * axes[0] + np.sin(angle) * axes[1])
        coordinates.append((name, tuple(coordinate), "C"))
    return _residue(number, "PHE", coordinates)


def _alanine(number: int, z_offset: float) -> Residue.Residue:
    return _residue(
        number,
        "ALA",
        [
            ("N", (-1.4, 0.0, z_offset), "N"),
            ("CA", (0.0, 0.0, z_offset), "C"),
            ("C", (1.5, 0.0, z_offset), "C"),
            ("O", (2.5, 0.0, z_offset), "O"),
            ("CB", (0.0, 1.5, z_offset), "C"),
        ],
    )


def _cysteine(number: int, x_offset: float, sulfur_x: float) -> Residue.Residue:
    return _residue(
        number,
        "CYS",
        [
            ("N", (x_offset - 1.4, 0.0, 0.0), "N"),
            ("CA", (x_offset, 0.0, 0.0), "C"),
            ("C", (x_offset + 1.5, 0.0, 0.0), "C"),
            ("O", (x_offset + 2.5, 0.0, 0.0), "O"),
            ("CB", (sulfur_x, 1.8, 0.0), "C"),
            ("SG", (sulfur_x, 0.0, 0.0), "S"),
        ],
    )


def test_hydrogen_bond_uses_hydrogen_as_angle_vertex() -> None:
    structure = _structure(
        [_residue(1, "SER", [("OG", (0.0, 0.0, 0.0), "O"), ("HG", (1.0, 0.0, 0.0), "H")])],
        [_residue(2, "ASP", [("OD1", (2.8, 0.0, 0.0), "O")])],
    )

    result = _contacts(structure, hydrogen_bond=True)

    assert len(result) == 1
    assert result[0]["interaction_type"] == "hydrogen_bond"
    assert result[0]["angle"] == 180.0
    assert result[0]["residue_a_name"] == "SER"
    assert result[0]["residue_b_name"] == "ASP"


def test_non_acceptor_nitrogen_does_not_form_hydrogen_bond() -> None:
    structure = _structure(
        [_residue(1, "SER", [("OG", (0.0, 0.0, 0.0), "O"), ("HG", (1.0, 0.0, 0.0), "H")])],
        [_residue(2, "LYS", [("NZ", (2.8, 0.0, 0.0), "N")])],
    )

    assert _contacts(structure, hydrogen_bond=True) == []


def test_reverse_salt_bridge_keeps_chain_a_and_b_orientation() -> None:
    structure = _structure(
        [_residue(10, "ASP", [("OD1", (-0.5, 0.0, 0.0), "O"), ("OD2", (0.5, 0.0, 0.0), "O")])],
        [_residue(20, "LYS", [("NZ", (4.0, 0.0, 0.0), "N")])],
    )

    result = _contacts(structure, salt_bridge=True)

    assert len(result) == 1
    assert result[0]["residue_a_num"] == 10
    assert result[0]["residue_a_name"] == "ASP"
    assert result[0]["residue_b_num"] == 20
    assert result[0]["residue_b_name"] == "LYS"


def test_protonated_acid_variant_is_not_a_salt_bridge_anion() -> None:
    structure = _structure(
        [_residue(10, "GLH", [("OE1", (-0.5, 0.0, 0.0), "O"), ("OE2", (0.5, 0.0, 0.0), "O")])],
        [_residue(20, "LYS", [("NZ", (4.0, 0.0, 0.0), "N")])],
    )

    assert _contacts(structure, salt_bridge=True) == []


def test_hydrophobic_side_chain_contact_is_detected() -> None:
    residue_a = _residue(
        1,
        "LEU",
        [("CA", (-3.0, 0.0, 0.0), "C"), ("CB", (-2.0, 0.0, 0.0), "C"),
         ("CG", (-1.0, 0.0, 0.0), "C"), ("CD1", (0.0, 0.0, 0.0), "C")],
    )
    residue_b = _residue(
        2,
        "VAL",
        [("CA", (-2.0, 3.5, 0.0), "C"), ("CB", (-1.0, 3.5, 0.0), "C"),
         ("CG1", (0.0, 3.5, 0.0), "C")],
    )

    result = _contacts(_structure([residue_a], [residue_b]), hydrophobic_contact=True)

    assert any(record["interaction_type"] == "hydrophobic_contact" for record in result)


def test_parallel_and_t_shaped_pi_stacking_are_independently_switchable() -> None:
    xy = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    yz = (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    ring_a = _phenylalanine(1, np.zeros(3), xy)
    parallel_b = _phenylalanine(2, np.array([0.0, 0.0, 4.5]), xy)
    t_shaped_b = _phenylalanine(3, np.array([0.0, 0.0, 4.5]), yz)

    parallel = _contacts(_structure([ring_a], [parallel_b]), pi_stacking_parallel=True)
    t_shaped = _contacts(_structure([ring_a.copy()], [t_shaped_b]), pi_stacking_t_shaped=True)
    disabled = _contacts(_structure([ring_a.copy()], [t_shaped_b.copy()]))

    assert [record["interaction_type"] for record in parallel] == ["pi_stacking_parallel"]
    assert [record["interaction_type"] for record in t_shaped] == ["pi_stacking_t_shaped"]
    assert disabled == []


def test_cation_pi_does_not_require_a_ring_on_the_cation_side() -> None:
    xy = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    structure = _structure(
        [_residue(1, "LYS", [("NZ", (0.0, 0.0, 4.0), "N")])],
        [_phenylalanine(2, np.zeros(3), xy)],
    )

    result = _contacts(structure, cation_pi_candidate=True)

    assert len(result) == 1
    assert result[0]["residue_a_name"] == "LYS"
    assert result[0]["residue_b_name"] == "PHE"


def test_water_bridge_requires_the_same_water_molecule() -> None:
    chain_a = [_residue(1, "ASP", [("OD1", (2.8, 0.0, 0.0), "O")])]
    chain_b = [_residue(2, "ASP", [("OD1", (-2.8, 0.0, 0.0), "O")])]
    shared_water = _residue(
        1,
        "HOH",
        [("O", (0.0, 0.0, 0.0), "O"), ("H1", (1.0, 0.0, 0.0), "H"),
         ("H2", (-1.0, 0.0, 0.0), "H")],
        "W",
    )

    shared_result = _contacts(_structure(chain_a, chain_b, [shared_water]), water_bridge=True)

    assert len(shared_result) == 1
    assert shared_result[0]["mediator"] == "W:1:HOH"

    separate_waters = [
        _residue(1, "HOH", [("O", (0.0, 0.0, 0.0), "O"), ("H1", (1.0, 0.0, 0.0), "H")], "W"),
        _residue(2, "HOH", [("O", (10.0, 0.0, 0.0), "O"), ("H1", (9.0, 0.0, 0.0), "H")], "W"),
    ]
    separated_structure = _structure(
        [_residue(1, "ASP", [("OD1", (2.8, 0.0, 0.0), "O")])],
        [_residue(2, "ASP", [("OD1", (7.2, 0.0, 0.0), "O")])],
        separate_waters,
    )

    assert _contacts(separated_structure, water_bridge=True) == []


def test_residue_mode_aggregates_atom_level_vdw_observations() -> None:
    structure = _structure(
        [_residue(1, "ALA", [("CA", (0.0, 0.0, 0.0), "C"), ("CB", (0.0, 1.0, 0.0), "C")])],
        [_residue(2, "ALA", [("CA", (3.0, 0.0, 0.0), "C"), ("CB", (3.0, 1.0, 0.0), "C")])],
    )
    flags = _DISABLED | {"van_der_waals_contact": True}

    atomic = characterize_chain_contacts(structure, "A", "B", atomic=True, **flags)
    residue = characterize_chain_contacts(structure, "A", "B", atomic=False, **flags)

    assert len(atomic) > 1
    assert len(residue) == 1
    assert residue[0]["distance"] == min(record["distance"] for record in atomic)


def test_invalid_topology_backend_is_rejected() -> None:
    structure = _structure([_alanine(1, 0.0)], [_alanine(2, 3.5)])

    with pytest.raises(ValueError, match="topology_backend"):
        characterize_chain_contacts(
            structure,
            "A",
            "B",
            topology_backend="unknown",
        )


def test_missing_openmm_backend_has_actionable_installation_message() -> None:
    structure = _structure([_alanine(1, 0.0)], [_alanine(2, 3.5)])
    original_import = builtins.__import__

    def import_without_openmm(name: str, *args, **kwargs):
        if name == "openmm" or name.startswith("openmm."):
            raise ModuleNotFoundError("No module named 'openmm'", name="openmm")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_openmm):
        with pytest.raises(ModuleNotFoundError, match=r"biotools\[contacts\]"):
            characterize_chain_contacts(
                structure,
                "A",
                "B",
                topology_backend="openmm",
            )


def test_openmm_and_template_backends_agree_for_standard_residue_bonds() -> None:
    pytest.importorskip("openmm")
    structure = _structure([_alanine(1, 0.0)], [_alanine(2, 3.5)])
    flags = _DISABLED | {"hydrophobic_contact": True}

    template_records = characterize_chain_contacts(
        structure,
        "A",
        "B",
        atomic=True,
        topology_backend="templates",
        **flags,
    )
    openmm_records = characterize_chain_contacts(
        structure,
        "A",
        "B",
        atomic=True,
        topology_backend="openmm",
        **flags,
    )

    assert openmm_records == template_records


def test_openmm_backend_adds_disulfide_bond() -> None:
    pytest.importorskip("openmm")
    cysteine_a = _cysteine(1, -2.0, 0.0)
    cysteine_b = _cysteine(2, 4.0, 2.05)
    structure = _structure([cysteine_a], [cysteine_b])

    adjacency = _build_openmm_bond_adjacency(
        structure[0],
        [cysteine_a, cysteine_b],
    )

    assert cysteine_b["SG"] in adjacency[cysteine_a["SG"]]
    assert cysteine_a["SG"] in adjacency[cysteine_b["SG"]]
