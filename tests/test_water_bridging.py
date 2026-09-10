"""Deterministic tests for local single-water bridge modeling."""

from __future__ import annotations

from itertools import count

import numpy as np
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools.structure import (
    evaluate_water_bridges,
    optimize_bridging_waters,
    orient_existing_waters,
    prepare_contact_system,
    propose_bridging_waters,
)


_SERIALS = count(1)


def _residue(number, name, atoms, hetero=" "):
    residue = Residue.Residue((hetero, number, " "), name, " ")
    for atom_name, coordinate, element in atoms:
        residue.add(
            Atom.Atom(
                atom_name,
                np.asarray(coordinate, dtype=float),
                10.0,
                0.75,
                " ",
                f"{atom_name:>4}",
                next(_SERIALS),
                element=element,
            )
        )
    return residue


def _bridge_structure(include_water=False):
    structure = Structure.Structure("water-bridge")
    model = Model.Model(0)
    structure.add(model)
    residues = (
        (
            "A",
            [_residue(1, "SER", [("OG", (-2.8, 0, 0), "O"), ("HG", (-1.8, 0, 0), "H")])],
        ),
        ("B", [_residue(2, "ASP", [("OD1", (2.8, 0, 0), "O")])]),
    )
    for chain_id, chain_residues in residues:
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue in chain_residues:
            chain.add(residue)
    if include_water:
        water_chain = Chain.Chain("W")
        model.add(water_chain)
        water_chain.add(_residue(9, "HOH", [("O", (0, 0, 0), "O")], "W"))
    return structure


def test_orient_existing_oxygen_preserves_site_and_both_bridge_legs() -> None:
    prepared = prepare_contact_system(_bridge_structure(True), ("A", "B"))

    result = orient_existing_waters(prepared)

    assert len(result.sites) == 1
    site = result.sites[0]
    assert site.oxygen_position == (0.0, 0.0, 0.0)
    assert site.crystal_occupancy == 0.75
    assert site.provenance == "modeled_orientation_at_observed_oxygen"
    assert any(orientation.accepted for orientation in site.orientations)
    assert {leg.partner for leg in site.orientations[0].legs} == {"a", "b"}


def test_proposal_optimization_and_independent_evaluation_are_reproducible() -> None:
    prepared = prepare_contact_system(_bridge_structure(), ("A", "B"))

    first = propose_bridging_waters(prepared)
    second = propose_bridging_waters(prepared)
    optimized = optimize_bridging_waters(prepared, first)
    evaluated = evaluate_water_bridges(prepared, optimized)

    assert first == second
    assert len(first.sites) == 1
    assert first.sites[0].oxygen_position == (0.0, 0.0, 0.0)
    assert optimized.sites[0].displacement == 0.0
    assert evaluated.sites[0].status == "accepted"
    assert all(len(orientation.legs) == 2 for orientation in evaluated.sites[0].orientations)


def test_dry_interface_without_compatible_anchors_has_explicit_diagnostic() -> None:
    structure = Structure.Structure("empty")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, position in (("A", 0.0), ("B", 20.0)):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        chain.add(_residue(1, "ALA", [("CB", (position, 0, 0), "C")]))
    prepared = prepare_contact_system(structure, ("A", "B"))

    result = propose_bridging_waters(prepared)

    assert result.sites == ()
    assert result.diagnostics[0].code == "no_water_proposals"
