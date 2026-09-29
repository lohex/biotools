"""Frame-wise contact frequencies and trajectory-reader regression tests."""

from __future__ import annotations

from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure
from biotite.structure.io import dcd, xtc

from biotools.structure.contacts import analyze_contacts, prepare_contact_system
from biotools.md_simulations.analysis import (
    TrajectoryFrame,
    analyze_trajectory_contacts,
    prepare_trajectory_contacts,
)

_SERIALS = count(1)


def _topology():
    structure = Structure.Structure("pair")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, number, positions in (
        ("P", 4, (0.0, 0.3)),
        ("M", 97, (3.0, 3.3)),
    ):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        residue = Residue.Residue((" ", number, " "), "VAL", " ")
        chain.add(residue)
        for name, x in zip(("CB", "CG1"), positions):
            residue.add(Atom.Atom(
                name, np.array((x, 0.0, 0.0)), 0.0, 1.0, " ",
                f"{name:>4}", next(_SERIALS), element="C",
            ))
    return structure


def _coordinates(structure):
    return np.array([atom.coord for atom in structure.get_atoms()])


def test_one_hit_per_pair_type_and_shared_denominator():
    structure = _topology()
    initial = _coordinates(structure)
    distant = initial.copy()
    distant[2:, 0] += 12
    result = analyze_trajectory_contacts(
        structure, (initial, initial, distant), partners=("P", "M"),
        contact_types=("hydrophobic_contact", "van_der_waals_contact"),
    )
    assert result.valid_frame_count == 3
    assert (result.first_analyzed_frame, result.last_analyzed_frame) == (0, 2)
    assert {row.interaction_type for row in result.frequencies} == {
        "hydrophobic_contact", "van_der_waals_contact",
    }
    for row in result.frequencies:
        assert (row.partner_a.residue_number, row.partner_b.residue_number) == (4, 97)
        assert row.positive_frame_count == 2
        assert row.valid_frame_count == 3
        assert row.frequency == pytest.approx(2 / 3)
        assert (row.first_observed_frame, row.last_observed_frame) == (0, 1)
    assert len(result.to_records()) == 2
    np.testing.assert_array_equal(_coordinates(structure), initial)


def test_prepared_frame_analysis_and_periodic_reimaging():
    structure = _topology()
    prepared = prepare_trajectory_contacts(
        structure, ("P", "M"), contact_types=("hydrophobic_contact",)
    )
    xyz = _coordinates(structure)
    xyz[2:, 0] += 6.7
    frame = TrajectoryFrame(xyz, np.diag((10.0, 10.0, 10.0)))
    assert not prepared.analyze_frame(frame)
    assert len(prepared.analyze_frame(frame, periodic=True)) == 1
    assert len(prepared.analyze_frame(frame, periodic=True)) == 1
    np.testing.assert_array_equal(_coordinates(structure), _coordinates(_topology()))


def test_invalid_frames_fail_instead_of_changing_denominator():
    structure = _topology()
    xyz = _coordinates(structure)
    invalid = xyz.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="Frame 1.*finite"):
        analyze_trajectory_contacts(
            structure, (xyz, invalid), partners=("P", "M"),
        )
    with pytest.raises(ValueError, match="Frame 0.*box vectors"):
        analyze_trajectory_contacts(
            structure, (xyz,), partners=("P", "M"), periodic=True,
        )
    with pytest.raises(ValueError, match="Frame 0.*shape"):
        analyze_trajectory_contacts(
            structure, (xyz[:-1],), partners=("P", "M"),
        )


@pytest.mark.parametrize("reader", (dcd.DCDFile, xtc.XTCFile))
def test_stream_dcd_and_xtc(reader, tmp_path):
    structure = _topology()
    initial = _coordinates(structure)
    distant = initial.copy()
    distant[2:, 0] += 12
    trajectory = reader()
    trajectory.set_coord(np.stack((initial, distant, initial)))
    trajectory.set_box(np.tile(np.diag((30.0, 30.0, 30.0)), (3, 1, 1)))
    if reader is xtc.XTCFile:
        trajectory.set_time(np.arange(3, dtype=float))
    path = tmp_path / ("run.dcd" if reader is dcd.DCDFile else "run.xtc")
    trajectory.write(path)
    result = analyze_trajectory_contacts(
        structure, path, partners=("P", "M"),
        contact_types=("hydrophobic_contact",),
        start=0, stop=3, step=1,
    )
    assert result.valid_frame_count == 3
    assert len(result.frequencies) == 1
    assert result.frequencies[0].positive_frame_count == 2
    sliced = analyze_trajectory_contacts(
        structure, path, partners=("P", "M"),
        contact_types=("hydrophobic_contact",),
        start=1, stop=3, step=2,
    )
    assert sliced.valid_frame_count == 1
    assert (sliced.first_analyzed_frame, sliced.last_analyzed_frame) == (1, 1)
    assert sliced.frequencies == ()
    writer = PDBIO()
    writer.set_structure(structure)
    topology_path = tmp_path / "topology.pdb"
    writer.save(str(topology_path))
    production = SimpleNamespace(output_path=topology_path, trajectory_path=path)
    from_result = analyze_trajectory_contacts(
        production, partners=("P", "M"),
        contact_types=("hydrophobic_contact",),
    )
    assert from_result.frequencies[0].positive_frame_count == 2


def test_water_bridge_frequency_is_site_based_not_water_identity():
    structure = Structure.Structure("water-exchange")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, number, name, atom_specs in (
        ("P", 4, "SER", (("OG", -2.8, "O"), ("HG", -1.8, "H"))),
        ("M", 97, "ASP", (("OD1", 2.8, "O"),)),
        ("W", 1, "HOH", (("O", 0.0, "O"),)),
        ("V", 2, "HOH", (("O", 20.0, "O"),)),
    ):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        residue = Residue.Residue(
            ("W" if name == "HOH" else " ", number, " "), name, " "
        )
        chain.add(residue)
        for atom_name, x, element in atom_specs:
            residue.add(Atom.Atom(
                atom_name, np.array((x, 0.0, 0.0)), 0.0, 1.0, " ",
                f"{atom_name:>4}", next(_SERIALS), element=element,
            ))
    initial = _coordinates(structure)
    exchanged = initial.copy()
    exchanged[3, 0], exchanged[4, 0] = exchanged[4, 0], exchanged[3, 0]
    result = analyze_trajectory_contacts(
        structure, (initial, exchanged), partners=("P", "M"),
        contact_types=("water_bridge",),
    )
    assert len(result.frequencies) == 1
    assert result.frequencies[0].interaction_type == "water_bridge"
    assert result.frequencies[0].positive_frame_count == 2
    assert result.frequencies[0].frequency == 1.0


def test_topology_bond_graph_is_built_once(monkeypatch):
    from biotools.structure import _contacts

    calls = 0
    original = _contacts._build_bond_adjacency

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_contacts, "_build_bond_adjacency", counted)
    structure = _topology()
    xyz = _coordinates(structure)
    result = analyze_trajectory_contacts(
        structure, (xyz, xyz, xyz), partners=("P", "M"),
        contact_types=("hydrophobic_contact",),
    )
    assert result.valid_frame_count == 3
    assert calls == 1


def test_default_types_match_single_structure_detector():
    structure = _topology()
    xyz = _coordinates(structure)
    snapshot = analyze_contacts(prepare_contact_system(structure, ("P", "M")))
    prepared = prepare_trajectory_contacts(structure, ("P", "M"))
    assert len(prepared.contact_types) == 8
    frame_types = {hit[2] for hit in prepared.analyze_frame(xyz)}
    assert frame_types == {item.interaction_type for item in snapshot.observations}
