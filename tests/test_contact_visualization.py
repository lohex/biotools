"""Focused tests for typed, interactive structure-contact views."""

from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools.structure import (
    AtomReference,
    ContactAnalysisResult,
    ContactObservation,
    ContactStyle,
    GeometryMeasurement,
    plot_structure_contacts,
    StructureContactView,
)
from biotools.pdbtools import plot_structure_contacts as legacy_plot


_SERIALS = count(1)


def _residue(number, name, points, hetero=" "):
    residue = Residue.Residue((hetero, number, " "), name, " ")
    for atom_name, point, element in points:
        residue.add(Atom.Atom(
            atom_name, np.asarray(point, dtype=float), 0.0, 1.0, " ",
            f"{atom_name:>4}", next(_SERIALS), element=element,
        ))
    return residue


def _ring(name, number, center):
    names = ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")
    points = [
        (atom_name, (
            center[0] + 1.4 * np.cos(index * np.pi / 3),
            center[1] + 1.4 * np.sin(index * np.pi / 3),
            center[2],
        ), "C")
        for index, atom_name in enumerate(names)
    ]
    return _residue(number, name, points)


def _fixture():
    structure = Structure.Structure("contact-view")
    model = Model.Model(0)
    structure.add(model)
    chain_a = Chain.Chain("A")
    chain_b = Chain.Chain("B")
    chain_w = Chain.Chain("W")
    model.add(chain_a)
    model.add(chain_b)
    model.add(chain_w)
    ser = _residue(1, "SER", [("OG", (0, 0, 0), "O"), ("CA", (-1, 0, 0), "C")])
    asp = _residue(2, "ASP", [("OD1", (3, 0, 0), "O"), ("CA", (4, 0, 0), "C")])
    phe = _ring("PHE", 3, (0, 5, 0))
    tyr = _ring("TYR", 4, (0, 5, 4))
    water = _residue(5, "HOH", [("O", (1.5, 1, 0), "O")], "W")
    for chain, residues in ((chain_a, (ser, phe)), (chain_b, (asp, tyr)), (chain_w, (water,))):
        for residue in residues:
            chain.add(residue)

    def refs(atoms):
        return tuple(AtomReference.from_atom(atom, structure.id) for atom in atoms)

    base = dict(criteria=(), rule_profile="refined")
    hbond = ContactObservation(
        interaction_type="hydrogen_bond",
        partner_a=refs((ser["OG"],)), partner_b=refs((asp["OD1"],)),
        role_a="donor", role_b="acceptor",
        geometry=(GeometryMeasurement("donor_acceptor_distance", 3.0, "angstrom"),),
        **base,
    )
    bridge = ContactObservation(
        interaction_type="water_bridge",
        partner_a=refs((ser["OG"],)), partner_b=refs((asp["OD1"],)),
        role_a="anchor", role_b="anchor",
        geometry=(GeometryMeasurement("leg_a_distance", 1.8, "angstrom"),),
        mediator_waters=refs((water["O"],)),
        **base,
    )
    stacking = ContactObservation(
        interaction_type="pi_stacking_parallel",
        partner_a=refs(tuple(phe.get_atoms())),
        partner_b=refs(tuple(tyr.get_atoms())),
        role_a="aromatic ring", role_b="aromatic ring",
        geometry=(GeometryMeasurement("centroid_distance", 4.0, "angstrom"),),
        **base,
    )
    return structure, (hbond, bridge, stacking)


def test_stable_pair_ids_and_type_pair_filters() -> None:
    structure, observations = _fixture()
    first = plot_structure_contacts(structure, observations)
    second = plot_structure_contacts(structure, reversed(observations))
    assert isinstance(first, StructureContactView)
    assert [item.pair_id for item in first.contacts] == [
        item.pair_id for item in second.contacts
    ]
    assert len({item.pair_id for item in first.contacts}) == 3
    selected = next(item for item in first.contacts if item.interaction_type == "hydrogen_bond")
    first.show_contact_types({"hydrogen_bond", "water_bridge"})
    first.show_contact_pairs({selected.pair_id})
    assert [item.pair_id for item in first.contacts] == [selected.pair_id]
    html = first.to_html()
    assert "pi_stacking_parallel" not in html.split("createBiotoolsContactController", 1)[1].split(");", 1)[0]
    assert selected.pair_id in html
    with pytest.raises(ValueError, match="Unknown pair IDs"):
        first.show_contact_pairs({"pair-missing"})
    assert legacy_plot is plot_structure_contacts

    filtered = plot_structure_contacts(
        structure, observations, contact_types={"water_bridge"}
    )
    assert len(filtered.contacts) == 1
    assert filtered.contacts[0].interaction_type == "water_bridge"
    assert "pi_stacking_parallel" not in filtered._payload()["styles"]
    typed_result = ContactAnalysisResult(
        observations=observations, diagnostics=(), coverage={},
        config_hash="test", rule_profile="refined",
        backend_versions={}, input_hash="test",
    )
    assert len(plot_structure_contacts(structure, typed_result).contacts) == 3


def test_master_type_pair_state_and_highlight_are_synchronized() -> None:
    structure, observations = _fixture()
    view = plot_structure_contacts(
        structure, observations, enabled_contact_types={"hydrogen_bond"},
        residue_labels={"A": "active", "B": "all"},
    )
    hbond = next(item for item in view.contacts if item.interaction_type == "hydrogen_bond")
    bridge = next(item for item in view.contacts if item.interaction_type == "water_bridge")
    assert view.checkbox_states()["master"] == "indeterminate"
    assert view.checkbox_states()["types"]["hydrogen_bond"] == "on"
    view.set_all_contacts_enabled(False)
    assert view.checkbox_states()["master"] == "off"
    assert not view.active_residues
    view.set_contact_pair_enabled(bridge.pair_id, True)
    assert view.checkbox_states()["master"] == "indeterminate"
    assert len(view.active_residues) == 2
    view.highlight_contact_pair(hbond.pair_id)
    assert hbond.pair_id in view.enabled_pair_ids
    assert view.highlighted_pair_id == hbond.pair_id
    view.set_contact_type_enabled("hydrogen_bond", False)
    assert view.highlighted_pair_id is None
    assert view.checkbox_states()["types"]["hydrogen_bond"] == "off"
    view.set_all_contacts_enabled(True)
    assert view.checkbox_states()["master"] == "on"
    view.clear_highlight()
    assert view.highlighted_pair_id is None



def test_multiple_highlights_keep_independent_visibility_and_filtering() -> None:
    structure, observations = _fixture()
    view = plot_structure_contacts(structure, observations)
    first, second, third = view.contacts
    view.highlight_contact_pair(first.pair_id)
    view.highlight_contact_pair(second.pair_id)
    assert view.highlighted_pair_ids == {first.pair_id, second.pair_id}
    assert set(view._payload()["highlighted"]) == view.highlighted_pair_ids
    assert view.highlighted_pair_id == second.pair_id
    view.set_contact_pair_highlighted(first.pair_id, False)
    assert first.pair_id in view.enabled_pair_ids
    assert view.highlighted_pair_ids == {second.pair_id}
    view.highlight_contact_pair(third.pair_id)
    view.show_contact_pairs({first.pair_id, third.pair_id})
    assert view.highlighted_pair_ids == {third.pair_id}
    view.set_contact_pair_enabled(third.pair_id, False)
    assert not view.highlighted_pair_ids
    view.highlight_contact_pair(first.pair_id)
    view.clear_highlight()
    assert not view.highlighted_pair_ids
    assert first.pair_id in view.enabled_pair_ids
    strict = plot_structure_contacts(
        structure, observations, enabled_contact_types=(), enable_on_select=False,
    )
    with pytest.raises(ValueError, match="disabled"):
        strict.highlight_contact_pair(strict.contacts[0].pair_id)


def test_contact_rows_identify_residues_and_controls_are_ordered() -> None:
    structure, observations = _fixture()
    view = plot_structure_contacts(structure, observations)
    contact = next(c for c in view._payload()["contacts"] if c["type"] == "hydrogen_bond")
    assert contact["pairLabel"] == "A SER 1 – B ASP 2 · 3.00 Å"
    html = view.to_html()
    assert html.index('data-views>') < html.index('data-chains>') < html.index('data-types>')
    assert view._payload()["activeSticks"] is True
    assert len(view._payload()["stickResidues"]) == 4


def test_residue_labels_ring_planes_and_water_geometry() -> None:
    structure, observations = _fixture()
    view = plot_structure_contacts(
        structure, observations, residue_labels={"A": "active", "B": "all"},
        chain_labels={"A": "Receptor", "B": "Ligand"},
    )
    stacking = next(item for item in view.contacts if item.interaction_type == "pi_stacking_parallel")
    water = next(item for item in view.contacts if item.interaction_type == "water_bridge")
    assert len(stacking.ring_planes) == 2
    assert all(plane.radius > 1 for plane in stacking.ring_planes)
    assert water.water == (1.5, 1.0, 0.0)
    assert water.anchor == water.water
    assert view._payload()["chainLabels"]["B"] == "Ligand"
    assert "W" not in view._payload()["chainLabels"]
    view.set_residue_label_mode("A", "off")
    assert view._payload()["labelModes"]["A"] == "off"
    view.set_residue_labels_enabled("A", True)
    assert view._payload()["labelModes"]["A"] == "all"
    view.set_residue_label_mode("A", "active")
    assert view._payload()["labelModes"]["A"] == "active"
    assert "fromCap:'flat',toCap:'flat'" in view.to_html()


def test_zero_contacts_escaping_transform_and_html_file(tmp_path: Path) -> None:
    structure, observations = _fixture()
    atom = next(structure.get_atoms())
    original = atom.coord.copy()
    matrix = np.eye(4)
    matrix[:3, 3] = (1, 2, 3)
    view = plot_structure_contacts(
        structure, (), coordinate_transform=matrix,
        residue_labels={"A": "all"}, initial_view="xz",
    )
    assert len(view.contacts) == 0
    assert view.checkbox_states()["master"] == "off"
    assert np.allclose(atom.coord, original)
    assert view._payload()["viewName"] == "xz"
    destination = view.write_html(tmp_path / "contacts.html")
    assert destination.read_text().startswith("<!doctype html>")
    assert "No contacts" in destination.read_text()
    injected = plot_structure_contacts(
        structure, observations[:1],
        chain_labels={"A": "<img src=x onerror=alert(1)>"},
        contact_styles={"hydrogen_bond": {"label": "<img src=x onerror=alert(1)>"}},
    )
    html = injected.to_html()
    assert "<img src=x onerror" not in html
    assert "\\u003cimg" in html


def test_dense_contacts_use_type_grouping_in_browser_controller() -> None:
    structure, observations = _fixture()
    hbond = observations[0]
    dense = tuple(
        ContactObservation(
            interaction_type="van_der_waals_contact",
            partner_a=hbond.partner_a, partner_b=hbond.partner_b,
            role_a="atom", role_b="atom",
            geometry=(GeometryMeasurement("atom_distance", 3.0 + index * 0.001, "angstrom"),),
            criteria=(), rule_profile="refined",
        )
        for index in range(200)
    )
    view = plot_structure_contacts(structure, dense)
    assert len(view.contacts) == 200
    html = view.to_html()
    assert "const shape = viewer.addShape" in html
    assert "shape.addDashedCylinder" in html
    assert html.count('"type":"van_der_waals_contact"') == 200
    with pytest.raises(ValueError, match="positive"):
        ContactStyle("bad", "#000", -1)


def test_browser_controls_labels_rotation_and_filled_ring_planes() -> None:
    """Optional Chromium check of the actual standalone HTML interaction flow."""
    playwright = pytest.importorskip("playwright.sync_api")
    from io import BytesIO
    from PIL import Image

    structure, observations = _fixture()
    view = plot_structure_contacts(
        structure, observations,
        residue_labels={"A": "active", "B": "active"},
        width=1100, height=720,
    )
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1120, "height": 740})
        page.set_content(view.to_html(), wait_until="load")
        page.wait_for_function(
            "Object.values(window.__biotoolsContactViews||{}).length === 1",
            timeout=30000,
        )
        state = "() => Object.values(window.__biotoolsContactViews)[0].state()"
        page.locator("[data-master]").uncheck()
        disabled = page.evaluate(state)
        assert disabled["enabled"] == []
        assert disabled["shapeCount"] == 0
        assert disabled["residueLabelCount"] == 0

        hbond = page.locator("[data-contact-type=hydrogen_bond]")
        hbond.locator("[data-expand]").click()
        hbond.locator("[data-pair-check]").check()
        one = page.evaluate(state)
        assert len(one["enabled"]) == 1
        assert len(one["activeResidues"]) == 2
        assert one["master"]["indeterminate"]
        assert page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          return v.selectedAtoms({chain:'A',resi:1}).every(a => !!a.style.stick) &&
            v.selectedAtoms({chain:'B',resi:2}).every(a => !!a.style.stick) &&
            v.selectedAtoms({chain:'A',resi:3}).every(a => !a.style.stick);
        }""")
        assert hbond.locator(".bc-pair-label").inner_text() == "A SER 1 – B ASP 2 · 3.00 Å"
        page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          window.__shapeBefore=v.shapes[0];
        }""")
        hbond.locator("[data-pair-select]").click()
        selected = page.evaluate(state)
        assert selected["highlightLabelCount"] == 1
        assert selected["shapeCount"] == one["shapeCount"]
        assert page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          return v.shapes[0] === window.__shapeBefore;
        }""")
        before = page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          return v.getView();
        }""")
        page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          v.rotate(35);v.render();
        }""")
        after = page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          return v.getView();
        }""")
        assert before != after
        assert page.evaluate(state)["highlightLabelCount"] == 1

        before_plane = np.asarray(Image.open(BytesIO(page.screenshot())).convert("RGB"))
        page.locator("[data-type-check=pi_stacking_parallel]").check()
        page.wait_for_timeout(200)
        after_plane = np.asarray(Image.open(BytesIO(page.screenshot())).convert("RGB"))
        changed = np.max(
            np.abs(after_plane[:600, :700].astype(int) - before_plane[:600, :700].astype(int)),
            axis=2,
        )
        assert np.count_nonzero(changed > 20) > 1000
        assert page.evaluate(state)["shapeCount"] == 3
        stacking = page.locator("[data-contact-type=pi_stacking_parallel]")
        stacking.locator("[data-expand]").click()
        stacking.locator("[data-pair-select]").click()
        assert page.evaluate(state)["highlightLabelCount"] == 2
        hbond.locator("[data-pair-select]").click()
        assert page.evaluate(state)["highlightLabelCount"] == 1
        assert hbond.locator("[data-pair-check]").is_checked()
        stacking.locator("[data-pair-check]").uncheck()
        assert page.evaluate(state)["highlightLabelCount"] == 0
        assert page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          return v.selectedAtoms({chain:'A',resi:3}).every(a => !a.style.stick);
        }""")
        page.locator("[data-master]").check()
        page.locator("[data-master]").uncheck()
        assert page.evaluate("""() => {
          const v=window[Object.keys(window).find(k=>/^viewer_[0-9]+$/.test(k))];
          return v.selectedAtoms({}).every(a => !a.style.stick);
        }""")

        empty_page = browser.new_page()
        empty_page.set_content(plot_structure_contacts(structure, ()).to_html())
        empty_page.wait_for_function(
            "Object.values(window.__biotoolsContactViews||{}).length === 1"
        )
        assert empty_page.get_by_text("No contacts").is_visible()
        assert empty_page.evaluate(state)["shapeCount"] == 0

        anchor = observations[0]
        dense = tuple(
            ContactObservation(
                interaction_type="van_der_waals_contact",
                partner_a=anchor.partner_a, partner_b=anchor.partner_b,
                role_a="atom", role_b="atom",
                geometry=(GeometryMeasurement("atom_distance", 3 + i * 0.001, "angstrom"),),
                criteria=(), rule_profile="refined",
            )
            for i in range(100)
        )
        dense_page = browser.new_page()
        dense_page.set_content(plot_structure_contacts(structure, dense).to_html())
        dense_page.wait_for_function(
            "Object.values(window.__biotoolsContactViews||{}).length === 1"
        )
        assert len(dense_page.evaluate(state)["enabled"]) == 100
        assert dense_page.evaluate(state)["shapeCount"] == 1
        browser.close()
