"""Tests for triangulated molecular surfaces and derived analyses."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools.structure import (
    AtomReference,
    calculate_molecular_surface,
    color_from_labels,
    find_surface_patches,
    map_electrostatic_potential,
    map_electrostatic_potential_apbs,
    map_electrostatic_potential_openmm,
    map_hydrophobicity,
    MolecularSurfaceMesh,
    MolecularSurfaceParameters,
    MolecularSurfaceResult,
    plot_molecular_surface,
    sample_surface,
    SurfaceAtom,
    SurfaceField,
)


def _structure() -> Structure.Structure:
    structure = Structure.Structure("mesh")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    model.add(chain)
    structure.add(model)
    for number, name, atom_name, element, coordinate in (
        (1, "ALA", "CA", "C", (0.0, 0.0, 0.0)),
        (2, "VAL", "CB", "C", (3.0, 0.0, 0.0)),
        (3, "HOH", "O", "O", (6.0, 0.0, 0.0)),
    ):
        hetero = "W" if name == "HOH" else " "
        residue = Residue.Residue((hetero, number, " "), name, " ")
        residue.add(
            Atom.Atom(
                atom_name,
                np.asarray(coordinate, dtype=float),
                0.0,
                1.0,
                " ",
                f"{atom_name:>4}",
                number,
                element=element,
            )
        )
        chain.add(residue)
    return structure


def test_calculate_molecular_surface_calls_msms_and_parses_mesh() -> None:
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        input_path = Path(command[command.index("-if") + 1])
        output_path = Path(command[command.index("-of") + 1])
        observed["xyzr"] = input_path.read_text()
        output_path.with_suffix(".vert").write_text(
            "0 0 1 0 0 1 1 1 3\n"
            "1 0 1 0 0 1 1 1 3\n"
            "0 1 1 0 0 1 1 2 3\n"
        )
        output_path.with_suffix(".face").write_text("1 2 3 3 1\n")
        return subprocess.CompletedProcess(
            command, 0, stdout="MSMS 2.6.1 completed", stderr=""
        )

    with patch(
        "biotools.structure.molecular_surface.subprocess.run",
        side_effect=run,
    ):
        result = calculate_molecular_surface(
            _structure(),
            density=2.0,
            executable="custom-msms",
        )

    assert observed["command"][0] == "custom-msms"
    assert observed["command"][-2:] == ["2.0", "-no_header"]
    assert observed["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }
    assert len(observed["xyzr"].splitlines()) == 2  # water omitted
    assert result.backend_version == "2.6.1"
    assert len(result.atoms) == 2
    np.testing.assert_array_equal(result.surface.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(result.surface.vertex_atom_indices, [0, 0, 1])
    np.testing.assert_allclose(result.surface.normals, [[0.0, 0.0, 1.0]] * 3)
    assert not result.surface.vertices.flags.writeable


def test_calculate_molecular_surface_reports_missing_msms() -> None:
    with patch(
        "biotools.structure.molecular_surface.subprocess.run",
        side_effect=FileNotFoundError("msms"),
    ), pytest.raises(FileNotFoundError, match="MSMS is not installed"):
        calculate_molecular_surface(_structure())


@pytest.mark.skipif(
    shutil.which("msms") is None,
    reason="MSMS executable is not available",
)
def test_calculate_molecular_surface_with_installed_msms() -> None:
    result = calculate_molecular_surface(_structure())

    assert result.backend_version == "2.6.1"
    assert result.surface.vertices.shape[1] == 3
    assert result.surface.normals.shape == result.surface.vertices.shape
    assert result.surface.faces.shape[1] == 3
    assert np.min(result.surface.faces) >= 0
    assert np.max(result.surface.faces) < len(result.surface.vertices)


def _reference(number: int, residue_name: str, atom_name: str) -> AtomReference:
    return AtomReference(
        structure_id="synthetic",
        model_id="0",
        chain_id="A",
        hetero_flag="",
        residue_number=number,
        insertion_code="",
        residue_name=residue_name,
        atom_name=atom_name,
        altloc="",
    )


def _surface_result() -> MolecularSurfaceResult:
    atoms = (
        SurfaceAtom(_reference(1, "ALA", "CA"), [0.0, 0.0, -1.0], 1.7, "C"),
        SurfaceAtom(_reference(2, "VAL", "CB"), [4.0, 0.0, -1.0], 1.7, "C"),
    )
    mesh = MolecularSurfaceMesh(
        component_id=0,
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
            ]
        ),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 6),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
        vertex_atom_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        vertex_patch_types=np.asarray([3] * 6),
        face_patch_types=np.asarray([3, 3]),
    )
    return MolecularSurfaceResult(
        components=(mesh,),
        atoms=atoms,
        parameters=MolecularSurfaceParameters(
            probe_radius=1.5,
            density=1.0,
            radii_model="test",
            include_hydrogens=False,
            include_heteroatoms=True,
            include_water=False,
            all_components=False,
            model_id="0",
        ),
    )


def test_map_hydrophobicity_maps_residue_scale_through_atom_indices() -> None:
    result = map_hydrophobicity(_surface_result())

    field = result.surface.get_field("hydrophobicity")
    np.testing.assert_allclose(field.values, [1.8, 1.8, 1.8, 4.2, 4.2, 4.2])
    assert field.source == "residue_scale:kyte_doolittle"
    assert _surface_result().surface.fields == ()


def test_map_electrostatic_potential_uses_all_atom_charges() -> None:
    result = map_electrostatic_potential(
        _surface_result(),
        [1.0, -1.0],
        dielectric=2.0,
    )

    values = result.surface.get_field("electrostatic_potential").values
    expected_first = 1389.35457644382 * (1.0 / 1.0 - 1.0 / np.sqrt(17.0)) / 2.0
    assert values[0] == pytest.approx(expected_first)
    assert result.surface.get_field("electrostatic_potential").units == (
        "kJ mol^-1 e^-1"
    )


def test_map_electrostatic_potential_accepts_extra_charge_sites() -> None:
    result = map_electrostatic_potential(
        _surface_result(),
        [1.0, -1.0, 0.5],
        charge_positions=[
            [0.0, 0.0, -1.0],
            [4.0, 0.0, -1.0],
            [0.0, 0.0, -2.0],
        ],
    )
    expected = 1389.35457644382 * (
        1.0 - 1.0 / np.sqrt(17.0) + 0.5 / 2.0
    )
    assert result.surface.get_field("electrostatic_potential").values[0] == (
        pytest.approx(expected)
    )


def test_map_electrostatic_potential_rejects_misaligned_charge_sites() -> None:
    with pytest.raises(ValueError, match="equal length"):
        map_electrostatic_potential(
            _surface_result(),
            [1.0],
            charge_positions=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        )


def _complete_dipeptide() -> Structure.Structure:
    """Two complete heavy-atom residues for an OpenMM charge integration test."""
    structure = Structure.Structure("dipeptide")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    structure.add(model)
    model.add(chain)
    records = (
        (
            "ALA",
            (
                ("N", "N", (0.0, 0.0, 0.0)),
                ("CA", "C", (1.45, 0.0, 0.0)),
                ("C", "C", (2.0, 1.42, 0.0)),
                ("O", "O", (1.3, 2.44, 0.0)),
                ("CB", "C", (1.9, -0.8, 1.2)),
            ),
        ),
        (
            "GLY",
            (
                ("N", "N", (3.33, 1.5, 0.0)),
                ("CA", "C", (4.0, 2.8, 0.0)),
                ("C", "C", (5.5, 2.7, 0.0)),
                ("O", "O", (6.0, 1.6, 0.0)),
                ("OXT", "O", (6.2, 3.8, 0.0)),
            ),
        ),
    )
    serial = 0
    for residue_number, (residue_name, atoms) in enumerate(records, 1):
        residue = Residue.Residue((" ", residue_number, " "), residue_name, " ")
        chain.add(residue)
        for name, symbol, xyz in atoms:
            serial += 1
            residue.add(
                Atom.Atom(
                    name,
                    np.asarray(xyz, dtype=float),
                    0.0,
                    1.0,
                    " ",
                    f"{name:>4}",
                    serial,
                    element=symbol,
                )
            )
    return structure


def test_openmm_potential_adds_hydrogen_charges_without_changing_surface() -> None:
    pytest.importorskip("openmm")
    structure = _complete_dipeptide()
    surface = calculate_molecular_surface(structure)
    original_coordinates = np.asarray(
        [atom.coord.copy() for atom in structure.get_atoms()]
    )
    from biotools.structure import electrostatic_potential as adapter

    with patch.object(
        adapter,
        "map_electrostatic_potential",
        wraps=map_electrostatic_potential,
    ) as forwarded:
        result = map_electrostatic_potential_openmm(
            structure,
            surface,
            dielectric=4.0,
        )

    forwarded.assert_called_once()
    assert len(forwarded.call_args.args[1]) > len(surface.atoms)
    assert len(forwarded.call_args.kwargs["charge_positions"]) == len(
        forwarded.call_args.args[1]
    )
    assert len(result.surface.get_field("electrostatic_potential").values) == len(
        surface.surface.vertices
    )
    assert "amber14-all.xml" in result.surface.get_field(
        "electrostatic_potential"
    ).source
    np.testing.assert_allclose(
        [atom.coord for atom in structure.get_atoms()], original_coordinates
    )
    assert surface.surface.fields == ()


def test_openmm_potential_rejects_structure_surface_mismatch() -> None:
    pytest.importorskip("openmm")
    structure = _complete_dipeptide()
    surface = calculate_molecular_surface(structure)
    structure[0]["A"][1]["CA"].coord += np.asarray([0.1, 0.0, 0.0])

    with pytest.raises(ValueError, match="coordinates differ"):
        map_electrostatic_potential_openmm(structure, surface)


def test_openmm_potential_reports_incomplete_residue() -> None:
    pytest.importorskip("openmm")
    structure = _complete_dipeptide()
    structure[0]["A"][2].detach_child("OXT")
    surface = calculate_molecular_surface(structure)

    with pytest.raises(ValueError, match="could not parameterize"):
        map_electrostatic_potential_openmm(structure, surface)


def test_apbs_potential_runs_pdb2pqr_and_interpolates_grid() -> None:
    observed = {}
    structure = _structure()
    structure.id = "synthetic"
    structure[0]["A"][1]["CA"].coord = np.asarray([0.0, 0.0, -1.0])
    structure[0]["A"][2]["CB"].coord = np.asarray([4.0, 0.0, -1.0])

    def run(command, **kwargs):
        if command[0] == "test-pdb2pqr":
            observed["pdb2pqr"] = command
            Path(command[-1]).write_text(
                "ATOM 1 CA ALA A 1 0.000 0.000 -1.000 0.000 1.700\n"
                "ATOM 2 CB VAL A 2 4.000 0.000 -1.000 0.000 1.700\n"
            )
        else:
            observed["apbs"] = command
            directory = Path(kwargs["cwd"])
            observed["input"] = (directory / command[1]).read_text()
            coordinates = np.arange(-10.0, 11.0, 10.0)
            values = [
                x + y + z for x in coordinates for y in coordinates for z in coordinates
            ]
            (directory / "potential-PE0.dx").write_text(
                "object 1 class gridpositions counts 3 3 3\n"
                "origin -10 -10 -10\n"
                "delta 10 0 0\n"
                "delta 0 10 0\n"
                "delta 0 0 10\n"
                "object 2 class gridconnections counts 3 3 3\n"
                "object 3 class array type double rank 0 items 27 data follows\n"
                + " ".join(map(str, values))
                + "\nattribute \"dep\" string \"positions\"\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    with patch(
        "biotools.structure.poisson_boltzmann.subprocess.run", side_effect=run
    ):
        result = map_electrostatic_potential_apbs(
            structure,
            _surface_result(),
            pdb2pqr_executable="test-pdb2pqr",
            apbs_executable="test-apbs",
            ionic_strength=0.15,
            grid_spacing=2.0,
        )

    assert observed["pdb2pqr"][0] == "test-pdb2pqr"
    assert "--titration-state-method" in observed["pdb2pqr"]
    assert observed["apbs"] == ["test-apbs", "apbs.in"]
    assert "lpbe" in observed["input"]
    assert "ion charge 1 conc 0.15 radius 2" in observed["input"]
    potential = result.surface.get_field("electrostatic_potential")
    conversion = 0.00831446261815324 * 298.15
    np.testing.assert_allclose(
        potential.values,
        np.sum(result.surface.vertices, axis=1) * conversion,
    )
    assert potential.units == "kJ mol^-1 e^-1"
    assert "ionic_strength=0.15 M" in potential.source


@pytest.mark.skipif(
    shutil.which("apbs") is None
    or shutil.which("msms") is None
    or importlib.util.find_spec("pdb2pqr") is None,
    reason="APBS, PDB2PQR, or MSMS is not available",
)
def test_apbs_potential_with_installed_backends() -> None:
    structure = _complete_dipeptide()
    surface = calculate_molecular_surface(structure)

    result = map_electrostatic_potential_apbs(
        structure,
        surface,
        grid_spacing=1.5,
        grid_padding=5.0,
        max_grid_points=65,
    )

    field = result.surface.get_field("electrostatic_potential")
    assert len(field.values) == len(surface.surface.vertices)
    assert np.all(np.isfinite(field.values))
    assert np.ptp(field.values) > 0.0


def test_find_surface_patches_labels_connected_thresholded_faces() -> None:
    result = _surface_result()
    field = SurfaceField(
        name="score",
        location="vertex",
        values=np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    )
    mesh = MolecularSurfaceMesh(
        component_id=result.surface.component_id,
        vertices=result.surface.vertices,
        normals=result.surface.normals,
        faces=result.surface.faces,
        vertex_atom_indices=result.surface.vertex_atom_indices,
        vertex_patch_types=result.surface.vertex_patch_types,
        face_patch_types=result.surface.face_patch_types,
        fields=(field,),
    )

    patches = find_surface_patches(
        mesh,
        field_name="score",
        threshold=0.5,
    )

    np.testing.assert_array_equal(patches.face_labels, [0, -1])
    assert len(patches.patches) == 1
    assert patches.patches[0].area == pytest.approx(0.5)
    np.testing.assert_allclose(patches.patches[0].centroid, [1 / 3, 1 / 3, 0])
    np.testing.assert_allclose(patches.patches[0].mean_normal, [0, 0, 1])


def test_color_from_labels_preserves_background_and_label_colors() -> None:
    colors = color_from_labels(np.asarray([-1, 0, 1, 0]))

    np.testing.assert_allclose(colors[0], [0.75, 0.75, 0.75, 0.0])
    np.testing.assert_allclose(colors[1], colors[3])
    assert not np.allclose(colors[1], colors[2])
    assert not colors.flags.writeable


def test_sample_surface_is_area_uniform_and_interpolates_fields() -> None:
    result = _surface_result()
    field = SurfaceField(
        name="score",
        location="vertex",
        values=np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
    )
    mesh = MolecularSurfaceMesh(
        component_id=result.surface.component_id,
        vertices=result.surface.vertices,
        normals=result.surface.normals,
        faces=result.surface.faces,
        vertex_atom_indices=result.surface.vertex_atom_indices,
        vertex_patch_types=result.surface.vertex_patch_types,
        face_patch_types=result.surface.face_patch_types,
        fields=(field,),
    )

    samples = sample_surface(
        mesh,
        100,
        face_indices=[0],
        field_names=("score",),
        seed=7,
    )

    assert samples.positions.shape == (100, 3)
    np.testing.assert_allclose(samples.positions[:, 2], 0.0)
    np.testing.assert_allclose(samples.normals, [[0.0, 0.0, 1.0]] * 100)
    np.testing.assert_allclose(np.sum(samples.barycentric, axis=1), 1.0)
    np.testing.assert_array_equal(samples.face_indices, np.zeros(100, dtype=int))
    expected = samples.barycentric @ np.asarray([0.0, 1.0, 2.0])
    np.testing.assert_allclose(samples.get_field("score").values, expected)
    assert not samples.positions.flags.writeable


def test_plot_molecular_surface_combines_structure_mesh_and_normals() -> None:
    surface = map_hydrophobicity(_surface_result())
    patches = find_surface_patches(
        surface,
        field_name="hydrophobicity",
        threshold=1.0,
    )
    samples = sample_surface(surface, 3, seed=4)

    view = plot_molecular_surface(
        _structure(),
        surface,
        patches=patches,
        samples=samples,
        surface_opacity=0.5,
    )

    assert ".addModel(" in view.startjs
    assert '"cartoon"' in view.startjs
    assert '"stick"' in view.startjs
    assert ".addCustom(" in view.startjs
    assert '"opacity": 0.5' in view.startjs
    assert view.startjs.count(".addArrow(") == 3


def test_plot_molecular_surface_embeds_field_colorbar() -> None:
    surface = map_hydrophobicity(_surface_result())

    view = plot_molecular_surface(
        _structure(),
        surface,
        field_name="hydrophobicity",
        cmap="coolwarm",
        field_range=(-4.5, 4.5),
        field_center=0.0,
        colorbar=True,
    )

    html = view._make_html()
    assert "hydrophobicity" in html
    assert "linear-gradient(to right" in html
    assert "-4.5" in html and "4.5" in html
    assert html.index("linear-gradient(to right") < html.index("</div>\n<script>")


def test_plot_molecular_surface_colors_only_selected_patch() -> None:
    surface = map_hydrophobicity(_surface_result())
    patches = find_surface_patches(
        surface, field_name="hydrophobicity", threshold=1.0
    )
    selected = max(patches.patches, key=lambda patch: patch.area)

    view = plot_molecular_surface(
        _structure(), surface, patch=selected,
        field_name="hydrophobicity", colorbar=True,
    )

    custom = view.startjs.split(".addCustom(", 1)[1].split(");", 1)[0]
    spec = json.loads(custom)
    assert len(spec["vertexArr"]) == 3
    assert spec["faceArr"] == [0, 1, 2]
    assert len(spec["color"]) == 3
    assert "linear-gradient(to right" in view.startjs



def test_plot_molecular_surface_wireframe_covers_only_patch_faces() -> None:
    mesh = replace(
        _surface_result().surface,
        fields=(SurfaceField(
            name="score", location="face", values=np.asarray([1.0, -1.0])
        ),),
    )
    patches = find_surface_patches(
        mesh, field_name="score", threshold=0.5
    )

    view = plot_molecular_surface(
        _structure(), mesh, patches=patches,
        patch_wireframe=True, patch_wireframe_color="#222222",
    )

    specs = [
        json.loads(call.split(");", 1)[0])
        for call in view.startjs.split(".addCustom(")[1:]
    ]
    assert len(specs) == 2
    assert specs[0]["wireframe"] is False
    assert len(specs[0]["faceArr"]) == 6
    assert specs[1]["wireframe"] is True
    assert specs[1]["faceArr"] == [0, 1, 2]
    assert specs[1]["color"] == "#222222"
    assert specs[1]["vertexArr"][0]["z"] == pytest.approx(0.02)


def test_plot_molecular_surface_wireframe_requires_patch_selection() -> None:
    with pytest.raises(ValueError, match="requires patch or patches"):
        plot_molecular_surface(
            _structure(), _surface_result(), patch_wireframe=True
        )

def test_plot_molecular_surface_rejects_colorbar_without_field() -> None:
    with pytest.raises(ValueError, match="requires field_name"):
        plot_molecular_surface(_structure(), _surface_result(), colorbar=True)
