"""Tests for triangulated molecular surfaces and derived analyses."""

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


def test_plot_molecular_surface_rejects_colorbar_without_field() -> None:
    with pytest.raises(ValueError, match="requires field_name"):
        plot_molecular_surface(_structure(), _surface_result(), colorbar=True)
