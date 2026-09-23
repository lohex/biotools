"""Solvent-screened electrostatic surface potentials from PDB2PQR and APBS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .contacts.models import AtomReference
from .molecular_surface import (
    MolecularSurfaceMesh,
    MolecularSurfaceResult,
    SurfaceField,
)

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure


_GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324


def _command_error(name: str, completed: subprocess.CompletedProcess[str]) -> RuntimeError:
    details = (completed.stderr or completed.stdout or "").strip()
    if len(details) > 1500:
        details = details[-1500:]
    message = f"{name} failed with exit status {completed.returncode}"
    if details:
        message += f":\n{details}"
    return RuntimeError(message)


def _surface_residue_keys(
    surface_obj: MolecularSurfaceResult,
) -> set[tuple[str, str, int, str, str]]:
    return {
        (
            atom.reference.chain_id,
            atom.reference.hetero_flag,
            atom.reference.residue_number,
            atom.reference.insertion_code,
            atom.reference.residue_name,
        )
        for atom in surface_obj.atoms
    }


def _write_surface_pdb(
    structure: Structure,
    surface_obj: MolecularSurfaceResult,
    path: Path,
) -> None:
    """Write and validate the structure subset from which the mesh was made."""
    from Bio.PDB import PDBIO, Select

    matching_models = [
        model for model in structure if str(model.id) == surface_obj.parameters.model_id
    ]
    if len(matching_models) != 1:
        raise ValueError(
            "The structure must contain exactly one model with ID "
            f"{surface_obj.parameters.model_id!r}"
        )
    model = matching_models[0]
    surface_atoms = {atom.reference: atom for atom in surface_obj.atoms}
    if len(surface_atoms) != len(surface_obj.atoms):
        raise ValueError("The surface contains duplicate atom references")
    residue_keys = _surface_residue_keys(surface_obj)
    matched: set[AtomReference] = set()

    for chain in model:
        for residue in chain:
            hetero_flag, residue_number, insertion_code = residue.id
            residue_key = (
                str(chain.id),
                str(hetero_flag).strip(),
                int(residue_number),
                str(insertion_code).strip(),
                str(residue.get_resname()).strip(),
            )
            if residue_key not in residue_keys:
                continue
            for source_atom in residue.get_atoms():
                reference = AtomReference.from_atom(source_atom, str(structure.id))
                if reference not in surface_atoms:
                    continue
                coordinates = np.asarray(source_atom.coord, dtype=float)
                surface_atom = surface_atoms[reference]
                element = str(getattr(source_atom, "element", "") or "").strip().upper()
                if element != surface_atom.element:
                    raise ValueError(
                        f"Surface and structure elements differ for {reference!r}"
                    )
                if not np.allclose(
                    coordinates, surface_atom.coordinates, rtol=0.0, atol=1e-3
                ):
                    raise ValueError(
                        f"Surface and structure coordinates differ for {reference!r}"
                    )
                matched.add(reference)

    if matched != set(surface_atoms):
        missing = next(iter(set(surface_atoms) - matched))
        raise ValueError(
            f"Surface atom {missing!r} was not found in the supplied structure"
        )

    model_id = surface_obj.parameters.model_id

    class SurfaceSelect(Select):
        def accept_model(self, selected_model):  # type: ignore[no-untyped-def]
            return str(selected_model.id) == model_id

        def accept_residue(self, residue):  # type: ignore[no-untyped-def]
            chain = residue.get_parent()
            hetero_flag, residue_number, insertion_code = residue.id
            key = (
                str(chain.id),
                str(hetero_flag).strip(),
                int(residue_number),
                str(insertion_code).strip(),
                str(residue.get_resname()).strip(),
            )
            return key in residue_keys

    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path), select=SurfaceSelect())


def _read_pqr_extent(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coordinates = []
    radii = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed PQR atom record: {line!r}")
        try:
            coordinates.append([float(value) for value in parts[-5:-2]])
            radii.append(float(parts[-1]))
        except ValueError as exc:
            raise ValueError(f"Malformed numeric PQR atom record: {line!r}") from exc
    coordinate_array = np.asarray(coordinates, dtype=float)
    radius_array = np.asarray(radii, dtype=float)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1:] != (3,):
        raise ValueError("PDB2PQR produced no usable atom coordinates")
    if (
        radius_array.shape != (len(coordinate_array),)
        or not np.all(np.isfinite(coordinate_array))
        or not np.all(np.isfinite(radius_array))
        or np.any(radius_array < 0.0)
    ):
        raise ValueError("PDB2PQR produced invalid atom coordinates or radii")
    return coordinate_array, radius_array


def _compatible_grid_count(length: float, spacing: float) -> int:
    requested = max(33, int(np.ceil(length / spacing)) + 1)
    return int(np.ceil((requested - 1) / 32.0) * 32 + 1)


def _write_apbs_input(
    path: Path,
    *,
    counts: NDArray[np.int64],
    fine_lengths: NDArray[np.float64],
    center: NDArray[np.float64],
    coarse_grid_factor: float,
    equation: Literal["linearized", "nonlinear"],
    protein_dielectric: float,
    solvent_dielectric: float,
    ionic_strength: float,
    ion_radius: float,
    solvent_radius: float,
    temperature: float,
) -> None:
    coarse_lengths = fine_lengths * coarse_grid_factor
    ion_lines = ""
    if ionic_strength > 0.0:
        ion_lines = (
            f"    ion charge 1 conc {ionic_strength:.10g} radius {ion_radius:.10g}\n"
            f"    ion charge -1 conc {ionic_strength:.10g} radius {ion_radius:.10g}\n"
        )
    equation_keyword = "lpbe" if equation == "linearized" else "npbe"
    path.write_text(
        "read\n"
        "    mol pqr molecule.pqr\n"
        "end\n"
        "elec name solvated\n"
        "    mg-auto\n"
        f"    dime {counts[0]} {counts[1]} {counts[2]}\n"
        f"    cglen {coarse_lengths[0]:.6f} {coarse_lengths[1]:.6f} "
        f"{coarse_lengths[2]:.6f}\n"
        f"    fglen {fine_lengths[0]:.6f} {fine_lengths[1]:.6f} "
        f"{fine_lengths[2]:.6f}\n"
        f"    cgcent {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n"
        f"    fgcent {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n"
        "    mol 1\n"
        f"    {equation_keyword}\n"
        "    bcfl sdh\n"
        f"    pdie {protein_dielectric:.10g}\n"
        f"    sdie {solvent_dielectric:.10g}\n"
        f"{ion_lines}"
        "    srfm smol\n"
        "    chgm spl2\n"
        "    sdens 10.0\n"
        f"    srad {solvent_radius:.10g}\n"
        "    swin 0.3\n"
        f"    temp {temperature:.10g}\n"
        "    calcenergy no\n"
        "    calcforce no\n"
        "    write pot dx potential\n"
        "end\n"
        "quit\n"
    )


def _read_open_dx(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lines = path.read_text().splitlines()
    counts = None
    origin = None
    deltas = []
    item_count = None
    data_start = None
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 8 and parts[:5] == [
            "object", "1", "class", "gridpositions", "counts"
        ]:
            counts = np.asarray(parts[5:8], dtype=int)
        elif len(parts) == 4 and parts[0] == "origin":
            origin = np.asarray(parts[1:], dtype=float)
        elif len(parts) == 4 and parts[0] == "delta":
            deltas.append([float(value) for value in parts[1:]])
        elif "data" in parts and "follows" in parts and "items" in parts:
            item_count = int(parts[parts.index("items") + 1])
            data_start = index + 1
            break
    if counts is None or origin is None or len(deltas) != 3:
        raise ValueError(f"OpenDX file {path} has incomplete grid metadata")
    delta_matrix = np.asarray(deltas, dtype=float)
    spacing = np.diag(delta_matrix)
    if (
        np.any(counts < 2)
        or not np.all(np.isfinite(origin))
        or not np.allclose(delta_matrix, np.diag(spacing), atol=1e-12)
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("Only finite axis-aligned OpenDX grids are supported")
    expected = int(np.prod(counts))
    if item_count != expected or data_start is None:
        raise ValueError("OpenDX item count does not match the grid dimensions")
    values: list[float] = []
    for line in lines[data_start:]:
        for token in line.split():
            if len(values) == expected:
                break
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) == expected:
            break
    if len(values) != expected or not np.all(np.isfinite(values)):
        raise ValueError("OpenDX potential data are incomplete or nonfinite")
    grid = np.asarray(values, dtype=float).reshape(tuple(counts))
    return origin, spacing, grid


def _interpolate_grid(
    points: NDArray[np.float64],
    origin: NDArray[np.float64],
    spacing: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    fractional = (points - origin) / spacing
    maximum = np.asarray(grid.shape, dtype=float) - 1.0
    tolerance = 1e-6
    if np.any(fractional < -tolerance) or np.any(fractional > maximum + tolerance):
        raise ValueError(
            "The APBS potential grid does not enclose all surface vertices; "
            "increase grid_padding"
        )
    fractional = np.clip(fractional, 0.0, maximum)
    lower = np.floor(fractional).astype(int)
    lower = np.minimum(lower, np.asarray(grid.shape) - 2)
    weight = fractional - lower
    x, y, z = lower.T
    wx, wy, wz = weight.T
    values = (
        grid[x, y, z] * (1 - wx) * (1 - wy) * (1 - wz)
        + grid[x + 1, y, z] * wx * (1 - wy) * (1 - wz)
        + grid[x, y + 1, z] * (1 - wx) * wy * (1 - wz)
        + grid[x, y, z + 1] * (1 - wx) * (1 - wy) * wz
        + grid[x + 1, y + 1, z] * wx * wy * (1 - wz)
        + grid[x + 1, y, z + 1] * wx * (1 - wy) * wz
        + grid[x, y + 1, z + 1] * (1 - wx) * wy * wz
        + grid[x + 1, y + 1, z + 1] * wx * wy * wz
    )
    return np.asarray(values, dtype=float)


def _replace_surface_field(
    mesh: MolecularSurfaceMesh,
    field: SurfaceField,
) -> MolecularSurfaceMesh:
    fields = tuple(existing for existing in mesh.fields if existing.name != field.name)
    return replace(mesh, fields=fields + (field,))


def map_electrostatic_potential_apbs(
    structure: Structure,
    surface_obj: MolecularSurfaceResult,
    *,
    ph: float = 7.0,
    forcefield: str = "PARSE",
    use_propka: bool = True,
    equation: Literal["linearized", "nonlinear"] = "linearized",
    protein_dielectric: float = 2.0,
    solvent_dielectric: float = 78.54,
    ionic_strength: float = 0.15,
    ion_radius: float = 2.0,
    solvent_radius: float = 1.4,
    temperature: float = 298.15,
    grid_spacing: float = 0.5,
    grid_padding: float = 12.0,
    coarse_grid_factor: float = 1.7,
    max_grid_points: int = 513,
    field_name: str = "electrostatic_potential",
    pdb2pqr_executable: str | None = None,
    apbs_executable: str = "apbs",
    pdb2pqr_options: Sequence[str] = (),
) -> MolecularSurfaceResult:
    """Map a solvent- and ion-screened Poisson--Boltzmann potential.

    PDB2PQR prepares protonation states, atomic charges, and PB radii. APBS
    solves the finite-difference Poisson--Boltzmann equation and writes a
    potential grid, which is trilinearly interpolated onto every mesh vertex.
    Returned values use ``kJ mol^-1 e^-1``.

    The input structure is not modified. It must be the structure used to
    calculate ``surface_obj``. Nonstandard residues generally require suitable
    PDB2PQR ligand or user-force-field options supplied via ``pdb2pqr_options``.
    """
    if not isinstance(surface_obj, MolecularSurfaceResult):
        raise TypeError("surface_obj must be a MolecularSurfaceResult")
    if not np.isfinite(ph) or not 0.0 <= ph <= 14.0:
        raise ValueError("ph must be between 0 and 14")
    if not isinstance(forcefield, str) or not forcefield.strip():
        raise ValueError("forcefield must be a non-empty string")
    if equation not in {"linearized", "nonlinear"}:
        raise ValueError("equation must be 'linearized' or 'nonlinear'")
    positive_values = {
        "protein_dielectric": protein_dielectric,
        "solvent_dielectric": solvent_dielectric,
        "ion_radius": ion_radius,
        "solvent_radius": solvent_radius,
        "temperature": temperature,
        "grid_spacing": grid_spacing,
        "grid_padding": grid_padding,
        "coarse_grid_factor": coarse_grid_factor,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(ionic_strength) or ionic_strength < 0.0:
        raise ValueError("ionic_strength must be finite and non-negative")
    if coarse_grid_factor <= 1.0:
        raise ValueError("coarse_grid_factor must be greater than 1")
    if (
        isinstance(max_grid_points, bool)
        or not isinstance(max_grid_points, int)
        or max_grid_points < 33
    ):
        raise ValueError("max_grid_points must be an integer of at least 33")
    if not isinstance(apbs_executable, str) or not apbs_executable.strip():
        raise ValueError("apbs_executable must be a non-empty string")
    if isinstance(pdb2pqr_options, str) or not all(
        isinstance(option, str) for option in pdb2pqr_options
    ):
        raise TypeError("pdb2pqr_options must be a sequence of strings")

    with TemporaryDirectory(prefix="biotools-apbs-") as temporary_directory:
        directory = Path(temporary_directory)
        pdb_path = directory / "structure.pdb"
        pqr_path = directory / "molecule.pqr"
        input_path = directory / "apbs.in"
        _write_surface_pdb(structure, surface_obj, pdb_path)

        if pdb2pqr_executable is None:
            pdb2pqr_command = [sys.executable, "-m", "pdb2pqr"]
        else:
            if not isinstance(pdb2pqr_executable, str) or not pdb2pqr_executable.strip():
                raise ValueError("pdb2pqr_executable must be a non-empty string")
            pdb2pqr_command = [pdb2pqr_executable]
        pdb2pqr_command.extend(
            ["--ff", forcefield.strip(), "--keep-chain", "--log-level", "WARNING"]
        )
        if not surface_obj.parameters.include_water:
            pdb2pqr_command.append("--drop-water")
        if use_propka:
            pdb2pqr_command.extend(
                ["--titration-state-method", "propka", "--with-ph", str(float(ph))]
            )
        pdb2pqr_command.extend(pdb2pqr_options)
        pdb2pqr_command.extend([str(pdb_path), str(pqr_path)])
        try:
            prepared = subprocess.run(
                pdb2pqr_command,
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "PDB2PQR is not installed or pdb2pqr_executable is invalid"
            ) from exc
        if prepared.returncode != 0 or not pqr_path.is_file():
            raise _command_error("PDB2PQR", prepared)

        pqr_coordinates, pqr_radii = _read_pqr_extent(pqr_path)
        all_vertices = np.concatenate(
            [mesh.vertices for mesh in surface_obj.components], axis=0
        )
        lower = np.minimum(
            np.min(all_vertices, axis=0),
            np.min(pqr_coordinates - pqr_radii[:, np.newaxis], axis=0),
        )
        upper = np.maximum(
            np.max(all_vertices, axis=0),
            np.max(pqr_coordinates + pqr_radii[:, np.newaxis], axis=0),
        )
        fine_lengths = upper - lower + 2.0 * grid_padding
        center = (lower + upper) / 2.0
        counts = np.asarray(
            [_compatible_grid_count(length, grid_spacing) for length in fine_lengths],
            dtype=int,
        )
        if np.any(counts > max_grid_points):
            raise ValueError(
                "The requested APBS grid exceeds max_grid_points; increase "
                "grid_spacing or max_grid_points"
            )
        _write_apbs_input(
            input_path,
            counts=counts,
            fine_lengths=fine_lengths,
            center=center,
            coarse_grid_factor=coarse_grid_factor,
            equation=equation,
            protein_dielectric=protein_dielectric,
            solvent_dielectric=solvent_dielectric,
            ionic_strength=ionic_strength,
            ion_radius=ion_radius,
            solvent_radius=solvent_radius,
            temperature=temperature,
        )
        try:
            solved = subprocess.run(
                [apbs_executable, input_path.name],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "APBS is not installed or apbs_executable is invalid"
            ) from exc
        if solved.returncode != 0:
            raise _command_error("APBS", solved)
        potential_paths = sorted(directory.glob("potential*.dx"))
        if len(potential_paths) != 1:
            raise RuntimeError(
                "APBS did not produce exactly one OpenDX potential grid"
            )
        origin, spacing, grid = _read_open_dx(potential_paths[0])
        conversion = _GAS_CONSTANT_KJ_MOL_K * temperature
        source = (
            f"apbs({equation}, pdie={protein_dielectric:g}, "
            f"sdie={solvent_dielectric:g}, ionic_strength={ionic_strength:g} M, "
            f"temperature={temperature:g} K); "
            f"pdb2pqr(forcefield={forcefield.strip()}, pH={ph:g}, "
            f"propka={use_propka})"
        )
        components = []
        for mesh in surface_obj.components:
            values = _interpolate_grid(mesh.vertices, origin, spacing, grid)
            field = SurfaceField(
                name=field_name,
                location="vertex",
                values=values * conversion,
                interpolation="linear",
                units="kJ mol^-1 e^-1",
                source=source,
            )
            components.append(_replace_surface_field(mesh, field))
    return replace(surface_obj, components=tuple(components))


__all__ = ["map_electrostatic_potential_apbs"]
