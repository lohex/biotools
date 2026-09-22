"""Triangulated molecular surfaces, scalar fields, patches, and sampling."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import numpy as np
from Bio.PDB.SASA import ATOMIC_RADII
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from numpy.typing import NDArray

from .contacts.models import AtomReference

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure


FieldLocation: TypeAlias = Literal["vertex", "face"]
FieldInterpolation: TypeAlias = Literal["linear", "nearest"]
PatchMode: TypeAlias = Literal["above", "below", "absolute_above"]
PatchConnectivity: TypeAlias = Literal["shared_edge", "shared_vertex"]
NormalMode: TypeAlias = Literal["interpolated", "face"]

_WATER_NAMES = {"HOH", "WAT", "H2O", "SOL", "TIP3", "TIP3P"}
_COULOMB_KJ_MOL_ANGSTROM_E2 = 1389.35457644382
_PATCH_TYPES = {1, 2, 3}
_HYDROPHOBICITY_SCALES = {
    "kyte_doolittle": {
        "ALA": 1.8,
        "ARG": -4.5,
        "ASN": -3.5,
        "ASP": -3.5,
        "CYS": 2.5,
        "GLN": -3.5,
        "GLU": -3.5,
        "GLY": -0.4,
        "HIS": -3.2,
        "ILE": 4.5,
        "LEU": 3.8,
        "LYS": -3.9,
        "MET": 1.9,
        "PHE": 2.8,
        "PRO": -1.6,
        "SER": -0.8,
        "THR": -0.7,
        "TRP": -0.9,
        "TYR": -1.3,
        "VAL": 4.2,
    }
}
_RESIDUE_ALIASES = {
    "ASH": "ASP",
    "CYM": "CYS",
    "CYX": "CYS",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "LYN": "LYS",
}


def _readonly_array(values: Any, dtype: Any | None = None) -> NDArray[Any]:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SurfaceAtom:
    """One atom represented as an input sphere in a molecular surface."""

    reference: AtomReference
    coordinates: NDArray[np.float64]
    radius: float
    element: str

    def __post_init__(self) -> None:
        coordinates = _readonly_array(self.coordinates, np.float64)
        if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("SurfaceAtom coordinates must be three finite values")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("SurfaceAtom radius must be finite and positive")
        if not self.element.strip():
            raise ValueError("SurfaceAtom element must not be empty")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "element", self.element.strip().upper())


@dataclass(frozen=True)
class SurfaceField:
    """Values associated with every vertex or face of a surface mesh."""

    name: str
    location: FieldLocation
    values: NDArray[Any]
    interpolation: FieldInterpolation = "linear"
    units: str | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SurfaceField name must not be empty")
        if self.location not in {"vertex", "face"}:
            raise ValueError("SurfaceField location must be 'vertex' or 'face'")
        if self.interpolation not in {"linear", "nearest"}:
            raise ValueError(
                "SurfaceField interpolation must be 'linear' or 'nearest'"
            )
        values = _readonly_array(self.values)
        if values.ndim != 1:
            raise ValueError("SurfaceField values must be one-dimensional")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class MolecularSurfaceMesh:
    """One connected component of a triangulated molecular surface."""

    component_id: int
    vertices: NDArray[np.float64]
    normals: NDArray[np.float64]
    faces: NDArray[np.int32]
    vertex_atom_indices: NDArray[np.int32]
    vertex_patch_types: NDArray[np.uint8]
    face_patch_types: NDArray[np.uint8]
    fields: tuple[SurfaceField, ...] = ()

    def __post_init__(self) -> None:
        vertices = _readonly_array(self.vertices, np.float64)
        normals = _readonly_array(self.normals, np.float64)
        faces = _readonly_array(self.faces, np.int32)
        atom_indices = _readonly_array(self.vertex_atom_indices, np.int32)
        vertex_types = _readonly_array(self.vertex_patch_types, np.uint8)
        face_types = _readonly_array(self.face_patch_types, np.uint8)

        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("vertices must have shape (n_vertices, 3)")
        if normals.shape != vertices.shape:
            raise ValueError("normals must have the same shape as vertices")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (n_faces, 3)")
        if atom_indices.shape != (len(vertices),):
            raise ValueError("vertex_atom_indices must contain one entry per vertex")
        if vertex_types.shape != (len(vertices),):
            raise ValueError("vertex_patch_types must contain one entry per vertex")
        if face_types.shape != (len(faces),):
            raise ValueError("face_patch_types must contain one entry per face")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
            raise ValueError("surface vertices and normals must be finite")
        if faces.size and (np.min(faces) < 0 or np.max(faces) >= len(vertices)):
            raise ValueError("faces contain an out-of-range vertex index")
        if atom_indices.size and np.min(atom_indices) < 0:
            raise ValueError("vertex_atom_indices must be zero-based")
        if not set(np.unique(vertex_types)).issubset(_PATCH_TYPES):
            raise ValueError("vertex_patch_types contain an unknown MSMS type")
        if not set(np.unique(face_types)).issubset(_PATCH_TYPES):
            raise ValueError("face_patch_types contain an unknown MSMS type")

        names: set[str] = set()
        for field in self.fields:
            if field.name in names:
                raise ValueError(f"Duplicate surface field {field.name!r}")
            names.add(field.name)
            expected = len(vertices) if field.location == "vertex" else len(faces)
            if len(field.values) != expected:
                raise ValueError(
                    f"Surface field {field.name!r} has {len(field.values)} values; "
                    f"expected {expected} for location {field.location!r}"
                )

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "vertex_atom_indices", atom_indices)
        object.__setattr__(self, "vertex_patch_types", vertex_types)
        object.__setattr__(self, "face_patch_types", face_types)
        object.__setattr__(self, "fields", tuple(self.fields))

    def get_field(self, name: str) -> SurfaceField:
        """Return a named field, raising ``KeyError`` if it is absent."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Surface field {name!r} is not present")


@dataclass(frozen=True)
class MolecularSurfaceParameters:
    """Inputs that define an MSMS surface calculation."""

    probe_radius: float
    density: float
    radii_model: str
    include_hydrogens: bool
    include_heteroatoms: bool
    include_water: bool
    all_components: bool
    model_id: str


@dataclass(frozen=True)
class MolecularSurfaceResult:
    """Complete molecular-surface geometry and its input atom mapping."""

    components: tuple[MolecularSurfaceMesh, ...]
    atoms: tuple[SurfaceAtom, ...]
    parameters: MolecularSurfaceParameters
    backend: str = "msms"
    backend_version: str | None = None

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("A molecular surface must contain a component")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("Molecular surface component IDs must be unique")
        for component in self.components:
            if component.vertex_atom_indices.size and (
                np.max(component.vertex_atom_indices) >= len(self.atoms)
            ):
                raise ValueError("Surface mesh refers to an unknown input atom")
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "atoms", tuple(self.atoms))

    @property
    def surface(self) -> MolecularSurfaceMesh:
        """Return the primary, external surface component."""
        return self.components[0]

    def get_component(self, component_id: int) -> MolecularSurfaceMesh:
        """Return a component by its MSMS component identifier."""
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(f"Surface component {component_id} is not present")


@dataclass(frozen=True)
class SurfacePatch:
    """One connected selection of faces from a thresholded surface field."""

    patch_id: int
    component_id: int
    face_indices: NDArray[np.int32]
    vertex_indices: NDArray[np.int32]
    atom_indices: NDArray[np.int32]
    area: float
    centroid: NDArray[np.float64]
    mean_normal: NDArray[np.float64]
    minimum: float
    maximum: float
    mean: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "face_indices", _readonly_array(self.face_indices, np.int32)
        )
        object.__setattr__(
            self, "vertex_indices", _readonly_array(self.vertex_indices, np.int32)
        )
        object.__setattr__(
            self, "atom_indices", _readonly_array(self.atom_indices, np.int32)
        )
        object.__setattr__(
            self, "centroid", _readonly_array(self.centroid, np.float64)
        )
        object.__setattr__(
            self, "mean_normal", _readonly_array(self.mean_normal, np.float64)
        )


@dataclass(frozen=True)
class SurfacePatchSet:
    """Connected patches obtained from one field and threshold rule."""

    component_id: int
    field_name: str
    face_labels: NDArray[np.int32]
    patches: tuple[SurfacePatch, ...]
    threshold: float
    mode: PatchMode
    connectivity: PatchConnectivity

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "face_labels", _readonly_array(self.face_labels, np.int32)
        )
        object.__setattr__(self, "patches", tuple(self.patches))


@dataclass(frozen=True)
class SampledSurfaceField:
    """Values of one surface field at sampled positions."""

    name: str
    values: NDArray[Any]
    units: str | None
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly_array(self.values))


@dataclass(frozen=True)
class SurfaceSamples:
    """Area-weighted points, normals, and optional values on one mesh."""

    component_id: int
    positions: NDArray[np.float64]
    normals: NDArray[np.float64]
    face_indices: NDArray[np.int32]
    barycentric: NDArray[np.float64]
    atom_indices: NDArray[np.int32]
    fields: tuple[SampledSurfaceField, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _readonly_array(self.positions, np.float64))
        object.__setattr__(self, "normals", _readonly_array(self.normals, np.float64))
        object.__setattr__(
            self, "face_indices", _readonly_array(self.face_indices, np.int32)
        )
        object.__setattr__(
            self, "barycentric", _readonly_array(self.barycentric, np.float64)
        )
        object.__setattr__(
            self, "atom_indices", _readonly_array(self.atom_indices, np.int32)
        )
        object.__setattr__(self, "fields", tuple(self.fields))

    def get_field(self, name: str) -> SampledSurfaceField:
        """Return a sampled field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Sampled surface field {name!r} is not present")


def _collect_surface_atoms(
    structure: Structure,
    *,
    model_index: int,
    include_hydrogens: bool,
    include_heteroatoms: bool,
    include_water: bool,
    radii: Mapping[str, float] | None,
) -> tuple[tuple[SurfaceAtom, ...], str]:
    models = list(structure)
    if isinstance(model_index, bool) or not isinstance(model_index, int):
        raise ValueError("model_index must be an integer")
    if model_index < 0 or model_index >= len(models):
        raise ValueError(
            f"model_index {model_index} is out of range for {len(models)} models"
        )
    model = models[model_index]
    overrides = {
        str(element).strip().upper(): float(radius)
        for element, radius in (radii or {}).items()
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in overrides.values()):
        raise ValueError("All custom atomic radii must be finite and positive")
    known_radii = dict(ATOMIC_RADII)
    atoms = []
    for atom in model.get_atoms():
        residue = atom.get_parent()
        residue_name = str(residue.get_resname()).strip().upper()
        hetero_flag = str(residue.id[0]).strip()
        element = str(getattr(atom, "element", "") or "").strip().upper()
        if not element:
            raise ValueError(
                f"Cannot determine element for atom {atom.get_full_id()!r}"
            )
        if not include_hydrogens and element in {"H", "D"}:
            continue
        if not include_heteroatoms and hetero_flag:
            continue
        if not include_water and residue_name in _WATER_NAMES:
            continue
        radius = overrides.get(element, known_radii.get(element))
        if radius is None:
            raise ValueError(
                f"No atomic radius is available for element {element!r}; "
                "provide it through the radii argument"
            )
        atoms.append(
            SurfaceAtom(
                reference=AtomReference.from_atom(atom, str(structure.id)),
                coordinates=np.asarray(atom.coord, dtype=float),
                radius=float(radius),
                element=element,
            )
        )
    if not atoms:
        raise ValueError("The selected model does not contain eligible atoms")
    radii_model = "biopython_atomic_radii"
    if overrides:
        radii_model += "+custom_overrides"
    return tuple(atoms), radii_model


def _write_xyzr(path: Path, atoms: tuple[SurfaceAtom, ...]) -> None:
    lines = [
        f"{atom.coordinates[0]:.8f} {atom.coordinates[1]:.8f} "
        f"{atom.coordinates[2]:.8f} {atom.radius:.8f}\n"
        for atom in atoms
    ]
    path.write_text("".join(lines), encoding="ascii")


def _parse_msms_mesh(
    vertex_path: Path,
    face_path: Path,
    component_id: int,
) -> MolecularSurfaceMesh:
    vertices = []
    normals = []
    atom_indices = []
    vertex_types = []
    for line_number, line in enumerate(
        vertex_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 9:
            raise ValueError(
                f"Invalid MSMS vertex record at {vertex_path}:{line_number}"
            )
        try:
            vertices.append([float(value) for value in columns[:3]])
            normals.append([float(value) for value in columns[3:6]])
            atom_indices.append(int(columns[7]) - 1)
            vertex_types.append(int(columns[8]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid MSMS vertex record at {vertex_path}:{line_number}"
            ) from exc

    faces = []
    face_types = []
    for line_number, line in enumerate(
        face_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 5:
            raise ValueError(
                f"Invalid MSMS face record at {face_path}:{line_number}"
            )
        try:
            faces.append([int(value) - 1 for value in columns[:3]])
            face_types.append(int(columns[3]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid MSMS face record at {face_path}:{line_number}"
            ) from exc

    if not vertices or not faces:
        raise ValueError(f"MSMS component {component_id} is empty")
    return MolecularSurfaceMesh(
        component_id=component_id,
        vertices=np.asarray(vertices, dtype=float),
        normals=np.asarray(normals, dtype=float),
        faces=np.asarray(faces, dtype=np.int32),
        vertex_atom_indices=np.asarray(atom_indices, dtype=np.int32),
        vertex_patch_types=np.asarray(vertex_types, dtype=np.uint8),
        face_patch_types=np.asarray(face_types, dtype=np.uint8),
    )


def _msms_component_id(path: Path, output_name: str) -> int:
    if path.name == f"{output_name}.vert":
        return 0
    match = re.fullmatch(rf"{re.escape(output_name)}_(\d+)\.vert", path.name)
    if match is None:
        raise ValueError(f"Unexpected MSMS output file {path.name!r}")
    return int(match.group(1))


def calculate_molecular_surface(
    structure: Structure,
    *,
    probe_radius: float = 1.5,
    density: float = 1.0,
    model_index: int = 0,
    include_hydrogens: bool = False,
    include_heteroatoms: bool = True,
    include_water: bool = False,
    all_components: bool = False,
    radii: Mapping[str, float] | None = None,
    executable: str = "msms",
) -> MolecularSurfaceResult:
    """Calculate a triangulated solvent-excluded surface with MSMS.

    Face and atom indices in the returned objects are converted from MSMS's
    one-based convention to zero-based NumPy indices. The input structure is
    not modified.
    """
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("executable must be a non-empty string")
    if not np.isfinite(probe_radius) or probe_radius <= 0.0:
        raise ValueError("probe_radius must be finite and positive")
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")

    atoms, radii_model = _collect_surface_atoms(
        structure,
        model_index=model_index,
        include_hydrogens=include_hydrogens,
        include_heteroatoms=include_heteroatoms,
        include_water=include_water,
        radii=radii,
    )
    model = list(structure)[model_index]
    with TemporaryDirectory(prefix="biotools-msms-") as temp_dir:
        directory = Path(temp_dir)
        input_path = directory / "surface.xyzr"
        output_prefix = directory / "surface"
        _write_xyzr(input_path, atoms)
        command = [
            executable,
            "-if",
            str(input_path),
            "-of",
            str(output_prefix),
            "-probe_radius",
            str(float(probe_radius)),
            "-density",
            str(float(density)),
            "-no_header",
        ]
        if all_components:
            command.append("-all_components")
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "MSMS is not installed or its executable is not available on PATH"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise OSError(
                f"MSMS failed with exit code {completed.returncode}: {detail}"
            )

        vertex_paths = sorted(
            directory.glob("surface*.vert"),
            key=lambda path: _msms_component_id(path, output_prefix.name),
        )
        if not vertex_paths:
            raise OSError("MSMS completed without producing a surface mesh")
        components = []
        for vertex_path in vertex_paths:
            component_id = _msms_component_id(vertex_path, output_prefix.name)
            face_path = vertex_path.with_suffix(".face")
            if not face_path.is_file():
                raise OSError(
                    f"MSMS did not produce faces for component {component_id}"
                )
            components.append(
                _parse_msms_mesh(vertex_path, face_path, component_id)
            )

    version_match = re.search(
        r"\bMSMS\s+(?:version\s+)?([0-9]+(?:\.[0-9]+)+)",
        "\n".join((completed.stdout or "", completed.stderr or "")),
        flags=re.IGNORECASE,
    )
    return MolecularSurfaceResult(
        components=tuple(components),
        atoms=atoms,
        parameters=MolecularSurfaceParameters(
            probe_radius=float(probe_radius),
            density=float(density),
            radii_model=radii_model,
            include_hydrogens=include_hydrogens,
            include_heteroatoms=include_heteroatoms,
            include_water=include_water,
            all_components=all_components,
            model_id=str(model.id),
        ),
        backend_version=version_match.group(1) if version_match else None,
    )


def _replace_field(mesh: MolecularSurfaceMesh, field: SurfaceField) -> MolecularSurfaceMesh:
    retained = tuple(item for item in mesh.fields if item.name != field.name)
    return replace(mesh, fields=retained + (field,))


def _resolve_atom_values(
    surface_obj: MolecularSurfaceResult,
    values: Sequence[float] | Mapping[AtomReference, float],
    *,
    name: str,
) -> NDArray[np.float64]:
    if isinstance(values, Mapping):
        missing = [atom.reference for atom in surface_obj.atoms if atom.reference not in values]
        if missing:
            raise ValueError(
                f"{name} is missing values for {len(missing)} surface atoms; "
                f"first missing atom: {missing[0]!r}"
            )
        array = np.asarray([values[atom.reference] for atom in surface_obj.atoms], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.shape != (len(surface_obj.atoms),):
        raise ValueError(
            f"{name} must contain one value for each of the "
            f"{len(surface_obj.atoms)} surface atoms"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must be finite")
    return array


def map_electrostatic_potential(
    surface_obj: MolecularSurfaceResult,
    charges: Sequence[float] | Mapping[AtomReference, float],
    *,
    charge_positions: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
    dielectric: float = 1.0,
    cutoff: float | None = None,
    field_name: str = "electrostatic_potential",
    chunk_size: int = 10_000,
    charge_source: str | None = None,
) -> MolecularSurfaceResult:
    """Map direct Coulomb potential from atom charges onto surface vertices.

    Charges are in elementary-charge units, coordinates in angstroms, and the
    returned potential is in ``kJ mol^-1 e^-1``. If ``charge_positions`` is
    supplied, charges may include atoms omitted from the mesh (such as added
    hydrogens); in that case charges must be a positional sequence. This is
    not a Poisson-Boltzmann calculation; solvent and ion screening are absent.
    """
    if not isinstance(surface_obj, MolecularSurfaceResult):
        raise TypeError("surface_obj must be a MolecularSurfaceResult")
    if not np.isfinite(dielectric) or dielectric <= 0.0:
        raise ValueError("dielectric must be finite and positive")
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0.0):
        raise ValueError("cutoff must be finite and positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if charge_positions is None:
        atom_charges = _resolve_atom_values(surface_obj, charges, name="charges")
        atom_coordinates = np.asarray(
            [atom.coordinates for atom in surface_obj.atoms], dtype=float
        )
    else:
        if isinstance(charges, Mapping):
            raise TypeError(
                "charges must be a positional sequence when charge_positions "
                "is supplied"
            )
        atom_coordinates = np.asarray(charge_positions, dtype=float)
        atom_charges = np.asarray(charges, dtype=float)
        if atom_coordinates.ndim != 2 or atom_coordinates.shape[1] != 3:
            raise ValueError("charge_positions must have shape (n_charges, 3)")
        if not len(atom_coordinates) or atom_charges.shape != (len(atom_coordinates),):
            raise ValueError("charges and charge_positions must have equal length")
        if not np.all(np.isfinite(atom_coordinates)) or not np.all(np.isfinite(atom_charges)):
            raise ValueError("charges and charge_positions must be finite")
    if not len(atom_charges):
        raise ValueError("At least one charge site is required")
    source = f"coulomb(dielectric={float(dielectric):g}, cutoff={cutoff})"
    if charge_source is not None:
        if not isinstance(charge_source, str) or not charge_source.strip():
            raise ValueError("charge_source must be a non-empty string")
        source += f"; charges={charge_source.strip()}"
    components = []
    # Keep temporary pair arrays bounded for realistic all-atom proteins.
    vertex_step = min(chunk_size, max(1, 500_000 // min(len(atom_charges), 2048)))
    charge_step = max(1, 500_000 // vertex_step)
    for mesh in surface_obj.components:
        potential = np.empty(len(mesh.vertices), dtype=float)
        for start in range(0, len(mesh.vertices), vertex_step):
            stop = min(start + vertex_step, len(mesh.vertices))
            accumulated = np.zeros(stop - start, dtype=float)
            for atom_start in range(0, len(atom_charges), charge_step):
                atom_stop = min(atom_start + charge_step, len(atom_charges))
                differences = (
                    mesh.vertices[start:stop, np.newaxis, :]
                    - atom_coordinates[np.newaxis, atom_start:atom_stop, :]
                )
                distances = np.linalg.norm(differences, axis=2)
                if np.any(distances <= np.finfo(float).eps):
                    raise ValueError("A surface vertex coincides with an atom center")
                contributions = atom_charges[np.newaxis, atom_start:atom_stop] / distances
                if cutoff is not None:
                    contributions = np.where(distances <= cutoff, contributions, 0.0)
                accumulated += np.sum(contributions, axis=1)
            potential[start:stop] = (
                _COULOMB_KJ_MOL_ANGSTROM_E2 * accumulated / dielectric
            )
        field = SurfaceField(
            name=field_name,
            location="vertex",
            values=potential,
            interpolation="linear",
            units="kJ mol^-1 e^-1",
            source=source,
        )
        components.append(_replace_field(mesh, field))
    return replace(surface_obj, components=tuple(components))


def map_hydrophobicity(
    surface_obj: MolecularSurfaceResult,
    *,
    scale: str = "kyte_doolittle",
    atom_values: Sequence[float] | Mapping[AtomReference, float] | None = None,
    unknown_value: float | None = np.nan,
    field_name: str = "hydrophobicity",
) -> MolecularSurfaceResult:
    """Map atom or residue hydrophobicity values onto surface vertices.

    Without ``atom_values``, every atom receives the Kyte-Doolittle value of
    its residue. Unsupported residues receive ``unknown_value``; pass ``None``
    to reject them instead.
    """
    if not isinstance(surface_obj, MolecularSurfaceResult):
        raise TypeError("surface_obj must be a MolecularSurfaceResult")
    if atom_values is not None:
        values = _resolve_atom_values(surface_obj, atom_values, name="atom_values")
        source = "custom_atom_values"
    else:
        try:
            residue_scale = _HYDROPHOBICITY_SCALES[scale]
        except KeyError as exc:
            choices = ", ".join(sorted(_HYDROPHOBICITY_SCALES))
            raise ValueError(
                f"Unknown hydrophobicity scale {scale!r}; choose from {choices}"
            ) from exc
        values_list = []
        unsupported = []
        for atom in surface_obj.atoms:
            residue_name = _RESIDUE_ALIASES.get(
                atom.reference.residue_name.upper(),
                atom.reference.residue_name.upper(),
            )
            value = residue_scale.get(residue_name)
            if value is None:
                unsupported.append(atom.reference)
                value = unknown_value
            values_list.append(value)
        if unsupported and unknown_value is None:
            raise ValueError(
                f"Hydrophobicity scale {scale!r} does not cover "
                f"{unsupported[0].residue_name!r}"
            )
        values = np.asarray(values_list, dtype=float)
        source = f"residue_scale:{scale}"

    components = []
    for mesh in surface_obj.components:
        field = SurfaceField(
            name=field_name,
            location="vertex",
            values=values[mesh.vertex_atom_indices],
            interpolation="linear",
            units=None,
            source=source,
        )
        components.append(_replace_field(mesh, field))
    return replace(surface_obj, components=tuple(components))


def _resolve_mesh(
    surface_obj: MolecularSurfaceResult | MolecularSurfaceMesh,
    component_id: int | None,
) -> MolecularSurfaceMesh:
    if isinstance(surface_obj, MolecularSurfaceMesh):
        if component_id is not None and surface_obj.component_id != component_id:
            raise ValueError(
                f"Mesh component {surface_obj.component_id} does not match "
                f"component_id={component_id}"
            )
        return surface_obj
    if isinstance(surface_obj, MolecularSurfaceResult):
        return (
            surface_obj.surface
            if component_id is None
            else surface_obj.get_component(component_id)
        )
    raise TypeError(
        "surface_obj must be a MolecularSurfaceResult or MolecularSurfaceMesh"
    )


def _triangle_geometry(
    mesh: MolecularSurfaceMesh,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    areas = lengths * 0.5
    normals = np.zeros_like(cross)
    nonzero = lengths > np.finfo(float).eps
    normals[nonzero] = cross[nonzero] / lengths[nonzero, np.newaxis]
    centroids = np.mean(triangles, axis=1)
    return areas, normals, centroids


def _field_face_values(
    mesh: MolecularSurfaceMesh,
    field: SurfaceField,
) -> NDArray[np.float64]:
    values = np.asarray(field.values, dtype=float)
    if field.location == "face":
        return values
    vertex_values = values[mesh.faces]
    valid = np.isfinite(vertex_values)
    counts = np.sum(valid, axis=1)
    sums = np.sum(np.where(valid, vertex_values, 0.0), axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(len(mesh.faces), np.nan, dtype=float),
        where=counts > 0,
    )


def _candidate_neighbors(
    faces: NDArray[np.int32],
    candidates: NDArray[np.bool_],
    connectivity: PatchConnectivity,
) -> dict[int, set[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for face_index in np.flatnonzero(candidates):
        face = faces[face_index]
        if connectivity == "shared_vertex":
            keys = [int(vertex) for vertex in face]
        else:
            a, b, c = (int(value) for value in face)
            keys = [tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a))]
        for key in keys:
            groups[key].append(int(face_index))
    neighbors: dict[int, set[int]] = defaultdict(set)
    for members in groups.values():
        for face_index in members:
            neighbors[face_index].update(
                other for other in members if other != face_index
            )
    return neighbors


def find_surface_patches(
    surface_obj: MolecularSurfaceResult | MolecularSurfaceMesh,
    *,
    field_name: str,
    threshold: float,
    mode: PatchMode = "above",
    connectivity: PatchConnectivity = "shared_edge",
    min_area: float = 0.0,
    component_id: int | None = None,
) -> SurfacePatchSet:
    """Find connected face patches after thresholding a scalar field."""
    mesh = _resolve_mesh(surface_obj, component_id)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if mode not in {"above", "below", "absolute_above"}:
        raise ValueError("mode must be 'above', 'below', or 'absolute_above'")
    if connectivity not in {"shared_edge", "shared_vertex"}:
        raise ValueError("connectivity must be 'shared_edge' or 'shared_vertex'")
    if not np.isfinite(min_area) or min_area < 0.0:
        raise ValueError("min_area must be finite and non-negative")
    field = mesh.get_field(field_name)
    face_values = _field_face_values(mesh, field)
    finite = np.isfinite(face_values)
    if mode == "above":
        candidates = finite & (face_values >= threshold)
    elif mode == "below":
        candidates = finite & (face_values <= threshold)
    else:
        candidates = finite & (np.abs(face_values) >= threshold)

    neighbors = _candidate_neighbors(mesh.faces, candidates, connectivity)
    connected = []
    remaining = set(int(index) for index in np.flatnonzero(candidates))
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        indices = []
        while queue:
            face_index = queue.popleft()
            indices.append(face_index)
            for neighbor in sorted(neighbors.get(face_index, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        connected.append(np.asarray(sorted(indices), dtype=np.int32))

    areas, face_normals, centroids = _triangle_geometry(mesh)
    labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    patches = []
    for indices in connected:
        patch_area = float(np.sum(areas[indices]))
        if patch_area < min_area:
            continue
        patch_id = len(patches)
        labels[indices] = patch_id
        vertex_indices = np.unique(mesh.faces[indices])
        atom_indices = np.unique(mesh.vertex_atom_indices[vertex_indices])
        if patch_area > 0.0:
            centroid = np.average(centroids[indices], axis=0, weights=areas[indices])
            mean_value = float(np.average(face_values[indices], weights=areas[indices]))
            normal = np.sum(face_normals[indices] * areas[indices, np.newaxis], axis=0)
        else:
            centroid = np.mean(centroids[indices], axis=0)
            mean_value = float(np.mean(face_values[indices]))
            normal = np.mean(face_normals[indices], axis=0)
        normal_length = float(np.linalg.norm(normal))
        if normal_length > np.finfo(float).eps:
            normal = normal / normal_length
        patches.append(
            SurfacePatch(
                patch_id=patch_id,
                component_id=mesh.component_id,
                face_indices=indices,
                vertex_indices=vertex_indices,
                atom_indices=atom_indices,
                area=patch_area,
                centroid=centroid,
                mean_normal=normal,
                minimum=float(np.min(face_values[indices])),
                maximum=float(np.max(face_values[indices])),
                mean=mean_value,
            )
        )
    return SurfacePatchSet(
        component_id=mesh.component_id,
        field_name=field_name,
        face_labels=labels,
        patches=tuple(patches),
        threshold=float(threshold),
        mode=mode,
        connectivity=connectivity,
    )


def color_from_labels(
    labels: Sequence[int] | NDArray[np.integer[Any]],
    *,
    cmap: str = "tab20",
    background_color: Any = (0.75, 0.75, 0.75, 0.0),
) -> NDArray[np.float64]:
    """Convert non-negative integer labels to deterministic RGBA colors."""
    label_array = np.asarray(labels)
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError("labels must contain integers")
    if np.any(label_array < -1):
        raise ValueError("labels may only use -1 as the background value")
    color_map = colormaps.get_cmap(cmap)
    background = np.asarray(to_rgba(background_color), dtype=float)
    colors = np.empty(label_array.shape + (4,), dtype=float)
    colors[...] = background
    for label in np.unique(label_array[label_array >= 0]):
        colors[label_array == label] = color_map(int(label) % color_map.N)
    return _readonly_array(colors, np.float64)


def sample_surface(
    surface_obj: MolecularSurfaceResult | MolecularSurfaceMesh,
    n_points: int,
    *,
    component_id: int | None = None,
    patch: SurfacePatch | None = None,
    face_indices: Sequence[int] | NDArray[np.integer[Any]] | None = None,
    field_names: Sequence[str] = (),
    normal_mode: NormalMode = "interpolated",
    seed: int | None = None,
) -> SurfaceSamples:
    """Sample points uniformly by triangle area from a surface selection."""
    mesh = _resolve_mesh(surface_obj, component_id)
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    if patch is not None and face_indices is not None:
        raise ValueError("Specify either patch or face_indices, not both")
    if patch is not None:
        if patch.component_id != mesh.component_id:
            raise ValueError("patch belongs to a different surface component")
        selected = np.asarray(patch.face_indices, dtype=np.int32)
    elif face_indices is None:
        selected = np.arange(len(mesh.faces), dtype=np.int32)
    else:
        selected = np.asarray(face_indices, dtype=np.int32)
        if selected.ndim != 1:
            raise ValueError("face_indices must be one-dimensional")
    if selected.size == 0:
        raise ValueError("No faces are available for sampling")
    if np.min(selected) < 0 or np.max(selected) >= len(mesh.faces):
        raise ValueError("face_indices contain an out-of-range index")
    if normal_mode not in {"interpolated", "face"}:
        raise ValueError("normal_mode must be 'interpolated' or 'face'")
    if isinstance(field_names, str):
        raise TypeError("field_names must be a sequence of field names")
    if len(field_names) != len(set(field_names)):
        raise ValueError("field_names must not contain duplicates")

    areas, face_normals, _ = _triangle_geometry(mesh)
    positive = areas[selected] > np.finfo(float).eps
    selected = selected[positive]
    selected_areas = areas[selected]
    if selected.size == 0:
        raise ValueError("Selected faces have zero area")
    probabilities = selected_areas / np.sum(selected_areas)
    rng = np.random.default_rng(seed)
    chosen_faces = rng.choice(selected, size=n_points, replace=True, p=probabilities)
    random_values = rng.random((n_points, 2))
    roots = np.sqrt(random_values[:, 0])
    barycentric = np.column_stack(
        (
            1.0 - roots,
            roots * (1.0 - random_values[:, 1]),
            roots * random_values[:, 1],
        )
    )
    face_vertices = mesh.faces[chosen_faces]
    triangles = mesh.vertices[face_vertices]
    positions = np.einsum("ni,nij->nj", barycentric, triangles)

    if normal_mode == "face":
        normals = face_normals[chosen_faces]
    else:
        normals = np.einsum(
            "ni,nij->nj", barycentric, mesh.normals[face_vertices]
        )
        lengths = np.linalg.norm(normals, axis=1)
        usable = lengths > np.finfo(float).eps
        normals[usable] /= lengths[usable, np.newaxis]
        normals[~usable] = face_normals[chosen_faces[~usable]]

    nearest_vertex = np.argmax(barycentric, axis=1)
    atom_indices = mesh.vertex_atom_indices[
        face_vertices[np.arange(n_points), nearest_vertex]
    ]
    sampled_fields = []
    for name in field_names:
        field = mesh.get_field(name)
        if field.location == "face":
            values = field.values[chosen_faces]
        elif field.interpolation == "nearest":
            values = field.values[
                face_vertices[np.arange(n_points), nearest_vertex]
            ]
        else:
            values = np.einsum(
                "ni,ni->n",
                barycentric,
                np.asarray(field.values[face_vertices], dtype=float),
            )
        sampled_fields.append(
            SampledSurfaceField(
                name=field.name,
                values=values,
                units=field.units,
                source=field.source,
            )
        )
    return SurfaceSamples(
        component_id=mesh.component_id,
        positions=positions,
        normals=normals,
        face_indices=chosen_faces,
        barycentric=barycentric,
        atom_indices=atom_indices,
        fields=tuple(sampled_fields),
    )


__all__ = [
    "calculate_molecular_surface",
    "color_from_labels",
    "find_surface_patches",
    "map_electrostatic_potential",
    "map_hydrophobicity",
    "MolecularSurfaceMesh",
    "MolecularSurfaceParameters",
    "MolecularSurfaceResult",
    "SampledSurfaceField",
    "sample_surface",
    "SurfaceAtom",
    "SurfaceField",
    "SurfacePatch",
    "SurfacePatchSet",
    "SurfaceSamples",
]
