# Structure module

[Back to the main README](../../README.md) · [Documentation overview](../README.md)

## Inhaltsverzeichnis

- [Installation and external tools](#installation-and-external-tools)
- [Structure I/O and metadata](#structure-io-and-metadata)
  - [Download and load structures](#download-and-load-structures)
  - [RCSB entry metadata](#rcsb-entry-metadata)
- [Chains and structural alignment](#chains-and-structural-alignment)
  - [Extract sequences and edit chains](#extract-sequences-and-edit-chains)
  - [Align homologous structures](#align-homologous-structures)
- [Secondary structure and solvent accessibility](#secondary-structure-and-solvent-accessibility)
  - [DSSP](#dssp)
  - [FreeSASA](#freesasa)
  - [Interaction-surface analysis](#interaction-surface-analysis)
  - [Molecular surface meshes](#molecular-surface-meshes)
- [Distances and contacts](#distances-and-contacts)
  - [Residue contacts by distance](#residue-contacts-by-distance)
  - [Geometric contact characterization](#geometric-contact-characterization)
  - [Interaction definitions and calculations](interactions.md)
  - [Distance matrices](#distance-matrices)
  - [Interaction matrices](#interaction-matrices)
- [Visualization and orientation](#visualization-and-orientation)
- [Compatibility imports](#compatibility-imports)

`biotools.structure` contains the public API for reading, transforming,
analyzing, and visualizing protein structures represented by Biopython.

## Installation and external tools

The module is part of the base installation:

```bash
python -m pip install -e .
```

The following external programs are only needed for their corresponding
analyses and must be available on `PATH`:

- DSSP as `dssp` or `mkdssp` for secondary-structure assignment;
- `freesasa` for SASA and interaction-surface calculations;
- `msms` for triangulated solvent-excluded surfaces; and
- `apbs` for solvent-screened Poisson--Boltzmann potentials. PDB2PQR is a
  Python dependency of the base package and prepares the corresponding PQR.

Contact characterization uses built-in protein bond templates without an
optional dependency. In Python packaging, an *extra* is a named group of
optional dependencies. The `contacts` extra is declared in `pyproject.toml`
and currently adds OpenMM; it is not a separate biotools module. Install a
released package together with that group using:

```bash
python -m pip install "biotools[contacts]"
```

From an editable source checkout, use:

```bash
python -m pip install -e ".[contacts]"
```

An ordinary `pip install biotools` omits this optional group. The `contacts`
extra does not install the CUDA packages or PDBFixer from `biotools[md]`. It
enables the OpenMM topology backend and the optional nonperiodic
`openmm_rigid_water` local-refinement backend. The default template topology
and `geometric` water backend do not require OpenMM.

## Structure I/O and metadata

### Download and load structures

`get_pdb_structure()` downloads a PDB entry and returns a Biopython
`Structure`:

```python
from biotools.structure import get_aa_sequence, get_pdb_structure

structure = get_pdb_structure("1crn", target_folder="structures")
sequences = get_aa_sequence(structure, show_gaps=False)

for chain_id, sequence in sequences.items():
    print(chain_id, sequence)
```

PDB/`.ent` is tried first and mmCIF is used as a fallback. Pass
`prefer_mmcif=True` to reverse this order. Use
`get_pdb_structure_as_pdb()` or `get_pdb_structure_as_mmcif()` when automatic
fallback is not desired.

Existing files can be loaded and structures can be written or converted with:

```python
from biotools.structure import (
    convert_cif_to_pdb,
    load_pdb_from_file,
    save_structure_to_file,
)

structure = load_pdb_from_file("input.pdb")
save_structure_to_file(structure, "copy.pdb")
convert_cif_to_pdb("input.cif", "converted.pdb")
```

### RCSB entry metadata

`get_pdb_metadata()` retrieves source organisms and dates from the RCSB Data
API:

```python
from biotools.structure import get_pdb_metadata

metadata = get_pdb_metadata("4HHB")

print(metadata.pdb_id)       # 4HHB
print(metadata.organisms)    # ("Homo sapiens",)
print(metadata.deposited)    # datetime.date(1984, 3, 7)
print(metadata.released)     # datetime.date(1984, 7, 17)
```

Organisms are the unique scientific source-organism names across the entry's
polymer entities. Entries without a published initial release date return
`released=None`.

## Chains and structural alignment

### Extract sequences and edit chains

The module can extract or trim chains, obtain amino-acid sequences, and change
chain or residue identifiers. Renaming uses collision-safe intermediate IDs,
so simultaneous swaps are supported:

```python
from biotools.structure import extract_chain, rename_chain, reset_index

selected = extract_chain(structure, ["A", "B"])
rename_chain(selected, {"A": "B", "B": "A"})
reset_index(selected)
```

These operations work on the object passed to them unless their function
documentation explicitly states that a copy is returned.

### Align homologous structures

`align_homologs()` determines a transformation from corresponding homologous
chains and applies it to a copy of the complete mobile structure:

```python
from biotools.structure import align_homologs, load_pdb_from_file

reference = load_pdb_from_file("reference.pdb")
mobile = load_pdb_from_file("mobile.pdb")

aligned = align_homologs(reference, mobile, chain1="A", chain2="B")
```

Lower-level helpers for RMSD, atom selections, transformations, and direct
superposition are also exported from `biotools.structure`.

## Secondary structure and solvent accessibility

### DSSP

DSSP must be installed separately and exposed as `dssp` or `mkdssp` on
`PATH`. `assign_secondary_structure()` accepts PDB and mmCIF files and calls
the executable directly:

```python
from biotools.structure import assign_secondary_structure

dssp = assign_secondary_structure("protein.pdb")

for residue in dssp.residues:
    print(
        residue.chain_id,
        residue.residue_id,
        residue.secondary_structure,
        residue.relative_accessibility,
        residue.absolute_accessibility,
    )

print(dssp.secondary_structure)
print(dssp.relative_sasa)
print(dssp.absolute_sasa)
```

`secondary_structure` concatenates the assignments for all returned residues.
`relative_sasa` and `absolute_sasa` contain values in the same order; absolute
SASA is reported in Å². The relative values use the selected Biopython
reference scale, `"Sander"` by default.

Some generated PDB files, including PeptideBuilder output, omit records DSSP
expects. When needed, biotools passes DSSP a temporary copy with compatibility
`HEADER` and dummy `CRYST1` records. The source file is not modified.

### FreeSASA

Use FreeSASA when only solvent accessibility is needed. Install the external
program and make its `freesasa` executable available on `PATH`:

```python
from biotools.structure import calculate_sasa

sasa = calculate_sasa(structure)

print(sasa.total_absolute_sasa)
print(sasa.chain_absolute_sasa)
for residue in sasa.residues:
    print(
        residue.chain_id,
        residue.residue_id,
        residue.absolute_sasa,
        residue.relative_sasa,
    )
```

Absolute values are reported in Å². Relative residue values are fractions, so
`1.0` corresponds to 100% of FreeSASA's reference accessibility. Values can be
larger than `1.0` for unusually exposed conformations.

### Interaction-surface analysis

`analyze_interaction_surface()` compares each selected chain in isolation
with the same chain in the two-chain complex:

```python
from biotools.structure import analyze_interaction_surface

surface = analyze_interaction_surface(
    structure,
    "A",
    "B",
    per_residue_scores=True,
    relative_sasa=True,
    absolute_sasa=True,
)

print(surface.total)
print(surface.chain_scores["A"])
print(surface.chain_scores["B"])
print(surface.per_residue_scores)
```

At every enabled level, `delta_sasa_absolute` is
`sasa_separated - sasa_complex`. `delta_sasa_relative` is this difference
divided by `sasa_separated`, i.e. the fraction of the originally accessible
surface buried during association. The total absolute delta is the two-sided
buried surface; the conventional interface area is half that value.

The analysis extracts copies of the two chains and removes water and other
hetero residues before calculating the isolated chains and their complex. It
does not modify the input structure. Set `per_residue_scores=False` to omit
residue records, or disable relative or absolute output independently.

### Molecular surface meshes

`calculate_molecular_surface()` calls MSMS and returns zero-based NumPy mesh
arrays, vertex normals, and a stable mapping from every vertex to the closest
input atom. MSMS must be available on `PATH`, or its path can be passed with
`executable=`:

```python
from biotools.structure import calculate_molecular_surface

surface = calculate_molecular_surface(
    structure,
    probe_radius=1.5,
    density=1.0,
    executable="msms",
)

mesh = surface.surface
print(mesh.vertices.shape, mesh.faces.shape, mesh.normals.shape)
```

By default the first structure model is used, hydrogens and water are omitted,
and other hetero atoms are retained. Atomic radii come from Biopython's atomic
radii table. Supply missing or replacement element radii with, for example,
`radii={"FE": 1.8}`. Use `all_components=True` to retain internal components
and cavities in addition to the primary external component.

Hydrophobicity can be assigned from the Kyte-Doolittle residue scale. Values
for unsupported residues are `NaN` unless `unknown_value` is supplied:

```python
from biotools.structure import map_hydrophobicity

surface = map_hydrophobicity(surface)
hydrophobicity = surface.surface.get_field("hydrophobicity").values
```

Electrostatic mapping requires one partial charge per `surface.atoms` entry.
The built-in calculation is an unscreened direct Coulomb potential, not a
Poisson-Boltzmann calculation. `dielectric` must therefore be selected for the
intended model:

```python
from biotools.structure import map_electrostatic_potential

surface = map_electrostatic_potential(
    surface,
    charges,
    dielectric=4.0,
)
```

For a complete protein PDB, use OpenMM to assign **per-atom force-field
partial charges** instead of assigning an entire residue charge to every
atom. Install `biotools[contacts]` for OpenMM, then pass the same Biopython
structure used to generate the surface:

```python
from biotools.structure import (
    calculate_molecular_surface,
    load_pdb_from_file,
    map_electrostatic_potential_openmm,
)

structure = load_pdb_from_file("prepared_protein.pdb")
surface = calculate_molecular_surface(
    structure,
    include_heteroatoms=False,
)
surface = map_electrostatic_potential_openmm(
    structure,
    surface,
    forcefield_files=("amber14-all.xml",),
    add_hydrogens=True,
    ph=7.0,
    dielectric=4.0,
)
```

The OpenMM adapter adds missing hydrogens to an in-memory copy and includes
their charges and positions in the Coulomb sum even when the MSMS surface
contains only heavy atoms. It does **not** repair missing heavy atoms or
parameterize unsupported ligands automatically: prepare those separately or
provide compatible force-field files. A regular PDB alone does not determine
protonation and partial charges uniquely. The resulting field is still a
constant-dielectric Coulomb approximation, not a solvent-screened potential.

For an ion- and solvent-screened potential, use the PDB2PQR/APBS backend.
PDB2PQR assigns protonation states, atomic charges, and PB radii; APBS solves
the linearized or nonlinear Poisson--Boltzmann equation on a finite-difference
grid. The OpenDX potential is interpolated onto the MSMS vertices and converted
from `kT/e` to `kJ mol^-1 e^-1`:

```python
from biotools.structure import map_electrostatic_potential_apbs

pb_surface = map_electrostatic_potential_apbs(
    structure,
    surface,
    ph=7.4,
    equation="linearized",
    protein_dielectric=2.0,
    solvent_dielectric=78.54,
    ionic_strength=0.15,  # mol/litre, symmetric monovalent salt
    grid_spacing=0.5,     # target maximum spacing in angstroms
)
```

`apbs` must be on `PATH`, or set `apbs_executable=`. By default PDB2PQR runs
through the active Python interpreter; `pdb2pqr_executable=` can select a
different command. `grid_padding`, `grid_spacing`, and `max_grid_points`
control grid extent, resolution, and the memory guard. Nonstandard residues
usually need ligand or user-force-field arguments via `pdb2pqr_options`.
APBS constructs its dielectric boundary from PDB2PQR radii; the returned field
is sampled on the independently calculated MSMS visualization mesh.

Thresholded fields can be segmented into connected face patches. Patch labels
are `-1` for background and otherwise index `patches.patches`:

```python
from biotools.structure import color_from_labels, find_surface_patches

patches = find_surface_patches(
    surface,
    field_name="hydrophobicity",
    threshold=1.5,
    min_area=10.0,
)
face_colors = color_from_labels(patches.face_labels)
```

`sample_surface()` chooses triangles proportional to their area and samples
uniform barycentric coordinates within them. Vertex fields and normals are
interpolated at the sampled positions:

```python
from biotools.structure import sample_surface

samples = sample_surface(
    surface,
    500,
    patch=patches.patches[0],
    field_names=("hydrophobicity", "electrostatic_potential"),
    seed=42,
)

probe_positions = samples.positions + 2.0 * samples.normals
potential = samples.get_field("electrostatic_potential").values
```

## Distances and contacts

### Residue contacts by distance

`get_interaction_residues()` uses a KD-tree to find residue pairs within a
cutoff. Its default is the minimum heavy-atom distance:

```python
from biotools.structure import get_interaction_residues

residue_contacts = get_interaction_residues(
    structure,
    "A",
    "B",
    cutoff=5.0,
    distance_metric="min_heavy_atom",
)
```

Available metrics are:

- `"min_heavy_atom"`: minimum distance after excluding hydrogen and
  deuterium atoms;
- `"min_atom"`: minimum distance across all atoms, preserving the former
  behavior;
- `"c_alpha"`: Cα-to-Cα distance;
- `"c_beta"`: Cβ-to-Cβ distance, with Cα used for glycine.

Cα and Cβ contact maps commonly use a larger cutoff such as 8 Å. Residues
without a required representative atom are skipped. The brute-force reference
implementation `get_interaction_residues_full()` supports the same metrics.

### Geometric contact characterization

`characterize_chain_contacts()` classifies snapshot-based geometric contact
candidates between two protein chains:

```python
from biotools.structure import characterize_chain_contacts

contacts = characterize_chain_contacts(
    structure,
    "A",
    "B",
    atomic=False,
    topology_backend="templates",
    profile="refined",
)
```

The supported categories are hydrogen bonds, salt bridges, hydrophobic
contacts, van der Waals contacts, parallel and T-shaped π stacking,
cation–π candidates, and single-water bridges. Every category has a Boolean
argument and is enabled by default. With `atomic=False`, one representative
observation is returned per residue pair and category; `atomic=True` retains
individual atom or group observations.

The complete per-type chemical rules, formulas, thresholds, refinements,
output fields, and single-water workflow are documented in
[Interaction types and calculations](interactions.md).

The default `"templates"` topology backend has no optional dependency. After
installing `biotools[contacts]`, the CPU-only OpenMM backend can provide
standard bond topology and disulfide assignments:

```python
contacts = characterize_chain_contacts(
    structure,
    "A",
    "B",
    topology_backend="openmm",
)
```

Selecting the backend explicitly keeps results independent of which optional
packages happen to be installed. These records describe geometric candidates,
not interaction energies or temporal stability. For contacts within a single
chain, use `characterize_intrachain_contacts()`; its default
`min_sequence_separation=2` excludes self-pairs and adjacent residues.

### Distance matrices

Intrachain matrices are square and symmetric. Interchain matrices use the
first chain for rows and the second for columns. Values are returned as NumPy
arrays in Å:

```python
from biotools.structure import (
    get_distance_matrix,
    get_interchain_distance_matrix,
)

intrachain = get_distance_matrix(
    structure,
    "A",
    distance_metric="min_heavy_atom",
)
interchain = get_interchain_distance_matrix(
    structure,
    "A",
    "B",
    distance_metric="c_beta",
)
```

Matrix functions support `"c_alpha"`, `"c_beta"`, and
`"min_heavy_atom"`. For the Cβ measure, glycine uses Cα. Other missing
representative atoms produce `NaN` without removing the residue from the
matrix.

Distance heatmaps return a Matplotlib figure and axes and do not call
`show()`:

```python
from biotools.structure import (
    plot_distance_matrix,
    plot_interchain_distance_matrix,
)

figure, axes = plot_distance_matrix(structure, "A", "c_alpha")
interface_figure, interface_axes = plot_interchain_distance_matrix(
    structure,
    "A",
    "B",
    "min_heavy_atom",
)
```

### Interaction matrices

Interaction plots accept either a distance measure, one geometric contact
type, or `"interaction_type"`:

```python
from biotools.structure import (
    plot_interaction_matrix,
    plot_interchain_interaction_matrix,
)

contact_figure, contact_axes = plot_interaction_matrix(
    structure,
    "A",
    interaction_measure="interaction_type",
    min_sequence_separation=2,
)
interface_figure, interface_axes = plot_interchain_interaction_matrix(
    structure,
    "A",
    "B",
    interaction_measure="interaction_type",
)
```

For intrachain plots, `min_sequence_separation=2` masks self-pairs and directly
adjacent residues. With `interaction_measure="interaction_type"`, each
observed set of contact types receives a discrete color. Cells with multiple
types get their own combination color, and the legend lists only categories
that occur in the plotted matrix. Pass `topology_backend="openmm"` to the
plotting functions to use the optional OpenMM topology backend.

## Visualization and orientation

`plot_structure()` creates an interactive py3Dmol view. The related
`plot_molecular_surface()` combines cartoon, stick, line, or custom structure
styles with an MSMS mesh. The mesh can be colored by one scalar field or by a
`SurfacePatchSet`; optional `SurfaceSamples` are drawn as normal-vector arrows.
With `colorbar=True`, a continuous field's color scale appears inside the
interactive view.
`move_to_center()` returns a translated copy whose center of mass is at the
origin, while `superimpose_PCA()` can center and orient a copy along the
principal axes of its Cα coordinates.

```python
from biotools.structure import move_to_center, plot_structure, superimpose_PCA

centered = move_to_center(structure)
oriented, shift, rotation = superimpose_PCA(centered)
view = plot_structure(oriented)
view.show()
```

```python
from biotools.structure import plot_molecular_surface

view = plot_molecular_surface(
    structure,
    surface,
    structure_style="cartoon+stick",
    patches=patches,
    patch_wireframe=True,
    samples=samples,
    surface_opacity=0.65,
)
view.show()
```

Pass `field_name="electrostatic_potential"` instead of `patches=` to use a
continuous surface field. `field_range=` and `field_center=` control its color
normalization. Patch colors are applied per face; continuous vertex fields are
interpolated by the WebGL renderer. With `patch_wireframe=True`, the MSMS
triangle edges are overlaid only on the faces selected by `patch=` or
`patches=`; `patch_wireframe_color=` controls their color. The grid follows
the original mesh resolution, so a lower MSMS `density` can make it easier to
read.

To display only the largest patch with continuous colors, pass the individual
`SurfacePatch` as `patch=` together with a `field_name`. The colorbar then uses
the displayed patch's value range unless `field_range` is set explicitly.

```python
largest = max(patches.patches, key=lambda patch: patch.area)
view = plot_molecular_surface(
    structure, surface, patch=largest,
    field_name="hydrophobicity", cmap="coolwarm", colorbar=True,
    structure_style={"cartoon": {"color": "blue"}},
)
view.show()
```

`colorbar=True` requires `field_name`; categorical `patches=` coloring does not
have a continuous color scale.

## Interactive contact views

`plot_structure_contacts()` visualizes existing typed `ContactObservation` or
`ContactAnalysisResult` objects; it does not detect contacts. The returned
`StructureContactView` displays in a notebook and writes an interactive HTML
file. The HTML loads 3Dmol.js from a CDN, so opening it requires network access.

```python
from biotools.structure import plot_structure_contacts

contact_view = plot_structure_contacts(
    structure,
    contact_result,
    chain_styles={
        "A": {"cartoon": {"color": "#d0d0d0", "opacity": 0.55}},
        "B": {"cartoon": {"color": "#d97721", "opacity": 0.90}},
    },
    chain_labels={"A": "Receptor", "B": "Ligand"},
    contact_types={"hydrogen_bond", "salt_bridge", "water_bridge"},
    enabled_contact_types={"hydrogen_bond", "salt_bridge"},
    residue_labels={"A": "active", "B": "all"},
    initial_view="xy",
)
contact_view.write_html("contacts.html")
```

`contact_types` excludes contact types from the document.
`enabled_contact_types` keeps included types available but initially hidden.
The Hide controls button at the top collapses the sidebar to a Show controls
button, preserving contact visibility, highlights, and expanded lists.
The sidebar puts Views first, followed by Residue labels and Interaction types.
Each type has a Show contacts button that expands a separated list of residue
pairs, formatted as chain, residue name, position, and distance.
The checkboxes control visibility. Each pair has an independent Highlight
toggle, so multiple visible contacts can have floating labels at once.
Highlighting does not change the line radius. Residues participating in visible
contacts are drawn as sticks by default (`active_contact_sticks=True`); hiding
their last visible contact removes these additional sticks. `show_contact_types()` and `show_contact_pairs()` change which
contacts are serialized the next time the view is displayed or saved.
`set_contact_type_enabled()`, `set_contact_pair_enabled()`,
`set_all_contacts_enabled()`, `highlight_contact_pair()`,
`set_contact_pair_highlighted()`, `clear_highlight()`,
`set_residue_label_mode()`, and `set_view()` change state from Python.
`highlighted_pair_ids` contains all highlighted pairs;
`highlighted_pair_id` retains the last highlighted pair for compatibility.
The exported HTML offers the same visibility and view controls in the browser.

The [1BRS interactions notebook](../../examples/interactions.ipynb) walks
through preparation, typed analysis, aggregation, 2D matrices, 3D controls,
and HTML export with a real protein complex.

Contact styles can be overridden through `contact_styles=`; ring planes use
`ring_opacity=`. A 4×4 rigid `coordinate_transform` or a callable that
returns an aligned copy can orient the structure before rendering. Named view
presets accept a four-number rotation quaternion or an eight-number 3Dmol view.

## Compatibility imports

The preferred public imports come from `biotools.structure`. The historical
`biotools.pdbtools` module remains a compatibility facade for existing code.
