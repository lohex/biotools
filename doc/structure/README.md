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
- [Distances and contacts](#distances-and-contacts)
  - [Residue contacts by distance](#residue-contacts-by-distance)
  - [Geometric contact characterization](#geometric-contact-characterization)
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
- `freesasa` for SASA and interaction-surface calculations.

Contact characterization uses built-in protein bond templates without an
optional dependency. A CPU-only OpenMM topology backend can be installed with:

```bash
python -m pip install -e ".[contacts]"
```

This extra does not install the CUDA packages from `biotools[md]`.

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
)
```

The supported categories are hydrogen bonds, salt bridges, hydrophobic
contacts, van der Waals contacts, parallel and T-shaped π stacking,
cation–π candidates, and single-water bridges. Every category has a Boolean
argument and is enabled by default. With `atomic=False`, one representative
observation is returned per residue pair and category; `atomic=True` retains
individual atom or group observations.

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

`plot_structure()` creates an interactive py3Dmol view. `move_to_center()`
returns a translated copy whose center of mass is at the origin, while
`superimpose_PCA()` can center and orient a copy along the principal axes of
its Cα coordinates.

```python
from biotools.structure import move_to_center, plot_structure, superimpose_PCA

centered = move_to_center(structure)
oriented, shift, rotation = superimpose_PCA(centered)
view = plot_structure(oriented)
view.show()
```

## Compatibility imports

The preferred public imports come from `biotools.structure`. The historical
`biotools.pdbtools` module remains a compatibility facade for existing code.
