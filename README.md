# biotools

![biotools logo showing an amino-acid sequence folding into a protein structure](assets/biotools-logo.png)

`biotools` is a small Python toolkit for recurring structural-biology and
bioinformatics tasks. It combines sequence alignment and FASTA utilities with
helpers for downloading, inspecting, transforming, aligning, and visualizing
protein structures through Biopython.

The project is intended as a practical collection of reusable building blocks
for analysis scripts and notebooks rather than as a command-line application.

## Features

### Protein structures

The `biotools.structure` package provides utilities to:

- download structures from the Protein Data Bank, with automatic PDB-to-mmCIF
  fallback;
- retrieve source organisms and deposition/release dates from the RCSB Data
  API;
- load and save PDB files and convert mmCIF structures to PDB;
- extract, rename, trim, and safely renumber chains and residues;
- obtain SEQRES records and amino-acid sequences;
- calculate RMSD and superimpose complete structures;
- align complete target structures from homologous chain correspondences;
- identify and geometrically characterize residue contacts between chains;
- assign secondary structure and solvent accessibility with DSSP;
- center structures or orient them along their principal axes; and
- create interactive `py3Dmol` structure views.

Operations that rename chains or residues use collision-safe intermediate IDs,
including chain swaps such as `A -> B` and `B -> A`.

### Sequences

The `biotools.sequence` package supports:

- global protein-sequence alignment using Biotite;
- alignment scores, identity, and similarity calculations;
- reading and writing simple FASTA files; and
- creating and searching local protein BLAST databases.

The implementation is organized into focused modules such as
`biotools.structure.alignment`, `biotools.structure.chains`,
`biotools.sequence.alignment`, and `biotools.sequence.blast`. The original
`biotools.pdbtools` and `biotools.seqtools` modules remain available as
backward-compatible import facades.

### Molecular dynamics

`biotools.mdtools` contains OpenMM-based molecular-dynamics preparation
helpers. Its `fix_pdb()` workflow uses PDBFixer to replace nonstandard
residues, remove unwanted heterogens, complete missing atoms, and add
hydrogens. `model_solvent()` adds pH-dependent hydrogens and explicit solvent;
`minimize()` performs force-field energy minimization.

## Requirements

- Python 3.11 or newer
- Biopython
- Biotite
- NumPy
- pandas
- py3Dmol

Local BLAST searches additionally require the NCBI BLAST+ executables
`makeblastdb` and `blastp` to be available on `PATH`. Secondary-structure
assignment requires DSSP to be installed, with either the `dssp` or `mkdssp`
executable available on `PATH`. The molecular-dynamics tools are optional and
require OpenMM, PDBFixer, Matplotlib, and the CUDA 12 support packages included
in the `md` extra. OpenMM-backed contact topology without CUDA is available
through the separate `contacts` extra.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd biotools
python -m pip install -e .
```

This default installation omits the MD tools and their large CUDA
dependencies. Contact characterization uses built-in protein bond templates by
default. Install the optional `contacts` extra to enable the OpenMM topology
backend without CUDA:

```bash
python -m pip install -e ".[contacts]"
```

To include `biotools.mdtools`, install the optional `md` extra:

```bash
python -m pip install -e ".[md]"
```

Both optional feature sets can be installed together. The package manager
merges the shared OpenMM requirement and installs it only once:

```bash
python -m pip install -e ".[md,contacts]"
```

With [uv](https://docs.astral.sh/uv/), the project environment can instead be
created without MD support from the lockfile:

```bash
uv sync
```

Add the optional MD dependencies with:

```bash
uv sync --extra md
```

For OpenMM contact topology, or both optional feature sets, use:

```bash
uv sync --extra contacts
uv sync --extra md --extra contacts
```

## Quick start

### Download and inspect a structure

```python
from biotools.structure import get_aa_sequence, get_pdb_structure

structure = get_pdb_structure("1crn", target_folder="structures")
sequences = get_aa_sequence(structure, show_gaps=False)

for chain_id, amino_acids in sequences.items():
    print(chain_id, amino_acids)
```

PDB/`.ent` is the default download format and mmCIF is used as a fallback. To
try mmCIF first, pass `prefer_mmcif=True`. The format-specific helpers
`get_pdb_structure_as_pdb()` and `get_pdb_structure_as_mmcif()` are also
available when no automatic fallback is desired.

### Fetch RCSB entry metadata

```python
from biotools.pdbtools import get_pdb_metadata

metadata = get_pdb_metadata("4HHB")

print(metadata.pdb_id)       # 4HHB
print(metadata.organisms)    # ("Homo sapiens",)
print(metadata.deposited)    # datetime.date(1984, 3, 7)
print(metadata.released)     # datetime.date(1984, 7, 17)
```

Organisms are the unique scientific source-organism names across the entry's
polymer entities. Entries without a published initial release date return
`released=None`.

### Assign secondary structure with DSSP

DSSP must be installed separately and exposed as `dssp` or `mkdssp` on
`PATH`. The function accepts PDB and mmCIF files:

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

`secondary_structure` concatenates the DSSP assignments for all returned
residues. `relative_sasa` and `absolute_sasa` contain the corresponding values
in the same order; absolute SASA is reported in Å². Some generated PDB files,
including files produced by PeptideBuilder, omit the records expected by DSSP.
When necessary, biotools passes DSSP a temporary copy with compatibility
`HEADER` and `CRYST1` records. The original PDB file is not modified.

### Align a complete structure from homologous chains

```python
from biotools.structure import align_homologs, load_pdb_from_file

reference = load_pdb_from_file("reference.pdb")
mobile = load_pdb_from_file("mobile.pdb")

# Chains A and B determine the transformation. The transformation itself is
# applied to a copy of the complete mobile structure.
aligned = align_homologs(reference, mobile, chain1="A", chain2="B")
```

### Safely rename and renumber chains

```python
from biotools.structure import rename_chain, reset_index

# Simultaneous swaps and chained renames are handled without ID collisions.
rename_chain(structure, {"A": "B", "B": "A"})
reset_index(structure)
```

### Characterize contacts between protein chains

The default topology backend uses built-in protein bond templates and requires
no optional dependencies:

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

After installing `biotools[contacts]`, OpenMM can provide the standard bond
topology and disulfide assignments. The backend is selected explicitly so that
results do not depend on which optional packages happen to be installed:

```python
contacts = characterize_chain_contacts(
    structure,
    "A",
    "B",
    topology_backend="openmm",
)
```

### Compare protein sequences

```python
from biotools.sequence import (
    global_alignment_identity,
    global_alignment_seqs,
    global_alignment_similarity,
)

sequence_a = "MKTAYIAKQRQISFVKSHFSRQ"
sequence_b = "MKTAYIAKQRTISFVKSHFSRQ"

aligned_a, aligned_b = global_alignment_seqs(sequence_a, sequence_b)
identity = global_alignment_identity(sequence_a, sequence_b)
similarity = global_alignment_similarity(sequence_a, sequence_b)

print(aligned_a)
print(aligned_b)
print(f"Identity: {identity:.1f}%")
print(f"Similarity: {similarity:.1f}%")
```

### Search a local BLAST database

```python
from biotools.sequence import BlastSearch

database = BlastSearch(
    input_fasta="proteins.fasta",
    db_name="proteins",
    description="Example protein database",
)
hits = database.search("query.fasta", evalue=1e-10, min_coverage=0.9)
print(hits)
```

### Repair a PDB for simulation

```python
from biotools.mdtools import fix_pdb

fixed_file = fix_pdb(
    "input.pdb",
    "fixed.pdb",
    keep_water=True,
    ph=7.0,
)
```

Missing whole residues are not modeled by default. Enable this explicitly when
appropriate for the intended simulation:

```python
fix_pdb("input.pdb", "fixed.pdb", add_missing_residues=True)
```

### Solvate and minimize a model

```python
from biotools.mdtools import minimize, model_solvent

model_solvent(
    "fixed.pdb",
    "solvated.pdb",
    ph=7.4,
    padding_nm=1.0,
    ionic_strength_molar=0.15,
)

minimize(
    "solvated.pdb",
    "minimized.pdb",
    max_iterations=1000,
)
```

OpenMM can occasionally stop its L-BFGS optimizer before reaching the
requested objective-gradient tolerance. Optional optimizer restarts continue
from the current coordinates in the same OpenMM context:

```python
result = minimize(
    "solvated.pdb",
    "minimized.pdb",
    max_iterations=1000,
    max_optimizer_restarts=2,
    return_diagnostics=True,
)

print(result.optimizer_restarts)
print(result.termination_reason)
print(result.converged)
```

With diagnostics enabled, the minimizer also retains the energy and objective
RMS-gradient history for plotting.

Restarts only occur for `optimizer_stopped` with satisfied constraints and an
objective RMS gradient above the requested tolerance. They are not attempted
after `max_iterations`, for unsatisfied constraints, or after convergence. The
default `max_optimizer_restarts=0` preserves the single-attempt behavior.

### Equilibrate a minimized structure

NVT and NPT are both common equilibration ensembles. A typical explicit-solvent
workflow first uses NVT to bring the system to the target temperature, then NPT
to relax density and box volume at the target pressure. NPT requires periodic
box vectors; NVT also supports nonperiodic structures.

```python
from biotools.mdtools import equilibrate, soft_equilibrate_nvt

soft_nvt = soft_equilibrate_nvt(
    "minimized.pdb",
    "nvt.pdb",
    initial_temperature_k=50.0,
    temperature_k=300.0,
    initial_timestep_fs=0.5,
    timestep_fs=2.0,
    heating_steps=50_000,
    max_steps=500_000,
    state_output_file="nvt-state.xml",
    checkpoint_output_file="nvt.chk",
)

npt = equilibrate(
    "nvt.pdb",
    "equilibrated.pdb",
    ensemble="NPT",
    state_input_file="nvt-state.xml",
    temperature_k=300.0,
    pressure_bar=1.0,
    max_steps=1_000_000,
    state_output_file="npt-state.xml",
    checkpoint_output_file="npt.chk",
)

print(npt.successful, npt.termination_reason, npt.steps)
print(npt.assessment.criteria)
print(npt.assessment.metrics)
print(npt.final_sample.density_g_ml)
print(npt.final_sample.pressure_bar)
print(npt.state_path, npt.checkpoint_path)
```

### Plot minimization and equilibration diagnostics

The optional `md` dependencies include Matplotlib. `plot_md_result()` creates
energy and objective-gradient plots for a minimization result. For an
equilibration result it creates temperature and energy plots, plus pressure for
NPT data:

```python
from biotools.mdtools import plot_md_result

min_figure, min_axes = plot_md_result(result)
min_figure.savefig("minimization.png", dpi=150)

npt_figure, npt_axes = plot_md_result(npt)
npt_figure.savefig("npt-equilibration.png", dpi=150)
```

The function returns the Matplotlib figure and axes without calling `show()`,
so notebooks, scripts, and GUI applications can decide how to display or save
the plot.

`soft_equilibrate_nvt()` starts by assigning velocities at the lower initial
temperature. It linearly increases both thermostat temperature and integrator
timestep over `heating_stages` without rebuilding the OpenMM Context or
reinitializing velocities. Adaptive convergence monitoring starts only after
the heating ramp reaches the requested target values. `max_steps` is the hard
limit for heating and subsequent adaptive equilibration combined.

The simulation runs in blocks (`check_interval_steps=5000` by default). After
each block, the default stability monitor checks a rolling window. NVT checks
the mean temperature and the temperature and potential-energy trends. NPT also
checks the relative box-volume trend and volume fluctuations. Several
consecutive stable windows are required, so a single favorable sample does not
stop the run. Instantaneous pressure is recorded for analysis but is not used
as a hard criterion because its equilibrium fluctuations are large. If the
criteria are not met, the run stops at `max_steps` and returns
`successful=False` while still writing the final structure and retaining all
sampled diagnostics.

Both equilibration functions always write the requested PDB. Optional
`state_output_file` and `checkpoint_output_file` arguments additionally save
an OpenMM XML State and binary checkpoint. The XML State includes positions,
velocities, periodic box vectors, and simulation time and is suitable for
transferring state. A checkpoint retains the complete Context and integrator
state for exact continuation with the same compatible OpenMM System.

`equilibrate()` can resume from either format. Use an XML State when changing
from NVT to NPT, since the NPT System adds a barostat. Use a checkpoint when
continuing the same ensemble with the same force field, constraints, integrator,
and barostat configuration:

```python
# Portable transfer of positions, velocities, box, time, and step count.
npt = equilibrate(
    "nvt.pdb",
    "npt.pdb",
    ensemble="NPT",
    state_input_file="nvt-state.xml",
)

# Exact continuation of a compatible NPT Context.
continued = equilibrate(
    "npt.pdb",
    "npt-continued.pdb",
    ensemble="NPT",
    checkpoint_input_file="npt.chk",
)
```

The two input options are mutually exclusive. `max_steps` always specifies the
additional budget for the current call. `result.initial_step` and
`result.final_step` expose the cumulative OpenMM step numbers, while
`result.steps` records how many new steps this call executed.

The tolerances can be changed with `EquilibrationCriteria`. For
system-specific convergence definitions, pass a callback as `monitor`. It is
called after every block with an `EquilibrationProgress` object:

```python
from biotools.mdtools import EquilibrationAssessment, equilibrate

def monitor(progress):
    if progress.current_step >= 100_000:
        return EquilibrationAssessment(
            stop=True,
            successful=True,
            reason="custom_criteria_met",
            criteria={"custom": True},
        )
    return None

result = equilibrate(
    "minimized.pdb",
    "equilibrated.pdb",
    ensemble="NVT",
    monitor=monitor,
)
```

## Logging

The library uses Python's standard `logging` package and does not configure
global handlers. Progress logging is enabled by default (`verbose=True`).
Pass `verbose=False` to suppress a function's messages. Applications must
also enable informational logging to display them:

```python
import logging
from biotools.mdtools import minimize

logging.basicConfig(level=logging.INFO)

minimize(
    "solvated.pdb",
    "minimized.pdb",
)
```

## Development status

`biotools` is under active development. APIs may still evolve, and test
coverage is currently limited. Validate results carefully before using them in
production analysis pipelines.
