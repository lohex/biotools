# Molecular-dynamics module

[Back to the main README](../../README.md) · [Documentation overview](../README.md)

## Inhaltsverzeichnis

- [Installation](#installation)
- [Workflow overview](#workflow-overview)
- [Prepare a structure](#prepare-a-structure)
  - [Repair a PDB](#repair-a-pdb)
  - [Add solvent and ions](#add-solvent-and-ions)
- [Energy minimization](#energy-minimization)
  - [Basic minimization](#basic-minimization)
  - [Diagnostics and optimizer restarts](#diagnostics-and-optimizer-restarts)
- [Equilibration](#equilibration)
  - [NVT and NPT](#nvt-and-npt)
  - [Gentle NVT heating](#gentle-nvt-heating)
  - [Default convergence monitoring](#default-convergence-monitoring)
  - [Custom convergence monitoring](#custom-convergence-monitoring)
- [States, checkpoints, and continuation](#states-checkpoints-and-continuation)
- [Diagnostic plots](#diagnostic-plots)
- [Platform selection](#platform-selection)
- [Logging](#logging)

`biotools.mdtools` provides OpenMM-based structure preparation, energy
minimization, NVT/NPT equilibration, continuation files, and diagnostic plots.

## Installation

The module is optional. Install it, including OpenMM, PDBFixer, and CUDA 12
support packages, with:

```bash
python -m pip install -e ".[md]"
```

With uv:

```bash
uv sync --extra md
```

The base installation deliberately omits these large dependencies. The
separate `contacts` extra is sufficient only for the optional CPU OpenMM
topology backend used by structure contact characterization; it does not
provide PDBFixer or the complete MD installation.

## Workflow overview

A typical explicit-solvent workflow is:

1. repair the input structure with `fix_pdb()`;
2. add hydrogens, water, and ions with `model_solvent()`;
3. remove unfavorable contacts with `minimize()`;
4. gently heat and equilibrate at constant volume with
   `soft_equilibrate_nvt()`; and
5. equilibrate density and box volume with `equilibrate(..., ensemble="NPT")`.

The appropriate preparation, force field, protonation state, ensembles, and
convergence criteria depend on the scientific system. The helpers do not
replace validation of the resulting model and trajectory.

## Prepare a structure

### Repair a PDB

`fix_pdb()` uses PDBFixer to replace nonstandard residues, remove unwanted
heterogens, complete missing atoms, and add hydrogens:

```python
from biotools.mdtools import fix_pdb

fixed_file = fix_pdb(
    "input.pdb",
    "fixed.pdb",
    keep_water=True,
    ph=7.0,
)
```

Missing whole residues are not modeled by default because reconstructing them
can materially change the model. Enable this explicitly when appropriate:

```python
fix_pdb("input.pdb", "fixed.pdb", add_missing_residues=True)
```

`add_missing_residues=True` requires missing-atom completion to remain
enabled.

### Add solvent and ions

`model_solvent()` adds pH-dependent hydrogens and an explicit solvent box:

```python
from biotools.mdtools import model_solvent

solvated_file = model_solvent(
    "fixed.pdb",
    "solvated.pdb",
    ph=7.4,
    padding_nm=1.0,
    ionic_strength_molar=0.15,
)
```

The defaults use Amber14 protein parameters and TIP3P-FB water, neutralize the
system, and use sodium and chloride ions. Force-field files, water model, ions,
and ID preservation can be configured through keyword arguments.

## Energy minimization

### Basic minimization

```python
from biotools.mdtools import minimize

minimized_file = minimize(
    "solvated.pdb",
    "minimized.pdb",
    max_iterations=1000,
)
```

Without diagnostics, `minimize()` returns the output `Path` and preserves its
original lightweight behavior.

### Diagnostics and optimizer restarts

OpenMM can stop its L-BFGS optimizer before reaching the requested objective
gradient tolerance. Diagnostics expose the termination reason and iteration
history. Optional optimizer restarts continue from the current coordinates in
the same OpenMM context:

```python
minimization = minimize(
    "solvated.pdb",
    "minimized.pdb",
    max_iterations=1000,
    max_optimizer_restarts=2,
    return_diagnostics=True,
)

print(minimization.initial_energy_kj_mol)
print(minimization.final_energy_kj_mol)
print(minimization.optimizer_restarts)
print(minimization.termination_reason)
print(minimization.converged)
```

With `return_diagnostics=True`, the returned `MinimizationResult` also retains
energy, objective RMS-gradient, restraint, and constraint-error samples for
plotting.

Restarts occur only after `optimizer_stopped` when constraints are satisfied
but the objective RMS gradient remains above the requested tolerance. They are
not attempted after `max_iterations`, after a constraint failure, or after
convergence. The default `max_optimizer_restarts=0` preserves single-attempt
behavior.

## Equilibration

### NVT and NPT

NVT keeps particle count, volume, and temperature fixed. NPT additionally
controls pressure and allows the periodic box to change. A common sequence is
NVT heating followed by NPT density relaxation. NPT requires periodic box
vectors; NVT can also run nonperiodic structures.

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

Both equilibration functions always write the requested final PDB, including
when the stability criteria are not reached before `max_steps`.

### Gentle NVT heating

`soft_equilibrate_nvt()` assigns velocities at the lower initial temperature.
It linearly increases thermostat temperature and integrator timestep over
`heating_stages`, without rebuilding the OpenMM context or reinitializing
velocities. Adaptive convergence monitoring starts after the ramp reaches the
target values. `max_steps` is the combined hard limit for heating and
subsequent equilibration.

### Default convergence monitoring

Equilibration runs in blocks (`check_interval_steps=5000` by default). After
each block, the default `StabilityMonitor` evaluates a rolling window:

- NVT checks mean temperature plus temperature and potential-energy trends;
- NPT additionally checks relative box-volume trend and volume fluctuations.

Several consecutive stable windows are required, so a single favorable sample
does not stop the run. Instantaneous pressure is recorded for analysis but is
not a hard criterion because its equilibrium fluctuations are large. If the
criteria are not met, the result has `successful=False` and
`termination_reason="max_steps"` while retaining all sampled diagnostics.

Use `EquilibrationCriteria` to adjust the tolerances:

```python
from biotools.mdtools import EquilibrationCriteria, equilibrate

criteria = EquilibrationCriteria(
    temperature_tolerance_k=5.0,
    required_stable_windows=3,
)
custom_result = equilibrate(
    "minimized.pdb",
    "equilibrated.pdb",
    ensemble="NVT",
    criteria=criteria,
)
```

### Custom convergence monitoring

For system-specific definitions, pass a callback as `monitor`. It is called
after every block with an `EquilibrationProgress` object:

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

A custom monitor and `criteria` are mutually exclusive.

## States, checkpoints, and continuation

Optional `state_output_file` and `checkpoint_output_file` arguments save two
different continuation formats:

- an OpenMM XML State stores positions, velocities, periodic box vectors,
  simulation time, and step count, and is suitable for transferring state;
- a binary checkpoint retains the complete context and integrator state for
  exact continuation with the same compatible OpenMM system.

Use an XML State when changing from NVT to NPT because the NPT system adds a
barostat. Use a checkpoint when continuing the same ensemble with matching
force field, constraints, integrator, and barostat configuration:

```python
from biotools.mdtools import equilibrate

# Portable transfer into a differently configured NPT system.
npt = equilibrate(
    "nvt.pdb",
    "npt.pdb",
    ensemble="NPT",
    state_input_file="nvt-state.xml",
)

# Exact continuation of a compatible NPT context.
continued = equilibrate(
    "npt.pdb",
    "npt-continued.pdb",
    ensemble="NPT",
    checkpoint_input_file="npt.chk",
)
```

The two input formats are mutually exclusive. `max_steps` is always the
additional step budget for the current call. `result.initial_step` and
`result.final_step` expose cumulative OpenMM step numbers, while
`result.steps` contains the number executed by that call.

## Diagnostic plots

`plot_md_result()` returns a Matplotlib figure and axes without calling
`show()`. Minimization results have energy and objective-gradient plots.
Equilibration results have temperature and energy plots, plus pressure when
NPT pressure samples are present:

```python
from biotools.mdtools import plot_md_result

min_figure, min_axes = plot_md_result(minimization)
min_figure.savefig("minimization.png", dpi=150)

npt_figure, npt_axes = plot_md_result(npt)
npt_figure.savefig("npt-equilibration.png", dpi=150)
```

Minimization plotting requires a result created with
`return_diagnostics=True` so its iteration samples are available.

## Platform selection

Simulation functions accept `platform_name` and `platform_properties` to
select and configure an OpenMM platform explicitly. If no platform is given,
OpenMM performs its normal platform selection. For reproducible execution,
record the chosen platform and relevant precision settings alongside the
simulation parameters.

## Logging

The MD helpers use Python's standard `logging` package. Progress logging is
enabled by default through `verbose=True`; pass `verbose=False` to an
individual call to suppress it. Applications must enable informational
logging to display progress:

```python
import logging
from biotools.mdtools import minimize

logging.basicConfig(level=logging.INFO)
minimize("solvated.pdb", "minimized.pdb")
```
