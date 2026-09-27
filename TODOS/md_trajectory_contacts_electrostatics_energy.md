# Trajectory Contacts, Electrostatics, and Direct Interaction Energies

**Status:** Draft. The APIs and scores described here have not yet been
implemented. The existing eight contact types remain available for snapshot
analysis.

## Goal and Scope

The MD module should provide a clean, efficient interface for evaluating the
eight existing contact types across trajectories. This should support a
separate electrostatics module and an optional energy score broken down into
individual components.

The three result levels must remain separate:

1. A **contact** is a geometric or chemical classification in one frame.
2. A **direct energy score** is the nonbonded interaction between two selected
   atom groups in a given conformation.
3. A **binding free energy** additionally accounts for solvent, reorganization,
   unbound states, and entropy, and is outside the scope of this work package.

A contact occupancy or mean trajectory energy must therefore not be described
as a binding probability or binding free energy.

## 1. Optimized Trajectory Analysis of the Eight Contact Types

### 1.1 Public API

Trajectory analysis belongs in the MD package and uses the existing contact
chemistry from `biotools.structure.contacts`. A possible interface is:

```python
from biotools.md_simulations.analysis import analyze_trajectory_contacts

result = analyze_trajectory_contacts(
    topology="system.pdb",
    trajectory="production.xtc",
    partners=("A", "B"),
    profile="refined",
    start=0,
    stop=None,
    step=10,
    periodic=True,
    store_frame_observations=False,
)
```

The core function should also accept a protocol for already opened trajectories
or arbitrary frame iterators. This keeps file formats and readers interchangeable
and avoids requiring large trajectory files in tests.

Proposed modules:

```text
src/biotools/md_simulations/analysis/
    models.py
    trajectory.py
    contacts.py
    electrostatics.py
    energies.py
```

### 1.2 Prepare Only Once

The current snapshot analysis must not be rebuilt in full for every frame.
Prepare the following information once from the topology and then reuse it:

- stable atom and residue indices and `AtomReference` mappings,
- partner selections,
- the bond graph and covalent exclusions,
- donor, acceptor, charged-group, and ring typing,
- ring memberships, atomic radii, and other invariant parameters,
- the contact profile and configuration hash.

Only coordinates and box vectors are updated per frame. Neighbor search,
geometric measurements, and the eight detectors then run on those data. The
prepared state must not mutate either the trajectory or the input topology.

The long-term internal interface should therefore separate chemical identity
from coordinates, for example:

```python
prepared = prepare_trajectory_contacts(topology, partners=("A", "B"))
observations = prepared.analyze_frame(coordinates, box_vectors=box)
```

### 1.3 Periodic Boundary Conditions

The analysis must explicitly define whether and how periodic boundary
conditions are applied. Each frame requires at least:

- reading and validating box vectors,
- placing both partners in a common periodic image,
- using minimum-image distances for atomic contacts,
- preserving molecular integrity rather than wrapping individual atoms apart,
- preserving water identities during transformations for water-bridge analysis.

Missing box information with `periodic=True` must produce a clear error.
Silently analyzing incorrect Cartesian distances is not acceptable.

### 1.4 Per-Frame Results and Online Aggregation

For each residue pair and interaction type, first determine `present=True/False`
in each valid frame. Occupancy is

```text
occupancy = positive_frame_count / valid_frame_count
```

Multiple atomic observations of the same residue pair and type count only once
per frame toward occupancy. Their multiplicity and representative geometries
may be aggregated separately.

The default path should calculate statistics online without keeping all atomic
frame observations in memory. Output must include at least:

```text
partner_a_residue
partner_b_residue
interaction_type
positive_frame_count
valid_frame_count
occupancy
first_observed_frame
last_observed_frame
```

Optionally, frame observations can be stored or streamed for long-format tables.
A single-frame input provides only `present`; a formally resulting occupancy of
0 or 1 should not be presented as temporal occupancy. Multiple independent
structures yield an ensemble frequency, likewise not a temporal occupancy.

### 1.5 Trajectory Adapters and Dependencies

The domain logic must not depend on a specific reader library. An optional
adapter can use MDAnalysis to read DCD/XTC and handle PBC. The base package must
remain importable without MDAnalysis; if needed, this dependency belongs in a
separate extra such as `biotools[trajectory-analysis]`.

The adapter must validate atom ordering, topology, and trajectory compatibility
before analysis. Mapping must not rely solely on mutable PDB serial numbers.

## 2. Separate Electrostatics Module

Electrostatic analysis should not be hidden as another geometric contact in
`ContactObservation`. It should have its own models, configuration, and
provenance while using the same stable atom and residue identities.

A possible interface is:

```python
result = analyze_electrostatics(
    source,
    partners=("A", "B"),
    backend="formal_charge_groups",
    decomposition="residue_pair",
)
```

### 2.1 Options Without OpenMM

#### A. Charged Groups and Geometry

The `formal_charge_groups` backend uses the already typed cationic and anionic
groups. It reports:

- group charges and charge signs,
- distance between charge centers,
- the shortest group-atom distance,
- `attractive`, `repulsive`, or `neutral`,
- optionally, a separate classification of like-charge proximity.

This extends salt-bridge analysis to repulsive charge pairs but does not
calculate energy.

#### B. Coulomb Proxy from Group Charges

An optional `screened_group_coulomb` backend can calculate the reproducible proxy

```text
E = k_e * Q_a * Q_b / (epsilon_r * r) * exp(-kappa * r)
```

The result must include `epsilon_r`, temperature, ionic strength, and the
resulting Debye length. Without screening, omit the exponential term.

This value is a heuristic electrostatics score. It must not be renamed
`electrostatic_energy_kj_mol`, because a group charge placed at its center is
not an atomic force-field energy.

#### C. User-Supplied Partial Charges

The `partial_charge_coulomb` backend accepts an explicit mapping from atom
references to partial charges and calculates pairwise Coulomb terms. Missing
or ambiguous charges produce coverage diagnostics or an error according to an
explicit strictness option. Partial charges must not be guessed from atom names.

This backend must also document whether a dielectric, screening, cutoff, PBC,
and nonbonded exclusions were used.

### 2.2 Options With OpenMM

The `openmm_nonbonded` backend reads charges, Lennard-Jones parameters,
exclusions, and exceptions from a fully parameterized OpenMM `System`. The
preferred input is a `system.xml` saved with the simulation; force-field
parameters cannot reliably be reconstructed from a trajectory alone.

For `NoCutoff`, direct pairwise Coulomb energy can be calculated exactly from
`NonbondedForce` parameters. Cutoff methods must consistently account for PBC,
cutoffs, and switching.

For PME, distinguish two result scopes:

- `direct_space`: the direct or real-space component between selected groups;
- `system_group_difference`: an optional, more expensive group calculation in
  the complete system.

The reciprocal PME component must not be attributed to individual residue pairs
without an unambiguous definition. Residue decomposition under PME must
therefore be labeled as a direct-space score by default. Result objects must
include at least `nonbonded_method`, `component_scope`, and
`reciprocal_electrostatics_included`.

Unsupported additional nonbonded forces, custom parameters, or multiple
conflicting `NonbondedForce` objects must not be silently ignored.

## 3. Direct Energy Score

The energy score is calculated separately from contact occupancies. For two
groups A and B, it includes at least:

```text
electrostatic_energy_kj_mol
lennard_jones_energy_kj_mol
direct_interaction_energy_kj_mol
```

with

```text
direct_interaction_energy_kj_mol =
    electrostatic_energy_kj_mol + lennard_jones_energy_kj_mol
```

For standard Lennard-Jones parameters, each atom pair follows:

```text
E_LJ = 4 * epsilon_ij * ((sigma_ij / r)^12 - (sigma_ij / r)^6)
sigma_ij = (sigma_i + sigma_j) / 2
epsilon_ij = sqrt(epsilon_i * epsilon_j)
```

OpenMM exceptions and covalent exclusions take precedence over these mixing
rules. Strongly positive Lennard-Jones values typically indicate steric overlap;
negative values indicate a favorable dispersion region.

A possible result object is:

```python
@dataclass(frozen=True)
class PairEnergyObservation:
    frame: int
    time_ps: float | None
    partner_a: tuple[AtomReference, ...]
    partner_b: tuple[AtomReference, ...]
    electrostatic_energy_kj_mol: float
    lennard_jones_energy_kj_mol: float
    direct_interaction_energy_kj_mol: float
    backend: str
    component_scope: str
```

For trajectories, aggregate at least the mean, standard deviation, median,
quantiles, number of valid frames, and fraction of energetically favorable
frames. For a single PDB structure, report the three values only for that snapshot.

Negative direct energy indicates a favorable direct nonbonded interaction in
the chosen model; positive energy indicates an unfavorable one. The score
contains no bond, angle, or torsion terms and, in general, no entropy or solvent
reorganization free energy. It is not `binding_free_energy_kj_mol`.

## 4. Combined Workflow

The recommended trajectory workflow is:

```text
Prepare topology and chemistry once
                 |
                 v
Analyze eight contact types per frame
                 |
                 v
Determine the union of relevant residue pairs
                 |
                 v
Calculate energy scores for these pairs in all frames
                 |
                 v
Join occupancies and energy statistics by residue identity
```

If contacts are used to preselect residue pairs, energies for those pairs must
be calculated across all analyzed frames, not only frames with a detected
contact. Otherwise, the results are systematically biased toward favorable
conformations. Record the selection rule and cutoff in the provenance.

A combined table may contain:

```text
partner_a_residue
partner_b_residue
contact_occupancy
salt_bridge_occupancy
vdw_contact_occupancy
mean_electrostatic_energy_kj_mol
mean_lennard_jones_energy_kj_mol
mean_direct_interaction_energy_kj_mol
direct_interaction_energy_std_kj_mol
valid_frame_count
```

Contact occupancy and Lennard-Jones energy are different quantities and must
have different field names and interpretations.

## 5. Reproducibility and MD Integration

The production pipeline should optionally save the serialized OpenMM `System`
as `system.xml` and include its path in `ProductionResult`. Retain the existing
`system_hash` for identity checks, but the hash alone is insufficient to restore
force-field parameters later.

Each analysis result records at least:

- hashes of topology, system, contact configuration, and energy configuration,
- force-field, OpenMM, and reader versions,
- partner selections and residue mappings,
- the analyzed frame range and time step,
- PBC, cutoff, screening, and PME settings,
- the number of skipped frames and coverage diagnostics.

## 6. Tests and Acceptance Criteria

### Trajectory Contacts

- One frame produces the same eight contact classifications as the snapshot API
  with identical coordinates and settings.
- Topology and chemical typing are prepared exactly once per analysis.
- Occupancy counts at most one hit per residue pair, type, and frame.
- Chunking and online aggregation produce the same results as storing all frames.
- Translation, rotation, and equivalent periodic re-imaging leave results unchanged.
- Missing boxes, atom mismatches, and invalid frames produce clear diagnostics.

### Electrostatics and Energy

- Analytical two-particle cases check the sign and magnitude of Coulomb energy.
- The Lennard-Jones test has its minimum at `r = 2**(1/6) * sigma`, with
  `E = -epsilon`.
- Direct energy equals the sum of both components within numerical tolerance.
- OpenMM `NoCutoff` results on CPU/Reference agree with analytical references.
- PBC, cutoffs, exceptions, and excluded pairs have dedicated boundary tests.
- Missing charges and unsupported OpenMM forces are not silently treated as zero.

## 7. Proposed Implementation Steps

1. Introduce trajectory and frame protocols, stable index mappings, and result
   objects.
2. Prepare contact chemistry once and adapt the eight detectors to replaceable
   coordinate arrays.
3. Implement online aggregation, occupancies, PBC handling, and an optional
   MDAnalysis adapter.
4. Add electrostatics without OpenMM, initially as charged-group analysis and a
   clearly labeled Coulomb proxy.
5. Save `system.xml` as an optional MD artifact and implement the
   `openmm_nonbonded` backend.
6. Add Lennard-Jones and direct energy scores and trajectory statistics.
7. Join contact and energy results through stable residue identities, document
   them, and validate them with small reference systems.
