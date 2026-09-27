# Strategy for Analyzing Peptide–MHC Binding

## Goal

The planned analysis should determine:

- which peptide and MHC residues contact each other,
- which chemical interaction types these contacts form, and
- how stable the interactions are across a molecular dynamics trajectory.

Three statements must be clearly distinguished:

1. A **contact** describes spatial proximity.
2. A **direct interaction energy** describes the nonbonded energy between
   selected atom groups in a given conformation.
3. A **contribution to binding free energy** additionally accounts for solvent,
   conformational changes, and entropy.

Steps 1–3 form the intended main workflow. Steps 4 and 5 are optional energy
extensions.

## Plan

### 1. Generate Trajectories With OpenMM

The starting points are repaired, protonated, solvated, and sufficiently
minimized pMHC structures. OpenMM should then perform equilibration and a
production simulation for each structure.

Save at least:

- a topology with stable identification of the MHC and peptide chains,
- a periodic trajectory, preferably DCD or XTC,
- simulation parameters such as temperature, pressure, time step, and force field,
- checkpoints or OpenMM States for continuation, and
- mappings from atom and residue indices to the original pMHCdb positions.

A single minimized structure may be used for an initial static analysis.
Statements about interaction stability should, however, rely on multiple frames
and preferably multiple independent simulations.

### 2. Trajectory Input and Geometric Analysis

An optional [MDAnalysis](https://www.mdanalysis.org/pages/documentation/)
adapter provides trajectory loading, selections, and periodic coordinate
processing. biotools performs the geometric and chemical contact analysis using
its own detectors. Keep the analysis core independent of the trajectory reader.

Periodic artifacts must be corrected before contact analysis. MHC and peptide
must be placed in the same periodic image, centered, and unwrapped if necessary.
Exclude water and ions from direct pMHC contacts, but retain them for
water-mediated contacts.

For each peptide–MHC residue pair, determine at least:

- the minimum heavy-atom distance per frame,
- contact presence based on a documented cutoff,
- contact occupancy as the fraction of analyzed frames,
- hydrogen-bond occupancy,
- salt-bridge occupancy,
- water-mediated hydrogen bonds, and
- optionally, the change in solvent-accessible surface area when MHC and peptide
  are brought together.

Cutoffs, angle criteria, PBC handling, and the analyzed frame range must be
stored with the results. Count contacts per pMHC structure and residue pair,
rather than as an unweighted collection of all atom pairs.

### 3. Chemical Interaction Analysis With biotools

Use the existing contact functionality in `biotools.structure.contacts`, built
around `prepare_contact_system()` and `analyze_contacts()`, as the basis for
trajectory analysis. Extend this functionality to reuse prepared topology and
chemical typing while updating coordinates and box vectors for each frame.
The trajectory API and occupancy aggregation still need to be implemented as
specified in [the trajectory-analysis plan](md_trajectory_contacts_electrostatics_energy.md).

Use the same eight interaction types and configurable `refined` or `legacy`
profiles as the snapshot analysis:

- hydrogen bonds with donor/acceptor direction,
- salt bridges between oppositely charged groups,
- hydrophobic contacts,
- van der Waals contacts,
- parallel π stacking,
- T-shaped π stacking,
- cation–π candidates, and
- water bridges, where they can be reliably determined for the particular run.

Retain biotools atom and residue identities, named geometry measurements,
chemical roles, evidence levels, and coverage diagnostics. Analyze waters
present in each frame for water-bridge occupancy; locally proposed or optimized
water sites must remain separately labeled modeling hypotheses.

Aggregate occupancy online by structure, peptide position, MHC position, and
interaction type. Count at most one positive observation per residue pair and
type in each valid frame. Optionally store or stream per-frame results in a
long-format table containing at least:

```text
structure_id
frame
peptide_position
peptide_residue
mhc_position
mhc_residue
interaction_type
present
minimum_distance_A
```

Also produce a compact table with the number and fraction of positive frames.
This can support contact and occupancy heatmaps along the standardized MHC
binding groove.

## Optional Extensions

### 4. Direct Nonbonded Interaction Energies With OpenMM

For selected peptide–MHC residue pairs, evaluate direct electrostatic and
Lennard-Jones contributions across multiple trajectory frames. OpenMM exposes
atomic parameters through `NonbondedForce`; `CustomNonbondedForce` and
interaction groups can restrict calculations to specific atom groups.

Report separately:

```text
mean_electrostatic_energy_kj_mol
mean_vdw_energy_kj_mol
mean_direct_interaction_energy_kj_mol
standard_deviation_kj_mol
```

Label these values as **direct interaction scores**. With PME, the reciprocal
electrostatic component cannot be uniquely assigned to individual residue pairs.
Water-mediated effects, reorganization, and entropy are also excluded. These
values must therefore not be described as residue contributions to binding
free energy.

### 5. Residue Energy Contributions and Mutation Analysis

Only after contact and occupancy analysis should particularly interesting
peptide or MHC positions receive more detailed energetic investigation.

Possible methods include:

- **FoldX or Rosetta:** rapid static interface evaluation and alanine scanning
  to prioritize candidates,
- **MM/PBSA or MM/GBSA:** approximate binding energies and per-residue
  decomposition from a trajectory ensemble,
- **alchemical calculations with OpenMM:** more expensive calculations of
  mutation contributions as a difference between bound and unbound states.

For alanine scanning, the relevant quantity is

\[
\Delta\Delta G_\mathrm{bind}
=
\Delta G_\mathrm{mutation,complex}
-
\Delta G_\mathrm{mutation,unbound}.
\]

Only this comparison accounts for a mutation changing both the bound complex
and the free state. Because of their computational cost, restrict these
calculations to positions prioritized through steps 1–3.

## Quality Control and Interpretation

- Explicitly validate MHC and peptide selections for every run.
- Analyses require consistent residue and position mappings despite solvation
  and newly generated chain IDs.
- Validate explicit hydrogens and bond information required by the selected
  biotools contact profile; report missing chemistry through coverage diagnostics.
- With identical coordinates and settings, single-frame trajectory analysis
  must reproduce the existing biotools snapshot classifications.
- Report contact occupancies with the number of frames and simulation duration.
- For multiple replicates, report the mean and variability or a confidence
  interval.
- Direct contacts, water bridges, interaction scores, and free energies must not
  be interpreted as interchangeable quantities.
