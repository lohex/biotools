# Interaction types and calculations

[Back to the structure guide](README.md) · [Documentation overview](../README.md)

`biotools.structure` reports snapshot-based geometric and chemical contact
candidates. A reported contact is neither a binding energy nor proof of a
stable interaction. All distances below are in Å, all angles in degrees, and
the D–H···A convention uses 180° for a linear hydrogen bond.

## Overview

| `interaction_type` | Participants | Default `refined` criterion | Principal reported geometry |
| --- | --- | --- | --- |
| `hydrogen_bond` | donor D–H and acceptor A | D–A ≤ 3.5; D–H···A ≥ 150 | D–A, H–A, angle, directed roles |
| `salt_bridge` | cationic and anionic groups | charge-center distance ≤ 5.5 | center and nearest group-atom distances |
| `hydrophobic_contact` | typed nonpolar C/S atoms | atom distance ≤ 4.0 | atom distance and surface gap |
| `van_der_waals_contact` | supported heavy atoms | `d ≤ r_a + r_b + 0.5` | distance, surface gap, overlap depth, clash flag |
| `pi_stacking_parallel` | two chemical rings | centers ≤ 5.5; planes ≤ 30; lateral offset ≤ 2.0 | angle, both heights/offsets, nearest atoms, planarity |
| `pi_stacking_t_shaped` | face ring and edge ring | centers ≤ 5.5; planes ≥ 60; edge-to-face projection valid | angle, heights/offsets, nearest atoms, assigned geometry |
| `cation_pi_candidate` | positive group and chemical ring | center ≤ 6.0; height ≥ 1.5; projection within ring+tolerance | center distance, plane height, lateral offset |
| `water_bridge` | anchors on both partners and one identical water site | both legs satisfy the hydrogen-bond profile | both leg distances/angles, roles, water identity, evidence |

The numerical values are a versioned starting configuration, not universal
physical constants. Pass a `ContactConfig` to use explicitly validated values
for a particular application.

## Shared preparation and exclusions

Every refined detector uses one prepared selection and one covalent graph:

```python
from biotools.structure import analyze_contacts, prepare_contact_system

prepared = prepare_contact_system(
    structure,
    partners=("A", "B"),
    topology_backend="templates",
)
result = analyze_contacts(prepared, profile="refined")
```

Stable atom references include structure/model, chain, the complete residue ID
(hetero flag, number, insertion code), residue and atom names, and altloc.
Preparation does not modify the input. The dependency-free template backend
assigns standard intramolecular bonds, geometry-supported peptide bonds and
disulfides. The optional OpenMM topology backend is selected explicitly.
Directly bonded (1–2) and angle-related (1–3) pairs are excluded from van der
Waals classification; 1–4 exclusion is configurable. Therefore an interchain
disulfide cannot also become a noncovalent sulfur contact.

Missing hydrogen atoms and unsupported residues are recorded in diagnostics
and coverage. Biopython's selected altloc is used; incompatible conformers are
not mixed. The current chemical support is protein-focused. Modified residues,
metals, halogen bonds, and general ligand chemistry are not silently inferred.

## Hydrogen bonds

For each typed donor atom D with a covalently bound explicit H and each typed
acceptor A, the detector computes

```text
d_DA = ||D - A||
d_HA = ||H - A||
theta = angle(D, H, A)
```

The refined profile accepts `d_DA ≤ 3.5` and `theta ≥ 150`. Donor and acceptor
are stored as structured partner roles, so exchanging or normalizing partners
does not lose direction. Each donor hydrogen initially yields its own atomic
observation. Missing donor H atoms cause a coverage diagnostic rather than an
invented directional bond.

Backbone and side-chain acceptors are typed by residue state. Amide N,
ammonium N, and protonated histidine N are not accepted as ordinary acceptors;
ASH/GLH oxygens bearing H are excluded. Sulfur behavior stays explicit and
conservative.

## Salt bridges

Charge is assigned to complete functional groups, not inferred from element
proximity. The detector calculates the Euclidean distance between group
centers and also the shortest distance between any atoms in the two groups.
Opposite signs and a center distance of at most 5.5 are required.

Supported groups include ARG guanidinium, LYS ammonium, ASP/GLU carboxylate,
and protonated HIP imidazolium. Neutral variants LYN, ASH, and GLH are kept
distinct. HIP remains positively typed even if an input file omits H atoms.
Termini are derived from peptide connectivity and explicit terminal atoms;
the first or last item of a residue subset alone does not create a charge.

## Hydrophobic contacts

Only conservatively typed nonpolar side-chain carbon environments and suitable
neutral sulfur environments participate. Carbonyl carbon and backbone C/CA
are excluded. CYM thiolate, CYX, and sulfur involved in an interresidue S–S
bond are not treated as hydrophobic. For atoms `a` and `b`, the detector stores

```text
surface_gap = d(a, b) - (r_a + r_b)
```

in addition to the ≤ 4.0 contact decision. A contact count is not a buried
nonpolar SASA and is not interpreted as a pairwise hydrophobic energy.

## van der Waals contacts and steric clashes

For supported heavy atoms, proximity is accepted when

```text
d(a, b) ≤ r_a + r_b + tolerance
surface_gap = d(a, b) - (r_a + r_b)
overlap_depth = max(0, -surface_gap)
```

The default tolerance is 0.5. An overlap deeper than 0.4 receives the
`steric_clash` quality flag. A negative gap alone is not an energy and does
not necessarily mean an unfavorable interaction; specific short polar
contacts require chemical interpretation. Unsupported elements are omitted
rather than assigned an arbitrary radius.

## Parallel π stacking

PHE, TYR, and histidine variants contribute one chemical ring. TRP contributes
separate five- and six-membered fused rings with stable ring IDs. Each ring is
fit by singular-value decomposition of its centered atom coordinates. The
normal is the least-variance singular vector; the largest absolute atom-plane
distance is the planarity error. Degenerate, incomplete, or excessively
nonplanar rings are not classified.

For each ring pair the normal sign is removed with `abs(n_a · n_b)`. The
detector calculates center distance, interplane angle, perpendicular height
and lateral offset relative to both planes, nearest ring-atom distance, and
both planarity errors. The default acceptance is center distance ≤ 5.5,
interplane angle ≤ 30, and maximum lateral offset ≤ 2.0.

## T-shaped π stacking

A perpendicular plane angle by itself is insufficient. Both possible
face-ring/edge-ring assignments are checked. The edge-ring center must project
within the face-ring radius plus 0.75, it must have nonzero height above the
face, and the face center must lie near the edge-ring plane. The distance and
angle prefilters are ≤ 5.5 and ≥ 60; the nearest ring atoms must remain at
least 1.5 apart to reject intersecting geometries. This rejects perpendicular
rings that are merely laterally close in center distance. The record retains
the selected face/edge assignment and the same full ring-pair geometry as
parallel stacking.

## Cation–π candidates

Only groups already classified as positively charged are tested. For cation
center `c`, ring center `p`, and unit ring normal `n`:

```text
height = |(c - p) · n|
lateral_offset = ||(c - p) - ((c - p) · n)n||
```

The center distance must be ≤ 6.0, height ≥ 1.5, and the projected center must
fall within the ring radius plus 0.75. Both sides of the ring are equivalent.
This directional filter rejects coplanar and strongly displaced cations that
the former center-distance-only rule accepted. Ammonium and extended
guanidinium group identities remain distinguishable in the record.

## Single-water bridges

A bridge is defined by one water oxygen/site and at least one anchor on each
partner. Every leg stores its own protein and water role:

- protein D–H / water acceptor; or
- water donor / protein acceptor.

Explicit waters are never combined across molecules. An oxygen without water
H can support a protein-donor leg, but a water-donor leg remains a
`distance_candidate` until an orientation is modeled. The compatibility
record therefore exposes `evidence_level` and `unoriented_water`; a typed
result preserves the mediator atom identity.

### When water-bridge analysis runs

The APIs distinguish three operations; none is triggered implicitly by
another one:

1. `characterize_chain_contacts(..., water_bridge=True)` and
   `analyze_contacts(..., water_bridge=True)` only evaluate water residues
   already present in the input model. They never insert a missing water.
2. `orient_existing_waters()` examines those existing oxygen positions and
   constructs possible rigid H₂O orientations while keeping every observed O
   fixed.
3. `propose_bridging_waters()` must be called explicitly to search for a
   missing single-water bridge. It can be used on a dry interface. Existing
   input waters, if any, remain part of the clash environment.

The proposal search tries typed donor/acceptor anchor pairs with one anchor on
each partner. A donor requires an explicit covalently bound H; acceptors follow
the selected contact profile. For each pair separated by no more than twice
the target anchor distance, the search samples the intersection of two
2.8-radius shells. It retains oxygen positions whose two anchor distances lie
between 2.6 and 3.5, removes massive clashes, limits candidate growth, clusters
positions within 0.75, and merges their anchor provenance. It then generates
eight rigid orientations per anchor pair by default. The supported pairings
include water accepting from protein donors, donating to protein acceptors, or
one leg of each kind.

No complete solvent box is needed. The proposal and `geometric` refinement
operate directly on the supplied protein/peptide complex. The
`openmm_rigid_water` backend also deliberately builds a nonperiodic system: it
requires a force-field-parameterizable *solute model*, not a periodically
solvated box. It inserts exactly one candidate water for each independent
evaluation. It does not add bulk solvent, and different proposed sites are not
optimized together.

The local modeling workflow is:

```python
from biotools.structure import (
    evaluate_water_bridges,
    optimize_bridging_waters,
    orient_existing_waters,
    propose_bridging_waters,
)

oriented = orient_existing_waters(prepared)
proposed = propose_bridging_waters(prepared)
optimized = optimize_bridging_waters(prepared, proposed, backend="geometric")
validated = evaluate_water_bridges(prepared, optimized)
```

Existing sites keep O fixed while rigid H₂O orientations use O–H = 0.9572 and
H–O–H = 104.52. Dry-interface proposals are deterministic intersections of
2.8-radius shells around cross-partner donor/acceptor anchors, restricted to
the configured 2.6–3.5 search interval. Massive clashes are rejected and
nearby sites are clustered while their anchor provenance is merged.

### `backend="geometric"`

This backend uses geometry only and is available in the base installation. It
does not construct an OpenMM system and does not calculate molecular-mechanics
energy.

At each oxygen position it creates a rigid water with O–H = 0.9572 and
H–O–H = 104.52. The default eight orientations are derived from directions to
protein acceptors and rotations around the first water-donor direction. Each
orientation is evaluated against both bridge legs and all supported local
heavy atoms. Its objective is

```text
distance_penalty = sum((leg_distance - 2.8)^2)
direction_penalty = sum((max(0, 150 - leg_angle) / 30)^2)
clash_penalty = sum(25 * (overlap - 0.6)^2) for overlap > 0.6
J_geometric = distance_penalty + direction_penalty
              + clash_penalty + capacity_penalty
```

An unavailable leg angle receives a fixed direction penalty of 4. The current
single-water implementation reports `capacity_penalty` separately but leaves
it at zero; explicit multi-anchor capacity modeling remains future work. A
massive oxygen overlap rejects the orientation regardless of the score.

Optimization begins at every proposed O position and tests moves of the oxygen
along ±x, ±y, and ±z. The initial step is 0.20. Whenever no move lowers
`J_geometric`, the step is halved. Search stops below 0.0125 or after 80
iterations. Every trial rebuilds the rigid H positions; solute atoms never
move. A final bridge is accepted only if both legs meet the selected
hydrogen-bond distance and angle criteria and no massive clash remains.

This is a deterministic geometric plausibility refinement, not a force-field
energy, free energy, or occupancy probability. Every score component and O
displacement is retained.

### `backend="openmm_rigid_water"`

This optional backend uses the same proposals, rigid-water orientations,
six-direction coordinate search, step sizes, and final geometric acceptance
rules as `geometric`. The difference is its optimization objective. It builds
one reusable OpenMM topology containing the complete supplied model plus one
candidate H₂O, parameterizes it with Amber14 and TIP3P, and evaluates

```text
J_OpenMM = total_potential_energy_kJ_per_mol + 100 * J_geometric
```

The factor of 100 keeps a low-energy trial from escaping the intended bridge
geometry. For every energy evaluation, all original coordinates are reset to
their unchanged input values and only the candidate O/H coordinates differ.
There is no solute minimization or side-chain relaxation. Existing waters in
the input model remain part of the environment; other proposed waters do not.

The OpenMM system uses `NoCutoff` nonbonded interactions and no periodic box.
It therefore avoids creating an arbitrary small solvent box, but its
unscreened electrostatics can overvalue charged pockets. The supplied solute
must have residue names, atoms, hydrogens, protonation states, and bonds that
Amber14 can parameterize. “Complete model” means complete in that force-field
sense—not surrounded by water. Missing parameters stop with an explicit error.

Total energies may only be compared among candidate positions evaluated with
the same unchanged solute, protonation, atom count, and boundary conditions.
They contain neither a bulk-water reference nor translational/orientational
entropy and must not be interpreted as binding free energies or occupancies.

Install OpenMM through the optional `contacts` dependency group:

```bash
python -m pip install "biotools[contacts]"
```

For an editable repository checkout, use
`python -m pip install -e ".[contacts]"`. A normal `pip install biotools`
does not install this optional group, and the `geometric` backend needs none of
it.

### Independent final evaluation

`evaluate_water_bridges()` recomputes both legs, angles, distances, and clashes
from the final coordinates without using the optimization backend's decision.
This prevents a favorable heuristic or OpenMM score from declaring an invalid
geometry to be a bridge. Source coordinates and experimental occupancy or
B-factor fields are never overwritten.

## Output, aggregation, and profiles

`characterize_chain_contacts()` retains its historical return value: a Python
list whose entries are dictionaries. Existing programs can therefore continue
indexing records such as `contacts[0]["interaction_type"]`. New code that
wants explicit types and IDE/type-checker support should use
`prepare_contact_system()` followed by `analyze_contacts()`; this returns the
`ContactAnalysisResult` dataclass containing `ContactObservation` dataclasses.
Changing the old function directly would break existing callers. With
`atomic=True`, the compatibility function returns individual observations.
Residue aggregation keeps the representative record plus observation count,
independent group count, and minimum/median/maximum distance. It does not add
categories as independent energies.

Compatibility records now also contain `geometry_metrics`, `geometry_units`,
`roles`, satisfied/violated criteria, `evidence_level`, `quality_flags`,
aggregation statistics, and `rule_profile`. The typed `analyze_contacts()`
result adds immutable identities, diagnostics, coverage, a configuration hash,
backend versions, and an input hash. Conversion to a plain dictionary and back
does not lose information:

```python
serialized = result.to_dict()
restored = ContactAnalysisResult.from_dict(serialized)
assert restored == result
```

The plain dictionary can, for example, be passed directly to `json.dumps()`
or stored by another serializer that supports ordinary Python containers.

`aggregate_contact_features()` produces the versioned
`contact_features_v1` schema with per-type counts and distance summaries while
retaining all contributing member IDs. Optional water results add distinct
site and accepted-site counts; alternative H orientations do not inflate the
site count.

Two built-in profiles make behavior changes explicit:

- `refined` is the default and applies the chemical/geometric refinements
  described here.
- `legacy` retains the former 3.0/150 hydrogen-bond rule and
  center-distance candidate behavior for aromatic interactions where
  applicable.

For reproducible analyses, always store the profile name and configuration
hash with downstream features.
