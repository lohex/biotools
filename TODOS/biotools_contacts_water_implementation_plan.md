# biotools: Contact Analysis and Locally Optimized Bridging Waters

Implementation plan and validation framework, September 8, 2026.

**Status (repository review, September 27, 2026):** Partially implemented.
Typed contact results, profiles, shared topology, refined detectors for all eight
contact types, feature aggregation, observed-water orientation, single-water
proposals, geometric refinement, and an optional OpenMM energy backend are
present. The full acceptance criteria below are not yet met: preparation and
atom-mapping policies, conflict handling, optimizer diagnostics and validation,
WaterKit integration, independent scientific benchmarks, and ensemble analysis
remain incomplete or absent. Scientific accuracy targets have not been
established by the existing unit tests. “Contacts” refers to geometric and
chemical candidates; detecting them does not measure a binding energy.

The original findings and API sketches below describe the September 8 baseline
and intended design. They are not a current API reference; implemented profiles
are named `legacy` and `refined`.

**Code baseline:** `lohex/biotools`, commit `ef52e57736c763319c3b4328f5710347f93db25d`. Files reviewed include `_contacts.py`, the structure documentation, contact and distance tests, `md_simulations/preparation.py`, and `pyproject.toml`. [Reference snapshot](https://github.com/lohex/biotools/tree/ef52e57736c763319c3b4328f5710347f93db25d)

## 1. Goals and Decisions

The existing analysis of protein-protein and protein-peptide contacts should become more chemically consistent, report uncertainty explicitly, and retain atomic observations for subsequent ML features. A local search for single-water bridges will add position and orientation optimization.

Recommended decisions:

1. Use a shared, prepared chemistry and topology graph for every detector.
2. Separate geometric observations, interaction classification, scores, and aggregation.
3. Initially preserve existing rules in a versioned `legacy` profile. Calibrate a new profile on a separate development dataset.
4. Orient existing waters first; then propose missing waters.
5. Provide geometric search and local OpenMM optimization as separate backends.
6. **Integrate WaterKit optionally**, initially as an external adapter and comparison baseline. Do not make it a mandatory core dependency.
7. Evaluate software correctness, experimental reconstruction, and usefulness for affinity models separately.

### What “Without Solvent” Can Mean Here

| Starting point | Suitable procedure | Interpretation |
| --- | --- | --- |
| Water oxygens are present, but H atoms are missing | Add H atoms and optimize orientations | A geometrically plausible orientation at a known water site |
| No waters are present; a full solvent box is not desired | Local candidate search, OpenMM optimization, optionally WaterKit | Hypotheses about local water sites |
| No explicit waters should be modeled, even locally | Implicit solvation | No assignment of specific single-water bridges |
| Occupancy probabilities or exchange kinetics are required | Sufficient solvent sampling and an appropriate reference | Model-dependent occupancy, or kinetics with suitable dynamics |

WaterKit uses explicit local waters. It can investigate selected protein regions without simulating the entire protein in a solvent box. This corresponds to the second row, not the third. [WaterKit paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10732097/)

## 2. Specific Findings in the Current Code

| Area | Observation | Implementation consequence |
| --- | --- | --- |
| Hydrogen bonds | Maximum donor-acceptor distance of 3.0 Å and minimum D-H-A angle of 150°; explicit H required | Preserve this profile and test alternative thresholds; distinguish missing H from negative findings |
| Acceptors | Residue/atom tables; some protonation variants are supported | Complete state-dependent rules, including termini and sulfur chemistry |
| Water | Waters without bonded H atoms are skipped entirely | Add a status for unoriented waters and an orientation module |
| Salt bridges | Group rules; terminal groups inferred from the first/last selected position | Derive termini from the covalent graph and capping, not selection order |
| Hydrophobicity | Selected C atoms, plus S from CYS/CYM/CYX/MET | Do not classify charged thiolate sulfur as hydrophobic indiscriminately |
| vdW | Only an upper bound of `r_i + r_j + 0.5 Å` | Distinguish clashes, ordinary contacts, and covalent neighbors |
| Aromaticity | All nine TRP ring atoms form one ring group | Model the two chemical rings separately; distinguish ring and residue aggregation |
| T-stacking | Ring-center distance and interplane angle | Add edge-to-face geometry and nearest-atom distance |
| Cation-pi | Ring-center distance without a directional filter | Add height, projection, and lateral offset |
| Topology | Template backend does not add interresidue bonds; OpenMM can provide standard/disulfide bonds | Implement shared exclusions for covalent contacts |
| Aggregation | One representative per residue pair and contact type | Also retain multiplicity and statistical summaries |
| Intrachain | Normalization swaps atom/residue fields, but leaves direction information in free text unchanged | Store donor/acceptor roles structurally and remap them correctly when swapping partners |

These are code findings, not a complete scientific validation of the detectors. [Contact implementation](https://github.com/lohex/biotools/blob/ef52e57736c763319c3b4328f5710347f93db25d/src/biotools/structure/_contacts.py)

## 3. Architecture and Data Model

Public functions in `biotools.structure` remain available. `_contacts.py` can serve as an internal facade during migration. New components have separate responsibilities:

| Proposed file | Responsibility |
| --- | --- |
| `structure/contacts/models.py` | Identities, observations, diagnostics, and result objects |
| `structure/contacts/config.py` | Versioned parameter profiles and serialization |
| `structure/contacts/chemistry.py` | Donors, acceptors, charged groups, rings, and protonation states |
| `structure/contacts/topology.py` | Templates/OpenMM, interresidue bonds, and covalent exclusions |
| `structure/contacts/neighbors.py` | Reusable neighbor search and, subsequently, PBC |
| `structure/contacts/detectors.py` | Pure detectors for all eight existing contact types |
| `structure/contacts/aggregate.py` | Atom, group, residue, and feature aggregation |
| `structure/water/models.py` | Water sites, orientations, bridge legs, and provenance |
| `structure/water/proposals.py` | Local oxygen candidates and deduplication |
| `structure/water/orientation.py` | Rigid water geometry and orientation optimization |
| `structure/water/optimize.py` | Local energy evaluation and optimization with OpenMM |
| `structure/water/networks.py` | Conflicts, alternative occupancies, and small water networks |
| `structure/water/backends/waterkit.py` | Optional external WaterKit adapter |
| `structure/contact_ensemble.py` | Site assignment and contact frequencies across frames |

Do not split modules solely to increase the file count. Small detectors can initially remain in one file.

### 3.1 Stable Identities

An atom reference includes structure, model, chain, full residue ID including hetero flag and insertion code, atom name, and altloc. A ring or charged group contains its constituent atom references. Keep coordinate indices separate from semantic identities.

Preparation must provide an `original_atom_id -> prepared_atom_id` mapping and provenance for added atoms. Do not identify atoms solely by sequential PDB numbers. Do not combine incompatible altlocs into artificial contacts.

### 3.2 Result Objects

`ContactObservation` should include at least:

- `interaction_type`, references to both partners, and atomic or group identities;
- structured roles such as `donor`, `acceptor`, `cation`, `anion`, and `ring`;
- named geometric quantities with units, rather than a single universal `distance`;
- satisfied/violated criteria and the rule profile used;
- `evidence_level`: `explicit_geometry`, `inferred_hydrogens`, or `distance_candidate`;
- optional quality flags and score components;
- a list of mediating water references or a water-site ID.

`ContactAnalysisResult` adds `observations`, `diagnostics`, `coverage`, `config_hash`, backend versions, and an input hash. Report missing chemistry at the system/site level without creating a record for every conceivable negative atom pair.

`WaterSite` contains the oxygen position, alternative H orientations, both bridge legs with their own distances/angles, model parameters, provenance, optimization status, and competing occupancies. A geometric score is not a probability. Name `crystal_occupancy` separately from any frequency estimated through sampling.

### 3.3 Aggregation Rules

- Atomic observations form the unchanged source layer.
- A geometric contact can satisfy several physical categories; do not treat these as independently additive binding energies.
- For each residue pair and type, retain the atomic count, number of independent groups, min/median/max of relevant quantities, and member IDs.
- For water bridges, count distinct water sites separately from anchor combinations. Do not inflate counts through water-H permutations.
- Generate backward-compatible lists as an explicit projection; preserve the complete new dataset.

## 4. Shared Chemical Preparation

Preparation is an explicit operation on a copy, not a silent side effect of contact analysis.

1. Select the model and biological chains; label crystal contacts separately.
2. Select altlocs consistently or treat them as separate conformations.
3. Build the covalent graph: add peptide bonds only where topology/geometry supports them, preserve chain breaks, and account for disulfides and existing links.
4. Preserve protonation states specified explicitly or supported by H atoms. Report contradictions. Record pH-based additions as modeling assumptions; they do not replace local pKa calculations.
5. Determine termini from the full graph and protecting groups. Selecting a subset of residues does not create new termini.
6. Add hydrogens deliberately. Missing H atoms must not imply that HIP is neutral or that an acceptor is automatically deprotonated.
7. Return `unsupported` for unsupported chemistry. Do not blindly apply standard-name templates to modified residues.

OpenMM can add missing H atoms and allows specific protonation variants. Relevant preparation already exists in this repository; preservation of states and atom mappings needs verification. [OpenMM Modeller](https://docs.openmm.org/latest/userguide/application/03_model_building_editing.html), [biotools preparation](https://github.com/lohex/biotools/blob/ef52e57736c763319c3b4328f5710347f93db25d/src/biotools/md_simulations/preparation.py)

**Covalent exclusions:** Exclude directly bonded and 1-3 pairs from noncovalent contact scoring. Configure the treatment of 1-4 pairs explicitly; apply force-field-specific scaling only during energy evaluation. This also applies across chain boundaries, for example to disulfides. Sequence separation alone is insufficient.

## 5. Refinement of All Existing Contact Types

All new thresholds below are **starting configurations for validation**, not universal physical constants. `legacy_v0_1` reproduces the old rules, including their documented limitations. Corrections to identities and diagnostics must not silently change scientific interpretation.

### 5.1 Hydrogen Bonds

**Chemistry:** Check donor capability together with a bonded H atom. Determine acceptor capability from charge, protonation, and functional group. Do not treat amide N, positively charged ammonium, or protonated histidine N as ordinary acceptors. Model carboxylate/carboxylic acid, phenol/phenolate, thiol/thiolate, and disulfides separately. Handle weak sulfur hydrogen bonds through a separate profile or flag.

**Geometry:** Retain D-A, H-A, and D-H-A. If the acceptor environment is known, also describe orientation relative to its bonding plane or approximate acceptor direction. Angle convention: 180° is linear. Multiple H atoms on one donor initially produce separate observations, followed by documented aggregation.

**Profiles:** Preserve the existing 3.0 Å/150° rule. Additional evaluation profiles could use 3.5 Å/150° and 3.5 Å/130°. Literature and tool definitions differ; ProLIF supports this kind of parameterization. [ProLIF interactions](https://prolif.readthedocs.io/en/latest/source/modules/interaction-fingerprint.html)

**Without H:** Either reconstruct explicitly or report only `distance_candidate`. Do not count the latter as a fully geometry-supported hydrogen bond. Water placement requires a consistent joint orientation.

**Tests:** Linear positive geometry; incorrect angle at the same D-A distance; nonacceptor N; protonation change; H/D isotopes; multiple H atoms; missing H; unsupported residue; threshold ±epsilon; roles after swapping chains.

### 5.2 Salt Bridges

Represent charge on functional groups, rather than inferring it from elements or proximity. Store the charge center, shortest distance between relevant group atoms, and group identities. Optionally record like-charge proximity as a separate repulsive neighborhood, not a salt bridge.

Handle LYS/LYN, ASP/ASH, GLU/GLH, HIS/HID/HIE/HIP, and true versus capped termini explicitly. Do not interpret missing H on explicitly charged HIP as evidence of neutrality. Define the arginine guanidinium center consistently and version that definition.

Initially retain the 5.5 Å group-distance threshold. Test a tighter atom-proximity criterion as an additional feature; do not silently substitute group and atom distances. Calculate Coulomb energy only with validated partial charges in a separate backend.

**Tests:** Both partner orders, protonation, protecting groups, cyclic peptides, chain fragments without artificial termini, and extended charged groups whose center distances differ from nearest-atom distances.

### 5.3 Hydrophobic Contacts

Maintain explicit chemical atom classes: nonpolar C environments, suitable neutral S environments, and optional future halogen classes. Do not indiscriminately include carbonyl C or other strongly polarized C environments. **Do not classify CYM as hydrophobic merely because it contains S.**

Initially retain the 4.0 Å threshold. Store atom distance and the normalized surface separation `d - (r_i + r_j)`. Evaluate solvent-accessible surface contributions separately: contact count and buried nonpolar SASA are different features.

Do not count covalent S-S bonds as noncovalent hydrophobic contacts. Hydrophobicity here is a descriptive classification, not an isolated pairwise hydrophobic energy.

**Tests:** Aliphatic and aromatic C contacts, carbonyl C, methionine, cysteine/thiolate, disulfides, and fully separated partners.

### 5.4 van der Waals Contacts and Clashes

Retain a configurable radius dataset. Add:

`surface_gap = distance - (radius_a + radius_b)`

Report proximity, overlap depth, and a chemistry-dependent `steric_clash` flag. A negative `surface_gap` does not automatically imply an unfavorable clash; short hydrogen bonds and other specific contacts require their own exceptions/tolerances. Check hydrogen-heavy-atom overlaps separately.

Exclude covalent 1-2/1-3 pairs. Do not use a single lower distance threshold as the entire collision model: distinguish radii, chemical context, and optional Lennard-Jones parameters. A geometric vdW contact is not equivalent to a negative Lennard-Jones energy.

**Tests:** Large separation, ordinary contact region, massive overlap, identical coordinates, a valid short hydrogen bond, unsupported element, and a disulfide connecting two chains.

### 5.5 Parallel Pi Stacking

Use chemical rings rather than arbitrary planar groups. Represent PHE/TYR/HIS as individual rings and TRP through its five- and six-membered rings; fused rings share atoms but have separate ring IDs.

For each ring, retain its centroid, SVD plane, normal vector, and planarity error. Do not claim a valid plane for degenerate or incomplete geometry. For a ring pair, calculate:

- the angle between plane normals, accounting for sign equivalence;
- center-to-center distance;
- perpendicular height relative to each plane;
- lateral offset relative to each plane;
- projected overlap of ring polygons and nearest-atom distance.

Keep the existing 5.5 Å/30°/2 Å values as the initial profile. Polygon projection can subsequently accommodate more realistic offset stacking without accepting every pair of nearby parallel planes. Calibrate alternative profiles on real ring contacts.

**Tests:** Centered and offset stacking, excessive offset, excessive height, inverted ring normal, TRP subrings, and nonplanar or incomplete groups.

### 5.6 T-Shaped Pi Stacking

An interplane angle near 90° and a center distance are insufficient. Check both possible `edge_ring`/`face_ring` assignments. Require a nearby edge region of one ring whose projection falls on or near the face of the other ring. Also retain height and nearest-atom distance.

The existing minimum angle of 60° can remain a search prefilter. Accept only after checking edge-to-face geometry and clashes. Label results as candidates until these rules have been calibrated against independent structures.

**Tests:** A genuine edge-to-face contact, perpendicular but laterally separated rings, rings geometrically intersecting each other, and role reversal.

### 5.7 Cation-Pi Contacts

Use only groups classified as positively charged. Use centroid distance as a prefilter, then check distance to the ring plane, lateral offset, and projection into the ring polygon plus tolerance. Both sides of a ring plane are allowed.

Describe point-like ammonium and extended guanidinium groups separately. For the latter, optionally include the group plane and atom-level geometry. The 6 Å distance initially remains a search radius, not the sole acceptance criterion.

**Tests:** Cation above/below the ring face, lateral displacement outside the ring, coplanar placement, a neutralized group, histidine variants, and fused rings.

### 5.8 Single-Water Bridges

A water must connect two anchors on different partners within one joint geometry. Store roles explicitly for each bridge leg: protein donor/water acceptor or water donor/protein acceptor.

The analysis distinguishes:

- an observed O site without a known H orientation;
- a modeled H orientation at an observed O;
- a newly proposed O site;
- a locally optimized water site;
- site occupancy estimated from an ensemble.

A water O without H can already be a distance candidate and an acceptor for a protein D-H group. Infer a complete bridge only if the required roles and joint water orientation are sufficiently determined. Keep predicted and observed provenance distinct.

A water can have more than two contacts. For the MVP, require at least one leg to each partner and retain every additional contact. Penalize implausible donor-capacity violations or multiple assignments along the same H direction; do not automatically declare possible bifurcated hydrogen bonds impossible. Virtual acceptor directions are approximations, not explicit lone pairs in a three-point water model.

**Tests:** All role pairings, the same versus different mediating waters, missing H, mutually incompatible orientations, permuted H labels, more than two anchors, and alternative waters at the same site.

### 5.9 Future New Categories

Halogen bonds, metal coordination, and general ligand chemistry are extensions beyond the eight existing types. They require their own chemical perception, rather than just additional element cutoffs. In the MVP, flag metal-bound waters and bridges near ions separately or exclude them from optimization until suitable parameters are available. Treat disulfides as covalent annotations and exclusions.

## 6. Bridging-Water Placement

### 6.1 Orienting Existing Waters

Initially keep each observed water oxygen O fixed. Place a rigid H2O geometry matching the selected model through rotation. Generate starting orientations from anchor geometries and additional broadly distributed rotations. For closely neighboring waters, optimize orientations jointly or iteratively with multiple restarts.

After selecting good orientations, optionally permit weakly restrained O displacement. Compare the result with the original O position and report the displacement. A benchmark with known O positions is a separate task, not a de novo water benchmark.

### 6.2 Generating New Oxygen Candidates

1. Collect donor/acceptor sites at the selected interface. Derive the ROI from the partners, never from held-out test waters.
2. Prefilter anchor pairs through neighbor search. Derive the search limit from the sum of maximum anchor-O distances, rather than using one fixed atom distance for every type.
3. Sample intersections of distance shells. Where donor directions are known, also generate directed starting points. A midpoint alone is not a sufficient production approach, although it can serve as a simple baseline.
4. Test 2.6-3.5 Å as an initial search interval for ordinary N/O anchors. Configure type-dependent values and search coverage separately from final classification.
5. Filter O candidates against the entire local structure. Do not systematically remove short N/O approaches through an unmodified hard sum-of-vdW-radii filter.
6. Cluster candidates spatially, merge their anchor provenance, and retain alternative orientations.

### 6.3 Orientation and Candidate Score

The degrees of freedom of a rigid single water are translation `t` and rotation `R`. Atom coordinates are generated as `x_k = R q_k + t`. O-H distances and the H-O-H angle come from the selected water model; geometric optimization must not distort them arbitrarily.

Proposed heuristic objective:

`J = clash_penalty + direction_penalty + capacity_penalty - bridge_geometry_reward`

Report every component separately. Use hard rules for massive clashes and smooth distance/angle functions for local optimization. Do not reward an unlimited number of redundant contacts to the same anchor linearly. A hypothetical water placement does not automatically receive a calibrated probability.

Rotating the entire system should yield equivalent results. Anchor-based local coordinate frames help achieve this. Finite orientation/grid sampling can nevertheless introduce small deviations; define numerical tolerances and sampling-convergence tests accordingly.

### 6.4 Local Optimization with OpenMM

**MVP:** Retain the fully parameterized protein/peptide system as the environment. All solute coordinates remain fixed; optimize the position and orientation of one rigid water per hypothesis. This avoids artificial charges created by truncating protein fragments.

Implementation outline:

1. Prepare the protein/peptide topology once, checking atom mappings and parameter coverage. The contact analysis module's OpenMM topology backend alone is not yet an energy backend.
2. Select one water force field consistent with the solute force field. Models with virtual sites require correct site updates; restrict the MVP to a validated three-point model.
3. Use OpenMM to evaluate energies/forces; optimize the rigid water externally over translation and a rotation vector. `scipy.optimize` is a possible optional dependency. This preserves fixed solute coordinates and rigid water geometry exactly.
4. Optimize multiple starting orientations. Configure iteration limits, translation range, and convergence criteria explicitly. A candidate moving outside the ROI is a failed bridge proposal; do not keep it artificially valid through strong anchor springs.
5. Set the full position arrays correctly at every energy evaluation. Solute coordinates remain unchanged. Initially verify gradients with finite differences; then derive translation gradients from summed forces and rotation gradients from torques.
6. For the initial nonperiodic model, enable neither PME nor an arbitrary small water box. An explicitly documented nonperiodic energy model serves as a local plausibility filter. Unscreened electrostatics can overvalue charged pockets; do not infer site occupancies from it.
7. After optimization, re-evaluate every contact, clash, and bridge criterion using the independent detector. Energy reduction alone does not establish success.

**Alternative second stage:** Conventional OpenMM minimization with rigid water constraints and positionally restrained nearby side chains. Fix the distant protein; report movable atoms, boundary conditions, and constraint violations. OpenMM enforces distance constraints during minimization, but the chosen tolerance must still be checked. [LocalEnergyMinimizer](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html)

**Comparable energy quantities:** Candidate energies can be compared relatively when atom count, protonation, boundary conditions, and positional restraints are identical. Calculate solute-water interaction energy separately if the backend can isolate it correctly. Do not rank total energies of different proteins or systems with different water counts as site scores. Local minimization includes neither the full bulk reference nor the loss of translational/orientational entropy.

**Performance:** Cache parameterization and unchanged data. An ROI restricts proposal generation; any additionally truncated energy environment must be evaluated separately against the full environment. Support CPU execution first; GPU execution is an optional accelerator.

### 6.5 Competition and Multiple Waters

Individually favorable candidates may be mutually incompatible. After optimization, build a conflict graph: nodes represent water hypotheses, and edges represent spatial overlap or incompatible occupancy. Retain multiple alternatives per site. For the MVP, use deterministic score-based selection with a conflict filter and label it as a heuristic.

At a later stage, optimize small clusters jointly, including water-water interactions and bridge networks. Single-water search necessarily misses bridges requiring two or more waters. Its target recall must therefore be defined specifically for single-water bridges.

### 6.6 API Sketch

The following names are proposed and are not currently executable:

```python
prepared = prepare_contact_system(
    structure,
    partners=("A", "B"),
    hydrogen_policy="require_or_prepare",
    protonation_policy="preserve_explicit",
    topology_backend="templates",
)

observed = analyze_contacts(prepared, profile="legacy_v0_1")
oriented = orient_existing_waters(prepared, config=orientation_config)
proposals = propose_bridging_waters(prepared, config=proposal_config)
optimized = optimize_bridging_waters(
    prepared,
    proposals,
    backend="openmm_rigid_water",
    config=optimization_config,
)
bridges = evaluate_water_bridges(prepared, optimized, config=bridge_config)
features = aggregate_contact_features(observed, bridges, schema=feature_schema)
```

Do not mutate `prepared` silently. An explicit export function writes a new structure with selected modeled waters and separate provenance metadata. Do not overwrite experimental occupancy/B-factor fields with uncalibrated prediction scores.

## 7. WaterKit as a Dependency

### Recommendation

**Yes as an optional hydration backend and benchmark; no as a core dependency for every water-bridge analysis.** biotools can analyze existing waters and optimize individual local hypotheses with its own geometric rules and optional OpenMM. WaterKit becomes useful when local water distributions and more detailed evaluation are required.

WaterKit is not a solvent-free contact classifier. It samples explicit waters locally and can use GIST for hydration analysis. Its published scope includes selected surface regions without simulating the entire solvated structure. [Method description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10732097/)

The current repository documents dependencies including OpenBabel, AutoDock Vina/AutoGrid, AmberTools, OpenMM, and ParmED, plus a special AutoGrid build. Adding a pip dependency alone therefore does not guarantee a working installation. [WaterKit installation](https://github.com/forlilab/waterkit)

### Adapter Contract

- Input: the unchanged prepared complete system, explicit ROI, seed/sampling parameters, specified water model, and output directory.
- Initially execute through a clearly isolated external environment/CLI. Record versions, command parameters, and logs; do not install tools implicitly during analysis.
- Output: water-O positions and, where available, orientations/frames and method-specific scores. Check units and the meaning of energies/densities before import.
- Subsequently use biotools rules to determine which sites actually connect both partners.
- Include both partners in the environment. Hydrating an isolated receptor may propose waters in positions occupied by the peptide/ligand in the complex.
- Report missing executables, incompatible versions, incomplete parameterization, timeouts, empty output, and invalid units as structured diagnostics.

### Packaging Sketch

Keep core analysis executable without importing OpenMM. Continue using the existing `contacts` extra for OpenMM topology. A new `water` extra can contain CPU OpenMM, SciPy, and required preparation tools. Document and pin the external WaterKit environment separately. Do not add a second automatic CUDA installation path for contact analysis.

Initially evaluate WaterKit in a pilot of, for example, ten curated complexes: installation, treatment of both partners, output formats, runtime, and reconstruction against simple baselines. Declare the adapter supported only after this pilot. This number is a proposed engineering pilot size, not a sufficient accuracy benchmark.

## 8. Quantifying Quality

The three levels address different questions. Strong agreement between tools is not experimental truth; strong water reconstruction is not evidence of improved affinity prediction.

| Level | Question | Primary measurements |
| --- | --- | --- |
| Implementation | Does the code implement the stated rules? | Analytical errors, invariance, expected classification, diagnostic coverage |
| Chemical detection | Are classified contacts plausible? | Per-type precision/recall/F1 on curated cases, agreement between tools |
| Water reconstruction | Can known sites be recovered? | One-to-one site matches, recall, model-relative precision, localization error |
| Optimization | Does the optimizer improve the local solution? | Convergence, constraint/clash errors, reconstruction before/after optimization |
| Downstream use | Do the features help with the actual task? | Error/ranking metrics on independent targets/scaffolds |

### 8.1 Small, Independent Unit Tests

Construct synthetic geometries analytically. Do not calculate expected values using the same helper being tested. Each contact type needs positive examples, hard negatives, and boundary cases. Existing tests provide a starting point but do not replace a chemical benchmark. [Existing tests](https://github.com/lohex/biotools/blob/ef52e57736c763319c3b4328f5710347f93db25d/tests/test_chain_contacts.py)

Required properties:

- Translation and rotation do not change distances, angles, or classification.
- Atom/residue order does not change results after stable sorting.
- Swapping chains maps partners and roles correctly.
- Swapping water H1/H2 does not create a new physical hypothesis.
- KD-tree and brute-force candidates agree within tolerance.
- Inputs are not mutated; identical parameters give reproducible results.
- Complete versus fragmented selections do not create false termini.
- Incomplete chemistry is visible instead of producing silent negatives.
- Model/altloc separation, insertion codes, and modified residues remain correct.
- Covalent exclusions also work across chains.

For well-conditioned float64 coordinates, deterministic geometry could be required to remain invariant within, for example, 1e-6 Å and 1e-5 degrees. These are proposed numerical test tolerances; unstable planes should instead produce a degeneracy status. Use looser tolerances for sampled water placement, determined through sampling-convergence tests.

### 8.2 Chemical Reference Cases

Build a small manually curated collection of protein/peptide motifs with expected groups, roles, and contact classes. Include neutral/charged variants, genuine and capped termini, cyclic peptides, TRP, disulfides, sulfur groups, water bridges, and incompatible geometries.

Use two separate test sets:

1. Fully specified geometry, including H. This tests the detectors.
2. Raw structures without H. This tests preparation plus detectors.

This prevents protonation errors from being mistaken for geometry errors. Report results by interaction type and state. Annotate ambiguous cases and exclude them from hard gold-standard scoring or report them separately.

### 8.3 Differential Tests Against Other Tools

Use external tools such as ProLIF, BINANA, or PLIP only as optional, independent benchmark references. These comparisons do not introduce a ProLIF integration or runtime dependency: production contact and trajectory analysis use the biotools detectors. Align versions, preparation, hydrogen policies, and cutoffs where scientifically appropriate. [ProLIF](https://prolif.readthedocs.io/en/latest/source/modules/interaction-fingerprint.html), [BINANA](https://durrantlab.pitt.edu/apps/binana/docs/INTERACTIONS.html), [PLIP](https://github.com/pharmai/plip/blob/master/DOCUMENTATION.md)

Measure agreement separately at atom/group and residue levels. Harmonize categories, such as directed HBDonor/HBAcceptor versus one common hydrogen-bond type. Calculate Jaccard/F1 for equivalent rules; where rules differ, classify disagreements by cause. Do not use a majority vote among tools as an experimental gold standard.

### 8.4 Water Benchmarks Without Information Leakage

Keep four tasks separate:

| Task | Available information | Held-out information |
| --- | --- | --- |
| O known | Water oxygen and solute | H orientation |
| One missing site | Solute and remaining waters | One individual water |
| Dry interface | Both partners and a defined ROI | All interface waters |
| Modeled complex | Predicted/docked partner pose | Experimental waters and native pose |

Removing one water is easier than reconstructing a dry interface. Do not combine these tasks into a single score. Remove held-out waters **before** solute-H optimization, site search, or any other data-dependent preparation. Otherwise, the target position may remain indirectly encoded in hydrogen orientations or preparation metadata.

**Proposed dataset selection:** Start with 30-50 development structures, followed by a sufficiently large independent set of protein-peptide and protein-protein complexes. Plan sample size based on confidence-interval width. Prefer high structural resolution, documented water occupancy, and good local model quality; test, for example, ≤2.0 Å as an initial filter. This is a quality filter, not proof of completeness.

For each structure, document experimental method, resolution, occupancy, local B-factors, altlocs, crystal contacts, and proximity to ions. Cluster frequently repeated homologous complexes. Split development/test sets by protein/interface family; for ligands, also consider chemical similarity. Do not optimize thresholds on the test set.

**H orientations:** Conventional X-ray water-O coordinates do not provide a complete H-orientation reference. Use curated neutron structures or clearly separate synthetic references for this purpose. Orientations reconstructed with the same hydrogen-bond rule are not independent experimental labels.

### 8.5 Site Matching and Metrics

Compare O positions after solute alignment. Assign predicted and reference waters through bipartite matching: first maximize the number of valid matches within radius epsilon, then minimize total distance. Each water can have at most one match. This prevents many candidates around one reference site from artificially inflating recall.

Report epsilon = 0.5, 1.0, and 1.5 Å separately; these radii are proposed benchmark conventions. Treat multiple H orientations of one hypothesis as one O site. In addition to O position, optionally require the correct bridge anchors and report a stricter `bridge_recall`.

- `recall = matched_reference_sites / all_reference_sites`
- `model_precision = matched_predictions / all_predictions`
- `F1_model` from these two quantities, with a clear reference definition
- Median, quantiles, and distribution of O-localization error
- Recall at a predefined candidate budget per interface or per unit interface area
- Number of unmatched proposals per interface
- Clash rate and fraction of proposals with valid bridge legs
- Coverage: fraction of systems, waters, and anchors that can be evaluated
- Runtime p50/p95, peak RAM, optimizer failures, and number of optimized candidates

**Limit of precision:** Unmatched means “not confirmed in the experimental structural model.” It does not prove that a water site is physically impossible. Accordingly, use the name `model_precision` and investigate selected additional sites using electron density, independent structures, or further sampling. Waters repeatedly observed in independent structures provide a stronger reference than a single structural model. WaterDock illustrates the use of consensus water sites as a benchmark principle. [WaterDock study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0032036)

Calculate metrics per complex first, then report macro averages and paired differences between methods. Bootstrap independent protein/interface clusters rather than individual waters; report 95% intervals. Individual water-rich complexes must not dominate the result. Explicitly define precision/recall behavior when predictions or references are absent, and report counts alongside metrics.

### 8.6 Optimization Benchmark

For exactly the same starting candidates, compare:

1. No optimization.
2. H-orientation optimization with fixed O.
3. Rigid-water translation and rotation.
4. Additional limited side-chain relaxation.

Measure site reconstruction, bridge reconstruction, clash rate, and runtime for each. Energy reduction, gradient/torque norm, constraint error, solute displacement, and convergence status are technical diagnostics. They are not the primary demonstration of biological success.

Hard quality gates: no nonfinite results; no accepted massive clash; fixed solute coordinates unchanged; water geometry within the declared tolerance; missing force-field parameters visibly stop optimization. Retain failed candidates with their status instead of reporting them as good bridges.

### 8.7 Baselines and Ablations

| Method | What does it test? |
| --- | --- |
| Random clash-free O sites in the same ROI with the same budget | Advantage over geometrically unspecific placement |
| Shell/distance search without directions | Added value of hydrogen-bond geometry |
| Directed search without optimization | Added value of local optimization |
| Geometric optimization | Added value of the force-field backend |
| OpenMM optimization with the same search budget | Local physics as an additional filter |
| WaterKit adapter | Comparison with a specialized local hydration method |
| Full solvation plus sampling, where available | A more expensive reference under documented simulation assumptions |

For methods with different computational budgets, also report quality/runtime curves. Disclose potential training overlap for pretrained methods. Initially compare WaterKit only within the chemistry/system scope actually supported.

### 8.8 Usefulness of Features for ML

Compare at least four otherwise identical modeling pipelines: existing features; corrected contact features; additional geometrically proposed waters; additional locally optimized waters. Optionally add ECIF-style distance histograms as a separate block.

Train on identical independent protein/scaffold splits. Perform feature selection and hyperparameter tuning only within inner training folds. For affinity regression, use MAE/RMSE and within-target rank correlation; for screening, use PR-AUC and early enrichment at a defined prevalence. Generate water features with the same information availability during training and deployment.

The quality of contact functions can be evaluated before an affinity dataset is available. An ML improvement would provide additional evidence of practical usefulness, not a substitute for geometric and chemical validation.

### 8.9 Ensembles, Uncertainty, and Sampling

Distinguish site occupancy from the lifetime of the same water molecule. Site occupancy measures whether any water occupies the spatial site and satisfies the bridge criteria. Molecular residence times require tracking identity across the trajectory.

For periodic systems, first center/unwrap correctly or support minimum-image geometry. Rings must not be split across a box boundary. Frames are correlated: use block averages, independent replicates, and convergence curves. Document the local alignment strategy for site comparisons. Variation across seeds or parameter sets is initially sensitivity, not automatically calibrated uncertainty.

## 9. Implementation Packages and Acceptance Criteria

| Package | Content | Acceptance |
| --- | --- | --- |
| A | Result objects, diagnostics, IDs, profiles, and preserved public facade | Old API explicitly reproducible; new metadata serializable |
| B | Shared chemistry graph, protonation, termini, and covalent exclusions | Curated state cases and topology tests across backends |
| C | Correct/refine all eight contact types | Analytical positive/negative cases, invariance, and documented behavior changes |
| D | Orient existing waters; retain both bridge legs | Fixed O, rigid water geometry, role/permutation invariance |
| E | De novo single waters, clusters/alternatives, and geometry scores | Reproducible proposals, fair baselines, no accepted massive clashes |
| F | OpenMM optimization | Parameter/unit checks, convergence diagnostics, fair before/after benchmark |
| G | WaterKit pilot and adapter | Reproducible external environment and correct handling of both partners |
| H | Experimental benchmark, feature aggregation, and optional ensemble analysis | Independent test clusters, confidence intervals, and reproducible report |

Do not promise a universal recall value before the pilot. After evaluating the development set, predeclare a minimum improvement or noninferiority margin and a runtime budget. Only then use the final test set once for the decision.

Suggested new tests: `test_contact_chemistry.py`, `test_contact_invariance.py`, `test_contact_clashes.py`, `test_aromatic_geometry.py`, `test_water_orientation.py`, `test_water_proposals.py`, `test_water_optimization.py`, and `test_water_matching.py`. Keep reference data small and deterministic in fast CI; run external tools and larger scientific benchmarks in separately triggered jobs.

Document every subsequent repository change under `Unreleased` in `doc/changes.md`, as required by the repository. Describe threshold/definition changes, including their effects on old results, in migration documentation. [Repository instructions](https://github.com/lohex/biotools/blob/ef52e57736c763319c3b4328f5710347f93db25d/AGENTS.md)

### 9.1 Delivery Order and Dependencies

Implement the plan as small, reviewable pull requests. Each PR must include its documentation and relevant tests. The sequence is A -> B -> C -> D -> E -> F. The WaterKit pilot G can start after E establishes the proposal interface; it is not a prerequisite for F. Build the benchmark infrastructure in H alongside A, and freeze the final evaluation protocol before tuning D-F.

1. **PR 1: Capture current behavior and introduce result contracts (A).** Inventory public imports, return formats, default thresholds, and optional dependency behavior. Add representative regression fixtures for currently supported inputs. Define immutable atom/group IDs, typed result records, configuration serialization, and diagnostic codes. Implement the legacy projection before routing public calls through the new facade. Acceptance: supported legacy calls retain their signatures and documented outputs under the legacy profile; new records round-trip through serialization.
2. **PR 2: Centralize chemical perception (B).** Move donor/acceptor typing, charged-group membership, ring perception, and topology exclusions behind one prepared-system interface. Preserve explicit protonation, track inferred atoms, and expose unsupported residues as coverage gaps. Add state-specific fixtures for histidine, termini, acidic groups, amides, disulfides, insertion codes, and alternative conformers. Acceptance: every detector uses the same chemical assignments and preparation preserves the original-to-prepared atom mapping.
3. **PR 3: Repair pairwise contact detectors (C, first part).** Implement hydrogen bonds, salt bridges, hydrophobic contacts, and van der Waals/clash classification as pure functions over prepared chemistry and coordinates. Share neighbor lists, but use separate physical criteria and named geometric outputs. Acceptance: independent boundary fixtures and rigid-transform tests pass; differences from legacy output are explained per category.
4. **PR 4: Repair group-based detectors (C, second part).** Add ring planarity checks, orientation-independent normal comparisons, lateral displacement, T-shaped geometry, and cation-to-ring measurements. Deduplicate by physical group identity. Acceptance: ring atom order and normal sign cannot change a result; deliberately displaced or nonplanar examples are rejected or explicitly flagged.
5. **PR 5: Evaluate and orient observed waters (D).** Represent both bridge legs, enumerate chemically permitted roles, and orient a rigid water while holding its experimental oxygen fixed. Preserve alternative orientations and distinguish an oxygen-only candidate from an orientation-supported bridge. Acceptance: water-H permutation gives the same site result, and improving one leg cannot hide failure of the other.
6. **PR 6: Propose waters in dry interfaces (E).** Select cross-partner anchor pairs, sample feasible oxygen regions, deduplicate positions, enumerate orientations, and evaluate both legs plus all local clashes. Add deterministic seeds, candidate limits, and an explicit empty-result diagnostic. Acceptance: identical inputs/configuration reproduce site IDs and ordering; the proposal stage runs without WaterKit or a full solvent box.
7. **PR 7: Add optional local energy refinement (F).** Create a reusable parameterized environment, insert one modeled water per optimization hypothesis, and expose rigid translation/rotation optimization through an optional backend. Return initial/final geometry, score components, convergence status, and displacement. Acceptance: solute coordinates and water geometry remain fixed within declared numerical tolerances; all accepted outputs pass the independent bridge detector after optimization.
8. **PR 8: Evaluate WaterKit and publish the benchmark (G/H).** Implement the adapter only after checking installation, licensing, model compatibility, and parameter coverage. Compare its proposals with local proposals under the same preparation and evaluation protocol. Publish configuration, dataset manifests, failure counts, runtime distributions, and cluster-level confidence intervals. Acceptance: the dependency recommendation follows measured benefit and operational cost; missing WaterKit does not break core imports.

### 9.2 Function-Level Implementation Contracts

- `prepare_contact_system`: validate units, models, conformers, partner selections, chemistry, and topology once. Return prepared coordinates, chemical groups, immutable IDs, atom mappings, and coverage diagnostics. Never infer successful parameterization from successful topology construction.
- `analyze_contacts`: reuse neighbor data and dispatch pure detectors. Collect atomic/group observations first; apply residue aggregation and legacy conversion afterward. Store the actual criterion profile with the result.
- `orient_existing_waters`: preserve oxygen coordinates and experimental metadata. Return alternative orientations with explicit bridge-leg roles and reasons for rejection; do not silently replace coordinates in the input structure.
- `propose_bridging_waters`: return geometric hypotheses with anchor IDs, oxygen position, candidate orientations, proposal score components, and provenance. Bound candidate growth before expensive refinement.
- `optimize_bridging_waters`: consume hypotheses without modifying them, validate backend coverage, cache the common environment, and optimize each hypothesis from multiple starts. Handle parameterization failure, nonfinite energy, nonconvergence, and escape from the search region explicitly. Do not accept an unoptimized fallback as an optimized success.
- `evaluate_water_bridges`: independently recompute distances, angles, chemical roles, and clashes on final coordinates. Use this same evaluator for local and WaterKit proposals so backend-specific scoring does not determine benchmark labels.
- Export and aggregation functions: retain site identity, alternative occupancy, and modeled/observed provenance. Version exported feature schemas so scientific changes cannot silently alter downstream ML inputs.

### 9.3 Concrete Verification and Release Workflow

For each detector PR, add analytical positive and negative geometries, near-threshold cases, invalid chemistry cases, and a small real-structure regression fixture. Geometry tests should derive expected results independently, rather than calling the production helper to compute the expected value. Document numerical tolerances and the angle convention explicitly.

For the optimizer PR, check translation and rotation gradients against finite differences on small parameterized systems; test fixed-solute and rigid-water invariants; include starts that converge, fail, and move away from both anchors. Compare multi-start refinement with the same candidates without refinement. Report the fraction of candidates that retain valid two-leg bridges as well as energy changes.

Fast CI should run dependency-light contact tests and deterministic small fixtures. Run the optional OpenMM suite in a separate environment with pinned backend versions. Run WaterKit integration and structural benchmarks as explicitly triggered jobs with archived inputs and machine-readable results. Separate skipped unsupported cases from tested negatives, and fail configuration checks if an intended benchmark silently runs no eligible cases.

Before release, generate a migration report comparing the legacy and revised profiles on the same structures. Include per-type count changes, reasons, unsupported chemistry, water-site recovery, bridge recovery, and runtime. Choose numerical scientific acceptance targets using development data, record them before opening the held-out test results, and require the documented compatibility and invariant checks regardless of benchmark performance.

### 9.4 Definition of Done

The first implementation milestone is complete when all eight existing contact categories have documented chemical/geometric criteria; observed and proposed single-water bridges preserve both legs; local optimization produces reproducible, independently revalidated results; and the public API exposes failures and coverage explicitly. Deliver the API documentation, migration notes, small test fixtures, benchmark runner, dataset manifest, and an example dry-interface workflow together. Multi-water networks, side-chain relaxation, occupancy prediction, and production WaterKit support require separate acceptance decisions after this milestone.

## 10. Open Decisions Before Production Release

- Which protonation states and modified peptides are within the guaranteed support scope?
- Will all solute atoms remain fixed during water optimization, or will local side-chain motion be supported in production?
- Should the first benchmark prioritize known O positions, dry interfaces, or both tasks?
- Which site/bridge metrics and runtime budgets will become release gates after the pilot?
- Does WaterKit improve upon the simpler local method within the specific protein/peptide scope?

These decisions do not block chemical corrections or the geometric prototype. They delimit which accuracy and applicability claims will be justified after implementation.
