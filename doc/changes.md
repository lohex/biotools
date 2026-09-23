# Changes

This file records user-visible changes since the latest release. Add new work
to `Unreleased` and move those entries into a dated version section when a
release is published.

## Unreleased

### Added

- Added `plot_structure_contacts()` with typed contact geometry, grouped
  dashed connections, aromatic planes, water bridges, synchronized pair/type
  controls, residue labels, view presets, and standalone interactive HTML.
- Added a `patch_wireframe=True` overlay that draws the MSMS triangle grid
  only on selected surface patches; the molecular-surface notebook now uses a
  coarser mesh and distinct patch colors for a clearer three-patch view.
- Added a PDB2PQR/APBS backend for linearized or nonlinear, solvent- and
  ion-screened Poisson--Boltzmann potentials interpolated onto surface meshes.
- Added an integrated colorbar for scalar-field molecular-surface plots and a
  `patch=` option to render one connected patch with continuous colormap colors.
- Added an optional OpenMM adapter that assigns per-atom force-field charges,
  adds missing hydrogens in memory, validates surface-to-structure atom
  alignment, and forwards all charge sites to Coulomb surface mapping.
- Reworked the molecular-surface notebook to download PDB entry 1CRN and
  demonstrate hydrophobic patches and OpenMM-derived surface potential on a
  real protein rather than a synthetic one-atom-per-residue model.
- Added MSMS-backed triangulated solvent-excluded surfaces with stable atom
  mappings, hydrophobic and Coulomb-potential vertex fields, connected surface
  patches, deterministic label colors, area-uniform point/normal sampling, and
  interactive py3Dmol structure/mesh/normal visualization.
- Added an implementation TODO for optimized trajectory analysis of all eight
  contact types, electrostatics with and without OpenMM, and direct
  electrostatic/Lennard-Jones interaction-energy scores.
- Added fixed-length NVT/NPT production runs with DCD or XTC trajectories,
  CSV thermodynamics reporting, periodic checkpoints, continuation support,
  and structured result metadata.
- Added a typed, serializable contact-analysis API with stable atom identities,
  named criterion profiles, structured roles and geometry, preparation
  diagnostics, coverage, and reproducibility hashes.
- Added deterministic local orientation, proposal, geometric refinement, and
  independent re-evaluation of single-water bridges without an external
  hydration dependency.
- Added an optional fixed-solute, rigid-water OpenMM energy-refinement backend
  with explicit parameterization and nonfinite-energy failures.
- Added a complete interaction reference covering all eight interaction types,
  their calculations, thresholds, refinements, aggregation, and migration
  behavior.

### Changed

- Made MD pipeline stages accept preceding result objects directly, with
  configurable automatic selection between coordinates, XML States, and
  configuration-verified checkpoints.
- Refined contact chemistry and geometry: topology-derived termini and
  covalent exclusions, explicit thiolate/disulfide handling, van der Waals gap
  and clash reporting, separate tryptophan rings, ring planarity and projection
  checks, directional cation-pi filtering, and explicit water-bridge evidence.
- Extended compatibility contact records with named/unit-bearing geometry,
  structured roles and criteria, quality flags, aggregation counts and distance
  statistics, and the applied rule profile.
- Simplified the built-in contact profile names to `refined` and `legacy`.
- Clarified optional dependency installation and documented exactly when
  existing or missing water bridges are evaluated, how both local-refinement
  backends score candidates, and why neither requires a complete solvent box.

### Fixed

- Added compatibility with both upstream and Ubuntu/Debian FreeSASA CLI depth
  options and JSON structure keys without masking unrelated FreeSASA errors.

## 0.1.1 - 2026-08-10

This is the first tagged GitHub release of `biotools`.

### Added

- Added geometric interchain and intrachain contact characterization for
  hydrogen bonds, salt bridges, hydrophobic and van der Waals contacts,
  aromatic interactions, cation-pi candidates, and water bridges.
- Added residue distance and interaction matrices with intrachain and
  interchain Matplotlib heatmaps.
- Added direct DSSP integration with secondary-structure strings, relative and
  absolute SASA, and temporary compatibility records for generated PDB files.
- Added FreeSASA-based per-residue and per-chain SASA plus two-chain buried
  interaction-surface analysis.
- Added RCSB structure metadata retrieval.
- Added OpenMM preparation, minimization, NVT/NPT equilibration, gentle NVT
  heating, convergence monitoring, continuation files, optimizer restarts, and
  diagnostic plots.

### Changed

- Set the minimum supported Python version to 3.11.
- Made CUDA-heavy molecular-dynamics dependencies optional through the `md`
  extra and added a CPU-only OpenMM topology backend through `contacts`.
- Added Matplotlib to the base installation for structural plots.
- Split structure and sequence functionality into focused packages while
  retaining `pdbtools`, `seqtools`, and `mdtools` compatibility imports.
- Reorganized the documentation into module guides and introduced the new
  interaction-analysis logo.

### Fixed

- Fixed protein alignment similarity scoring by using Biotite's
  `SubstitutionMatrix.get_score()` API.
