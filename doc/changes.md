# Changes

This file records user-visible changes since the latest release. Add new work
to `Unreleased` and move those entries into a dated version section when a
release is published.

## Unreleased

No changes yet.

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
