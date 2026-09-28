# pMHC contact viewer: limitations and remaining work

**Status:** The biotools viewer API extensions below are implemented in this
repository. pMHCdb integration tasks remain open and are not part of this
repository. The original limitation descriptions document the state before
these extensions and do not imply that pMHCdb now uses them.

## 1. Virtual mediator waters

**Original limitation:** CSV water XYZ coordinates are preserved as geometry
measurements, but water-bridge observations without explicit mediator atoms
are skipped with a structured warning. These external coordinates are neither
transformed nor rendered. In-memory observations referencing real water atoms
are supported and transformed with the structure.

- [x] biotools: expose a typed mediator-point/display-coordinate override.
- [ ] pMHCdb: pass modeled-water coordinates through that public API and apply
  the same binding-cleft transform used for the structure.
- [x] Test two dashed bridge legs and a mediator sphere without adding a water
  residue to the caller's structure.

Example: `9pkf_A_P` currently displays 28 contacts and skips one virtual-water
bridge with a diagnostic.

## 2. HTML title, metadata and alignment status

**Original limitation:** validated metadata is available as
`view.pmhc_metadata`, but the renderer has no public title/subtitle API.
The HTML title remains `Structure contacts`; structure ID, HLA allele, peptide
sequence, resolution and alignment status are not presented as the requested
pMHC page header.

- [x] biotools: provide public title/subtitle or structured metadata fields,
  with HTML escaping in the rendering layer.
- [ ] pMHCdb: configure `Peptide–MHC interactions`, non-empty metadata fields
  and alignment status.
- [x] Test escaping of title and subtitle text rendered in the HTML header.

## 3. Excluded-contact notes and diagnostics in the viewer

**Original limitation:** excluded types are recorded in
`view.pmhc_excluded_types`; unresolvable observations appear in
`view.pmhc_diagnostics` and as warnings. CSV validation errors emit structured
warnings. The CLI reports skipped contacts and exclusions, but the HTML sidebar
contains neither the excluded-type note nor geometry warnings.

- [x] biotools: expose public configuration for exclusion notes and structured
  diagnostics in the controls.
- [ ] pMHCdb: supply exclusions and both CSV and geometry diagnostics.
- [x] Verify that intentional filtering is distinguishable from skipped,
  invalid geometry in the generated page.

## 4. Different residue-label universes for HLA and peptide

**Original limitation:** biotools accepts one label universe shared by all
chains. The adapter uses `all` so every peptide residue can be labeled.
Consequently, enabling HLA All labels labels the entire HLA chain instead of
only residues occurring in included contacts. Initial HLA labels are disabled,
with All retained as the stored mode; peptide All labels are enabled.

- [x] biotools: allow a separate label universe per chain.
- [ ] pMHCdb: use included-contact residues for HLA and all standard residues
  for peptide.
- [x] Test these universes together with master/type/pair-driven Active labels.