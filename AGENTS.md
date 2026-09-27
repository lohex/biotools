# Repository instructions

## Change documentation

- Document every new feature and bug fix in `doc/changes.md` under `Unreleased`
  as part of the same change, before committing or pushing.
- Update `doc/changes.md` with every change that affects users, packaging,
  documentation, tests, or development workflows.
- Record ongoing work under `Unreleased` so the file always describes what has
  changed since the latest release.
- When publishing a release, move the accumulated entries into a dated version
  section and restore an empty `Unreleased` section.

## Package versioning

- Increment the package subversion (the second numeric component) by one for
  every commit and every push, for example `0.1` -> `0.2` or `0.12` -> `0.13`.
- If the user explicitly announces a new version, increment the major version
  instead and reset the subversion to zero, for example `0.12` -> `1.0`.
- Keep all package version declarations consistent when updating the version.
