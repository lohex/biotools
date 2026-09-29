# Repository instructions

## Change documentation

- Document every new feature and bug fix in `doc/changes.md` as part of the
  same change. Record ongoing work under `Unreleased`.
- Update `doc/changes.md` with every change that affects users, packaging,
  documentation, tests, or development workflows.
- Whenever the package version increases, move the accumulated entries into a
  dated `## <version> - YYYY-MM-DD` section before committing. Leave
  `Unreleased` ready for new work.
- Do not add version increments as changelog bullet points; the version heading
  records the increment.

## Package versioning

- For each commit requested by the user, increment the package subversion
  (the second numeric component) exactly once and by exactly one, for example
  `0.7.1` -> `0.8.1`. A later push of that commit does not increment the
  version again. Do not count the commit and its push as two version changes,
  even when both are requested together.
- If the user explicitly announces a new version, increment the major version
  instead and reset the subversion to zero, for example `0.12` -> `1.0`.
- Keep all package version declarations consistent when updating the version.
