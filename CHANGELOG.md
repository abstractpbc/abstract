# Changelog

## 0.1.0 — unreleased

Initial standalone release candidate, derived in part from Robocurve inspect-robots and
inspect-robots-yam; see THIRD_PARTY_NOTICES.md for bases and attribution.

- Preserve fixed policy semantics and explicit callable contracts during binding.
- Align gateway camera names with YAM and collect stateful history at control cadence.
- Validate gateway sessions, dimensions, numeric limits, URLs and response semantics.
- Scope cloud credentials to their origin, reject literal secret arguments and redact setup.
- Retain raw data when export completion cannot be verified; recheck file hashes at deletion.
- Route CLI termination through teardown and bound inference/arm-initialization waits.
- Share Python/CLI guardrails, seed and refresh measured references, and preflight native failsafes.
- Preserve terminal metadata through callable conversion and temporal ensembling.
- Distinguish requested/approved actions, final commands and transitions without execution.
- Bundle third-party notices and typing metadata; separate the immutable native-driver group.
- Exercise minimal installs, built distributions and both supported Python versions in CI.

Before 1.0, public API changes may occur between minor versions. Patch releases should
preserve documented interfaces except when a correction is required to refuse unsafe or
invalid input; record those changes here. Underscore-prefixed modules are internal.
