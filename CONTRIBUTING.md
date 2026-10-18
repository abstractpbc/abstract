# Contributing

Use Python 3.12 or 3.13 and uv. Keep tests independent of real hardware, model weights,
GPUs, credentials and paid APIs. Use fake drivers, camera readers and HTTP transports.

## Development checks

```sh
uv sync --locked --group typing
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
uv build --no-sources
uv run --no-sync python scripts/check_dist.py dist
```

The configured type-checking scope currently covers credential handling, termination,
inference supervision, shared safety construction and the action-policy boundary. Expand
that scope with changes; it is not a claim that every module is strictly typed.

For optional runtime tests, sync with `--extra robot`. For the complete YAM/kinematics
suite, use `uv sync --locked --extra yam --group yam-driver`. The core/controller/agent
tests are always collected. Only tests that actually need an optional dependency skip
when it is absent. Linux CI runs core and full robot suites on both supported Python versions.
Loopback sockets and POSIX shared memory must be available for the RPC and capture tests.
A sandbox denial is a failed validation prerequisite; do not turn it into a product test skip.

A change to semantics, units, buffering, cancellation, guardrails, export or deletion needs
a regression test at the real integration boundary. Preserve failure records and test
that refused or late commands never reach a fake actuator. Changes to public interfaces
also update the changelog and examples.

## Release evidence

The nested `.github/workflows/ci.yml` is for the standalone export. The monorepo uses
`../../.github/workflows/harness.yml`; keep their jobs equivalent. Both build and inspect
artifacts and run a fresh installed-wheel smoke outside the source tree. Neither publishes.
Run the manual `lerobot-reader` workflow before a release to check the upstream reader
against LeRobot 0.6.1. This heavier dependency set is intentionally separate from normal development.

A release needs green CI from a clean checkout, an exact export/history secret scan,
verified upstream license provenance, and an enabled private vulnerability-reporting channel.
Record the source and driver revisions and dependency lock with the release. Revalidate
claimed dependency floors before changing them; normal CI tests the locked resolution.

Hardware changes need a dated, operator-run acceptance record: rig identity and geometry,
limits, driver/firmware revisions, first target, reset/parking, thermal trip, camera/CAN loss,
blocked inference, cancellation, restart and measured stop latency. Source tests do not
replace that evidence. Agents and unattended CI must never energize the arms.

Upstream bases are recorded in THIRD_PARTY_NOTICES.md. For an upstream update, review its
license and security changes, retain local boundary regressions and describe any divergent
patches. Preserve original attribution in derivative code and tests.

By contributing, you license your original contributions under Apache-2.0; pre-existing
third-party notices continue to apply.
