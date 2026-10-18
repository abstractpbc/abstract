# Security

Report vulnerabilities privately through this repository's Security tab and its
“Report a vulnerability” action. Maintainers must enable GitHub private vulnerability
reporting before publishing the first release. If it is unavailable, use a private
maintainer contact rather than posting exploit details publicly.

Include the affected revision, module, reproduction with fake hardware, expected and
observed behavior, and impact. Do not include real credentials or customer recordings.

In scope are credential leakage, RPC authorization, command execution, run-store
isolation, billing/accounting defects and software defects that can cause unintended
physical commands, prevent cancellation or corrupt safety-relevant records. Physical
certification and hardware design are outside this package's assurance, but that does
not exclude software defects affecting physical behavior.

Only the latest 0.1 release line receives fixes during the initial release period.
See [docs/SAFETY.md](docs/SAFETY.md) for execution guarantees and limitations.
