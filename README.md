# chisel-platform

The platform side of [Chisel Docs](https://github.com/WyattAu/chisel-doc): the
crates that turn a single-player PDF editor into a system with identity,
assets, and an audit story that survives the machine it ran on.

| Crate | Composes (kit estate) | Status |
| --- | --- | --- |
| [`chisel-audit`](crates/chisel-audit) | `tamper-audit` (SHA-256 chain, verification, queries) | document vocabulary over the chain: open / edit / redact / sign / export |
| [`chisel-kernel`](crates/chisel-kernel) | `config-kit` (layered typed config, redacted secrets) | `ServiceSettings` (strict) + `DesktopSettings` (lenient) + validated ids |
| [`chisel-assets`](crates/chisel-assets) | `cas-kit` (BLAKE3, verify-on-read, GC) | **live** — content-addressed store; dedup by construction, tamper fails the read (`blobkit`/S3 is the service slice) |
| `chisel-auth` | `oauth-toolkit`, `tokenkit`, `webauthn-kit`, `barbican`, `accessctl` | planned — OIDC/PKCE, sessions, RBAC |
| `chisel-jobs` | `outbox-kit`, `idempotency-kit`, `worker-kit` | planned — render/redaction job orchestration |
| `chisel-observe` | `telemetry-init`, `metrics-kit`, `healthkit`, `percentile-kit` | planned |

## Why the wrappers are thin

On purpose. `chisel-audit` is ~200 lines over `tamper-audit` because chaining,
verification, and querying are the kit's tested surface; what the platform adds
is the **vocabulary** (a closed action set a reviewer can query), the
document-scoped error type, and the seam the service continues. The same rule
will govern `chisel-assets` and `chisel-jobs`: the kit owns the mechanism, the
platform owns the nouns.

## Gates

Born blocking: every crate runs the full estate matrix at Tier A
(`.github/workflows/estate-gate.yml`) — locked build, clippy `-D warnings`
plus the unwrap/expect/panic/indexing denials, coverage ≥ 90%, semver, vet.
New code has no ratchet excuse.

## consuming chisel-doc

The editor will consume `chisel-audit` for its in-app audit panel and
`chisel-assets` for the team library; those pins are what graduate this repo
from `support` to `core` in the estate manifest.
