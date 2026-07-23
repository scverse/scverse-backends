# Release notes

## 0.0.3 — 2026-07-23

### Whole-class dispatch

- Added `BackendDispatcher.backend_class` for APIs whose backend replaces a
  complete class.
- Class construction supports the active setting and a per-instance
  `backend=` override, with CPU fallback when an adapter does not implement the
  class.
- Host signatures, documentation, custom metaclasses, class APIs, and normal
  subclass construction are preserved.
- Backend adapter exports are validated as classes, and recursive
  self-registration is rejected with a clear error.

### Reliability and packaging

- Invalid dispatcher identity and trusted-provider configuration now fail
  early with actionable errors.
- One broken adapter registration no longer prevents other entry points from
  being discovered.
- The conformance runner raises an explicit `ValueError` when a backend cannot
  be resolved or a requested function filter is invalid.
- Function dispatch rejects unsupported methods and variadic positional host
  signatures instead of silently misrouting values.
- Positional-only backend calls preserve omitted defaults, and host-only
  default comparisons no longer assume scalar equality.
- Runtime type-hint introspection now resolves every public annotation.
- Discovery can be retried after metadata or signature-update failures, and
  adapter metadata failures cannot leave a partially registered backend.
- Installed trusted backends report host-configured aliases consistently.
- CI and publishing configuration now exercise the same lint, documentation,
  coverage, build, and metadata checks used for release validation.
