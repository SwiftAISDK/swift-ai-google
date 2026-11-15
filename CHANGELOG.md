# Changelog

## 1.3.1

- Map Interactions common reasoning to the model thinking level, honor explicit provider thinkingLevel and return compatibility warnings on fallback. Agent requests report dropped reasoning. Direct streams begin with stream-start even with empty warnings. Caller headers pass through without an obsolete injected API revision. Exact release JSON and stream fixtures retain their values and have verified Git blob hashes. Reviewed npm delta: @ai-sdk/google 4.0.120 to 4.0.121. Validation passed: node Scripts/test-packages.mjs (55 owning packages, 2451 owning tests and 1736 compatibility tests), generation checks, 27 tooling tests, isolation/version checks, deterministic exports and minimum-dependency consumers. Documentation check/build passed (104 pages). Publication is gated by exact-source Swift 6.3.3 CI; subsequent public refs and clients are recorded in Docs/ReleaseVerification20261115.json. Paid/live-service checks were not run; this delta does not claim full-package parity.
  Compatibility: Compatible request/stream/header corrections. Existing public selectors remain. Use explicit request headers for any caller-selected API revision.
