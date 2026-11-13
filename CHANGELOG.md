# Changelog

## 1.3.0

- Add native Gemini Decisions and typed extraction. Preserve legacy nondecision-ID adapters and provide nativeDecisionModel for custom native IDs. Apply current thinking floors/efforts, tier metadata fallback, resource-accurate sources and dotted-index accumulation. Reviewed npm delta: @ai-sdk/google 4.0.114 to 4.0.120. Validation: `node Scripts/test-packages.mjs` passed all 52 owning suites and 1731 facade tests; tooling 27/27, generation/isolation checks, standalone exports and 106 minimum/latest consumers, documentation check and 101-page build passed. Publication requires green exact-source Swift 6.3.3 CI. Paid/live verification and full-package parity are not claimed.
  Compatibility: Additive native APIs or capabilities under the project minor-release convention. npm baselines and Swift SemVer are independent; this reviewed delta does not establish full-package or live-service parity. Required direct minimums: swift-ai-core 1.3.0 to 1.4.0.
