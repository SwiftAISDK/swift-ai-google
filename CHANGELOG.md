# Changelog

## 1.5.0

- Add the Interactions MCP server tool with header redaction in telemetry, Neon reasoning, prompt blocking metadata and early validation of unsupported batch response schemas. Reviewed npm: @ai-sdk/google 4.0.121 to 4.0.124. Offline validation passed: node Scripts/test-packages.mjs (56 packages, 3672 owning tests and 1760 compatibility tests), generation, tooling, isolation, and version selection. Standalone Developer API transcription and speech translation remain outside the native surface. Vertex Live provides the tested VAD behavior. See Docs/Archive/WeeklyUpstreamAudit20261122.md for retained gaps. Paid provider calls were skipped.
  Compatibility: Use JSON mode or disable structured outputs for Google Batch requests that previously supplied a wire schema. Requires SwiftAI Core 1.7.0 for contextual credential redaction.
