# Changelog

## 1.4.0

- Tool results are sent as `functionResponse.response {name, content}`; each system message is its own `systemInstruction` part; thought text and signatures are separate reasoning parts; empty candidates return empty content; server tools no longer turn the finish reason into `tool-calls`; absent metadata fields are `null`. `tunedModels/` and `endpoints/` IDs are used as-is, Gemini 2.5 penalties and Vertex-only image config warn, YouTube/Files API/https URLs pass through, the `ai-sdk-google` user agent is always added, video provider options override call parameters and function tools send only their input schema.
  Compatibility: Source-compatible; changes are additive. Request bodies, results, warnings and errors change to match the reviewed upstream package.
