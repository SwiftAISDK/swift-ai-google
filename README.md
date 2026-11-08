<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-google 1.2.0

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-google.git", from: "1.2.0")
```

Products: `SwiftAIGoogle`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# SwiftAIGoogle

Swift-native port of the Google adapter from the Vercel AI SDK.

Canonical sources and behavior tests are maintained in
[SwiftAISDK/SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK). This directory is
exported as an independently versioned Swift package. Swift releases and npm
baselines are separate; published release records are in the canonical `Releases/sets`.

```swift
import SwiftAIGoogle
let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "..."))
let model = try provider.languageModel("gemini-3.6-flash")
```

The reviewed upstream baseline is `@ai-sdk/google` 4.0.114. Gemini speech forwards
an explicit language (except `auto`) for both single and multiple voices.
Interactions accepts string-valued `labels`, maps function and provider tools,
and omits every tool when `toolChoice` is `none`. GenerateContent retains
function calls in their original text order and preserves empty signed thought
blocks when replaying messages. Sparse streamed array indexes are ignored.

Gemini Live supports native WebSockets with ephemeral client tokens or resolved
server credentials:

```swift
let realtime = provider.realtimeModel("gemini-3.8-live")
let server = try await realtime.getResolvedServerWebSocketConfig()
let secret = try await realtime.doCreateClientSecret(.init(expiresAfterSeconds: 60))
```

Use `AIRealtimeSession.connectServer` from `SwiftAI` for the shared native socket
lifecycle. The conversation endpoint rejects transcription intent. Each session
gets a separate parser; `generationComplete` does not close a turn, late output
transcripts retain the preceding turn ID, and unassociated usage snapshots stay
separate from completion. `AIRealtimeServerEvent.usage` preserves reported totals
and known modality counts without inventing missing counts. Server connections
resolve async credentials immediately before use. The synchronous descriptor
is intended for static credentials; use its async counterpart with a supplier.

Browser-owned media capture/playback, speech translation and the separate Google
transcription product remain outside this package's native implementation. Mock
tests verify the reviewed contracts; no live-service parity is claimed.

See [provider usage and capabilities](https://github.com/SwiftAISDK/SwiftAISDK/blob/main/docs-site/scripts/provider-guides/google-generative-ai.md) for authentication, options and supported surfaces.
