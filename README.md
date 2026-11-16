<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-google 1.4.0

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-google.git", from: "1.4.0")
```

Products: `SwiftAIGoogle`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# Google Generative AI (Gemini API) for Swift

`SwiftAIGoogle` adds Google Generative AI (Gemini API) to [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK), an independent Swift port of the Vercel AI SDK. It ports `@ai-sdk/google` and supports text generation, embeddings, image generation, speech, video generation, evaluation, decisions, extraction and provider files.

Provider guide with models, options, streaming and errors:

- [Google Generative AI (Gemini API)](https://swiftaisdk.github.io/SwiftAISDK/providers/google-generative-ai/)

## Usage

Add the `SwiftAI` product from `swift-ai-core` and the `SwiftAIGoogle` product from this package to your target, then create the provider:

```swift
import SwiftAI
import SwiftAIGoogle

let provider = try AIProviders.google(configuration: GoogleProviderSettings())
```

By default the provider reads `GOOGLE_GENERATIVE_AI_API_KEY`, `GOOGLE_API_KEY` and `GEMINI_API_KEY` from the environment; values set in `GoogleProviderSettings` take precedence.

To use every provider through one dependency, add the complete [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK) package instead; see [Install](https://swiftaisdk.github.io/SwiftAISDK/getting-started/install/).

## Contributing

This package is exported from `Packages/swift-ai-google` in the canonical [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK) repository. Open issues and pull requests there.
