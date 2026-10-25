@testable import SwiftAIGoogle
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func googleRequestUsesGenerateContentShape() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"gemini"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    #expect(model.providerID == "google.generative-ai")
    let result = try await model.generate(LanguageModelRequest(messages: [.user("Ping")]))

    #expect(result.text == "gemini")
    #expect(result.usage?.totalTokens == 3)
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
    #expect(request.headers["x-goog-api-key"] == "gemini-key")
    #expect(request.headers["user-agent"] == "ai-sdk-google/4.0.105")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["contents"]?[0]?["role"]?.stringValue == "user")
}
@Test func googleGenerateContentSurfacesProviderResponseID() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"responseId":"google-response-123","candidates":[{"content":{"parts":[{"text":"gemini"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-3.6-flash")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Ping")]))

    #expect(result.responseMetadata.id == "google-response-123")
}
@Test func googleAppendsVersionedUserAgentToCustomHeader() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"gemini"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(
        apiKey: "gemini-key",
        headers: ["User-Agent": "CustomApp/1.0"],
        transport: transport
    ))
    let model = try provider.languageModel("gemini-2.5-flash")

    _ = try await model.generate(LanguageModelRequest(messages: [.user("Ping")]))

    let request = try #require(await transport.requests().first)
    #expect(request.headers["x-goog-api-key"] == "gemini-key")
    #expect(request.headers["user-agent"] == "CustomApp/1.0 ai-sdk-google/4.0.105")
}
@Test func googleCustomXGoogAPIKeyOverridesConfiguredAPIKeyLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"gemini"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(
        apiKey: "configured-key",
        headers: ["x-goog-api-key": "custom-key"],
        transport: transport
    ))
    let model = try provider.languageModel("gemini-2.5-flash")

    _ = try await model.generate(LanguageModelRequest(messages: [.user("Ping")]))

    let request = try #require(await transport.requests().first)
    #expect(request.headers["x-goog-api-key"] == "custom-key")
}
@Test func googleGenerateContentResolvesTopLevelInlineMediaType() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"saw image"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")
    let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A])

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [.data(mimeType: "image/*", data: png)])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["contents"]?[0]?["parts"]?[0]?["inlineData"]?["mimeType"]?.stringValue == "image/png")
}
@Test func googleGemmaPrependsSystemInstructionToFirstUserMessage() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"gemma"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemma-3-12b-it")

    let result = try await model.generate(LanguageModelRequest(messages: [.system("Be precise."), .user("Hello")]))

    #expect(result.warnings.isEmpty)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["systemInstruction"] == nil)
    #expect(body["contents"]?[0]?["role"]?.stringValue == "user")
    #expect(body["contents"]?[0]?["parts"]?[0]?["text"]?.stringValue == "Be precise.\n\n")
    #expect(body["contents"]?[0]?["parts"]?[1]?["text"]?.stringValue == "Hello")
}
@Test func googleGeminiStillSendsSystemInstruction() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"gemini"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-pro")

    _ = try await model.generate(LanguageModelRequest(messages: [.system("Be precise."), .user("Hello")]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["systemInstruction"]?["parts"]?[0]?["text"]?.stringValue == "Be precise.")
    #expect(body["contents"]?[0]?["parts"]?[0]?["text"]?.stringValue == "Hello")
}
@Test func googleLanguageMapsStandardStructuredResponseFormat() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"{\\"location\\":\\"Tokyo\\"}"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Where?")],
        responseFormat: .json(schema: [
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": ["location": ["type": "string"]],
            "required": ["location"],
            "additionalProperties": false
        ])
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["generationConfig"]?["responseMimeType"]?.stringValue == "application/json")
    #expect(body["generationConfig"]?["responseJsonSchema"]?["type"]?.stringValue == "object")
    #expect(body["generationConfig"]?["responseJsonSchema"]?["properties"]?["location"]?["type"]?.stringValue == "string")
    #expect(body["generationConfig"]?["responseJsonSchema"]?["required"]?[0]?.stringValue == "location")
    #expect(body["generationConfig"]?["responseJsonSchema"]?["additionalProperties"]?.boolValue == false)
    #expect(body["generationConfig"]?["responseJsonSchema"]?["$schema"]?.stringValue == "http://json-schema.org/draft-07/schema#")
    #expect(body["responseFormat"] == nil)
}
@Test func googleLanguageOmitsResponseSchemaWhenStructuredOutputsDisabled() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"{\\"location\\":\\"Tokyo\\"}"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Where?")],
        responseFormat: .json(schema: [
            "type": "object",
            "properties": ["location": ["type": "string"]]
        ]),
        extraBody: ["google": ["structuredOutputs": false]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["generationConfig"]?["responseMimeType"]?.stringValue == "application/json")
    #expect(body["generationConfig"]?["responseJsonSchema"] == nil)
    #expect(body["structuredOutputs"] == nil)
    #expect(body["google"] == nil)
}
@Test func googleLanguageMapsProviderOptionsSamplingAndReasoning() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"options"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-3-pro")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Options.")],
        topK: 7,
        presencePenalty: 0.2,
        frequencyPenalty: 0.3,
        seed: 123,
        reasoning: "xhigh",
        providerOptions: [
            "google": .object([
                "serviceTier": "flex",
                "sharedRequestType": "priority",
                "requestType": "shared",
                "thinkingConfig": ["includeThoughts": true],
                "responseModalities": ["TEXT"],
                "unsupportedProperty": "drop-me"
            ])
        ]
    ))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    let generationConfig = try #require(body["generationConfig"])
    #expect(generationConfig["topK"]?.intValue == 7)
    #expect(generationConfig["presencePenalty"]?.doubleValue == 0.2)
    #expect(generationConfig["frequencyPenalty"]?.doubleValue == 0.3)
    #expect(generationConfig["seed"]?.intValue == 123)
    #expect(generationConfig["thinkingConfig"]?["thinkingLevel"]?.stringValue == "high")
    #expect(generationConfig["thinkingConfig"]?["includeThoughts"]?.boolValue == true)
    #expect(generationConfig["responseModalities"]?[0]?.stringValue == "TEXT")
    #expect(body["serviceTier"]?.stringValue == "flex")
    #expect(body["sharedRequestType"] == nil)
    #expect(body["requestType"] == nil)
    #expect(body["unsupportedProperty"] == nil)
    #expect(request.headers["X-Vertex-AI-LLM-Shared-Request-Type"] == nil)
    #expect(result.warnings.contains(AIWarning(
        type: "compatibility",
        feature: "reasoning",
        message: "reasoning \"xhigh\" is not directly supported by this model. mapped to effort \"high\"."
    )))
    #expect(result.warnings.contains { $0.type == "other" && ($0.message?.contains("'sharedRequestType' and 'requestType'") ?? false) })
}
@Test func googleLanguageMapsGemini25ReasoningToThinkingBudget() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"budget"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-pro")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Think.")],
        reasoning: "high"
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["generationConfig"]?["thinkingConfig"]?["thinkingBudget"]?.intValue == 32768)
}

@Test func googleGemini37FlashCoercesNoneAndMinimalReasoningToLow() {
    for modelID in ["gemini-3.7-flash", "models/gemini-3.7-flash", "MODELS/GEMINI-3.7-FLASH"] {
        var noneWarnings: [AIWarning] = []
        let none = googleThinkingConfig(for: "none", modelID: modelID, warnings: &noneWarnings)
        #expect(none?["thinkingLevel"]?.stringValue == "low")
        #expect(noneWarnings.isEmpty)

        var minimalWarnings: [AIWarning] = []
        let minimal = googleThinkingConfig(for: "minimal", modelID: modelID, warnings: &minimalWarnings)
        #expect(minimal?["thinkingLevel"]?.stringValue == "low")
        #expect(minimalWarnings.contains(AIWarning(
            type: "compatibility",
            feature: "reasoning",
            message: "reasoning \"minimal\" is not directly supported by this model. mapped to effort \"low\"."
        )))
    }

    var previewWarnings: [AIWarning] = []
    let preview = googleThinkingConfig(
        for: "minimal",
        modelID: "gemini-3.7-flash-preview",
        warnings: &previewWarnings
    )
    #expect(preview?["thinkingLevel"]?.stringValue == "low")
    #expect(previewWarnings.contains(AIWarning(
        type: "compatibility",
        feature: "reasoning",
        message: "reasoning \"minimal\" is not directly supported by this model. mapped to effort \"low\"."
    )))
}
@Test func googleModelCapabilitiesKeepLegacyModelsAndDefaultFutureModelsToNewestBehavior() {
    #expect(googleModelCapabilities(for: "gemini-pro") == GoogleModelCapabilities(
        supportsGemini2Tools: false,
        supportsFileSearch: false,
        usesGemini3Features: false
    ))
    #expect(googleModelCapabilities(for: "gemini-1.5-flash") == GoogleModelCapabilities(
        supportsGemini2Tools: false,
        supportsFileSearch: false,
        usesGemini3Features: false
    ))
    #expect(googleModelCapabilities(for: "gemini-2.5-flash") == GoogleModelCapabilities(
        supportsGemini2Tools: true,
        supportsFileSearch: true,
        usesGemini3Features: false
    ))
    for modelID in [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-99-pro-preview",
        "publishers/google/models/gemini-ultra-latest"
    ] {
        #expect(googleModelCapabilities(for: modelID) == GoogleModelCapabilities(
            supportsGemini2Tools: true,
            supportsFileSearch: true,
            usesGemini3Features: true
        ))
    }
    #expect(googleModelCapabilities(for: "nano-banana-pro-preview") == GoogleModelCapabilities(
        supportsGemini2Tools: true,
        supportsFileSearch: false,
        usesGemini3Features: false
    ))
}
@Test func googleUnknownFutureGeminiUsesNewestRequestBehavior() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-99-pro-preview")
    let call = AIToolCall(
        id: "chart-call",
        name: "createWeatherChart",
        arguments: #"{"location":"San Francisco"}"#
    )
    let result = AIToolResult(
        toolCallID: "chart-call",
        toolName: "createWeatherChart",
        result: [:],
        modelOutput: [
            "type": "content",
            "value": [
                ["type": "text", "text": "Weather chart"],
                [
                    "type": "file",
                    "mediaType": "image/png",
                    "data": ["data": "aW1hZ2U="]
                ]
            ]
        ]
    )

    let generation = try await model.generate(LanguageModelRequest(
        messages: [
            .user("Create a weather chart."),
            .assistant(toolCalls: [call]),
            .toolResult(result)
        ],
        reasoning: "high",
        tools: [
            "createWeatherChart": [
                "type": "object",
                "properties": ["location": ["type": "string"]]
            ],
            "google.file_search": GoogleTools.fileSearch(
                fileSearchStoreNames: ["fileSearchStores/example-store"]
            )
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["generationConfig"]?["thinkingConfig"]?["thinkingLevel"]?.stringValue == "high")
    #expect(body["contents"]?[1]?["parts"]?[0]?["thoughtSignature"]?.stringValue == "skip_thought_signature_validator")
    #expect(body["contents"]?[2]?["parts"]?[0]?["functionResponse"]?["parts"]?[0]?["inlineData"]?["mimeType"]?.stringValue == "image/png")
    let tools = try #require(body["tools"]?.arrayValue)
    #expect(tools.contains { $0["fileSearch"] != nil })
    #expect(tools.contains { $0["functionDeclarations"]?[0]?["name"]?.stringValue == "createWeatherChart" })
    #expect(!generation.warnings.contains {
        $0.type == "unsupported" && $0.feature == "combination of function and provider-defined tools"
    })
}
@Test func googleLanguageWarnsAndDropsVertexOnlyStreamFunctionCallArguments() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"ignored"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-3-pro")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Options.")],
        providerOptions: [
            "google": .object([
                "streamFunctionCallArguments": true
            ])
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["toolConfig"] == nil)
    #expect(body["streamFunctionCallArguments"] == nil)
    #expect(result.warnings.contains { $0.type == "other" && ($0.message?.contains("'streamFunctionCallArguments' is only supported on the Vertex AI API") ?? false) })
}
@Test func googleEmbeddingPreservesRequestMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"embedding":{"values":[0.1,0.2]}}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.embeddingModel("gemini-embedding-001")

    let result = try await model.embed(EmbeddingRequest(values: ["hello"], headers: ["x-client": "swift"]))

    #expect(result.embeddings == [[0.1, 0.2]])
    #expect(result.requestMetadata.body?["content"]?["parts"]?[0]?["text"]?.stringValue == "hello")
    #expect(result.requestMetadata.headers["x-client"] == "swift")
    #expect(result.requestMetadata.headers["x-goog-api-key"] == nil)
}
@Test func googleEmbeddingMapsProviderOptionsAndMultimodalContentLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.embeddingModel("gemini-embedding-001")

    let result = try await model.embed(EmbeddingRequest(
        values: ["first", ""],
        providerOptions: [
            "google": .object([
                "outputDimensionality": 128,
                "taskType": "RETRIEVAL_DOCUMENT",
                "content": [
                    [
                        ["inlineData": ["mimeType": "image/png", "data": "image-data"]]
                    ],
                    [
                        ["fileData": ["fileUri": "https://generativelanguage.googleapis.com/v1beta/files/file-1", "mimeType": "application/pdf"]]
                    ]
                ]
            ])
        ]
    ))

    #expect(result.embeddings == [[0.1, 0.2], [0.3, 0.4]])
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["requests"]?[0]?["content"]?["role"]?.stringValue == "user")
    #expect(body["requests"]?[0]?["content"]?["parts"]?[0]?["text"]?.stringValue == "first")
    #expect(body["requests"]?[0]?["content"]?["parts"]?[1]?["inlineData"]?["data"]?.stringValue == "image-data")
    #expect(body["requests"]?[0]?["outputDimensionality"]?.intValue == 128)
    #expect(body["requests"]?[0]?["taskType"]?.stringValue == "RETRIEVAL_DOCUMENT")
    #expect(body["requests"]?[1]?["content"]?["parts"]?[0]?["fileData"]?["fileUri"]?.stringValue == "https://generativelanguage.googleapis.com/v1beta/files/file-1")
}
@Test func googleEmbeddingRejectsTooManyValuesAndMismatchedMultimodalContentLikeUpstream() async throws {
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: RecordingTransport(response: jsonResponse("{}"))))
    let model = try provider.embeddingModel("gemini-embedding-001")

    #expect(model.maxEmbeddingsPerCall == 100)
    let tooManyValues = Array(repeating: "x", count: 101)
    await #expect(throws: AITooManyEmbeddingValuesForCallError(
        provider: "google.generative-ai",
        modelID: "gemini-embedding-001",
        maxEmbeddingsPerCall: 100,
        values: tooManyValues
    )) {
        _ = try await model.embed(EmbeddingRequest(values: tooManyValues))
    }
    await #expect(throws: AIError.invalidArgument(
        argument: "providerOptions.google.content",
        message: "The number of multimodal content entries (1) must match the number of values (2)."
    )) {
        _ = try await model.embed(EmbeddingRequest(
            values: ["a", "b"],
            providerOptions: ["google": ["content": [[["text": "extra"]]]]]
        ))
    }
}
@Test func googleLanguageMapsFunctionToolsAndToolChoice() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"tool-ready"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Use lookup.")],
        tools: [
            "lookup": [
                "type": "object",
                "description": "Look up a value.",
                "properties": ["query": ["type": "string"]],
                "required": ["query"],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            ]
        ],
        extraBody: ["toolChoice": ["type": "tool", "toolName": "lookup"]]
    ))

    #expect(result.text == "tool-ready")
    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    let declaration = try #require(body["tools"]?[0]?["functionDeclarations"]?[0])
    #expect(declaration["name"]?.stringValue == "lookup")
    #expect(declaration["description"]?.stringValue == "Look up a value.")
    #expect(declaration["parametersJsonSchema"]?["properties"]?["query"]?["type"]?.stringValue == "string")
    #expect(declaration["parametersJsonSchema"]?["required"]?[0]?.stringValue == "query")
    #expect(declaration["parametersJsonSchema"]?["additionalProperties"]?.boolValue == false)
    #expect(declaration["parametersJsonSchema"]?["$schema"]?.stringValue == "http://json-schema.org/draft-07/schema#")
    #expect(body["toolConfig"]?["functionCallingConfig"]?["mode"]?.stringValue == "ANY")
    #expect(body["toolConfig"]?["functionCallingConfig"]?["allowedFunctionNames"]?[0]?.stringValue == "lookup")
    #expect(body["toolChoice"] == nil)
}
@Test func googleLanguageMapsProviderDefinedTools() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"grounded"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Search.")],
        tools: [
            "google.google_search": GoogleTools.googleSearch(searchTypes: ["imageSearch": [:]]),
            "google.code_execution": GoogleTools.codeExecution()
        ],
        extraBody: ["toolChoice": "auto"]
    ))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    let tools = try #require(body["tools"]?.arrayValue)
    #expect(tools.contains { $0["googleSearch"]?["searchTypes"]?["imageSearch"] != nil })
    #expect(tools.contains { $0["codeExecution"]?.objectValue?.isEmpty == true })
    #expect(body["toolConfig"] == nil)
    #expect(body["toolChoice"] == nil)
}
@Test func googleLanguageWarnsForUnsupportedProviderDefinedToolsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"plain"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-pro")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Search.")],
        tools: [
            "google.google_search": GoogleTools.googleSearch(),
            "google.unknown": GoogleTools.providerTool(id: "google.unknown", name: "unknown")
        ]
    ))

    #expect(result.warnings.contains {
        $0.type == "unsupported" && $0.feature == "provider-defined tool google.google_search" && $0.message == "Google Search requires Gemini 2.0 or newer."
    })
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "provider-defined tool google.unknown")))
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["tools"] == nil)
}
@Test func googleLanguageWarnsForMixedFunctionAndProviderToolsBeforeGemini3() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"mixed"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Use tools.")],
        tools: [
            "lookup": ["type": "object", "properties": ["query": ["type": "string"]]],
            "google.google_search": GoogleTools.googleSearch()
        ]
    ))

    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "combination of function and provider-defined tools")))
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let tools = try #require(body["tools"]?.arrayValue)
    #expect(tools.contains { $0["googleSearch"]?.objectValue?.isEmpty == true })
    #expect(!tools.contains { $0["functionDeclarations"] != nil })
}
@Test func googleLanguageWarnsForVertexRagStoreOnNonVertexProvider() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"candidates":[{"content":{"parts":[{"text":"rag"}]},"finishReason":"STOP"}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Use RAG.")],
        tools: [
            "google.vertex_rag_store": GoogleTools.vertexRagStore(ragCorpus: "projects/p/locations/l/ragCorpora/rag-1", topK: 2)
        ]
    ))

    #expect(result.warnings.contains {
        $0.type == "other" && ($0.message?.contains("vertex_rag_store") == true)
    })
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["tools"]?[0]?["retrieval"]?["vertex_rag_store"]?["similarity_top_k"]?.intValue == 2)
}
