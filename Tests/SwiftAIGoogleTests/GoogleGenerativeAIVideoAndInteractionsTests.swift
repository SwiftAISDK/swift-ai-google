@testable import SwiftAIGoogle
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func googleVideoMapsStandardImageAndProviderOptionsLikeUpstream() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"name":"operations/video-2","done":false}"#),
        jsonResponse(#"{"name":"operations/video-2","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"https://generativelanguage.googleapis.com/files/video-456.mp4?alt=media"}}]}}}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.videoModel("veo-3.1-generate-preview")

    let result = try await model.generateVideo(VideoGenerationRequest(
        prompt: "cat running",
        image: ImageInputFile(data: Data("frame".utf8), mediaType: "image/png"),
        resolution: "1280x720",
        seed: 7,
        providerOptions: [
            "google": [
                "referenceImages": [
                    ["bytesBase64Encoded": "reference-image"],
                    ["gcsUri": "gs://bucket/reference.png"]
                ],
                "personGeneration": "allow_adult",
                "negativePrompt": "rain",
                "pollIntervalMs": 0
            ]
        ]
    ))

    #expect(result.urls == ["https://generativelanguage.googleapis.com/files/video-456.mp4?alt=media&key=gemini-key"])
    #expect(result.providerMetadata["google"]?["videos"]?[0]?["uri"]?.stringValue == "https://generativelanguage.googleapis.com/files/video-456.mp4?alt=media")
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["instances"]?[0]?["image"]?["bytesBase64Encoded"]?.stringValue == Data("frame".utf8).base64EncodedString())
    #expect(body["instances"]?[0]?["image"]?["mimeType"]?.stringValue == "image/png")
    #expect(body["instances"]?[0]?["referenceImages"]?[0]?["bytesBase64Encoded"]?.stringValue == "reference-image")
    #expect(body["instances"]?[0]?["referenceImages"]?[1]?["gcsUri"]?.stringValue == "gs://bucket/reference.png")
    #expect(body["parameters"]?["resolution"]?.stringValue == "720p")
    #expect(body["parameters"]?["seed"]?.intValue == 7)
    #expect(body["parameters"]?["personGeneration"]?.stringValue == "allow_adult")
    #expect(body["parameters"]?["negativePrompt"]?.stringValue == "rain")
    #expect(body["parameters"]?["pollIntervalMs"] == nil)
}
@Test func googleVideoWarnsAndIgnoresURLImageLikeUpstream() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"name":"operations/video-url","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"https://generativelanguage.googleapis.com/files/video-url.mp4?alt=media"}}]}}}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.videoModel("veo-3.1-generate-preview")

    let result = try await model.generateVideo(VideoGenerationRequest(
        prompt: "cat running",
        image: ImageInputFile(url: "https://example.com/frame.png")
    ))

    #expect(result.warnings.contains { $0.type == "unsupported" && $0.feature == "URL-based image input" })
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["instances"]?[0]?["image"] == nil)
}

@Test func googleVideoMapsFrameImagesLikeUpstream() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"name":"operations/video-frames","done":false}"#),
        jsonResponse(#"{"name":"operations/video-frames","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"https://generativelanguage.googleapis.com/files/video-frames.mp4?alt=media"}}]}}}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.videoModel("veo-3.1-generate-preview")

    _ = try await model.generateVideo(VideoGenerationRequest(
        prompt: "cat running",
        image: ImageInputFile(data: Data("legacy-image".utf8), mediaType: "image/png"),
        frameImages: [
            VideoFrameImage(image: ImageInputFile(data: Data("first-frame-data".utf8), mediaType: "image/png"), frameType: .firstFrame),
            VideoFrameImage(image: ImageInputFile(data: Data("last-frame-data".utf8), mediaType: "image/jpeg"), frameType: .lastFrame)
        ],
        providerOptions: ["google": ["pollIntervalMs": 0]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["instances"]?[0]?["image"]?["mimeType"]?.stringValue == "image/png")
    #expect(body["instances"]?[0]?["image"]?["bytesBase64Encoded"]?.stringValue == Data("first-frame-data".utf8).base64EncodedString())
    #expect(body["instances"]?[0]?["lastFrame"]?["mimeType"]?.stringValue == "image/jpeg")
    #expect(body["instances"]?[0]?["lastFrame"]?["bytesBase64Encoded"]?.stringValue == Data("last-frame-data".utf8).base64EncodedString())
}

@Test func googleVideoMapsInputReferencesLikeUpstream() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"name":"operations/video-refs","done":false}"#),
        jsonResponse(#"{"name":"operations/video-refs","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"https://generativelanguage.googleapis.com/files/video-refs.mp4?alt=media"}}]}}}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.videoModel("veo-3.1-generate-preview")

    _ = try await model.generateVideo(VideoGenerationRequest(
        prompt: "cat running",
        inputReferences: [
            ImageInputFile(data: Data("reference-from-input".utf8), mediaType: "image/png")
        ],
        providerOptions: [
            "google": [
                "referenceImages": [
                    ["bytesBase64Encoded": "provider-reference"]
                ],
                "pollIntervalMs": 0
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["instances"]?[0]?["referenceImages"]?.arrayValue?.count == 1)
    #expect(body["instances"]?[0]?["referenceImages"]?[0]?["referenceType"]?.stringValue == "asset")
    #expect(body["instances"]?[0]?["referenceImages"]?[0]?["image"]?["mimeType"]?.stringValue == "image/png")
    #expect(body["instances"]?[0]?["referenceImages"]?[0]?["image"]?["bytesBase64Encoded"]?.stringValue == Data("reference-from-input".utf8).base64EncodedString())
}
@Test func googleInteractionsUsesInteractionsEndpointAndInputShape() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"completed","service_tier":"standard","model":"gemini-2.5-flash","usage":{"total_tokens":58,"total_input_tokens":7,"total_output_tokens":19,"total_thought_tokens":32,"total_cached_tokens":0},"steps":[{"type":"thought","summary":[{"type":"text","text":"thinking"}]},{"type":"model_output","content":[{"type":"text","text":"Hello from interactions"}]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.system("Be helpful."), .user("Hello")],
        temperature: 0.3,
        topP: 0.8,
        topK: 10,
        presencePenalty: 0.4,
        frequencyPenalty: 0.5,
        seed: 42,
        maxOutputTokens: 64,
        extraBody: [
            "previousInteractionId": "interaction-old",
            "serviceTier": "flex",
            "store": false,
            "responseModalities": ["text", "image"],
            "responseFormat": [
                ["type": "image", "mimeType": "image/png", "aspectRatio": "1:1", "imageSize": "1K"]
            ],
            "thinkingLevel": "high",
            "thinkingSummaries": true
        ]
    ))

    #expect(result.text == "Hello from interactions")
    #expect(result.finishReason == "stop")
    #expect(result.usage?.inputTokens == 7)
    #expect(result.usage?.outputTokens == 51)
    #expect(result.usage?.totalTokens == 58)
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/interactions")
    #expect(request.headers["x-goog-api-key"] == "gemini-key")
    #expect(request.headers["Api-Revision"] == "2026-05-20")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["model"]?.stringValue == "gemini-2.5-flash")
    #expect(body["system_instruction"]?.stringValue == "Be helpful.")
    #expect(body["input"]?[0]?["type"]?.stringValue == "user_input")
    #expect(body["input"]?[0]?["content"]?[0]?["type"]?.stringValue == "text")
    #expect(body["input"]?[0]?["content"]?[0]?["text"]?.stringValue == "Hello")
    #expect(body["generation_config"]?["temperature"]?.doubleValue == 0.3)
    #expect(body["generation_config"]?["top_p"]?.doubleValue == 0.8)
    #expect(body["generation_config"]?["top_k"]?.intValue == 10)
    #expect(body["generation_config"]?["seed"]?.intValue == 42)
    #expect(body["generation_config"]?["frequency_penalty"] == nil)
    #expect(body["generation_config"]?["presence_penalty"] == nil)
    #expect(body["generation_config"]?["max_output_tokens"]?.intValue == 64)
    #expect(body["generation_config"]?["thinking_level"]?.stringValue == "high")
    #expect(body["generation_config"]?["thinking_summaries"]?.boolValue == true)
    #expect(body["previous_interaction_id"]?.stringValue == "interaction-old")
    #expect(body["service_tier"]?.stringValue == "flex")
    #expect(body["store"]?.boolValue == false)
    #expect(body["response_modalities"]?[0]?.stringValue == "text")
    #expect(body["response_format"]?[0]?["mime_type"]?.stringValue == "image/png")
    #expect(body["response_format"]?[0]?["aspect_ratio"]?.stringValue == "1:1")
    #expect(body["response_format"]?[0]?["image_size"]?.stringValue == "1K")
    #expect(result.warnings == [
        AIWarning(
            type: "other",
            message: "google.interactions: providerOptions.google.previousInteractionId was set together with store: false. These are incoherent (the prior interaction cannot be referenced when nothing was stored on the server); the full history will be sent and previous_interaction_id will still be emitted."
        ),
        AIWarning(type: "unsupported", feature: "frequencyPenalty"),
        AIWarning(type: "unsupported", feature: "presencePenalty")
    ])
}

@Test func googleInteractionsCompactsLinkedAssistantAndToolResultHistory() throws {
    let previousInteractionID = "v1_prev-interaction-abc"
    let call = AIToolCall(
        id: "call-old",
        name: "getWeather",
        arguments: #"{"location":"Boston"}"#
    )
    let result = AIToolResult(
        toolCallID: call.id,
        toolName: call.name,
        result: "sunny"
    )
    let prepared = try googleInteractionsPreparedCall(
        for: LanguageModelRequest(
            messages: [
                .user("q1"),
                AIMessage(role: .assistant, content: [
                    .text(
                        "old answer",
                        providerMetadata: ["google": ["interactionId": .string(previousInteractionID)]]
                    ),
                    .toolCall(call)
                ]),
                .toolResult(result),
                .user("q2")
            ],
            providerOptions: [
                "google": ["previousInteractionId": .string(previousInteractionID)]
            ]
        ),
        modelID: "gemini-3.8-flash",
        agent: nil,
        stream: false
    )

    let input = try #require(prepared.body["input"]?.arrayValue)
    #expect(input.map { $0["type"]?.stringValue } == ["user_input", "user_input"])
    #expect(input[0]["content"]?[0]?["text"]?.stringValue == "q1")
    #expect(input[1]["content"]?[0]?["text"]?.stringValue == "q2")
    #expect(prepared.body["previous_interaction_id"]?.stringValue == previousInteractionID)
    #expect(prepared.warnings.isEmpty)
}

@Test func googleInteractionsPreservesHistoryAndWarnsWhenPreviousInteractionIsNotStored() throws {
    let previousInteractionID = "v1_prev-interaction-abc"
    let prepared = try googleInteractionsPreparedCall(
        for: LanguageModelRequest(
            messages: [
                .user("q1"),
                AIMessage(role: .assistant, content: [
                    .text(
                        "old answer",
                        providerMetadata: ["google": ["interactionId": .string(previousInteractionID)]]
                    )
                ]),
                .user("q2")
            ],
            providerOptions: [
                "google": [
                    "previousInteractionId": .string(previousInteractionID),
                    "store": false
                ]
            ]
        ),
        modelID: "gemini-3.8-flash",
        agent: nil,
        stream: false
    )

    let input = try #require(prepared.body["input"]?.arrayValue)
    #expect(input.map { $0["type"]?.stringValue } == ["user_input", "model_output", "user_input"])
    #expect(prepared.body["previous_interaction_id"]?.stringValue == previousInteractionID)
    #expect(prepared.body["store"]?.boolValue == false)
    #expect(prepared.warnings == [AIWarning(
        type: "other",
        message: "google.interactions: providerOptions.google.previousInteractionId was set together with store: false. These are incoherent (the prior interaction cannot be referenced when nothing was stored on the server); the full history will be sent and previous_interaction_id will still be emitted."
    )])
}

@Test func googleInteractionsRejectsProviderReferenceWithoutGoogleEntry() {
    let reference = ["openai": "file-openai-only"]
    #expect(throws: AINoSuchProviderReferenceError(provider: "google", reference: reference)) {
        try googleInteractionsPreparedCall(
            for: LanguageModelRequest(messages: [
                AIMessage(role: .user, content: [
                    .providerReference(mimeType: "image/png", reference: reference)
                ])
            ]),
            modelID: "gemini-3.8-flash",
            agent: nil,
            stream: false
        )
    }
}

@Test func googleInteractionsThreadsValidatedMediaResolutionOntoImageBlocks() throws {
    let prepared = try googleInteractionsPreparedCall(
        for: LanguageModelRequest(
            messages: [
                AIMessage(role: .user, content: [
                    .data(mimeType: "image/png", data: Data([1, 2, 3, 4]))
                ])
            ],
            providerOptions: ["google": ["mediaResolution": "high"]]
        ),
        modelID: "gemini-3.8-flash",
        agent: nil,
        stream: false
    )

    let image = try #require(prepared.body["input"]?[0]?["content"]?[0])
    #expect(image["type"]?.stringValue == "image")
    #expect(image["data"]?.stringValue == "AQIDBA==")
    #expect(image["mime_type"]?.stringValue == "image/png")
    #expect(image["resolution"]?.stringValue == "high")
}

@Test func googleInteractionsRejectsInvalidMediaResolution() {
    #expect(throws: AIError.invalidArgument(
        argument: "providerOptions.google.mediaResolution",
        message: "Google Interactions mediaResolution must be low, medium, high, or ultra_high."
    )) {
        try googleInteractionsPreparedCall(
            for: LanguageModelRequest(
                messages: [.user("Describe this")],
                providerOptions: ["google": ["mediaResolution": "maximum"]]
            ),
            modelID: "gemini-3.8-flash",
            agent: nil,
            stream: false
        )
    }
}

@Test func googleInteractionsUsesProviderSystemInstructionAsFallback() throws {
    let prepared = try googleInteractionsPreparedCall(
        for: LanguageModelRequest(
            messages: [.user("Hello")],
            providerOptions: [
                "google": ["systemInstruction": "Be concise."]
            ]
        ),
        modelID: "gemini-3.8-flash",
        agent: nil,
        stream: false
    )

    #expect(prepared.body["system_instruction"]?.stringValue == "Be concise.")
    #expect(prepared.warnings.isEmpty)
}

@Test func googleInteractionsPrefersSystemMessageAndWarnsOnProviderConflict() throws {
    let prepared = try googleInteractionsPreparedCall(
        for: LanguageModelRequest(
            messages: [.system("Use the message."), .user("Hello")],
            providerOptions: [
                "google": ["systemInstruction": "Use the provider option."]
            ]
        ),
        modelID: "gemini-3.8-flash",
        agent: nil,
        stream: false
    )

    #expect(prepared.body["system_instruction"]?.stringValue == "Use the message.")
    #expect(prepared.warnings == [AIWarning(
        type: "other",
        message: "google.interactions: both AI SDK system message and providerOptions.google.systemInstruction were set; using the AI SDK system message."
    )])
}

@Test func googleInteractionsMapsStandardStructuredResponseFormat() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"{\\"name\\":\\"Ada\\",\\"age\\":36}"}]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Person?")],
        responseFormat: .json(schema: [
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": [
                "name": ["type": "string", "description": "Full name."],
                "age": ["type": "number", "description": "Age in years."]
            ],
            "required": ["name", "age"],
            "additionalProperties": false
        ])
    ))

    #expect(result.text == "{\"name\":\"Ada\",\"age\":36}")
    #expect(result.warnings.isEmpty)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["response_format"]?[0]?["type"]?.stringValue == "text")
    #expect(body["response_format"]?[0]?["mime_type"]?.stringValue == "application/json")
    #expect(body["response_format"]?[0]?["schema"]?["$schema"]?.stringValue == "http://json-schema.org/draft-07/schema#")
    #expect(body["response_format"]?[0]?["schema"]?["additionalProperties"]?.boolValue == false)
    #expect(body["response_format"]?[0]?["schema"]?["properties"]?["name"]?["description"]?.stringValue == "Full name.")
    #expect(body["responseFormat"] == nil)
}
@Test func googleInteractionsCombinesCallAndProviderResponseFormats() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"{\\"ok\\":true}"}]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("JSON and image.")],
        responseFormat: .json(),
        extraBody: [
            "google": [
                "responseFormat": [
                    ["type": "image", "mimeType": "image/png", "aspectRatio": "1:1"]
                ]
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["response_format"]?[0]?["type"]?.stringValue == "text")
    #expect(body["response_format"]?[0]?["mime_type"]?.stringValue == "application/json")
    #expect(body["response_format"]?[0]?["schema"] == nil)
    #expect(body["response_format"]?[1]?["type"]?.stringValue == "image")
    #expect(body["response_format"]?[1]?["mime_type"]?.stringValue == "image/png")
    #expect(body["response_format"]?[1]?["aspect_ratio"]?.stringValue == "1:1")
    #expect(body["google"] == nil)
}
@Test func googleInteractionsAgentDropsStandardResponseFormatWithWarning() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"id":"agent-interaction","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"agent done"}]}]}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsAgent("deep-research")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Research.")],
        responseFormat: .json(schema: ["type": "object"])
    ))

    #expect(result.text == "agent done")
    #expect(result.warnings == [
        AIWarning(
            type: "other",
            message: "google.interactions: structured output (responseFormat) is not supported when an agent is set; responseFormat will be ignored."
        )
    ])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["agent"]?.stringValue == "deep-research")
    #expect(body["response_format"] == nil)
}
@Test func googleInteractionsResolvesTopLevelInlineMediaType() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"saw image"}]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")
    let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A])

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [.data(mimeType: "image", data: png)])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["input"]?[0]?["content"]?[0]?["type"]?.stringValue == "image")
    #expect(body["input"]?[0]?["content"]?[0]?["mime_type"]?.stringValue == "image/png")
}
@Test func googleInteractionsExtractsSourcesAndProviderMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"completed","service_tier":"standard","usage":{"total_tokens":12,"total_input_tokens":3,"total_output_tokens":4},"steps":[{"type":"model_output","content":[{"type":"text","text":"Grounded answer","annotations":[{"type":"url_citation","url":"https://example.com/article","title":"Example Article"},{"type":"file_citation","document_uri":"gs://bucket/path/report.pdf","file_name":"report.pdf"},{"type":"place_citation","url":"https://maps.google.com/?q=foo","name":"Foo Place"}]}]},{"type":"url_context_result","call_id":"url-1","result":[{"url":"https://context.example.com/a","status":"success"},{"url":"https://context.example.com/b","status":"error"}]},{"type":"google_search_result","call_id":"search-1","result":[{"url":"https://news.example.com/1","title":"Article 1"},{"search_suggestions":"<html/>"}]},{"type":"file_search_result","call_id":"file-1","result":[{"file_name":"notes.md","source":"fileSearchStores/x/notes.md"},{"document_uri":"https://storage.example.com/file.txt"}]},{"type":"google_maps_result","call_id":"maps-1","result":[{"places":[{"name":"Bar Cafe","url":"https://maps.google.com/?q=bar"},{"name":"No URL"}]}]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Ground it.")]))

    #expect(result.text == "Grounded answer")
    #expect(result.providerMetadata["google"]?["interactionId"]?.stringValue == "interaction-1")
    #expect(result.providerMetadata["google"]?["serviceTier"]?.stringValue == "standard")
    #expect(result.sources.count == 8)
    #expect(result.sources[0].sourceType == "url")
    #expect(result.sources[0].url == "https://example.com/article")
    #expect(result.sources[0].title == "Example Article")
    #expect(result.sources[1].sourceType == "document")
    #expect(result.sources[1].title == "report.pdf")
    #expect(result.sources[1].mediaType == "application/pdf")
    #expect(result.sources[1].filename == "report.pdf")
    #expect(result.sources[2].sourceType == "url")
    #expect(result.sources[2].url == "https://maps.google.com/?q=foo")
    #expect(result.sources[2].title == "Foo Place")
    #expect(result.sources[3].url == "https://context.example.com/a")
    #expect(result.sources[4].url == "https://news.example.com/1")
    #expect(result.sources[4].title == "Article 1")
    #expect(result.sources[5].sourceType == "document")
    #expect(result.sources[5].title == "notes.md")
    #expect(result.sources[5].mediaType == "text/markdown")
    #expect(result.sources[5].filename == "notes.md")
    #expect(result.sources[6].sourceType == "url")
    #expect(result.sources[6].url == "https://storage.example.com/file.txt")
    #expect(result.sources[7].sourceType == "url")
    #expect(result.sources[7].url == "https://maps.google.com/?q=bar")
    #expect(result.sources[7].title == "Bar Cafe")
}
@Test func googleInteractionsStreamsTextAndFinishUsage() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"interaction":{"id":"interaction-1","status":"in_progress"},"event_type":"interaction.created"}

    data: {"index":0,"step":{"type":"model_output"},"event_type":"step.start"}

    data: {"index":0,"delta":{"type":"text","text":"hello "},"event_type":"step.delta"}

    data: {"index":0,"delta":{"type":"text","text":"world"},"event_type":"step.delta"}

    data: {"interaction":{"id":"interaction-1","status":"completed","usage":{"total_tokens":12,"total_input_tokens":3,"total_output_tokens":4,"total_thought_tokens":5}},"event_type":"interaction.completed"}

    data: [DONE]

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    var deltas: [String] = []
    var finishReason: String?
    var totalTokens: Int?
    var outputTokens: Int?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hello")])) {
        switch part {
        case let .textDeltaPart(_, delta, _):
            deltas.append(delta)
        case let .finishMetadata(reason, usage, _):
            finishReason = reason
            totalTokens = usage?.totalTokens
            outputTokens = usage?.outputTokens
        default:
            break
        }
    }

    #expect(deltas == ["hello ", "world"])
    #expect(finishReason == "stop")
    #expect(totalTokens == 12)
    #expect(outputTokens == 9)
    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["stream"] == true)
    #expect(body["model"]?.stringValue == "gemini-2.5-flash")
}
@Test func googleInteractionsStreamsSourcesAndMetadata() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"interaction":{"id":"interaction-1","status":"in_progress","service_tier":"standard"},"event_type":"interaction.created"}

    data: {"index":0,"step":{"type":"model_output","content":[{"type":"text","text":"","annotations":[{"type":"url_citation","url":"https://example.com/article","title":"Example Article"}]}]},"event_type":"step.start"}

    data: {"index":0,"delta":{"type":"text","text":"hello"},"event_type":"step.delta"}

    data: {"index":0,"delta":{"type":"text_annotation","annotations":[{"type":"url_citation","url":"https://example.com/article","title":"Example Article"},{"type":"file_citation","document_uri":"gs://bucket/report.pdf","file_name":"report.pdf"}]},"event_type":"step.delta"}

    data: {"index":1,"step":{"type":"google_search_result","call_id":"search-1","result":[{"url":"https://news.example.com/1","title":"Article 1"}]},"event_type":"step.start"}

    data: {"interaction":{"id":"interaction-1","status":"completed","service_tier":"priority","usage":{"total_tokens":12,"total_input_tokens":3,"total_output_tokens":4,"total_thought_tokens":5}},"event_type":"interaction.completed"}

    data: [DONE]

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    var text: [String] = []
    var sources: [AISource] = []
    var metadata: [[String: JSONValue]] = []
    var totalTokens: Int?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hello")])) {
        switch part {
        case let .textDeltaPart(_, delta, _):
            text.append(delta)
        case let .source(source):
            sources.append(source)
        case let .metadata(value):
            metadata.append(value)
        case let .finishMetadata(_, usage, _):
            totalTokens = usage?.totalTokens
        default:
            break
        }
    }

    #expect(text == ["hello"])
    #expect(sources.count == 3)
    #expect(sources[0].url == "https://example.com/article")
    #expect(sources[0].title == "Example Article")
    #expect(sources[1].sourceType == "document")
    #expect(sources[1].mediaType == "application/pdf")
    #expect(sources[1].filename == "report.pdf")
    #expect(sources[2].url == "https://news.example.com/1")
    #expect(sources[2].title == "Article 1")
    #expect(metadata.first?["google"]?["interactionId"]?.stringValue == "interaction-1")
    #expect(metadata.first?["google"]?["serviceTier"]?.stringValue == "standard")
    #expect(metadata.last?["google"]?["serviceTier"]?.stringValue == "priority")
    #expect(totalTokens == 12)
}
@Test func googleInteractionsParsesFunctionCallSteps() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"requires_action","usage":{"total_tokens":109,"total_input_tokens":53,"total_output_tokens":15,"total_thought_tokens":41},"steps":[{"type":"thought","signature":"sig"},{"id":"zggxzq8r","type":"function_call","name":"getWeather","arguments":{"location":"San Francisco"}}],"model":"gemini-2.5-flash"}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Weather?")]))

    #expect(result.text == "")
    #expect(result.finishReason == "tool-calls")
    #expect(result.usage?.totalTokens == 109)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].id == "zggxzq8r")
    #expect(result.toolCalls[0].name == "getWeather")
    #expect(try decodeJSONBody(Data(result.toolCalls[0].arguments.utf8))["location"]?.stringValue == "San Francisco")
}
@Test func googleInteractionsStreamsFunctionCallSteps() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"interaction":{"id":"interaction-1","status":"in_progress"},"event_type":"interaction.created"}

    data: {"index":1,"step":{"id":"61nzpsv4","signature":"","type":"function_call","name":"getWeather","arguments":{}},"event_type":"step.start"}

    data: {"index":1,"delta":{"arguments":"{\\"location\\":\\"San Francisco\\"}","type":"arguments_delta"},"event_type":"step.delta"}

    data: {"index":1,"event_type":"step.stop"}

    data: {"interaction":{"id":"interaction-1","status":"requires_action","usage":{"total_tokens":133,"total_input_tokens":53,"total_output_tokens":15,"total_thought_tokens":65}},"event_type":"interaction.completed"}

    data: [DONE]

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsModel("gemini-2.5-flash")

    var deltas: [String] = []
    var inputLifecycle: [String] = []
    var finalCall: AIToolCall?
    var finishReason: String?
    var totalTokens: Int?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Weather?")])) {
        switch part {
        case let .toolInputStart(id, name, _, _, _, _):
            inputLifecycle.append("start:\(id):\(name)")
        case let .toolInputDelta(id, delta, _):
            inputLifecycle.append("delta:\(id):\(delta)")
        case let .toolInputEnd(id, _):
            inputLifecycle.append("end:\(id)")
        case let .toolCallDelta(_, _, argumentsDelta, _):
            deltas.append(argumentsDelta)
        case let .toolCall(call):
            finalCall = call
        case let .finishMetadata(reason, usage, _):
            finishReason = reason
            totalTokens = usage?.totalTokens
        default:
            break
        }
    }

    let call = try #require(finalCall)
    #expect(deltas == [#"{"location":"San Francisco"}"#])
    #expect(inputLifecycle == [
        "start:61nzpsv4:getWeather",
        #"delta:61nzpsv4:{"location":"San Francisco"}"#,
        "end:61nzpsv4"
    ])
    #expect(call.id == "61nzpsv4")
    #expect(call.name == "getWeather")
    #expect(try decodeJSONBody(Data(call.arguments.utf8))["location"]?.stringValue == "San Francisco")
    #expect(finishReason == "tool-calls")
    #expect(totalTokens == 133)
}
@Test func googleInteractionsAgentUsesAgentAndBackgroundBody() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"id":"agent-interaction","status":"in_progress"}"#),
        jsonResponse(#"{"id":"agent-interaction","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"agent done"}]}],"usage":{"total_tokens":4,"total_input_tokens":1,"total_output_tokens":3}}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsAgent("deep-research")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Research")],
        extraBody: [
            "background": true,
            "agentConfig": ["type": "deep-research", "thinkingSummaries": true, "collaborativePlanning": false],
            "environment": ["type": "remote"]
        ]
    ))

    #expect(result.text == "agent done")
    let requests = await transport.requests()
    #expect(requests.count == 2)
    let body = try decodeJSONBody(try #require(requests[0].body))
    #expect(body["agent"]?.stringValue == "deep-research")
    #expect(body["model"] == nil)
    #expect(body["background"]?.boolValue == true)
    #expect(body["agent_config"]?["type"]?.stringValue == "deep-research")
    #expect(body["agent_config"]?["thinking_summaries"]?.boolValue == true)
    #expect(body["agent_config"]?["collaborative_planning"]?.boolValue == false)
    #expect(body["environment"]?["type"]?.stringValue == "remote")
    #expect(body["generation_config"] == nil)
    #expect(requests[1].method == "GET")
    #expect(requests[1].url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/interactions/agent-interaction")
}

@Test func googleInteractionsAgentWarnsForEveryDroppedGenerationSetting() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"agent-interaction","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"agent done"}]}]}"#))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = provider.interactionsAgent("deep-research")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Research")],
        temperature: 0.5,
        topP: 0.9,
        topK: 10,
        presencePenalty: 0.4,
        frequencyPenalty: 0.3,
        seed: 42,
        maxOutputTokens: 100,
        stopSequences: ["stop"],
        providerOptions: [
            "google": [
                "thinkingLevel": "high",
                "thinkingSummaries": true,
                "imageConfig": ["aspectRatio": "1:1"]
            ]
        ]
    ))

    #expect(result.warnings == [AIWarning(
        type: "other",
        message: "google.interactions: temperature, topP, topK, frequencyPenalty, presencePenalty, seed, stopSequences, maxOutputTokens, thinkingLevel, thinkingSummaries, imageConfig are not supported when an agent is set; use providerOptions.google.agentConfig instead. Dropped from the request body."
    )])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["generation_config"] == nil)
}

@Test func googleInteractionsMapsVideoProcessingAndReplaysProcessingCustomParts() async throws {
    let transport = RecordingTransport(response: jsonResponse(
        #"{"id":"interaction-next","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"ok"}]}]}"#
    ))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let model = provider.interactionsModel("gemini-3.8-flash")
    let assistant = AIMessage(role: .assistant, content: [
        .custom(
            ["kind": "google.processing_call"],
            providerMetadata: ["google": ["processingId": "processing-1", "signature": "call-signature"]]
        ),
        .custom(
            ["kind": "google.processing_result"],
            providerMetadata: ["google": ["processingCallId": "processing-1", "signature": "result-signature"]]
        )
    ])
    let result = try await model.generate(LanguageModelRequest(
        messages: [
            AIMessage(role: .user, content: [
                .providerReference(
                    mimeType: "video",
                    reference: ["google": "https://www.youtube.com/watch?v=abc123"],
                    providerMetadata: ["google": ["processing": "agentic"]]
                ),
                .data(
                    mimeType: "video/mp4",
                    data: Data([5, 6, 7]),
                    providerMetadata: ["google": ["processing": [
                        "type": "static",
                        "startOffset": 1_200,
                        "endOffset": 1_500,
                        "fps": 0.5
                    ]]]
                ),
                .providerReference(
                    mimeType: "video/mp4",
                    reference: ["google": "https://example.test/invalid.mp4"],
                    providerMetadata: ["google": ["processing": "invalid"]]
                )
            ]),
            assistant
        ],
        extraBody: [
            "responseModalities": ["video"],
            "responseFormat": [[
                "type": "video",
                "aspectRatio": "16:9",
                "resolution": "360p",
                "duration": "4s",
                "delivery": "uri",
                "gcsUri": "gs://video-output/clip.mp4"
            ]]
        ]
    ))

    #expect(result.text == "ok")
    #expect(result.warnings.contains {
        $0.type == "other" && $0.message?.contains("invalid providerOptions.google.processing") == true
    })
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let videos = try #require(body["input"]?[0]?["content"]?.arrayValue)
    #expect(videos[0]["type"]?.stringValue == "video")
    #expect(videos[0]["mime_type"] == nil)
    #expect(videos[0]["processing"]?.stringValue == "agentic")
    #expect(videos[1]["data"]?.stringValue == "BQYH")
    #expect(videos[1]["processing"]?["type"]?.stringValue == "static")
    #expect(videos[1]["processing"]?["start_offset"]?.stringValue == "1200s")
    #expect(videos[1]["processing"]?["end_offset"]?.stringValue == "1500s")
    #expect(videos[1]["processing"]?["fps"]?.doubleValue == 0.5)
    #expect(videos[2]["processing"] == nil)
    #expect(body["response_modalities"]?[0]?.stringValue == "video")
    #expect(body["response_format"]?[0]?["type"]?.stringValue == "video")
    #expect(body["response_format"]?[0]?["aspect_ratio"]?.stringValue == "16:9")
    #expect(body["response_format"]?[0]?["gcs_uri"]?.stringValue == "gs://video-output/clip.mp4")
    #expect(body["input"]?[1]?["type"]?.stringValue == "processing_call")
    #expect(body["input"]?[1]?["id"]?.stringValue == "processing-1")
    #expect(body["input"]?[1]?["signature"]?.stringValue == "call-signature")
    #expect(body["input"]?[2]?["type"]?.stringValue == "processing_result")
    #expect(body["input"]?[2]?["call_id"]?.stringValue == "processing-1")
    #expect(body["input"]?[2]?["signature"]?.stringValue == "result-signature")
}

@Test func googleInteractionsParsesVideoAndProcessingStepsWithMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-1","status":"completed","usage":{"total_output_tokens":10,"output_tokens_by_modality":[{"modality":"video","tokens":8},{"modality":"text","tokens":2}]},"steps":[{"type":"processing_call","id":"processing-1","signature":"call-signature"},{"type":"processing_result","call_id":"processing-1","signature":"result-signature"},{"type":"model_output","content":[{"type":"video","mime_type":"video/mp4","data":"BQYH"},{"type":"video","uri":"https://example.test/clip.mp4"}]}]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let result = try await provider.interactionsModel("gemini-3.8-flash").generate(
        LanguageModelRequest(messages: [.user("Make video")])
    )

    #expect(result.content.count == 4)
    guard case let .custom(call, callMetadata) = result.content[0] else {
        Issue.record("Expected processing call custom part")
        return
    }
    #expect(call["kind"]?.stringValue == "google.processing_call")
    #expect(callMetadata["google"]?["interactionId"]?.stringValue == "interaction-1")
    #expect(callMetadata["google"]?["processingId"]?.stringValue == "processing-1")
    #expect(callMetadata["google"]?["signature"]?.stringValue == "call-signature")
    guard case let .custom(processingResult, resultMetadata) = result.content[1] else {
        Issue.record("Expected processing result custom part")
        return
    }
    #expect(processingResult["kind"]?.stringValue == "google.processing_result")
    #expect(resultMetadata["google"]?["processingCallId"]?.stringValue == "processing-1")
    guard case let .file(inlineVideo) = result.content[2],
          case let .file(urlVideo) = result.content[3] else {
        Issue.record("Expected inline and URL video files")
        return
    }
    #expect(inlineVideo.mediaType == "video/mp4")
    #expect(inlineVideo.data == Data([5, 6, 7]))
    #expect(inlineVideo.providerMetadata["google"]?["interactionId"]?.stringValue == "interaction-1")
    #expect(urlVideo.mediaType == "video/mp4")
    #expect(urlVideo.url == "https://example.test/clip.mp4")
    #expect(result.providerMetadata["google"]?["outputTokensByModality"]?["video"]?.intValue == 8)
    #expect(result.providerMetadata["google"]?["outputTokensByModality"]?["text"]?.intValue == 2)
}

@Test func googleInteractionsStreamsVideoAndProcessingCustomParts() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"interaction":{"id":"interaction-1","status":"in_progress"},"event_type":"interaction.created"}

    data: {"index":0,"step":{"type":"processing_call","id":"processing-1","signature":"call-signature"},"event_type":"step.start"}

    data: {"index":0,"event_type":"step.stop"}

    data: {"index":1,"step":{"type":"processing_result","call_id":"processing-1","signature":"result-signature"},"event_type":"step.start"}

    data: {"index":1,"event_type":"step.stop"}

    data: {"index":2,"step":{"type":"model_output"},"event_type":"step.start"}

    data: {"index":2,"delta":{"type":"video","data":"BQYH","mime_type":"video/mp4"},"event_type":"step.delta"}

    data: {"index":2,"event_type":"step.stop"}

    data: {"interaction":{"id":"interaction-1","status":"completed","usage":{"total_output_tokens":10,"output_tokens_by_modality":[{"modality":"video","tokens":8},{"modality":"text","tokens":2}]}},"event_type":"interaction.completed"}

    data: [DONE]
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    var files: [AIStreamFile] = []
    var custom: [(JSONValue, [String: JSONValue])] = []
    var finishMetadata: [String: JSONValue] = [:]
    var semanticOrder: [String] = []
    for try await part in provider.interactionsModel("gemini-3.8-flash").stream(
        LanguageModelRequest(messages: [.user("Make video")])
    ) {
        if case let .file(file) = part {
            files.append(file)
            semanticOrder.append("file")
        }
        if case let .custom(value, metadata) = part {
            custom.append((value, metadata))
            semanticOrder.append("custom")
        }
        if case let .finishMetadata(_, _, metadata) = part {
            finishMetadata = metadata
            semanticOrder.append("finish")
        }
    }

    #expect(custom.count == 2)
    #expect(custom[0].0["kind"]?.stringValue == "google.processing_call")
    #expect(custom[0].1["google"]?["processingId"]?.stringValue == "processing-1")
    #expect(custom[1].0["kind"]?.stringValue == "google.processing_result")
    #expect(custom[1].1["google"]?["processingCallId"]?.stringValue == "processing-1")
    #expect(files.count == 1)
    #expect(files[0].mediaType == "video/mp4")
    #expect(files[0].data == Data([5, 6, 7]))
    #expect(files[0].providerMetadata["google"]?["interactionId"]?.stringValue == "interaction-1")
    #expect(finishMetadata["google"]?["outputTokensByModality"]?["video"]?.intValue == 8)
    #expect(semanticOrder == ["custom", "custom", "file", "finish"])
}

@Test func googleInteractionsReplaysAssistantHistoryAsOrderedStandaloneSteps() async throws {
    let transport = RecordingTransport(response: jsonResponse(
        #"{"id":"interaction-next","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"done"}]}]}"#
    ))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let call = AIToolCall(
        id: "call-weather",
        name: "getWeather",
        arguments: #"{"location":"Tokyo"}"#,
        providerMetadata: ["google": ["signature": "call-signature"]]
    )
    let toolResult = AIToolResult(
        toolCallID: "call-weather",
        toolName: "getWeather",
        result: .null,
        modelOutput: ["type": "json", "value": ["condition": "sunny", "temperature": 24]],
        providerMetadata: ["google": ["signature": "result-signature"]]
    )

    _ = try await provider.interactionsModel("gemini-3.8-flash").generate(LanguageModelRequest(messages: [
        .user("Weather?"),
        AIMessage(role: .assistant, content: [
            .text("Before thought."),
            .reasoning("Check the forecast.", providerMetadata: ["google": ["signature": "thought-signature"]]),
            .reasoning("", providerMetadata: ["google": ["signature": "empty-thought-signature"]]),
            .text("Before tool."),
            .toolCall(call),
            .text("After tool.")
        ]),
        .toolResult(toolResult)
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let input = try #require(body["input"]?.arrayValue)
    #expect(input.map { $0["type"]?.stringValue } == [
        "user_input", "model_output", "thought", "thought", "model_output", "function_call", "model_output", "user_input"
    ])
    #expect(input[1]["content"]?[0]?["text"]?.stringValue == "Before thought.")
    #expect(input[2]["signature"]?.stringValue == "thought-signature")
    #expect(input[2]["summary"]?[0]?["text"]?.stringValue == "Check the forecast.")
    #expect(input[3]["signature"]?.stringValue == "empty-thought-signature")
    #expect(input[3]["summary"] == nil)
    #expect(input[4]["content"]?[0]?["text"]?.stringValue == "Before tool.")
    #expect(input[5]["id"]?.stringValue == "call-weather")
    #expect(input[5]["name"]?.stringValue == "getWeather")
    #expect(input[5]["arguments"]?["location"]?.stringValue == "Tokyo")
    #expect(input[5]["signature"]?.stringValue == "call-signature")
    #expect(input[6]["content"]?[0]?["text"]?.stringValue == "After tool.")
    let functionResult = try #require(input[7]["content"]?[0])
    #expect(functionResult["type"]?.stringValue == "function_result")
    #expect(functionResult["call_id"]?.stringValue == "call-weather")
    #expect(functionResult["name"]?.stringValue == "getWeather")
    #expect(functionResult["result"]?.stringValue == #"{"condition":"sunny","temperature":24}"#)
    #expect(functionResult["signature"]?.stringValue == "result-signature")
}

@Test func googleInteractionsStreamsThoughtSummaryAndUpdatedSignature() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"interaction":{"id":"interaction-reasoning","status":"in_progress"},"event_type":"interaction.created"}

    data: {"index":0,"step":{"type":"thought","signature":"initial-signature","summary":[{"type":"text","text":"Initial thought. "}]},"event_type":"step.start"}

    data: {"index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"Continued thought."}},"event_type":"step.delta"}

    data: {"index":0,"delta":{"type":"thought_signature","signature":"final-signature"},"event_type":"step.delta"}

    data: {"index":0,"event_type":"step.stop"}

    data: {"interaction":{"id":"interaction-reasoning","status":"completed"},"event_type":"interaction.completed"}

    data: [DONE]
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    var lifecycle: [String] = []
    var endMetadata: [String: JSONValue] = [:]

    for try await part in provider.interactionsModel("gemini-3.8-flash").stream(
        LanguageModelRequest(messages: [.user("Think")])
    ) {
        switch part {
        case let .reasoningStart(id, _):
            lifecycle.append("start:\(id)")
        case let .reasoningDeltaPart(id, delta, _):
            lifecycle.append("delta:\(id):\(delta)")
        case let .reasoningEnd(id, metadata):
            lifecycle.append("end:\(id)")
            endMetadata = metadata
        default:
            break
        }
    }

    #expect(lifecycle == [
        "start:reasoning-0",
        "delta:reasoning-0:Initial thought. ",
        "delta:reasoning-0:Continued thought.",
        "end:reasoning-0"
    ])
    #expect(endMetadata["google"]?["interactionId"]?.stringValue == "interaction-reasoning")
    #expect(endMetadata["google"]?["signature"]?.stringValue == "final-signature")
}

@Test func googleInteractionsParsesBuiltInToolStepsInWireOrder() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"interaction-tools","status":"completed","steps":[
      {"type":"google_search_call","id":"search-1","arguments":{"queries":["Swift AI SDK"]}},
      {"type":"google_search_result","call_id":"search-1","result":[{"url":"https://example.test/result","title":"Result"}]},
      {"type":"mcp_server_tool_call","id":"mcp-1","name":"lookup","arguments":{"query":"status"}},
      {"type":"mcp_server_tool_result","call_id":"mcp-1","name":"lookup","result":{"answer":"ready"}},
      {"type":"model_output","content":[{"type":"text","text":"Done."}]}
    ]}
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let result = try await provider.interactionsModel("gemini-3.8-flash").generate(
        LanguageModelRequest(messages: [.user("Search")])
    )

    #expect(result.finishReason == "stop")
    #expect(result.content.count == 6)
    guard case let .toolCall(searchCall) = result.content[0],
          case let .toolResult(searchResult) = result.content[1],
          case let .source(source) = result.content[2],
          case let .toolCall(mcpCall) = result.content[3],
          case let .toolResult(mcpResult) = result.content[4],
          case let .text(text, _) = result.content[5] else {
        Issue.record("Expected built-in tool calls, results, source, and text in wire order")
        return
    }
    #expect(searchCall.id == "search-1")
    #expect(searchCall.name == "google_search")
    #expect(searchCall.providerExecuted)
    #expect(searchResult.toolCallID == "search-1")
    #expect(searchResult.toolName == "google_search")
    #expect(searchResult.providerExecuted)
    #expect(source.url == "https://example.test/result")
    #expect(mcpCall.name == "lookup")
    #expect(mcpCall.providerExecuted)
    #expect(mcpResult.toolName == "lookup")
    #expect(mcpResult.providerExecuted)
    #expect(text == "Done.")
}

@Test func googleInteractionsStreamsBuiltInToolPartsBeforeTheirSources() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"interaction":{"id":"interaction-tools","status":"in_progress"},"event_type":"interaction.created"}

    data: {"index":0,"step":{"type":"google_search_call","id":"search-1"},"event_type":"step.start"}

    data: {"index":0,"delta":{"type":"google_search_call","arguments":{"queries":["Swift AI SDK"]}},"event_type":"step.delta"}

    data: {"index":0,"event_type":"step.stop"}

    data: {"index":1,"step":{"type":"google_search_result","call_id":"search-1"},"event_type":"step.start"}

    data: {"index":1,"delta":{"type":"google_search_result","result":[{"url":"https://example.test/result","title":"Result"}],"is_error":false},"event_type":"step.delta"}

    data: {"index":1,"event_type":"step.stop"}

    data: {"interaction":{"id":"interaction-tools","status":"completed"},"event_type":"interaction.completed"}

    data: [DONE]
    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    var semanticOrder: [String] = []
    var streamedCall: AIToolCall?
    var streamedResult: AIToolResult?
    var finishReason: String?

    for try await part in provider.interactionsModel("gemini-3.8-flash").stream(
        LanguageModelRequest(messages: [.user("Search")])
    ) {
        switch part {
        case let .toolCall(call):
            semanticOrder.append("tool-call")
            streamedCall = call
        case let .toolResult(result):
            semanticOrder.append("tool-result")
            streamedResult = result
        case .source:
            semanticOrder.append("source")
        case let .finishMetadata(reason, _, _):
            semanticOrder.append("finish")
            finishReason = reason
        default:
            break
        }
    }

    #expect(semanticOrder == ["tool-call", "tool-result", "source", "finish"])
    #expect(streamedCall?.id == "search-1")
    #expect(streamedCall?.name == "google_search")
    #expect(streamedCall?.providerExecuted == true)
    #expect(streamedCall?.arguments == #"{"queries":["Swift AI SDK"]}"#)
    #expect(streamedResult?.toolCallID == "search-1")
    #expect(streamedResult?.toolName == "google_search")
    #expect(streamedResult?.providerExecuted == true)
    #expect(finishReason == "stop")
}
