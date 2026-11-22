import Foundation
import SwiftAI
import SwiftAIProvider
import SwiftAIProviderUtilities
import SwiftAITransport
import Testing
@_spi(Vertex) @testable import SwiftAIGoogle

private actor GoogleInteractionsTelemetryRecorder: Telemetry.Integration {
    private(set) var events: [Telemetry.Event] = []
    private(set) var typedErrorBody: JSONValue?

    func record(_ event: Telemetry.Event) { events.append(event) }

    func executeLanguageModelCall<Output: Sendable>(_ context: Telemetry.LanguageModelCallContext<Output>) async throws -> Output {
        do {
            return try await context.execute()
        } catch let error as AIError {
            typedErrorBody = error.apiCallError?.requestBody
            throw error
        }
    }
}

// @ai-sdk/google 4.0.124 at 9b8c25e582bb2ca97e459ba3f8f6b325cb9dfb22.
@Suite struct Weekly20261122GoogleTests {
    private static let prompt = [AIMessage.user("Hello, how are you?")]
    private static let promptFeedback: JSONValue = [
        "blockReason": "PROHIBITED_CONTENT",
        "blockReasonMessage": "The prompt is blocked due to prohibited contents"
    ]
    private static let blockedResponse: JSONValue = [
        "promptFeedback": promptFeedback,
        "usageMetadata": ["promptTokenCount": 9, "totalTokenCount": 9]
    ]
    private static let secret = "test-connector-secret"

    @Test(arguments: [false, true])
    func includesPromptBlockReasonMessageInProviderMetadata(streaming: Bool) async throws {
        let json = String(decoding: try encodeJSONBody(Self.blockedResponse), as: UTF8.self)
        let transport = RecordingTransport(response: streaming ? sseResponse("data: \(json)\n\n") : jsonResponse(json))
        let model = try provider(transport).languageModel("gemini-2.5-pro")
        let request = LanguageModelRequest(messages: Self.prompt)
        if streaming {
            var finishFeedback: JSONValue?
            for try await part in model.stream(request) {
                if case let .finishMetadata(_, _, metadata) = part { finishFeedback = metadata["google"]?["promptFeedback"] }
            }
            #expect(finishFeedback == Self.promptFeedback)
        } else {
            let result = try await model.generate(request)
            #expect(result.providerMetadata["google"]?["promptFeedback"] == Self.promptFeedback)
        }
    }

    @Test(arguments: [("xhigh", "xhigh"), ("max", "xhigh"), ("minimal", "low")])
    func neonMapsReasoningToSupportedThinkingLevels(reasoning: String, expected: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}"#))
        let model = try provider(transport).languageModel("gemini-4-neon")
        _ = try await model.generate(.init(messages: Self.prompt, reasoning: reasoning))
        let body = try requestBody(await transport.requests())
        #expect(body["generationConfig"]?["thinkingConfig"]?["thinkingLevel"] == .string(expected))
    }

    @Test(arguments: ["gemini-4-hydrogen", "gemini-4-hydrogen-eap", "gemini-4-neon"])
    func interactionsPassesExplicitXhighThinkingLevel(modelID: String) async throws {
        let transport = RecordingTransport(response: try basicInteractionsResponse())
        let result = try await provider(transport).interactionsModel(modelID).generate(.init(
            messages: Self.prompt,
            providerOptions: ["google": ["thinkingLevel": "xhigh"]]
        ))
        let body = try requestBody(await transport.requests())
        #expect(body["model"] == .string(modelID))
        #expect(body["generation_config"]?["thinking_level"] == "xhigh")
        #expect(result.warnings.isEmpty)
    }

    @Test func telemetryRedactsEveryMCPToolHeaderWithoutMutatingTheTool() throws {
        let tool = GoogleTools.mcpServer(name: "local-mcp", url: "https://example.com/mcp", headers: [
            "Authorization": "Bearer mcp-private-token", "X-Custom-Key": "mcp-private-key"
        ], allowedTools: ["getWeather"])
        let original = tool
        let redacted = googleRedactInteractionsTelemetry(tool)
        #expect(redacted["args"] == [
            "name": "local-mcp", "url": "https://example.com/mcp",
            "headers": ["Authorization": "[redacted]", "X-Custom-Key": "[redacted]"],
            "allowedTools": ["getWeather"]
        ])
        #expect(tool == original)
        #expect(tool["args"]?["headers"]?["X-Custom-Key"] == "mcp-private-key")
    }

    @Test(arguments: [
        ("google.mcp_server", JSONValue.object(["url": "https://example.com/mcp"])),
        ("google.mcp_server", JSONValue.object(["headers": .null])),
        ("google.google_search", JSONValue.object(["headers": ["Authorization": "Bearer token"]])),
        ("other.tool", JSONValue.object(["headers": ["Authorization": "Bearer token"]]))
    ])
    func redactionPreservesOtherToolDefinitions(id: String, args: JSONValue) {
        let tool: JSONValue = ["id": .string(id), "args": args]
        #expect(googleRedactInteractionsTelemetry(tool) == tool)
    }

    @Test(arguments: [
        JSONValue.object([:]), ["input": []], ["tools": [["type": "google_search"]]],
        ["tools": [["type": "mcp_server", "url": "https://example.com/mcp"]]]
    ])
    func redactionPreservesRequestBodiesWithoutMCPCredentials(body: JSONValue) {
        #expect(googleRedactInteractionsTelemetry(body) == body)
    }

    @Test func redactsConfigAndWireHeadersWhilePreservingTheWirePayload() throws {
        let tool = GoogleTools.mcpServer(name: "local-mcp", url: "https://example.com/mcp", headers: [
            "Authorization": "test-secret", "X-Custom-Key": "test-secret"
        ], allowedTools: ["getWeather"])
        let original = tool
        let prepared = googleInteractionsPrepareNativeTools(["connector": tool], choice: nil)
        let body: JSONValue = ["model": "gemini-2.5-flash", "input": [["type": "user_input", "content": []]], "tools": .array(prepared.tools)]
        #expect(prepared.tools == [[
            "type": "mcp_server", "name": "local-mcp", "url": "https://example.com/mcp",
            "headers": ["Authorization": "test-secret", "X-Custom-Key": "test-secret"],
            "allowed_tools": ["getWeather"]
        ]])
        for value in [tool, body] {
            let raw = try jsonString(value)
            #expect(raw.contains("test-secret"))
            #expect(try jsonString(googleRedactInteractionsTelemetry(value)) == raw.replacingOccurrences(of: "test-secret", with: "[redacted]"))
        }
        let withSearch: JSONValue = ["tools": .array(prepared.tools + [["type": "google_search"]])]
        #expect(googleRedactInteractionsTelemetry(withSearch)["tools"]?[1] == ["type": "google_search"])
        #expect(tool == original)
    }

    @Test func generateSendsCredentialsAndRedactsTelemetryOnly() async throws {
        let transport = RecordingTransport(response: try basicInteractionsResponse())
        let model = try provider(transport).interactionsModel("gemini-2.5-flash")
        let recorder = GoogleInteractionsTelemetryRecorder()
        let result = try await AI.generateText(model: model, request: mcpRequest(), retryPolicy: .none, telemetry: .init(integrations: [recorder]))
        let body = try requestBody(await transport.requests())
        #expect(body["tools"]?[0]?["type"] == "mcp_server")
        #expect(body["tools"]?[0]?["headers"]?["Authorization"] == .string("Bearer \(Self.secret)"))
        #expect(result.requestMetadata.body == body)
        try expectTelemetryRedaction(await recorder.events)
    }

    @Test func generateContentDropsTheConnectorBeforeSendingAndStillRedactsTelemetry() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}"#))
        let model = try provider(transport).languageModel("gemini-2.5-flash")
        let recorder = GoogleInteractionsTelemetryRecorder()
        let result = try await AI.generateText(model: model, request: mcpRequest(), retryPolicy: .none, telemetry: .init(integrations: [recorder]))
        let body = try requestBody(await transport.requests())
        #expect(body["tools"] == nil)
        #expect(result.warnings.contains { $0.feature == "provider-defined tool google.mcp_server" })
        try expectTelemetryRedaction(await recorder.events)
    }

    @Test func streamSendsCredentialsAndRedactsTelemetryOnly() async throws {
        let transport = RecordingTransport(response: sseResponse("""
        data: {"event_type":"interaction.created","event_id":"evt-1","interaction":{"id":"v1_credential-test","status":"in_progress"}}

        data: {"event_type":"step.start","event_id":"evt-2","index":0,"step":{"type":"model_output"}}

        data: {"event_type":"step.delta","event_id":"evt-3","index":0,"delta":{"type":"text","text":"OK"}}

        data: {"event_type":"step.stop","event_id":"evt-4","index":0,"step":{"type":"model_output"}}

        data: {"event_type":"interaction.completed","event_id":"evt-5","interaction":{"id":"v1_credential-test","status":"completed"}}

        """))
        let model = try provider(transport).interactionsModel("gemini-2.5-flash")
        let recorder = GoogleInteractionsTelemetryRecorder()
        var metadata: AIRequestMetadata?
        for try await part in AI.streamText(model: model, request: mcpRequest(), retryPolicy: .none, telemetry: .init(integrations: [recorder])) {
            if case let .requestMetadata(value) = part { metadata = value }
        }
        let body = try requestBody(await transport.requests())
        #expect(body["stream"] == true)
        #expect(body["tools"]?[0]?["type"] == "mcp_server")
        #expect(body["tools"]?[0]?["headers"]?["Authorization"] == .string("Bearer \(Self.secret)"))
        #expect(metadata?.body == body)
        try expectTelemetryRedaction(await recorder.events)
    }

    @Test func httpErrorRetainsCallerDiagnosticsAndRedactsTelemetry() async throws {
        let transport = RecordingTransport(response: .init(statusCode: 400, body: Data(#"{"error":{"code":400,"message":"invalid request","status":"INVALID_ARGUMENT"}}"#.utf8)))
        let model = try provider(transport).interactionsModel("gemini-2.5-flash")
        let recorder = GoogleInteractionsTelemetryRecorder()
        do {
            _ = try await AI.generateText(model: model, request: mcpRequest(), retryPolicy: .none, telemetry: .init(integrations: [recorder]))
            Issue.record("Expected an API call error")
        } catch let error as AIError {
            let call = try #require(error.apiCallError)
            #expect(call.statusCode == 400)
            let body = try #require(call.requestBody)
            #expect(try jsonString(body).contains(Self.secret))
            let sanitized = try #require(await recorder.typedErrorBody)
            #expect(sanitized["tools"]?[0]?["headers"]?["Authorization"] == "[redacted]")
            #expect(try !jsonString(sanitized).contains(Self.secret))
        }
        try expectTelemetryRedaction(await recorder.events)
    }

    @Test(arguments: [false, true])
    func batchRejectsResponseSchemasBeforeSending(providerOwned: Bool) async throws {
        let transport = RecordingTransport(response: jsonResponse("{}"))
        let request = LanguageModelRequest(messages: [.user("Hello")], responseFormat: .json(schema: ["type": "object", "properties": [:]]))
        do {
            try await startBatch(provider(transport), request: request, id: "schema", providerOwned: providerOwned)
            Issue.record("Expected an unsupported batch schema error")
        } catch let error as AIError {
            guard case let .invalidArgument(functionality, message) = error else { throw error }
            #expect(functionality == "structured output schemas in Google batches")
            #expect(message.contains("providerOptions.google.structuredOutputs"))
        }
        #expect(await transport.requests().isEmpty)
    }

    @Test(arguments: [false, true])
    func batchAllowsJSONModeAndCanExplicitlyOmitASchema(disableSchema: Bool) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"name":"batches/batch-123","done":false,"metadata":{"state":"BATCH_STATE_PENDING"}}"#))
        let request = LanguageModelRequest(
            messages: [.user("Hello")],
            responseFormat: .json(schema: disableSchema ? ["type": "object", "properties": [:]] : nil),
            providerOptions: disableSchema ? ["google": ["structuredOutputs": false]] : [:]
        )
        try await startBatch(provider(transport), request: request, id: disableSchema ? "schema" : "json", providerOwned: true)
        let body = try requestBody(await transport.requests())
        let generation = body["batch"]?["inputConfig"]?["requests"]?["requests"]?[0]?["request"]?["generationConfig"]
        #expect(generation?["responseMimeType"] == "application/json")
        #expect(generation?["responseJsonSchema"] == nil)
    }

    private func startBatch(_ provider: GoogleGenerativeAIProvider, request: LanguageModelRequest, id: String, providerOwned: Bool) async throws {
        let modelID = "gemini-2.5-flash"
        if providerOwned {
            _ = try await provider.experimentalBatch().startBatch(.init(requests: [.text(.init(id: id, modelID: modelID, request: request))]))
        } else {
            _ = try await provider.batchLanguageModel(modelID).startBatch(.init(requests: [.init(id: id, request: request)]))
        }
    }

    private func provider(_ transport: RecordingTransport) throws -> GoogleGenerativeAIProvider {
        try GoogleGenerativeAIProvider(configuration: .init(apiKey: "test-api-key", transport: transport))
    }

    private func basicInteractionsResponse() throws -> AIHTTPResponse {
        let fixture = try #require(Bundle.module.url(forResource: "interactions-basic", withExtension: "json"))
        return .init(statusCode: 200, headers: ["content-type": "application/json"], body: try Data(contentsOf: fixture))
    }

    private func mcpRequest() -> LanguageModelRequest {
        .init(messages: Self.prompt, tools: ["connector": GoogleTools.mcpServer(
            name: "local-mcp", url: "https://example.com/mcp", headers: ["Authorization": "Bearer \(Self.secret)"], allowedTools: ["getWeather"]
        )])
    }

    private func requestBody(_ requests: [AIHTTPRequest]) throws -> JSONValue {
        try decodeJSONBody(#require(requests.first?.body))
    }

    private func jsonString(_ value: JSONValue) throws -> String {
        String(decoding: try encodeJSONBody(value), as: UTF8.self)
    }

    private func expectTelemetryRedaction(_ events: [Telemetry.Event]) throws {
        let views = try events.flatMap { event in
            try [event.input, event.output, event.responseMetadata.body].compactMap { $0 }.map(jsonString)
        }
        #expect(views.contains { $0.contains("[redacted]") })
        #expect(views.allSatisfy { !$0.contains(Self.secret) })
        #expect(events.allSatisfy { $0.errorDescription?.contains(Self.secret) != true })
    }
}
