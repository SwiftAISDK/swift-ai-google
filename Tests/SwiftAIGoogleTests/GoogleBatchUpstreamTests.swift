@testable import SwiftAIGoogle
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func googleBatchStartsInlineWithPreparedRequestsHeadersAndAbort() async throws {
    let transport = RecordingTransport(response: jsonResponse(googleBatchTestOperation(
        state: "BATCH_STATE_PENDING",
        done: false,
        total: 1,
        successful: 0,
        failed: 0,
        pending: 1
    )))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(
        apiKey: "test-api-key",
        headers: ["Provider-Header": "provider"],
        transport: transport
    ))
    let model = provider.batchLanguageModel("gemini-2.5-flash")
    let controller = AIAbortController()

    let result = try await model.startBatch(AIBatchStartOptions(
        requests: [AILanguageModelBatchRequest(
            id: "france",
            request: LanguageModelRequest(
                messages: [.system("Only the city."), .user("Capital of France?")],
                temperature: 0.2,
                maxOutputTokens: 20
            )
        )],
        abortSignal: controller.signal,
        headers: ["Operation-Header": "operation"],
        webhookURL: "https://example.com/hook"
    ))

    #expect(result.batchID == "batches/batch-123")
    #expect(result.status.status == AIBatchLifecycleStatus.pending)
    #expect(result.status.requestCounts == AIBatchRequestCounts(total: 1, pending: 1, completed: 0, failed: 0))
    #expect(result.providerMetadata.isEmpty)
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:batchGenerateContent")
    #expect(normalizeHeaders(request.headers)["provider-header"] == "provider")
    #expect(normalizeHeaders(request.headers)["operation-header"] == "operation")
    #expect(request.abortSignal === controller.signal)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["batch"]?["webhookConfig"]?["uris"]?[0]?.stringValue == "https://example.com/hook")
    #expect(body["batch"]?["inputConfig"]?["requests"]?["requests"]?[0]?["metadata"]?["key"]?.stringValue == "france")
    let prepared = body["batch"]?["inputConfig"]?["requests"]?["requests"]?[0]?["request"]
    #expect(prepared?["systemInstruction"]?["parts"]?[0]?["text"]?.stringValue == "Only the city.")
    #expect(prepared?["contents"]?[0]?["parts"]?[0]?["text"]?.stringValue == "Capital of France?")
    #expect(prepared?["generationConfig"]?["maxOutputTokens"]?.intValue == 20)
}

@Test func googleBatchPreservesQualifiedModelPaths() async throws {
    let transport = RecordingTransport(response: jsonResponse(googleBatchTestOperation()))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    _ = try await provider.batchLanguageModel("tunedModels/custom-model").startBatch(AIBatchStartOptions(
        requests: [AILanguageModelBatchRequest(
            id: "qualified",
            request: LanguageModelRequest(messages: [.user("Hello")])
        )]
    ))

    #expect((await transport.requests()).first?.url.absoluteString ==
        "https://generativelanguage.googleapis.com/v1beta/tunedModels/custom-model:batchGenerateContent")
}

@Test func googleBatchUploadsJSONLAtInlineLimitAndReturnsInputFileMetadata() async throws {
    let transport = RecordingTransport(responses: [
        AIHTTPResponse(
            statusCode: 200,
            headers: ["x-goog-upload-url": "https://upload.example.com/session"],
            body: Data()
        ),
        jsonResponse(#"{"file":{"name":"files/batch-input","expirationTime":"2026-08-27T12:00:00Z"}}"#),
        jsonResponse(googleBatchTestOperation())
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(
        apiKey: "test-api-key",
        headers: ["Provider-Header": "provider"],
        transport: transport
    ))
    let model = provider.batchLanguageModel("gemini-2.5-flash")
    let controller = AIAbortController()
    let largePrompt = String(repeating: "a", count: 20_000_000)

    let result = try await model.startBatch(AIBatchStartOptions(
        requests: [
            AILanguageModelBatchRequest(id: "small", request: LanguageModelRequest(messages: [.user("small")])),
            AILanguageModelBatchRequest(id: "large", request: LanguageModelRequest(messages: [.user(largePrompt)]))
        ],
        abortSignal: controller.signal,
        headers: ["Operation-Header": "operation"]
    ))

    #expect(result.providerMetadata["google"]?["inputFileId"]?.stringValue == "files/batch-input")
    #expect(result.providerMetadata["google"]?["inputFileExpiresAt"]?.stringValue == "2026-08-27T12:00:00Z")
    let requests = await transport.requests()
    #expect(requests.map(\.url.absoluteString) == [
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        "https://upload.example.com/session",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:batchGenerateContent"
    ])
    #expect(requests.allSatisfy { $0.abortSignal === controller.signal })
    #expect(normalizeHeaders(requests[0].headers)["provider-header"] == "provider")
    #expect(normalizeHeaders(requests[0].headers)["operation-header"] == "operation")
    #expect(requests[0].headers["X-Goog-Upload-Header-Content-Type"] == "application/jsonl")
    #expect(requests[1].headers["X-Goog-Upload-Command"] == "upload, finalize")
    #expect(normalizeHeaders(requests[1].headers)["x-goog-api-key"] == nil)
    let uploadBody = try #require(requests[1].body)
    let jsonl = try #require(String(data: uploadBody, encoding: .utf8))
    #expect(jsonl.hasPrefix(#"{"key":"small","request":"#))
    #expect(jsonl.contains("\n{\"key\":\"large\",\"request\":"))
    #expect(jsonl.hasSuffix("\n"))
    let creation = try decodeJSONBody(try #require(requests[2].body))
    #expect(creation["batch"]?["inputConfig"]?["fileName"]?.stringValue == "files/batch-input")
}

@Test func googleBatchMapsLifecycleCountsAndRPCError() async throws {
    let transport = RecordingTransport(response: jsonResponse(googleBatchTestOperation(
        state: "JOB_STATE_FAILED",
        done: true,
        total: 7,
        successful: 2,
        failed: 3,
        pending: 2,
        error: ["code": 3, "message": "Bad batch.", "status": "INVALID_ARGUMENT"]
    )))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let status = try await provider.batchLanguageModel("gemini-2.5-flash").getBatchStatus(
        AIBatchOperationOptions(batchID: "batches/batch-123")
    )

    #expect(status.status == .failed)
    #expect(status.rawStatus == "JOB_STATE_FAILED")
    #expect(status.requestCounts == AIBatchRequestCounts(total: 7, pending: 2, completed: 2, failed: 3))
    #expect(status.error == AIBatchError(message: "Bad batch.", type: "INVALID_ARGUMENT", code: "3"))
    #expect(status.createdAt == "2026-08-04T12:34:56.123Z")
}

@Test func googleBatchResultsKeepToolContentAndIsolateInvalidAndBlockedItems() async throws {
    let operation: JSONValue = [
        "name": "batches/batch-123",
        "done": true,
        "metadata": [
            "state": "BATCH_STATE_SUCCEEDED",
            "output": [
                "inlinedResponses": [
                    "inlinedResponses": [
                        [
                            "metadata": ["key": "tool"],
                            "response": googleBatchTestResponse(parts: [
                                ["functionCall": ["id": "call-1", "name": "weather", "args": ["city": "Paris"]]],
                                ["executableCode": ["language": "PYTHON", "code": "print(1)"]],
                                ["codeExecutionResult": ["outcome": "OUTCOME_OK", "output": "1\n"]]
                            ])
                        ],
                        [
                            "metadata": ["key": "blocked"],
                            "response": ["candidates": [], "promptFeedback": ["blockReason": "SAFETY"]]
                        ],
                        [
                            "metadata": ["key": "invalid"],
                            "response": googleBatchTestResponse(parts: [["text": 42]])
                        ],
                        [
                            "metadata": ["key": "valid"],
                            "response": googleBatchTestResponse(parts: [["text": "Paris"]])
                        ]
                    ]
                ]
            ]
        ]
    ]
    let transport = RecordingTransport(response: jsonResponse(try googleBatchTestJSONString(operation)))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let stream = try await provider.batchLanguageModel("gemini-2.5-flash").getBatchResults(
        AIBatchOperationOptions(batchID: "batches/batch-123")
    )
    var items: [AIBatchItemResult<TextGenerationResult>] = []
    for try await item in stream { items.append(item) }

    #expect(items.count == 4)
    guard case let .succeeded(toolID, toolResult) = items[0] else {
        Issue.record("Expected tool batch item to succeed")
        return
    }
    #expect(toolID == "tool")
    #expect(toolResult.toolCalls.map(\.name) == ["weather", "code_execution"])
    #expect(toolResult.toolResults.count == 1)
    #expect(toolResult.toolResults[0].toolCallID == "google-code-execution-1")
    #expect(toolResult.content.count == 3)
    guard case let .failed(blockedID, blockedError, metadata) = items[1] else {
        Issue.record("Expected blocked item failure")
        return
    }
    #expect(blockedID == "blocked")
    #expect(blockedError.code == "prompt_blocked")
    #expect(metadata["google"]?["promptFeedback"]?["blockReason"]?.stringValue == "SAFETY")
    guard case let .failed(invalidID, invalidError, _) = items[2] else {
        Issue.record("Expected invalid item failure")
        return
    }
    #expect(invalidID == "invalid")
    #expect(invalidError.code == "invalid_response")
    guard case let .succeeded(validID, validResult) = items[3] else {
        Issue.record("Expected later valid item to succeed")
        return
    }
    #expect(validID == "valid")
    #expect(validResult.text == "Paris")
}

@Test func googleBatchDownloadsEncodedJSONLResultFileAndMapsCancellation() async throws {
    let operation: JSONValue = [
        "name": "batches/batch-123",
        "done": true,
        "metadata": [
            "state": "BATCH_STATE_SUCCEEDED",
            "output": ["responsesFile": "files/output?alt=json#fragment"]
        ]
    ]
    let lines = """
    {"key":"cancelled","error":{"code":1,"message":"Cancelled."}}
    {"key":"valid","response":{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}}
    """
    let transport = RecordingTransport(responses: [
        jsonResponse(try googleBatchTestJSONString(operation)),
        AIHTTPResponse(statusCode: 200, body: Data(lines.utf8))
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let stream = try await provider.batchLanguageModel("gemini-2.5-flash").getBatchResults(
        AIBatchOperationOptions(batchID: "batches/batch-123", headers: ["Operation-Header": "operation"])
    )
    var items: [AIBatchItemResult<TextGenerationResult>] = []
    for try await item in stream { items.append(item) }

    #expect(items.count == 2)
    guard case let .cancelled(id, error, _) = items[0] else {
        Issue.record("Expected cancelled item")
        return
    }
    #expect(id == "cancelled")
    #expect(error?.code == "1")
    guard case let .succeeded(id, result) = items[1] else {
        Issue.record("Expected successful item")
        return
    }
    #expect(id == "valid")
    #expect(result.text == "done")
    let requests = await transport.requests()
    #expect(requests[1].url.absoluteString == "https://generativelanguage.googleapis.com/download/v1beta/files/output%3Falt%3Djson%23fragment:download?alt=media")
    #expect(normalizeHeaders(requests[1].headers)["operation-header"] == "operation")
}

@Test func googleBatchRejectsInlineResultWithoutAStringMetadataKey() async throws {
    let operation: JSONValue = [
        "name": "batches/batch-123",
        "done": true,
        "metadata": [
            "state": "BATCH_STATE_SUCCEEDED",
            "output": [
                "inlinedResponses": [
                    "inlinedResponses": [[
                        "metadata": [:],
                        "response": googleBatchTestResponse(parts: [["text": "orphaned"]])
                    ]]
                ]
            ]
        ]
    ]
    let transport = RecordingTransport(response: jsonResponse(try googleBatchTestJSONString(operation)))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))

    await #expect(throws: AIError.invalidResponse(
        provider: "google.generative-ai",
        message: "Invalid Google batch result row: expected a non-empty string key."
    )) {
        _ = try await provider.batchLanguageModel("gemini-2.5-flash").getBatchResults(
            AIBatchOperationOptions(batchID: "batches/batch-123")
        )
    }
}

@Test func googleBatchRejectsDownloadedJSONLResultWithoutAStringKey() async throws {
    let operation: JSONValue = [
        "name": "batches/batch-123",
        "done": true,
        "metadata": [
            "state": "BATCH_STATE_SUCCEEDED",
            "output": ["responsesFile": "files/output"]
        ]
    ]
    let transport = RecordingTransport(responses: [
        jsonResponse(try googleBatchTestJSONString(operation)),
        AIHTTPResponse(
            statusCode: 200,
            body: Data(#"{"key":42,"response":{"candidates":[{"content":{"parts":[{"text":"orphaned"}]}}]}}"#.utf8)
        )
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let stream = try await provider.batchLanguageModel("gemini-2.5-flash").getBatchResults(
        AIBatchOperationOptions(batchID: "batches/batch-123")
    )

    await #expect(throws: AIError.invalidResponse(
        provider: "google.generative-ai",
        message: "Invalid Google batch result row: expected a non-empty string key."
    )) {
        for try await _ in stream {}
    }
}

@Test func googleBatchMergesEmptyTextThoughtSignatureIntoPreviousContent() async throws {
    let operation: JSONValue = [
        "name": "batches/batch-123",
        "done": true,
        "metadata": [
            "state": "BATCH_STATE_SUCCEEDED",
            "output": [
                "inlinedResponses": [
                    "inlinedResponses": [[
                        "metadata": ["key": "signed-call"],
                        "response": googleBatchTestResponse(parts: [
                            ["functionCall": ["id": "call-1", "name": "weather", "args": ["city": "Tokyo"]]],
                            ["text": "", "thoughtSignature": "trailing-signature"]
                        ])
                    ]]
                ]
            ]
        ]
    ]
    let transport = RecordingTransport(response: jsonResponse(try googleBatchTestJSONString(operation)))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport))
    let stream = try await provider.batchLanguageModel("gemini-2.5-flash").getBatchResults(
        AIBatchOperationOptions(batchID: "batches/batch-123")
    )
    var items: [AIBatchItemResult<TextGenerationResult>] = []
    for try await item in stream { items.append(item) }

    guard case let .succeeded(id, result) = try #require(items.first),
          case let .toolCall(call) = try #require(result.content.first) else {
        Issue.record("Expected a successful batch tool call")
        return
    }
    #expect(id == "signed-call")
    #expect(result.content.count == 1)
    #expect(call.id == "call-1")
    #expect(call.providerMetadata["google"]?["thoughtSignature"]?.stringValue == "trailing-signature")
}

private func googleBatchTestOperation(
    state: String = "BATCH_STATE_SUCCEEDED",
    done: Bool = true,
    total: Int = 2,
    successful: Int = 2,
    failed: Int = 0,
    pending: Int = 0,
    error: JSONValue? = nil
) -> String {
    var operation: [String: JSONValue] = [
        "name": "batches/batch-123",
        "done": .bool(done),
        "metadata": [
            "state": .string(state),
            "createTime": "2026-08-04T12:34:56.123Z",
            "batchStats": [
                "requestCount": .string(String(total)),
                "successfulRequestCount": .string(String(successful)),
                "failedRequestCount": .string(String(failed)),
                "pendingRequestCount": .string(String(pending))
            ]
        ]
    ]
    if let error { operation["error"] = error }
    return (try? googleBatchTestJSONString(.object(operation))) ?? "{}"
}

private func googleBatchTestResponse(parts: [JSONValue]) -> JSONValue {
    [
        "responseId": "response-1",
        "candidates": [[
            "content": ["role": "model", "parts": .array(parts)],
            "finishReason": "STOP"
        ]],
        "usageMetadata": ["promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2]
    ]
}

private func googleBatchTestJSONString(_ value: JSONValue) throws -> String {
    try #require(String(data: encodeJSONBody(value), encoding: .utf8))
}
