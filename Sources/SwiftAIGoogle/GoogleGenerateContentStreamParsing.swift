import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleStreamParts(from raw: JSONValue) -> [LanguageStreamPart] {
    var parts: [LanguageStreamPart] = []
    var currentTextID: String?
    var currentReasoningID: String?
    var blockCounter = 0
    let contentParts = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []
    for contentPart in contentParts {
        guard let text = contentPart["text"]?.stringValue else { continue }
        let metadata = googleThoughtSignatureProviderMetadata(from: contentPart)
        if contentPart["thought"]?.boolValue == true {
            if let id = currentTextID {
                parts.append(.textEnd(id: id))
                currentTextID = nil
            }
            if currentReasoningID == nil {
                currentReasoningID = String(blockCounter)
                blockCounter += 1
                parts.append(.reasoningStart(id: currentReasoningID!, providerMetadata: metadata))
            }
            parts.append(.reasoningDeltaPart(id: currentReasoningID!, delta: text, providerMetadata: metadata))
        } else {
            if let id = currentReasoningID {
                parts.append(.reasoningEnd(id: id))
                currentReasoningID = nil
            }
            if currentTextID == nil {
                currentTextID = String(blockCounter)
                blockCounter += 1
                parts.append(.textStart(id: currentTextID!, providerMetadata: metadata))
            }
            parts.append(.textDeltaPart(id: currentTextID!, delta: text, providerMetadata: metadata))
        }
    }
    if let id = currentReasoningID {
        parts.append(.reasoningEnd(id: id))
    }
    if let id = currentTextID {
        parts.append(.textEnd(id: id))
    }
    if raw["candidates"]?[0]?["finishReason"]?.stringValue != nil || raw["usageMetadata"] != nil {
        parts.append(.finishMetadata(
            reason: googleGenerateContentFinishReason(from: raw, hasToolCalls: false) ?? "other",
            usage: googleGenerateContentUsage(from: raw),
            providerMetadata: googleFinalizeGenerateContentProviderMetadata(
                googleGenerateContentProviderMetadata(from: raw, includeNullDefaults: false)
            )
        ))
    }
    return parts
}

struct GoogleGenerateContentStreamState {
    private var responseMetadata: AIResponseMetadata
    private let includeRawChunks: Bool
    private let warnings: [AIWarning]
    private let toolNameMapping: AIToolNameMapping
    private var jsonResponseConverter: GoogleJSONResponseToolStreamConverter?
    private var toolCalls = GoogleGenerateContentStreamingToolCalls()
    private var lastCodeExecutionToolCallID: String?
    private var lastServerToolCallID: String?
    private var latestFinishReason: String?
    private var confirmedPromptBlockReason: String?
    private var latestUsage: TokenUsage?
    private var currentUsageMetadata: JSONValue?
    private var continuedUsageMetadata: JSONValue?
    private var currentRequestGroundingMetadata: JSONValue?
    private var groundingMetadataPerRequest: [JSONValue] = []
    private var initialPromptTokenCount: JSONValue?
    private var didContinue = false
    private var requestHop = 0
    private var receivedProviderError = false
    private var latestProviderMetadata: [String: JSONValue] = [:]
    private var emittedSourceKeys: Set<String> = []
    private var sawToolCalls = false
    private var hasResponseID = false
    private var emittedInitialParts = false
    private var currentTextID: String?
    private var currentReasoningID: String?
    private var blockCounter = 0

    init(
        response: AIHTTPResponse,
        includeRawChunks: Bool,
        modelID: String?,
        warnings: [AIWarning],
        toolNameMapping: AIToolNameMapping = AIToolNameMapping(),
        jsonResponseToolName: String? = nil
    ) {
        self.responseMetadata = aiResponseMetadata(response: response, modelID: modelID)
        self.includeRawChunks = includeRawChunks
        self.warnings = warnings
        self.toolNameMapping = toolNameMapping
        self.jsonResponseConverter = jsonResponseToolName.map(GoogleJSONResponseToolStreamConverter.init)
        self.toolCalls = GoogleGenerateContentStreamingToolCalls(jsonResponseToolName: jsonResponseToolName)
    }

    mutating func apply(_ raw: JSONValue) -> [LanguageStreamPart] {
        var parts: [LanguageStreamPart] = []
        if !hasResponseID, let responseID = raw["responseId"]?.stringValue {
            hasResponseID = true
            responseMetadata.id = responseID
        }
        if !emittedInitialParts {
            emittedInitialParts = true
            parts.append(.responseMetadata(responseMetadata))
            parts.append(.streamStart(warnings: warnings))
        }
        if includeRawChunks {
            parts.append(.raw(raw))
        }
        if let error = raw["error"], error != .null {
            receivedProviderError = true
            latestFinishReason = nil
            parts.append(.providerError(AIStreamProviderError(
                message: error["message"]?.stringValue ?? "Google stream error",
                type: error["status"]?.stringValue,
                code: error["code"] == .null ? nil : error["code"],
                statusCode: error["code"]?.intValue,
                data: error
            )))
            return parts
        }
        latestProviderMetadata = googleMergeProviderMetadata(
            latestProviderMetadata,
            googleGenerateContentProviderMetadata(from: raw, includeNullDefaults: false)
        )
        if let grounding = raw["candidates"]?[0]?["groundingMetadata"], grounding != .null {
            currentRequestGroundingMetadata = grounding
        }
        if let usageMetadata = raw["usageMetadata"], usageMetadata != .null {
            currentUsageMetadata = usageMetadata
            latestUsage = googleGenerateContentUsage(fromUsageMetadata: usageMetadata)
        }
        if confirmedPromptBlockReason == nil {
            confirmedPromptBlockReason = googleConfirmedPromptBlockReason(
                raw["promptFeedback"]?["blockReason"]?.stringValue
            )
        }

        // A confirmed prompt block is terminal for generated content, but later
        // chunks can still contribute usage and provider metadata.
        if confirmedPromptBlockReason != nil {
            return parts
        }

        for var source in googleGenerateContentSources(from: raw) {
            let key = googleSourceDeduplicationKey(source)
            guard !emittedSourceKeys.contains(key) else { continue }
            source.id = "grounding-\(emittedSourceKeys.count)"
            emittedSourceKeys.insert(key)
            parts.append(.source(source))
        }

        let contentParts = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []
        for (index, contentPart) in contentParts.enumerated() {
            if let executableCode = contentPart["executableCode"],
               executableCode["code"]?.stringValue != nil {
                let id = "google-code-execution-\(index)" + (requestHop == 0 ? "" : "-hop-\(requestHop)")
                lastCodeExecutionToolCallID = id
                sawToolCalls = true
                let toolName = toolNameMapping.toCustomToolName("code_execution")
                parts.append(.toolInputStart(id: id, name: toolName, providerExecuted: true))
                let input = googleGenerateContentArguments(executableCode)
                parts.append(.toolInputDelta(id: id, delta: input))
                parts.append(.toolInputEnd(id: id))
                parts.append(.toolCall(AIToolCall(
                    id: id,
                    name: toolName,
                    arguments: input,
                    providerExecuted: true,
                    rawValue: contentPart
                )))
                continue
            }
            if let codeExecutionResult = contentPart["codeExecutionResult"] {
                let id = lastCodeExecutionToolCallID ?? "google-code-execution-result-\(index)"
                parts.append(.toolResult(AIToolResult(
                    toolCallID: id,
                    toolName: toolNameMapping.toCustomToolName("code_execution"),
                    result: googleCodeExecutionResultJSON(codeExecutionResult)
                )))
                continue
            }
            if let text = contentPart["text"]?.stringValue {
                let metadata = googleThoughtSignatureProviderMetadata(from: contentPart)
                if contentPart["thought"]?.boolValue == true {
                    if !text.isEmpty {
                        if let id = currentTextID {
                            parts.append(.textEnd(id: id))
                            currentTextID = nil
                        }
                        if currentReasoningID == nil {
                            currentReasoningID = String(blockCounter)
                            blockCounter += 1
                            parts.append(.reasoningStart(id: currentReasoningID!, providerMetadata: metadata))
                        }
                        parts.append(.reasoningDeltaPart(id: currentReasoningID!, delta: text, providerMetadata: metadata))
                    }
                } else if !text.isEmpty {
                    if let id = currentReasoningID {
                        parts.append(.reasoningEnd(id: id))
                        currentReasoningID = nil
                    }
                    if currentTextID == nil {
                        currentTextID = String(blockCounter)
                        blockCounter += 1
                        parts.append(.textStart(id: currentTextID!, providerMetadata: metadata))
                    }
                    parts.append(.textDeltaPart(id: currentTextID!, delta: text, providerMetadata: metadata))
                } else if !metadata.isEmpty, let id = currentTextID {
                    parts.append(.textDeltaPart(id: id, delta: "", providerMetadata: metadata))
                }
            }
            if let inlineData = contentPart["inlineData"],
               let mediaType = inlineData["mimeType"]?.stringValue,
               let base64 = inlineData["data"]?.stringValue {
                if let id = currentTextID {
                    parts.append(.textEnd(id: id))
                    currentTextID = nil
                }
                if let id = currentReasoningID {
                    parts.append(.reasoningEnd(id: id))
                    currentReasoningID = nil
                }
                let file = AIStreamFile(
                    mediaType: mediaType,
                    data: Data(base64Encoded: base64),
                    providerMetadata: googleInlineDataProviderMetadata(from: contentPart),
                    rawValue: contentPart
                )
                if contentPart["thought"]?.boolValue == true {
                    parts.append(.reasoningFile(file))
                } else {
                    parts.append(.file(file))
                }
            }
            if let functionCall = contentPart["functionCall"] {
                if jsonResponseConverter != nil {
                    if let id = currentTextID { parts.append(.textEnd(id: id)); currentTextID = nil }
                    if let id = currentReasoningID { parts.append(.reasoningEnd(id: id)); currentReasoningID = nil }
                }
                sawToolCalls = true
                parts.append(contentsOf: toolCalls.apply(functionCall: functionCall, rawValue: contentPart))
            }
            if let serverToolCall = contentPart["toolCall"],
               let toolType = serverToolCall["toolType"]?.stringValue {
                let id = serverToolCall["id"]?.stringValue ?? ("google-server-tool-\(index)" + (requestHop == 0 ? "" : "-hop-\(requestHop)"))
                lastServerToolCallID = id
                sawToolCalls = true
                let name = "server:\(toolType)"
                let input = googleGenerateContentArguments(serverToolCall["args"])
                let metadata = googleServerToolProviderMetadata(id: id, type: toolType, part: contentPart)
                parts.append(.toolInputStart(id: id, name: name, providerExecuted: true, dynamic: true, providerMetadata: metadata))
                parts.append(.toolInputDelta(id: id, delta: input, providerMetadata: metadata))
                parts.append(.toolInputEnd(id: id, providerMetadata: metadata))
                parts.append(.toolCall(AIToolCall(
                    id: id,
                    name: name,
                    arguments: input,
                    providerExecuted: true,
                    dynamic: true,
                    providerMetadata: metadata,
                    rawValue: contentPart
                )))
            }
            if let serverToolResponse = contentPart["toolResponse"],
               let toolType = serverToolResponse["toolType"]?.stringValue {
                let id = lastServerToolCallID ?? serverToolResponse["id"]?.stringValue ?? "google-server-tool-response-\(index)"
                let metadata = googleServerToolProviderMetadata(id: id, type: toolType, part: contentPart)
                parts.append(.toolResult(AIToolResult(
                    toolCallID: id,
                    toolName: "server:\(toolType)",
                    result: serverToolResponse["response"] ?? .object([:]),
                    dynamic: true,
                    providerMetadata: metadata
                )))
                lastServerToolCallID = nil
            }
        }

        if let reason = raw["candidates"]?[0]?["finishReason"]?.stringValue {
            latestFinishReason = reason
        }
        return convertJSONResponseParts(parts)
    }

    mutating func completeRequest(willContinue: Bool) {
        guard willContinue else { return }
        if !didContinue { initialPromptTokenCount = currentUsageMetadata?["promptTokenCount"] }
        didContinue = true
        requestHop += 1
        continuedUsageMetadata = googleAddUsageMetadata(continuedUsageMetadata, currentUsageMetadata)
        currentUsageMetadata = nil
    }

    mutating func startContinuationRequest() {
        // Commit a grounding slot only after the next HTTP request succeeds.
        // A failed continuation has no response to add to the ordered history.
        groundingMetadataPerRequest.append(currentRequestGroundingMetadata ?? .null)
        currentRequestGroundingMetadata = nil
        // This runs after all prior response events have been consumed, including
        // its CONTINUATION reason, and even when the next response is empty.
        latestFinishReason = nil
        confirmedPromptBlockReason = nil
    }

    mutating func recordParseError() {
        if latestFinishReason == nil { receivedProviderError = true }
    }

    mutating func finish() -> [LanguageStreamPart] {
        var parts: [LanguageStreamPart] = []
        if !emittedInitialParts {
            emittedInitialParts = true
            parts.append(.responseMetadata(responseMetadata))
            parts.append(.streamStart(warnings: warnings))
        }
        let finishedToolParts = toolCalls.finishedParts()
        parts.append(contentsOf: finishedToolParts)
        if let id = currentReasoningID {
            parts.append(.reasoningEnd(id: id))
        }
        currentReasoningID = nil
        if let id = currentTextID {
            parts.append(.textEnd(id: id))
        }
        currentTextID = nil
        let usageMetadata = googleAddUsageMetadata(continuedUsageMetadata, currentUsageMetadata)
        latestUsage = usageMetadata.map(googleGenerateContentUsage(fromUsageMetadata:))
        var google = latestProviderMetadata["google"]?.objectValue ?? [:]
        google["usageMetadata"] = usageMetadata ?? .null
        google["serviceTier"] = usageMetadata?["serviceTier"] ?? .null
        if !groundingMetadataPerRequest.isEmpty {
            google["groundingMetadataPerRequest"] = .array(
                groundingMetadataPerRequest + [currentRequestGroundingMetadata ?? .null]
            )
        }
        if latestFinishReason != "MAX_TOKENS" && latestFinishReason != "CONTINUATION" { google.removeValue(forKey: "continuationToken") }
        if didContinue, let initialPromptTokenCount, initialPromptTokenCount != .null { google["initialPromptTokenCount"] = initialPromptTokenCount }
        latestProviderMetadata["google"] = .object(google)
        let finishReason = receivedProviderError ? "error" : confirmedPromptBlockReason != nil
            ? "content-filter"
            : googleGenerateContentFinishReason(
                latestFinishReason,
                hasToolCalls: sawToolCalls || !finishedToolParts.isEmpty
            ) ?? "interrupted"
        let providerMetadata = googleFinalizeGenerateContentProviderMetadata(latestProviderMetadata)
        parts.append(.finishMetadata(reason: finishReason, usage: latestUsage, providerMetadata: providerMetadata))
        return convertJSONResponseParts(parts)
    }

    private mutating func convertJSONResponseParts(_ parts: [LanguageStreamPart]) -> [LanguageStreamPart] {
        guard var converter = jsonResponseConverter else { return parts }
        let converted = parts.flatMap { converter.apply($0) }
        jsonResponseConverter = converter
        return converted
    }
}

/// Matches the provider-utils response-tool conversion while preserving the
/// Swift streaming adapter's legacy tool-call delta events for application tools.
private struct GoogleJSONResponseToolStreamConverter {
    let name: String
    private var responseIDs: Set<String> = []
    private var IDsWithDeltas: Set<String> = []
    private var hasResponseTool = false
    private var hasApplicationTool = false

    init(_ name: String) { self.name = name }

    mutating func apply(_ part: LanguageStreamPart) -> [LanguageStreamPart] {
        switch part {
        case .textStart, .textDelta, .textDeltaPart, .textEnd:
            return []
        case let .toolInputStart(id, toolName, _, _, _, metadata) where toolName == name:
            responseIDs.insert(id)
            return [.textStart(id: id, providerMetadata: metadata)]
        case let .toolInputDelta(id, delta, metadata) where responseIDs.contains(id):
            if !delta.isEmpty { IDsWithDeltas.insert(id) }
            return [.textDeltaPart(id: id, delta: delta, providerMetadata: metadata)]
        case let .toolInputEnd(id, _) where responseIDs.contains(id):
            return []
        case let .toolCallDelta(id, toolName, _, _) where toolName == name || (id.map(responseIDs.contains) ?? false):
            return []
        case let .toolCall(call) where call.name == name:
            var result: [LanguageStreamPart] = []
            if !IDsWithDeltas.contains(call.id) {
                result.append(.textDeltaPart(id: call.id, delta: call.arguments, providerMetadata: call.providerMetadata))
            }
            result.append(.textEnd(id: call.id, providerMetadata: call.providerMetadata))
            hasResponseTool = true
            return result
        case let .toolCall(call):
            if !call.providerExecuted { hasApplicationTool = true }
        case let .finishMetadata(reason, usage, metadata) where hasResponseTool && !hasApplicationTool && reason == "tool-calls":
            return [.finishMetadata(reason: "stop", usage: usage, providerMetadata: metadata)]
        default:
            break
        }
        return [part]
    }
}

/// Joins the provider's long-decode hops while maintaining one stream lifecycle.
public func googleStreamWithContinuation(
    body: JSONValue,
    request: LanguageModelRequest,
    modelID: String,
    providerID: String,
    prepared: GoogleGenerateContentPreparedCall,
    jsonResponseToolName: String?,
    continuation: AsyncThrowingStream<LanguageStreamPart, Error>.Continuation,
    errorHandler: (@Sendable (AIHTTPResponse) -> AIError)? = nil,
    open: (JSONValue) async throws -> (request: AIHTTPRequest, response: AIHTTPStreamResponse)
) async throws {
    var nextBody = body
    var state: GoogleGenerateContentStreamState?
    while true {
        try Task.checkCancellation()
        try request.abortSignal?.throwIfAborted()
        let opened: (request: AIHTTPRequest, response: AIHTTPStreamResponse)
        do {
            opened = try await open(nextBody)
            guard (200..<300).contains(opened.response.statusCode) else {
                let failure = try await bufferedHTTPResponse(from: opened.response, request: opened.request)
                throw errorHandler?(failure) ?? apiCallError(provider: providerID, response: failure)
            }
        } catch {
            try Task.checkCancellation()
            try request.abortSignal?.throwIfAborted()
            guard state != nil, !(error is CancellationError), !(error is AIAbortError) else { throw error }
            // Preserve the last continuation token so a failed hop is resumable.
            continuation.yield(googleContinuationStreamError(error))
            break
        }
        if state == nil {
            state = GoogleGenerateContentStreamState(
                response: httpResponseHead(from: opened.response, request: opened.request),
                includeRawChunks: request.includeRawChunks,
                modelID: modelID,
                warnings: prepared.warnings,
                toolNameMapping: prepared.toolNameMapping,
                jsonResponseToolName: jsonResponseToolName
            )
        } else {
            state!.startContinuationRequest()
        }
        var pendingToken: String?
        var receivedProviderError = false
        for try await event in serverSentEvents(from: opened.response.body) {
            try Task.checkCancellation()
            try request.abortSignal?.throwIfAborted()
            if event.data == "[DONE]" { break }
            let raw: JSONValue
            do { raw = try decodeJSONBody(Data(event.data.utf8)) }
            catch {
                state!.recordParseError()
                continuation.yield(.error(message: error.localizedDescription))
                continue
            }
            if raw["error"] != nil && raw["error"] != .null { receivedProviderError = true }
            if let candidate = raw["candidates"]?[0], candidate["finishReason"] == "CONTINUATION", let token = candidate["continuationToken"]?.stringValue {
                pendingToken = token
            }
            for part in state!.apply(raw) { continuation.yield(part) }
        }
        guard !receivedProviderError, let token = pendingToken else { break }
        state!.completeRequest(willContinue: true)
        var resumed = body.objectValue ?? [:]
        resumed["continuationToken"] = .string(token)
        nextBody = .object(resumed)
    }
    for part in state?.finish() ?? [] { continuation.yield(part) }
}

private func googleContinuationStreamError(_ error: Error) -> LanguageStreamPart {
    guard let error = error as? AIError, case let .apiCall(call) = error else {
        return .error(message: String(describing: error), rawValue: nil)
    }
    let body = try? decodeJSONBody(Data(call.responseBody.utf8))
    return .providerError(AIStreamProviderError(
        message: call.message ?? body?["error"]?["message"]?.stringValue ?? call.description,
        type: body?["error"]?["status"]?.stringValue,
        code: body?["error"]?["code"],
        statusCode: call.statusCode,
        isRetryable: call.isRetryable,
        data: .object([
            "responseBody": .string(call.responseBody),
            "responseHeaders": .object(call.responseHeaders.mapValues(JSONValue.string)),
            "error": body?["error"] ?? .null
        ])
    ))
}