import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleConvertJSONResponseTool(_ raw: JSONValue, name: String?) -> JSONValue {
    guard let name, var object = raw.objectValue,
          var candidates = object["candidates"]?.arrayValue,
          !candidates.isEmpty, var candidate = candidates[0].objectValue,
          var content = candidate["content"]?.objectValue,
          let parts = content["parts"]?.arrayValue else { return raw }
    content["parts"] = .array(parts.compactMap { part in
        if let call = part["functionCall"], call["name"]?.stringValue == name {
            var text: [String: JSONValue] = ["text": .string(call["args"]?.stringValue ?? googleGenerateContentArguments(call["args"]))]
            if let signature = part["thoughtSignature"] { text["thoughtSignature"] = signature }
            return .object(text)
        }
        if part["text"] != nil, part["thought"]?.boolValue != true { return nil }
        return part
    })
    candidate["content"] = .object(content)
    candidates[0] = .object(candidate)
    object["candidates"] = .array(candidates)
    return .object(object)
}

func googleJSONResponseToolContent(
    from raw: JSONValue,
    toolNameMapping: AIToolNameMapping
) -> [AIResultContentPart] {
    var content: [AIResultContentPart] = []
    var calls = googleGenerateContentToolCalls(from: raw, toolNameMapping: toolNameMapping).makeIterator()
    var results = googleGenerateContentToolResults(from: raw, toolNameMapping: toolNameMapping).makeIterator()
    for part in raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? [] {
        if let text = part["text"]?.stringValue {
            let metadata = googleThoughtSignatureProviderMetadata(from: part)
            content.append(part["thought"]?.boolValue == true ? .reasoning(text, providerMetadata: metadata) : .text(text, providerMetadata: metadata))
        } else if part["functionCall"] != nil || part["executableCode"] != nil || part["toolCall"] != nil {
            if let call = calls.next() { content.append(.toolCall(call)) }
        } else if part["codeExecutionResult"] != nil || part["toolResponse"] != nil {
            if let result = results.next() { content.append(.toolResult(result)) }
        } else if let data = part["inlineData"], let mediaType = data["mimeType"]?.stringValue,
                  let bytes = data["data"]?.stringValue {
            let file = AIStreamFile(mediaType: mediaType, data: Data(base64Encoded: bytes), providerMetadata: googleInlineDataProviderMetadata(from: part), rawValue: part)
            content.append(part["thought"]?.boolValue == true ? .reasoningFile(file) : .file(file))
        }
    }
    content.append(contentsOf: googleGenerateContentSources(from: raw).map(AIResultContentPart.source))
    return content
}

func googleGenerateContentText(from raw: JSONValue, includeThoughts: Bool = true) -> String? {
    let text = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue?.compactMap { part in
        includeThoughts || part["thought"]?.boolValue != true ? part["text"]?.stringValue : nil
    }.joined()
    return text
}

func googleGenerateContentReasoning(from raw: JSONValue) -> String {
    (raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []).compactMap { part in
        part["thought"]?.boolValue == true ? part["text"]?.stringValue : nil
    }.joined()
}

func googleGenerateContentToolCalls(
    from raw: JSONValue,
    toolNameMapping: AIToolNameMapping = AIToolNameMapping()
) -> [AIToolCall] {
    var calls: [AIToolCall] = []
    let parts = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []
    for (index, part) in parts.enumerated() {
        if let executableCode = part["executableCode"],
           executableCode["code"]?.stringValue != nil {
            calls.append(AIToolCall(
                id: "google-code-execution-\(index)",
                name: toolNameMapping.toCustomToolName("code_execution"),
                arguments: googleGenerateContentArguments(executableCode),
                providerExecuted: true,
                rawValue: part
            ))
            continue
        }
        if let functionCall = part["functionCall"],
           let name = functionCall["name"]?.stringValue {
            calls.append(AIToolCall(
                id: resolvedToolCallID(functionCall["id"]?.stringValue, whenMissing: "tool-call-\(index)"),
                name: name,
                arguments: googleGenerateContentArguments(functionCall["args"]),
                providerMetadata: googleThoughtSignatureProviderMetadata(from: part),
                rawValue: part
            ))
            continue
        }
        if let serverToolCall = part["toolCall"],
           let toolType = serverToolCall["toolType"]?.stringValue {
            let id = serverToolCall["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
                ?? "google-server-tool-\(index)"
            calls.append(AIToolCall(
                id: id,
                name: "server:\(toolType)",
                arguments: googleGenerateContentArguments(serverToolCall["args"]),
                providerExecuted: true,
                dynamic: true,
                providerMetadata: googleServerToolProviderMetadata(id: id, type: toolType, part: part),
                rawValue: part
            ))
        }
    }
    return calls
}

func googleGenerateContentToolResults(
    from raw: JSONValue,
    toolNameMapping: AIToolNameMapping = AIToolNameMapping()
) -> [AIToolResult] {
    var results: [AIToolResult] = []
    var lastCodeExecutionToolCallID: String?
    var lastServerToolCallID: String?
    let parts = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []
    for (index, part) in parts.enumerated() {
        if let executableCode = part["executableCode"],
           executableCode["code"]?.stringValue != nil {
            lastCodeExecutionToolCallID = "google-code-execution-\(index)"
            continue
        }
        if let codeExecutionResult = part["codeExecutionResult"] {
            results.append(AIToolResult(
                toolCallID: lastCodeExecutionToolCallID ?? "google-code-execution-result-\(index)",
                toolName: toolNameMapping.toCustomToolName("code_execution"),
                result: googleCodeExecutionResultJSON(codeExecutionResult)
            ))
            continue
        }
        if let serverToolCall = part["toolCall"],
           serverToolCall["toolType"]?.stringValue != nil {
            lastServerToolCallID = serverToolCall["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
                ?? "google-server-tool-\(index)"
            continue
        }
        if let serverToolResponse = part["toolResponse"],
            let toolType = serverToolResponse["toolType"]?.stringValue {
            let id = lastServerToolCallID
                ?? serverToolResponse["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
                ?? "google-server-tool-response-\(index)"
            results.append(AIToolResult(
                toolCallID: id,
                toolName: "server:\(toolType)",
                result: serverToolResponse["response"] ?? .object([:]),
                dynamic: true,
                providerMetadata: googleServerToolProviderMetadata(id: id, type: toolType, part: part)
            ))
            lastServerToolCallID = nil
        }
    }
    return results
}

func googleGenerateContentSources(from raw: JSONValue) -> [AISource] {
    let chunks = raw["candidates"]?[0]?["groundingMetadata"]?["groundingChunks"]?.arrayValue ?? []
    return chunks.enumerated().compactMap { index, chunk in
        googleGroundingChunkSource(from: chunk, index: index)
    }
}

func googleGenerateContentProviderMetadata(from raw: JSONValue, includeNullDefaults: Bool = true) -> [String: JSONValue] {
    let candidate = raw["candidates"]?[0]
    var google: [String: JSONValue] = [:]
    if let safetyRatings = candidate?["safetyRatings"] {
        google["safetyRatings"] = safetyRatings
    }
    if let promptFeedback = raw["promptFeedback"] {
        google["promptFeedback"] = promptFeedback
    }
    if let groundingMetadata = candidate?["groundingMetadata"],
       includeNullDefaults || groundingMetadata != .null {
        google["groundingMetadata"] = groundingMetadata
    }
    if let urlContextMetadata = candidate?["urlContextMetadata"] {
        google["urlContextMetadata"] = urlContextMetadata
    }
    if let token = candidate?["continuationToken"], token != .null {
        google["continuationToken"] = token
    }
    if let finishMessage = candidate?["finishMessage"] {
        google["finishMessage"] = finishMessage
    } else if includeNullDefaults {
        google["finishMessage"] = .null
    }
    if let usageMetadata = raw["usageMetadata"] {
        google["usageMetadata"] = usageMetadata
    } else if includeNullDefaults {
        google["usageMetadata"] = .null
    }
    if let serviceTier = raw["usageMetadata"]?["serviceTier"] {
        google["serviceTier"] = serviceTier
    } else if includeNullDefaults {
        google["serviceTier"] = .null
    }
    guard !google.isEmpty else { return [:] }
    return ["google": .object(google)]
}

func googleFinalizeGenerateContentProviderMetadata(_ providerMetadata: [String: JSONValue]) -> [String: JSONValue] {
    var google = providerMetadata["google"]?.objectValue ?? [:]
    google["finishMessage"] = google["finishMessage"] ?? .null
    google["usageMetadata"] = google["usageMetadata"] ?? .null
    google["serviceTier"] = google["serviceTier"] ?? .null
    return ["google": .object(google)]
}

func googleMergeProviderMetadata(_ current: [String: JSONValue], _ incoming: [String: JSONValue]) -> [String: JSONValue] {
    var output = current
    for (provider, value) in incoming {
        if var existing = output[provider]?.objectValue,
           let incomingObject = value.objectValue {
            existing.merge(incomingObject) { _, new in new }
            output[provider] = .object(existing)
        } else {
            output[provider] = value
        }
    }
    return output
}


public func googleGenerateContentUsage(from raw: JSONValue) -> TokenUsage? {
    guard let usage = raw["usageMetadata"] else { return nil }
    return googleGenerateContentUsage(fromUsageMetadata: usage)
}

public func googleGenerateContentUsage(fromUsageMetadata usage: JSONValue) -> TokenUsage {
    let promptTokens = usage["promptTokenCount"]?.intValue ?? 0
    let toolUsePromptTokens = usage["toolUsePromptTokenCount"]?.intValue ?? 0
    let cachedContentTokens = usage["cachedContentTokenCount"]?.intValue ?? 0
    let candidateTokens = usage["candidatesTokenCount"]?.intValue ?? 0
    let thoughtTokens = usage["thoughtsTokenCount"]?.intValue ?? 0
    let inputTokens = promptTokens + toolUsePromptTokens
    return TokenUsage(
        inputTokens: inputTokens,
        outputTokens: candidateTokens + thoughtTokens,
        totalTokens: usage["totalTokenCount"]?.intValue,
        inputTokensNoCache: inputTokens - cachedContentTokens,
        inputTokensCacheRead: cachedContentTokens,
        outputTextTokens: candidateTokens,
        outputReasoningTokens: thoughtTokens,
        rawValue: usage
    )
}

public func googleGenerateContentFinishReason(_ reason: String?, hasToolCalls: Bool) -> String? {
    switch reason {
    case "STOP":
        return hasToolCalls ? "tool-calls" : "stop"
    case "MAX_TOKENS", "CONTINUATION":
        return "length"
    case "IMAGE_SAFETY", "RECITATION", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
        return "content-filter"
    case "MALFORMED_FUNCTION_CALL":
        return "error"
    case nil:
        return nil
    default:
        return "other"
    }
}

func googleConfirmedPromptBlockReason(_ reason: String?) -> String? {
    guard let reason,
          !reason.isEmpty,
          reason != "BLOCK_REASON_UNSPECIFIED",
          reason != "BLOCKED_REASON_UNSPECIFIED" else {
        return nil
    }
    return reason
}

public func googleGenerateContentFinishReason(from raw: JSONValue, hasToolCalls: Bool) -> String? {
    let candidateFinishReason = raw["candidates"]?[0]?["finishReason"]?.stringValue
    let promptBlockReason = googleConfirmedPromptBlockReason(raw["promptFeedback"]?["blockReason"]?.stringValue)
    if candidateFinishReason == nil, promptBlockReason != nil {
        return "content-filter"
    }
    return googleGenerateContentFinishReason(candidateFinishReason ?? promptBlockReason, hasToolCalls: hasToolCalls)
}

/// Preserves per-request billing usage across automatically continued responses.
func googleAddUsageMetadata(_ first: JSONValue?, _ second: JSONValue?) -> JSONValue? {
    guard let first, first != .null else { return second }
    guard let second, second != .null else { return first }
    var combined = first.objectValue ?? [:]
    combined.merge(second.objectValue ?? [:]) { _, later in later }
    for key in ["promptTokenCount", "candidatesTokenCount", "toolUsePromptTokenCount", "totalTokenCount", "cachedContentTokenCount", "thoughtsTokenCount"] {
        let a = first[key]?.doubleValue
        let b = second[key]?.doubleValue
        if a != nil || b != nil { combined[key] = .number((a ?? 0) + (b ?? 0)) }
    }
    for key in ["promptTokensDetails", "cacheTokensDetails", "candidatesTokensDetails", "toolUsePromptTokensDetails"] {
        guard let a = first[key]?.arrayValue, let b = second[key]?.arrayValue else {
            combined[key] = (second[key] != nil && second[key] != .null) ? second[key] : first[key]
            continue
        }
        var order: [String] = []
        var counts: [String: Double] = [:]
        for detail in a + b {
            guard let modality = detail["modality"]?.stringValue, let count = detail["tokenCount"]?.doubleValue else { continue }
            if counts[modality] == nil { order.append(modality) }
            counts[modality, default: 0] += count
        }
        combined[key] = .array(order.map { .object(["modality": .string($0), "tokenCount": .number(counts[$0] ?? 0)]) })
    }
    return .object(combined)
}

public struct GoogleContinuedGenerateContentResponse {
    public var responses: [JSONValue]
    public var response: AIHTTPResponse
    public var usageMetadata: JSONValue?

    public func result(providerID: String, modelID: String, prepared: GoogleGenerateContentPreparedCall, jsonResponseToolName: String?) throws -> TextGenerationResult {
        let raw = responses.last ?? .object([:])
        var converted = raw.objectValue ?? [:]
        converted["usageMetadata"] = usageMetadata
        let parsed = responses.map { googleConvertJSONResponseTool($0, name: jsonResponseToolName) }
        let finishReason = googleGenerateContentFinishReason(from: raw, hasToolCalls: !(parsed.last.map { googleGenerateContentToolCalls(from: $0) } ?? []).isEmpty)
        let textParts = parsed.compactMap { googleGenerateContentText(from: $0, includeThoughts: responses.count == 1 && jsonResponseToolName == nil) }
        let content = parsed.enumerated().flatMap { hop, response in
            googleContinuationContent(googleJSONResponseToolContent(from: response, toolNameMapping: prepared.toolNameMapping), hop: hop)
        }
        let toolCalls = resultToolCalls(from: content)
        let toolResults = resultToolResults(from: content)
        guard !textParts.isEmpty || !content.isEmpty || finishReason == "content-filter" else {
            throw AIError.invalidResponse(provider: providerID, message: providerID.contains("vertex") ? "No candidate text found in Vertex response." : "No candidate text found in Google response.")
        }
        var metadata = googleGenerateContentProviderMetadata(from: .object(converted))
        if responses.count > 1 {
            var google = metadata["google"]?.objectValue ?? [:]
            google["groundingMetadataPerRequest"] = .array(responses.map {
                $0["candidates"]?[0]?["groundingMetadata"] ?? .null
            })
            google["groundingMetadata"] = raw["candidates"]?[0]?["groundingMetadata"] ?? .null
            if let count = responses.first?["usageMetadata"]?["promptTokenCount"], count != .null {
                google["initialPromptTokenCount"] = count
            }
            metadata["google"] = .object(google)
        }
        let hasMedia = content.contains { if case .file = $0 { return true }; if case .reasoningFile = $0 { return true }; return false }
        return TextGenerationResult(
            text: textParts.joined(),
            content: responses.count == 1 && jsonResponseToolName == nil && !hasMedia ? [] : content,
            reasoning: responses.count == 1 && jsonResponseToolName == nil ? "" : parsed.map(googleGenerateContentReasoning).joined(),
            finishReason: finishReason,
            usage: usageMetadata.map(googleGenerateContentUsage(fromUsageMetadata:)),
            toolCalls: toolCalls,
            toolResults: toolResults,
            sources: resultSources(from: content),
            providerMetadata: metadata,
            rawValue: raw,
            warnings: prepared.warnings,
            responseMetadata: googleGenerateContentResponseMetadata(from: raw, response: response, modelID: modelID)
        )
    }
    public init(responses: [JSONValue], response: AIHTTPResponse, usageMetadata: JSONValue? = nil) {
        self.responses = responses
        self.response = response
        self.usageMetadata = usageMetadata
    }

}

private func googleContinuationContent(_ content: [AIResultContentPart], hop: Int) -> [AIResultContentPart] {
    guard hop > 0 else { return content }
    var ids: [String: String] = [:]
    for case let .toolCall(call) in content {
        let explicitID = call.rawValue?["functionCall"]?["id"]?.stringValue ?? call.rawValue?["toolCall"]?["id"]?.stringValue
        if explicitID == nil || explicitID?.isEmpty == true { ids[call.id] = "\(call.id)-hop-\(hop)" }
    }
    return content.map { part in
        switch part {
        case var .source(source):
            source.id += "-hop-\(hop)"
            return .source(source)
        case var .toolCall(call):
            if let id = ids[call.id] {
                call.id = id
                if var google = call.providerMetadata["google"]?.objectValue, google["serverToolCallId"] != nil {
                    google["serverToolCallId"] = .string(id)
                    call.providerMetadata["google"] = .object(google)
                }
            }
            return .toolCall(call)
        case var .toolResult(result):
            if let id = ids[result.toolCallID] {
                result.toolCallID = id
                if var google = result.providerMetadata["google"]?.objectValue, google["serverToolCallId"] != nil {
                    google["serverToolCallId"] = .string(id)
                    result.providerMetadata["google"] = .object(google)
                }
            }
            return .toolResult(result)
        default:
            return part
        }
    }
}

public func googleGenerateContentWithContinuation(
    body: JSONValue,
    abortSignal: AIAbortSignal?,
    send: (JSONValue) async throws -> (json: JSONValue, response: AIHTTPResponse)
) async throws -> GoogleContinuedGenerateContentResponse {
    var nextBody = body
    var responses: [JSONValue] = []
    var usage: JSONValue?
    while true {
        try Task.checkCancellation()
        try abortSignal?.throwIfAborted()
        let response = try await send(nextBody)
        responses.append(response.json)
        usage = googleAddUsageMetadata(usage, response.json["usageMetadata"])
        guard let candidate = response.json["candidates"]?[0],
              candidate["finishReason"] == "CONTINUATION",
              let token = candidate["continuationToken"]?.stringValue else {
            return GoogleContinuedGenerateContentResponse(responses: responses, response: response.response, usageMetadata: usage)
        }
        var resumed = body.objectValue ?? [:]
        resumed["continuationToken"] = .string(token)
        nextBody = .object(resumed)
    }
}