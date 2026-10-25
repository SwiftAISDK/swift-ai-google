import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleInteractionsHasFunctionCall(_ raw: JSONValue) -> Bool {
    (raw["steps"]?.arrayValue ?? []).contains { step in
        step["type"]?.stringValue == "function_call" || step["type"]?.stringValue == "google_search_call" || step["type"]?.stringValue == "code_execution_call"
    }
}

func googleInteractionsToolCalls(from raw: JSONValue) -> [AIToolCall] {
    (raw["steps"]?.arrayValue ?? []).compactMap { step in
        guard step["type"]?.stringValue == "function_call",
              let name = step["name"]?.stringValue else {
            return nil
        }
        return AIToolCall(
            id: resolvedToolCallID(step["id"]?.stringValue, whenMissing: "tool-call-\(name)"),
            name: name,
            arguments: googleInteractionsArguments(step["arguments"]),
            rawValue: step
        )
    }
}

struct GoogleInteractionsToolCallBuffer {
    var id: String
    var name: String
    var arguments: String
    var inputStarted: Bool
    var google: [String: JSONValue]
    var rawValue: JSONValue?
}

public struct GoogleInteractionsStreamingToolCalls {
    private var buffers: [Int: GoogleInteractionsToolCallBuffer] = [:]

    public mutating func start(step: JSONValue?, index: Int?, interactionID: String?) -> [LanguageStreamPart] {
        let key = index ?? 0
        guard let step,
              let name = step["name"]?.stringValue else {
            return []
        }
        let id = step["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? "tool-call-\(key)"
        let arguments = googleInteractionsArguments(step["arguments"])
        buffers[key] = GoogleInteractionsToolCallBuffer(
            id: id,
            name: name,
            arguments: arguments == "{}" ? "" : arguments,
            inputStarted: true,
            google: googleInteractionsStreamingMetadata(
                interactionID: interactionID,
                signature: step["signature"]?.stringValue
            ),
            rawValue: step
        )
        var parts: [LanguageStreamPart] = [.toolInputStart(id: id, name: name)]
        if arguments != "{}" {
            parts.append(.toolCallDelta(id: id, name: name, argumentsDelta: arguments, index: key))
            parts.append(.toolInputDelta(id: id, delta: arguments))
        }
        return parts
    }

    public mutating func delta(_ delta: JSONValue, index: Int?, interactionID: String?) -> [LanguageStreamPart] {
        let key = index ?? 0
        guard var buffer = buffers[key] else {
            return []
        }
        let argumentsDelta = delta["arguments"]?.stringValue ?? ""
        buffer.arguments += argumentsDelta
        if let id = delta["id"]?.stringValue, !id.isEmpty {
            buffer.id = id
        }
        if let interactionID {
            buffer.google["interactionId"] = .string(interactionID)
        }
        if let signature = delta["signature"]?.stringValue {
            buffer.google["signature"] = .string(signature)
        }
        buffer.rawValue = delta
        buffers[key] = buffer
        var parts: [LanguageStreamPart] = [.toolCallDelta(id: buffer.id, name: buffer.name, argumentsDelta: argumentsDelta, index: key)]
        if !argumentsDelta.isEmpty {
            parts.append(.toolInputDelta(id: buffer.id, delta: argumentsDelta))
        }
        return parts
    }

    public mutating func stop(index: Int?) -> [LanguageStreamPart] {
        let key = index ?? 0
        guard let buffer = buffers.removeValue(forKey: key) else {
            return []
        }
        return [
            .toolInputEnd(id: buffer.id),
            .toolCall(AIToolCall(
            id: buffer.id,
            name: buffer.name,
            arguments: buffer.arguments.isEmpty ? "{}" : buffer.arguments,
            providerMetadata: buffer.google.isEmpty ? [:] : ["google": .object(buffer.google)],
            rawValue: buffer.rawValue
            ))
        ]
    }
    public init() {

    }

}

public struct GoogleInteractionsStreamingContent {
    private var openTextIDs: Set<String> = []
    private var reasoning: [Int: GoogleInteractionsStreamingReasoning] = [:]
    private var media: [Int: GoogleInteractionsStreamingMedia] = [:]
    private var custom: [Int: GoogleInteractionsStreamingCustom] = [:]
    private var builtinCalls: [Int: GoogleInteractionsStreamingBuiltinCall] = [:]
    private var builtinResults: [Int: GoogleInteractionsStreamingBuiltinResult] = [:]

    public mutating func start(_ step: JSONValue?, index: Int?, interactionID: String?) -> [LanguageStreamPart] {
        let key = index ?? 0
        guard let step else { return [] }
        switch step["type"]?.stringValue {
        case "thought":
            let id = "reasoning-\(key)"
            reasoning[key] = GoogleInteractionsStreamingReasoning(
                id: id,
                google: googleInteractionsStreamingMetadata(
                    interactionID: interactionID,
                    signature: step["signature"]?.stringValue
                )
            )
            var parts: [LanguageStreamPart] = [.reasoningStart(id: id)]
            for item in step["summary"]?.arrayValue ?? [] {
                if item["type"]?.stringValue == "text",
                   let text = item["text"]?.stringValue {
                    parts.append(.reasoningDeltaPart(id: id, delta: text))
                }
            }
            return parts
        case "processing_call":
            var google = googleInteractionsStreamingMetadata(
                interactionID: interactionID,
                signature: step["signature"]?.stringValue
            )
            google["processingId"] = .string(
                step["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? "processing-\(key)"
            )
            custom[key] = GoogleInteractionsStreamingCustom(
                kind: "google.processing_call",
                google: google
            )
        case "processing_result":
            var google = googleInteractionsStreamingMetadata(
                interactionID: interactionID,
                signature: step["signature"]?.stringValue
            )
            google["processingCallId"] = .string(
                step["call_id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? "processing-\(key)"
            )
            custom[key] = GoogleInteractionsStreamingCustom(
                kind: "google.processing_result",
                google: google
            )
        case "model_output":
            if let block = step["content"]?.arrayValue?.first,
               block["type"]?.stringValue == "video" || block["type"]?.stringValue == "image" {
                media[key] = googleInteractionsStreamingMedia(
                    from: block,
                    interactionID: interactionID
                )
            }
        default:
            guard let type = step["type"]?.stringValue else { break }
            if googleInteractionsBuiltinToolCallTypes.contains(type) {
                builtinCalls[key] = GoogleInteractionsStreamingBuiltinCall(
                    type: type,
                    id: step["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? generateId(),
                    name: googleInteractionsBuiltinToolName(type: type, explicitName: step["name"]?.stringValue),
                    arguments: step["arguments"] ?? .object([:]),
                    rawValue: step
                )
            } else if googleInteractionsBuiltinToolResultTypes.contains(type) {
                builtinResults[key] = GoogleInteractionsStreamingBuiltinResult(
                    type: type,
                    callID: step["call_id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? generateId(),
                    name: googleInteractionsBuiltinToolName(type: type, explicitName: step["name"]?.stringValue),
                    result: step["result"] ?? .null,
                    isError: step["is_error"]?.boolValue ?? false
                )
            }
        }
        return []
    }

    public mutating func delta(_ delta: JSONValue, index: Int?, interactionID: String?) -> [LanguageStreamPart] {
        let key = index ?? 0
        var parts: [LanguageStreamPart] = []
        if var pending = reasoning[key] {
            if let interactionID {
                pending.google["interactionId"] = .string(interactionID)
            }
            switch delta["type"]?.stringValue {
            case "thought_summary":
                if delta["content"]?["type"]?.stringValue == "text",
                   let text = delta["content"]?["text"]?.stringValue {
                    parts.append(.reasoningDeltaPart(id: pending.id, delta: text))
                }
            case "thought_signature":
                if let signature = delta["signature"]?.stringValue {
                    pending.google["signature"] = .string(signature)
                }
            default:
                break
            }
            reasoning[key] = pending
            return parts
        }
        if var pending = builtinCalls[key] {
            guard delta["type"]?.stringValue == pending.type else { return [] }
            if let id = delta["id"]?.stringValue, !id.isEmpty {
                pending.id = id
            }
            if let arguments = delta["arguments"] {
                pending.arguments = arguments
            }
            if pending.type == "mcp_server_tool_call",
               let name = delta["name"]?.stringValue {
                pending.name = name
            }
            pending.rawValue = delta
            builtinCalls[key] = pending
            return []
        }
        if var pending = builtinResults[key] {
            guard delta["type"]?.stringValue == pending.type else { return [] }
            if let callID = delta["call_id"]?.stringValue, !callID.isEmpty {
                pending.callID = callID
            }
            if let result = delta["result"] {
                pending.result = result
            }
            if let isError = delta["is_error"]?.boolValue {
                pending.isError = isError
            }
            if pending.type == "mcp_server_tool_result",
               let name = delta["name"]?.stringValue {
                pending.name = name
            }
            builtinResults[key] = pending
            return []
        }
        if delta["type"]?.stringValue == "video" || delta["type"]?.stringValue == "image" {
            let pending = googleInteractionsStreamingMedia(from: delta, interactionID: interactionID)
            media.removeValue(forKey: key)
            return pending.file.map { [.file($0)] } ?? []
        }
        if var pending = custom[key] {
            if let signature = delta["signature"]?.stringValue {
                pending.google["signature"] = .string(signature)
            }
            if delta["type"]?.stringValue == "processing_call",
               let processingID = delta["id"]?.stringValue,
               !processingID.isEmpty {
                pending.google["processingId"] = .string(processingID)
            }
            if delta["type"]?.stringValue == "processing_result",
               let processingCallID = delta["call_id"]?.stringValue,
               !processingCallID.isEmpty {
                pending.google["processingCallId"] = .string(processingCallID)
            }
            custom[key] = pending
            return parts
        }
        if let text = delta["text"]?.stringValue, !text.isEmpty {
            let id = "text-\(key)"
            if openTextIDs.insert(id).inserted {
                parts.append(.textStart(id: id))
            }
            parts.append(.textDeltaPart(id: id, delta: text))
        }
        if let summary = delta["summary"]?.stringValue, !summary.isEmpty {
            let id = "reasoning-\(key)"
            if reasoning[key] == nil {
                reasoning[key] = GoogleInteractionsStreamingReasoning(
                    id: id,
                    google: googleInteractionsStreamingMetadata(
                        interactionID: interactionID,
                        signature: nil
                    )
                )
                parts.append(.reasoningStart(id: id))
            }
            parts.append(.reasoningDeltaPart(id: id, delta: summary))
        }
        return parts
    }

    public mutating func stop(
        index: Int?,
        sourceCounter: inout Int,
        emittedSourceKeys: inout Set<String>
    ) -> [LanguageStreamPart] {
        let key = index ?? 0
        var parts: [LanguageStreamPart] = []
        if let pending = reasoning.removeValue(forKey: key) {
            parts.append(.reasoningEnd(
                id: pending.id,
                providerMetadata: pending.google.isEmpty ? [:] : ["google": .object(pending.google)]
            ))
        }
        let textID = "text-\(key)"
        if openTextIDs.remove(textID) != nil {
            parts.append(.textEnd(id: textID))
        }
        if let pending = media.removeValue(forKey: key),
           let file = pending.file {
            parts.append(.file(file))
        }
        if let pending = custom.removeValue(forKey: key) {
            parts.append(.custom(
                .object(["kind": .string(pending.kind)]),
                providerMetadata: ["google": .object(pending.google)]
            ))
        }
        if let pending = builtinCalls.removeValue(forKey: key) {
            parts.append(.toolCall(AIToolCall(
                id: pending.id,
                name: pending.name,
                arguments: googleInteractionsArguments(pending.arguments),
                providerExecuted: true,
                rawValue: pending.rawValue
            )))
        }
        if let pending = builtinResults.removeValue(forKey: key) {
            parts.append(contentsOf: googleInteractionsStreamingBuiltinResultParts(
                pending,
                sourceCounter: &sourceCounter,
                emittedSourceKeys: &emittedSourceKeys
            ))
        }
        return parts
    }

    public mutating func finishParts(
        sourceCounter: inout Int,
        emittedSourceKeys: inout Set<String>
    ) -> [LanguageStreamPart] {
        var parts: [LanguageStreamPart] = []
        for key in reasoning.keys.sorted() {
            guard let pending = reasoning[key] else { continue }
            parts.append(.reasoningEnd(
                id: pending.id,
                providerMetadata: pending.google.isEmpty ? [:] : ["google": .object(pending.google)]
            ))
        }
        reasoning.removeAll()
        for id in openTextIDs.sorted() {
            parts.append(.textEnd(id: id))
        }
        openTextIDs.removeAll()
        for key in media.keys.sorted() {
            if let file = media[key]?.file {
                parts.append(.file(file))
            }
        }
        media.removeAll()
        for key in custom.keys.sorted() {
            guard let pending = custom[key] else { continue }
            parts.append(.custom(
                .object(["kind": .string(pending.kind)]),
                providerMetadata: ["google": .object(pending.google)]
            ))
        }
        custom.removeAll()
        for key in builtinCalls.keys.sorted() {
            guard let pending = builtinCalls[key] else { continue }
            parts.append(.toolCall(AIToolCall(
                id: pending.id,
                name: pending.name,
                arguments: googleInteractionsArguments(pending.arguments),
                providerExecuted: true,
                rawValue: pending.rawValue
            )))
        }
        builtinCalls.removeAll()
        for key in builtinResults.keys.sorted() {
            guard let pending = builtinResults[key] else { continue }
            parts.append(contentsOf: googleInteractionsStreamingBuiltinResultParts(
                pending,
                sourceCounter: &sourceCounter,
                emittedSourceKeys: &emittedSourceKeys
            ))
        }
        builtinResults.removeAll()
        return parts
    }
    public init() {

    }

}

private struct GoogleInteractionsStreamingReasoning {
    var id: String
    var google: [String: JSONValue]
}

private struct GoogleInteractionsStreamingCustom {
    var kind: String
    var google: [String: JSONValue]
}

private struct GoogleInteractionsStreamingBuiltinCall {
    var type: String
    var id: String
    var name: String
    var arguments: JSONValue
    var rawValue: JSONValue
}

private struct GoogleInteractionsStreamingBuiltinResult {
    var type: String
    var callID: String
    var name: String
    var result: JSONValue
    var isError: Bool
}

private func googleInteractionsStreamingBuiltinResultParts(
    _ pending: GoogleInteractionsStreamingBuiltinResult,
    sourceCounter: inout Int,
    emittedSourceKeys: inout Set<String>
) -> [LanguageStreamPart] {
    var parts: [LanguageStreamPart] = [.toolResult(AIToolResult(
        toolCallID: pending.callID,
        toolName: pending.name,
        result: pending.result,
        isError: pending.isError,
        providerExecuted: true
    ))]
    let sourceStep: JSONValue = .object([
        "type": .string(pending.type),
        "call_id": .string(pending.callID),
        "result": pending.result
    ])
    parts.append(contentsOf: googleInteractionsBuiltinToolResultSources(
        from: sourceStep,
        sourceCounter: &sourceCounter,
        emittedKeys: &emittedSourceKeys
    ).map(LanguageStreamPart.source))
    return parts
}

private struct GoogleInteractionsStreamingMedia {
    var mediaType: String
    var data: String?
    var uri: String?
    var providerMetadata: [String: JSONValue]
    var rawValue: JSONValue

    var file: AIStreamFile? {
        if let data, !data.isEmpty {
            return AIStreamFile(
                mediaType: mediaType,
                data: Data(base64Encoded: data),
                providerMetadata: providerMetadata,
                rawValue: rawValue
            )
        }
        if let uri, !uri.isEmpty {
            return AIStreamFile(
                mediaType: mediaType,
                url: uri,
                providerMetadata: providerMetadata,
                rawValue: rawValue
            )
        }
        return nil
    }
}

private func googleInteractionsStreamingMedia(
    from value: JSONValue,
    interactionID: String?
) -> GoogleInteractionsStreamingMedia {
    let defaultMediaType = value["type"]?.stringValue == "video" ? "video/mp4" : "image/png"
    return GoogleInteractionsStreamingMedia(
        mediaType: value["mime_type"]?.stringValue ?? defaultMediaType,
        data: value["data"]?.stringValue,
        uri: value["uri"]?.stringValue,
        providerMetadata: googleInteractionsPartProviderMetadata(interactionID: interactionID),
        rawValue: value
    )
}

private func googleInteractionsStreamingMetadata(
    interactionID: String?,
    signature: String?
) -> [String: JSONValue] {
    var google: [String: JSONValue] = [:]
    if let interactionID { google["interactionId"] = .string(interactionID) }
    if let signature { google["signature"] = .string(signature) }
    return google
}

func googleInteractionsArguments(_ value: JSONValue?) -> String {
    guard let value else { return "{}" }
    guard let data = try? encodeJSONBody(value),
          let text = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return text
}

public func googleInteractionsFinishReason(status: String?, hasFunctionCall: Bool) -> String? {
    switch status {
    case "completed":
        return hasFunctionCall ? "tool-calls" : "stop"
    case "requires_action":
        return "tool-calls"
    case "failed":
        return "error"
    case "incomplete":
        return "length"
    case "cancelled":
        return "other"
    default:
        return "other"
    }
}

public func googleInteractionsStreamProviderError(from raw: JSONValue) -> AIStreamProviderError? {
    guard raw["event_type"]?.stringValue == "error" else { return nil }
    let error = raw["error"] ?? raw
    let code: JSONValue?
    if error["code"]?.stringValue != nil || error["code"]?.doubleValue != nil {
        code = error["code"]
    } else {
        code = nil
    }
    return AIStreamProviderError(
        message: error["message"]?.stringValue ?? "Unknown interaction error",
        type: "error",
        code: code,
        data: raw
    )
}

public func googleInteractionsUsage(from raw: JSONValue) -> TokenUsage? {
    guard let usage = raw["usage"] else { return nil }
    let output = (usage["total_output_tokens"]?.intValue ?? 0) + (usage["total_thought_tokens"]?.intValue ?? 0)
    return TokenUsage(
        inputTokens: usage["total_input_tokens"]?.intValue,
        outputTokens: output == 0 && usage["total_output_tokens"] == nil && usage["total_thought_tokens"] == nil ? nil : output,
        totalTokens: usage["total_tokens"]?.intValue
    )
}

public func googleInteractionsIsTerminal(_ status: String?) -> Bool {
    switch status {
    case "completed", "failed", "incomplete", "cancelled":
        return true
    default:
        return false
    }
}

public func googleInteractionsPollTimeout(raw: JSONValue) -> UInt64 {
    let milliseconds = raw["pollingTimeoutMs"]?.intValue ?? 600_000
    return UInt64(milliseconds) * 1_000_000
}