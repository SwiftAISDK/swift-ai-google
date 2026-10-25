import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public struct GoogleInteractionsParsedOutput {
    public var content: [AIResultContentPart]
    public var text: String
    public var reasoning: String
    public var hasFunctionCall: Bool
    public init(content: [AIResultContentPart], text: String, reasoning: String, hasFunctionCall: Bool) {
        self.content = content
        self.text = text
        self.reasoning = reasoning
        self.hasFunctionCall = hasFunctionCall
    }

}

public func googleInteractionsParsedOutput(from raw: JSONValue) -> GoogleInteractionsParsedOutput {
    let interactionID = raw["id"]?.stringValue
    var content: [AIResultContentPart] = []
    var text = ""
    var reasoning = ""
    var hasFunctionCall = false
    var sourceCounter = 0
    var emittedSourceKeys: Set<String> = []

    for step in raw["steps"]?.arrayValue ?? [] {
        switch step["type"]?.stringValue {
        case "model_output":
            for block in step["content"]?.arrayValue ?? [] {
                switch block["type"]?.stringValue {
                case "text":
                    let value = block["text"]?.stringValue ?? ""
                    text += value
                    content.append(.text(
                        value,
                        providerMetadata: googleInteractionsPartProviderMetadata(interactionID: interactionID)
                    ))
                    content.append(contentsOf: googleInteractionsSources(
                        fromAnnotations: block["annotations"]?.arrayValue,
                        sourceCounter: &sourceCounter,
                        emittedKeys: &emittedSourceKeys
                    ).map(AIResultContentPart.source))
                case "video", "image":
                    guard let file = googleInteractionsOutputFile(from: block, interactionID: interactionID) else {
                        continue
                    }
                    content.append(.file(file))
                default:
                    continue
                }
            }
        case "thought":
            let value = (step["summary"]?.arrayValue ?? []).compactMap { item in
                item["type"]?.stringValue == "text" ? item["text"]?.stringValue : nil
            }.joined(separator: "\n")
            reasoning += value
            content.append(.reasoning(
                value,
                providerMetadata: googleInteractionsPartProviderMetadata(
                    interactionID: interactionID,
                    signature: step["signature"]?.stringValue
                )
            ))
        case "processing_call":
            content.append(.custom(
                .object(["kind": .string("google.processing_call")]),
                providerMetadata: googleInteractionsPartProviderMetadata(
                    interactionID: interactionID,
                    signature: step["signature"]?.stringValue,
                    processingID: step["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? generateId()
                )
            ))
        case "processing_result":
            content.append(.custom(
                .object(["kind": .string("google.processing_result")]),
                providerMetadata: googleInteractionsPartProviderMetadata(
                    interactionID: interactionID,
                    signature: step["signature"]?.stringValue,
                    processingCallID: step["call_id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? generateId()
                )
            ))
        case "function_call":
            guard let name = step["name"]?.stringValue else { continue }
            hasFunctionCall = true
            content.append(.toolCall(AIToolCall(
                id: resolvedToolCallID(step["id"]?.stringValue, whenMissing: "tool-call-\(name)"),
                name: name,
                arguments: googleInteractionsArguments(step["arguments"]),
                providerMetadata: googleInteractionsPartProviderMetadata(
                    interactionID: interactionID,
                    signature: step["signature"]?.stringValue
                ),
                rawValue: step
            )))
        default:
            guard let type = step["type"]?.stringValue else { continue }
            if googleInteractionsBuiltinToolCallTypes.contains(type) {
                let name = googleInteractionsBuiltinToolName(type: type, explicitName: step["name"]?.stringValue)
                content.append(.toolCall(AIToolCall(
                    id: step["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? generateId(),
                    name: name,
                    arguments: googleInteractionsArguments(step["arguments"]),
                    providerExecuted: true,
                    rawValue: step
                )))
            } else if googleInteractionsBuiltinToolResultTypes.contains(type) {
                let name = googleInteractionsBuiltinToolName(type: type, explicitName: step["name"]?.stringValue)
                content.append(.toolResult(AIToolResult(
                    toolCallID: step["call_id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? generateId(),
                    toolName: name,
                    result: step["result"] ?? .null,
                    isError: step["is_error"]?.boolValue ?? false,
                    providerExecuted: true
                )))
                content.append(contentsOf: googleInteractionsBuiltinToolResultSources(
                    from: step,
                    sourceCounter: &sourceCounter,
                    emittedKeys: &emittedSourceKeys
                ).map(AIResultContentPart.source))
            }
        }
    }

    return GoogleInteractionsParsedOutput(
        content: content,
        text: text,
        reasoning: reasoning,
        hasFunctionCall: hasFunctionCall
    )
}

let googleInteractionsBuiltinToolCallTypes: Set<String> = [
    "google_search_call",
    "code_execution_call",
    "url_context_call",
    "file_search_call",
    "google_maps_call",
    "mcp_server_tool_call"
]

public let googleInteractionsBuiltinToolResultTypes: Set<String> = [
    "google_search_result",
    "code_execution_result",
    "url_context_result",
    "file_search_result",
    "google_maps_result",
    "mcp_server_tool_result"
]

func googleInteractionsBuiltinToolName(type: String, explicitName: String?) -> String {
    if type == "mcp_server_tool_call" || type == "mcp_server_tool_result" {
        return explicitName ?? "mcp_server_tool"
    }
    if type.hasSuffix("_call") {
        return String(type.dropLast("_call".count))
    }
    if type.hasSuffix("_result") {
        return String(type.dropLast("_result".count))
    }
    return explicitName ?? type
}

func googleInteractionsPartProviderMetadata(
    interactionID: String?,
    signature: String? = nil,
    processingID: String? = nil,
    processingCallID: String? = nil
) -> [String: JSONValue] {
    var google: [String: JSONValue] = [:]
    if let interactionID { google["interactionId"] = .string(interactionID) }
    if let signature { google["signature"] = .string(signature) }
    if let processingID { google["processingId"] = .string(processingID) }
    if let processingCallID { google["processingCallId"] = .string(processingCallID) }
    return google.isEmpty ? [:] : ["google": .object(google)]
}

private func googleInteractionsOutputFile(from block: JSONValue, interactionID: String?) -> AIStreamFile? {
    let type = block["type"]?.stringValue
    let defaultMediaType = type == "video" ? "video/mp4" : "image/png"
    let mediaType = block["mime_type"]?.stringValue ?? defaultMediaType
    let metadata = googleInteractionsPartProviderMetadata(interactionID: interactionID)
    if let base64 = block["data"]?.stringValue, !base64.isEmpty {
        return AIStreamFile(
            mediaType: mediaType,
            data: Data(base64Encoded: base64),
            providerMetadata: metadata,
            rawValue: block
        )
    }
    if let uri = block["uri"]?.stringValue, !uri.isEmpty {
        return AIStreamFile(
            mediaType: mediaType,
            url: uri,
            providerMetadata: metadata,
            rawValue: block
        )
    }
    return nil
}

func googleInteractionsText(from raw: JSONValue) -> String {
    (raw["steps"]?.arrayValue ?? []).compactMap { step in
        guard step["type"]?.stringValue == "model_output" else { return nil }
        return step["content"]?.arrayValue?.compactMap { block in
            block["type"]?.stringValue == "text" ? block["text"]?.stringValue : nil
        }.joined()
    }.joined()
}

public func googleInteractionsProviderMetadata(from raw: JSONValue) -> [String: JSONValue] {
    var google: [String: JSONValue] = [:]
    if let id = raw["id"] {
        google["interactionId"] = id
    }
    if let serviceTier = raw["service_tier"] {
        google["serviceTier"] = serviceTier
    }
    var outputTokensByModality: [String: JSONValue] = [:]
    for entry in raw["usage"]?["output_tokens_by_modality"]?.arrayValue ?? [] {
        guard let modality = entry["modality"]?.stringValue,
              let tokens = normalizedBatchJSONInteger(entry["tokens"]) else {
            continue
        }
        outputTokensByModality[modality] = .number(Double(tokens))
    }
    if !outputTokensByModality.isEmpty {
        google["outputTokensByModality"] = .object(outputTokensByModality)
    }
    guard !google.isEmpty else { return [:] }
    return ["google": .object(google)]
}

public func googleInteractionsSources(from raw: JSONValue) -> [AISource] {
    var sourceCounter = 0
    var emittedKeys: Set<String> = []
    return googleInteractionsSources(from: raw, sourceCounter: &sourceCounter, emittedKeys: &emittedKeys)
}

public func googleInteractionsSources(from raw: JSONValue, sourceCounter: inout Int, emittedKeys: inout Set<String>) -> [AISource] {
    var sources: [AISource] = []

    if let steps = raw["steps"]?.arrayValue {
        for step in steps {
            sources.append(contentsOf: googleInteractionsSources(fromStep: step, sourceCounter: &sourceCounter, emittedKeys: &emittedKeys))
        }
    }

    if let step = raw["step"] {
        sources.append(contentsOf: googleInteractionsSources(fromStep: step, sourceCounter: &sourceCounter, emittedKeys: &emittedKeys))
    }

    if let delta = raw["delta"],
       delta["type"]?.stringValue == "text_annotation" || delta["type"]?.stringValue == "text_annotation_delta" {
        sources.append(contentsOf: googleInteractionsSources(fromAnnotations: delta["annotations"]?.arrayValue, sourceCounter: &sourceCounter, emittedKeys: &emittedKeys))
    }

    return sources
}

public func googleInteractionsSources(fromStep step: JSONValue, sourceCounter: inout Int, emittedKeys: inout Set<String>) -> [AISource] {
    var sources: [AISource] = []
    if step["type"]?.stringValue == "model_output" {
        for block in step["content"]?.arrayValue ?? [] where block["type"]?.stringValue == "text" {
            sources.append(contentsOf: googleInteractionsSources(fromAnnotations: block["annotations"]?.arrayValue, sourceCounter: &sourceCounter, emittedKeys: &emittedKeys))
        }
    }
    sources.append(contentsOf: googleInteractionsBuiltinToolResultSources(from: step, sourceCounter: &sourceCounter, emittedKeys: &emittedKeys))
    return sources
}

public func googleInteractionsSources(fromAnnotations annotations: [JSONValue]?, sourceCounter: inout Int, emittedKeys: inout Set<String>) -> [AISource] {
    annotations?.compactMap { annotation in
        guard let source = googleInteractionsAnnotationSource(from: annotation, id: "interactions-source-\(sourceCounter)") else {
            return nil
        }
        let key = googleInteractionsSourceKey(source)
        guard !emittedKeys.contains(key) else {
            return nil
        }
        emittedKeys.insert(key)
        sourceCounter += 1
        return source
    } ?? []
}

func googleInteractionsAnnotationSource(from annotation: JSONValue, id: String) -> AISource? {
    switch annotation["type"]?.stringValue {
    case "url_citation":
        guard let url = annotation["url"]?.stringValue, !url.isEmpty else { return nil }
        return AISource(id: id, sourceType: "url", url: url, title: annotation["title"]?.stringValue, rawValue: annotation)
    case "file_citation":
        guard let uri = annotation["url"]?.stringValue ?? annotation["document_uri"]?.stringValue ?? annotation["file_name"]?.stringValue, !uri.isEmpty else {
            return nil
        }
        if googleInteractionsIsHTTP(uri) {
            return AISource(id: id, sourceType: "url", url: uri, title: annotation["file_name"]?.stringValue, rawValue: annotation)
        }
        let filename = annotation["file_name"]?.stringValue ?? googleInteractionsBasename(uri)
        return AISource(
            id: id,
            sourceType: "document",
            title: annotation["file_name"]?.stringValue ?? filename ?? uri,
            mediaType: googleInteractionsDocumentMediaType(uri),
            filename: filename,
            rawValue: annotation
        )
    case "place_citation":
        guard let url = annotation["url"]?.stringValue, !url.isEmpty else { return nil }
        return AISource(id: id, sourceType: "url", url: url, title: annotation["name"]?.stringValue, rawValue: annotation)
    default:
        return nil
    }
}

func googleInteractionsBuiltinToolResultSources(from step: JSONValue, sourceCounter: inout Int, emittedKeys: inout Set<String>) -> [AISource] {
    guard let type = step["type"]?.stringValue else { return [] }
    let rawSources: [AISource]
    switch type {
    case "url_context_result":
        rawSources = (step["result"]?.arrayValue ?? []).compactMap { entry in
            guard let url = entry["url"]?.stringValue, !url.isEmpty else { return nil }
            if let status = entry["status"]?.stringValue, status != "success" { return nil }
            return AISource(id: "interactions-source-\(sourceCounter)", sourceType: "url", url: url, rawValue: entry)
        }
    case "google_search_result":
        rawSources = (step["result"]?.arrayValue ?? []).compactMap { entry in
            guard let url = entry["url"]?.stringValue, !url.isEmpty else { return nil }
            return AISource(id: "interactions-source-\(sourceCounter)", sourceType: "url", url: url, title: entry["title"]?.stringValue, rawValue: entry)
        }
    case "google_maps_result":
        rawSources = (step["result"]?.arrayValue ?? []).flatMap { entry in
            (entry["places"]?.arrayValue ?? []).compactMap { place in
                guard let url = place["url"]?.stringValue, !url.isEmpty else { return nil }
                return AISource(id: "interactions-source-\(sourceCounter)", sourceType: "url", url: url, title: place["name"]?.stringValue, rawValue: place)
            }
        }
    case "file_search_result":
        rawSources = (step["result"]?.arrayValue ?? []).compactMap { entry in
            guard let uri = entry["url"]?.stringValue ?? entry["document_uri"]?.stringValue ?? entry["file_name"]?.stringValue ?? entry["source"]?.stringValue, !uri.isEmpty else {
                return nil
            }
            if googleInteractionsIsHTTP(uri) {
                return AISource(id: "interactions-source-\(sourceCounter)", sourceType: "url", url: uri, title: entry["title"]?.stringValue, rawValue: entry)
            }
            let filename = entry["file_name"]?.stringValue ?? googleInteractionsBasename(uri)
            return AISource(
                id: "interactions-source-\(sourceCounter)",
                sourceType: "document",
                title: entry["title"]?.stringValue ?? entry["file_name"]?.stringValue ?? filename ?? uri,
                mediaType: googleInteractionsDocumentMediaType(uri),
                filename: filename,
                rawValue: entry
            )
        }
    default:
        return []
    }

    var sources: [AISource] = []
    for var source in rawSources {
        source.id = "interactions-source-\(sourceCounter)"
        let key = googleInteractionsSourceKey(source)
        guard !emittedKeys.contains(key) else { continue }
        emittedKeys.insert(key)
        sourceCounter += 1
        sources.append(source)
    }
    return sources
}

func googleInteractionsSourceKey(_ source: AISource) -> String {
    if source.sourceType == "url", let url = source.url {
        return "url:\(url)"
    }
    return "doc:\(source.filename ?? source.title ?? source.id)"
}

func googleInteractionsIsHTTP(_ value: String) -> Bool {
    value.hasPrefix("http://") || value.hasPrefix("https://")
}

func googleInteractionsBasename(_ value: String) -> String? {
    value.split(separator: "/").last.map(String.init)
}

func googleInteractionsDocumentMediaType(_ value: String) -> String {
    let lower = value.lowercased()
    if lower.hasSuffix(".pdf") { return "application/pdf" }
    if lower.hasSuffix(".txt") { return "text/plain" }
    if lower.hasSuffix(".md") || lower.hasSuffix(".markdown") { return "text/markdown" }
    if lower.hasSuffix(".doc") { return "application/msword" }
    if lower.hasSuffix(".docx") { return "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }
    return "application/octet-stream"
}