import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

/// Sentinel Google documents for replaying `functionCall` parts whose original
/// `thoughtSignature` is not available to the client.
let googleSkipThoughtSignatureValidator = "skip_thought_signature_validator"

/// Options that mirror upstream `convertToGoogleMessages`.
struct GooglePromptConversionOptions {
    var modelID: String
    var providerOptionsNames: [String] = ["google"]
    var includeFunctionCallIDs: Bool = true
    var supportsGoogleCloudStorageToolResults: Bool = false

    var isGemini3Model: Bool { googleMessageTargetIsGemini3(modelID) }
    var supportsFunctionResponseParts: Bool { googleMessageTargetIsGemini3(modelID) }
    var isVertexLike: Bool { !providerOptionsNames.contains("google") }
}

/// Mutable state shared by every message of one request.
struct GooglePromptConversionState {
    var missingSignatureToolNames: [String] = []
}

/// Converts a whole prompt like upstream `convertToGoogleMessages`: system
/// messages are only accepted before other messages, server-side tool results
/// are attached to the preceding model content, and a single warning lists
/// every replayed unsigned Gemini 3 function call.
func googleGenerateContentPrompt(
    _ messages: [AIMessage],
    options: GooglePromptConversionOptions,
    warnings: inout [AIWarning]
) throws -> (contents: [JSONValue], systemInstruction: JSONValue?) {
    var systemParts: [String] = []
    var contents: [[String: JSONValue]] = []
    var systemMessagesAllowed = true
    var state = GooglePromptConversionState()

    for message in messages {
        switch message.role {
        case .system:
            guard systemMessagesAllowed else {
                throw AIError.invalidArgument(
                    argument: "prompt",
                    message: "'system messages are only supported at the beginning of the conversation' functionality not supported."
                )
            }
            systemParts.append(message.combinedText)
        case .tool:
            systemMessagesAllowed = false
            var parts: [JSONValue] = []
            for part in message.content {
                guard case let .toolResult(result) = part else { continue }
                if let serverTool = googleServerToolMetadata(from: result.providerMetadata, providerOptionsNames: options.providerOptionsNames),
                   let last = contents.indices.last,
                   contents[last]["role"]?.stringValue == "model" {
                    var lastParts = contents[last]["parts"]?.arrayValue ?? []
                    lastParts.append(googleServerToolResponsePart(result, serverTool: serverTool, providerOptionsNames: options.providerOptionsNames))
                    contents[last]["parts"] = .array(lastParts)
                    continue
                }
                parts.append(contentsOf: googleGenerateContentToolResultParts(result, options: options))
            }
            contents.append(["role": "user", "parts": .array(parts)])
        default:
            systemMessagesAllowed = false
            let converted = try googleGenerateContentMessageObject(message, options: options, state: &state)
            contents.append(converted)
        }
    }

    if googleIsGemmaModel(options.modelID),
       !systemParts.isEmpty,
       !contents.isEmpty,
       contents[0]["role"]?.stringValue == "user" {
        var parts = contents[0]["parts"]?.arrayValue ?? []
        parts.insert(.object(["text": .string(systemParts.joined(separator: "\n\n") + "\n\n")]), at: 0)
        contents[0]["parts"] = .array(parts)
    }

    if let warning = googleMissingThoughtSignatureWarning(toolNames: state.missingSignatureToolNames) {
        warnings.append(warning)
    }

    let systemInstruction: JSONValue? = systemParts.isEmpty || googleIsGemmaModel(options.modelID)
        ? nil
        : .object(["parts": .array(systemParts.map { .object(["text": .string($0)]) })])
    return (contents.map(JSONValue.object), systemInstruction)
}

/// Converts one non-system message. Kept for source compatibility; whole
/// requests use the prompt converter so the replay warning is emitted once.
public func googleGenerateContentMessageJSON(
    _ message: AIMessage,
    modelID: String,
    includeFunctionCallIDs: Bool = true,
    supportsGoogleCloudStorageToolResults: Bool = false,
    providerOptionsNames: [String] = ["google"],
    warnings: inout [AIWarning]
) throws -> JSONValue {
    let options = GooglePromptConversionOptions(
        modelID: modelID,
        providerOptionsNames: providerOptionsNames,
        includeFunctionCallIDs: includeFunctionCallIDs,
        supportsGoogleCloudStorageToolResults: supportsGoogleCloudStorageToolResults
    )
    var state = GooglePromptConversionState()
    let converted: [String: JSONValue]
    if message.role == .tool {
        var parts: [JSONValue] = []
        for case let .toolResult(result) in message.content {
            if let serverTool = googleServerToolMetadata(from: result.providerMetadata, providerOptionsNames: providerOptionsNames) {
                parts.append(googleServerToolResponsePart(result, serverTool: serverTool, providerOptionsNames: providerOptionsNames))
            } else {
                parts.append(contentsOf: googleGenerateContentToolResultParts(result, options: options))
            }
        }
        converted = ["role": "user", "parts": .array(parts)]
    } else {
        converted = try googleGenerateContentMessageObject(message, options: options, state: &state)
    }
    if let warning = googleMissingThoughtSignatureWarning(toolNames: state.missingSignatureToolNames) {
        warnings.append(warning)
    }
    return .object(converted)
}

private func googleGenerateContentMessageObject(
    _ message: AIMessage,
    options: GooglePromptConversionOptions,
    state: inout GooglePromptConversionState
) throws -> [String: JSONValue] {
    if message.role == .assistant {
        return ["role": "model", "parts": .array(try googleAssistantParts(message.content, options: options, state: &state))]
    }
    var parts: [JSONValue] = []
    for part in message.content {
        parts.append(contentsOf: try googleUserParts(part, options: options))
    }
    return ["role": "user", "parts": .array(parts)]
}

private func googleUserParts(_ part: AIContentPart, options: GooglePromptConversionOptions) throws -> [JSONValue] {
    func withVideoMetadata(_ value: [String: JSONValue]) throws -> [JSONValue] {
        var value = value
        if let metadata = try googleVideoMetadata(part.providerMetadata, providerOptionsNames: options.providerOptionsNames) {
            value["videoMetadata"] = metadata
        }
        return [.object(value)]
    }
    switch part {
    case let .text(text, _), let .reasoning(text, _):
        return [.object(["text": .string(text)])]
    case let .imageURL(url, _):
        return try withVideoMetadata(["fileData": .object(["fileUri": .string(url)])])
    case let .data(mimeType, data, _), let .file(mimeType, data, _, _):
        return try withVideoMetadata([
            "inlineData": .object([
                "mimeType": .string(try resolveFullMediaType(mediaType: mimeType, data: data)),
                "data": .string(data.base64EncodedString())
            ])
        ])
    case let .providerReference(mimeType, reference, _, _):
        guard !options.isVertexLike else {
            throw AIError.invalidArgument(argument: "prompt", message: "'file parts with provider references' functionality not supported.")
        }
        return try withVideoMetadata([
            "fileData": .object([
                "mimeType": .string(mimeType),
                "fileUri": .string(try resolveProviderReference(reference, provider: "google"))
            ])
        ])
    case let .toolCall(call):
        var state = GooglePromptConversionState()
        return [googleFunctionCallPart(call, options: options, skipMissingSignatureMitigation: true, state: &state)]
    case let .toolResult(result):
        return googleGenerateContentToolResultParts(result, options: options)
    case .reasoningFile, .custom, .toolApprovalRequest, .toolApprovalResponse:
        return []
    }
}

private func googleAssistantParts(
    _ content: [AIContentPart],
    options: GooglePromptConversionOptions,
    state: inout GooglePromptConversionState
) throws -> [JSONValue] {
    var parts: [JSONValue] = []
    var modelResponseHasSignedFunctionCall = false
    for part in content {
        let providerOptions = googlePartProviderOptions(part.providerMetadata, providerOptionsNames: options.providerOptionsNames)
        let thoughtSignature = providerOptions?["thoughtSignature"].flatMap(googleSignatureString).map(JSONValue.string)
        let thoughtFlag = providerOptions?["thought"]?.boolValue == true
        switch part {
        case let .text(text, _):
            guard !text.isEmpty || thoughtSignature != nil else { continue }
            var value: [String: JSONValue] = ["text": .string(text)]
            value["thoughtSignature"] = thoughtSignature
            parts.append(.object(value))
        case let .reasoning(text, _):
            guard !text.isEmpty || thoughtSignature != nil else { continue }
            var value: [String: JSONValue] = ["text": .string(text), "thought": true]
            value["thoughtSignature"] = thoughtSignature
            parts.append(.object(value))
        case let .reasoningFile(file):
            let fileOptions = googlePartProviderOptions(file.providerMetadata, providerOptionsNames: options.providerOptionsNames)
            guard let data = file.data else {
                throw AIError.invalidArgument(argument: "prompt", message: "'File data URLs in assistant messages are not supported' functionality not supported.")
            }
            var value: [String: JSONValue] = [
                "inlineData": .object(["mimeType": .string(file.mediaType), "data": .string(data.base64EncodedString())]),
                "thought": true
            ]
            value["thoughtSignature"] = fileOptions?["thoughtSignature"].flatMap(googleSignatureString).map(JSONValue.string)
            parts.append(.object(value))
        case .imageURL:
            throw AIError.invalidArgument(argument: "prompt", message: "'File data URLs in assistant messages are not supported' functionality not supported.")
        case let .data(mimeType, data, _), let .file(mimeType, data, _, _):
            var value: [String: JSONValue] = [
                "inlineData": .object(["mimeType": .string(mimeType), "data": .string(data.base64EncodedString())])
            ]
            if thoughtFlag { value["thought"] = true }
            value["thoughtSignature"] = thoughtSignature
            parts.append(.object(value))
        case let .providerReference(mimeType, reference, _, _):
            guard !options.isVertexLike else {
                throw AIError.invalidArgument(argument: "prompt", message: "'file parts with provider references' functionality not supported.")
            }
            var value: [String: JSONValue] = [
                "fileData": .object([
                    "mimeType": .string(mimeType),
                    "fileUri": .string(try resolveProviderReference(reference, provider: "google"))
                ])
            ]
            if thoughtFlag { value["thought"] = true }
            value["thoughtSignature"] = thoughtSignature
            parts.append(.object(value))
        case let .toolCall(call):
            let isServerToolCall = googleServerToolMetadata(from: call.providerMetadata, providerOptionsNames: options.providerOptionsNames) != nil
            let skip = !isServerToolCall && thoughtSignature == nil && modelResponseHasSignedFunctionCall
            parts.append(googleFunctionCallPart(call, options: options, skipMissingSignatureMitigation: skip, state: &state))
            if !isServerToolCall, thoughtSignature != nil, !(call.providerExecuted && call.name == "code_execution") {
                modelResponseHasSignedFunctionCall = true
            }
        case let .toolResult(result):
            let output = googleNormalizedToolOutput(result)
            if result.toolName == "code_execution", output.type == "json", let value = output.value {
                parts.append(.object(["codeExecutionResult": googleCodeExecutionResultJSON(value)]))
            } else if let serverTool = googleServerToolMetadata(from: result.providerMetadata, providerOptionsNames: options.providerOptionsNames) {
                parts.append(googleServerToolResponsePart(result, serverTool: serverTool, providerOptionsNames: options.providerOptionsNames))
            }
        case .custom, .toolApprovalRequest, .toolApprovalResponse:
            continue
        }
    }
    return parts
}

private func googleFunctionCallPart(
    _ call: AIToolCall,
    options: GooglePromptConversionOptions,
    skipMissingSignatureMitigation: Bool,
    state: inout GooglePromptConversionState
) -> JSONValue {
    if call.providerExecuted, call.name == "code_execution" {
        return .object(["executableCode": googleToolArguments(call.arguments)])
    }
    let providerOptions = googlePartProviderOptions(call.providerMetadata, providerOptionsNames: options.providerOptionsNames)
    let thoughtSignature = providerOptions?["thoughtSignature"].flatMap(googleSignatureString)
    var effectiveSignature = thoughtSignature
    if effectiveSignature == nil, options.isGemini3Model, !skipMissingSignatureMitigation {
        state.missingSignatureToolNames.append(call.name)
        effectiveSignature = googleSkipThoughtSignatureValidator
    }

    var output: [String: JSONValue]
    if let serverTool = googleServerToolMetadata(from: call.providerMetadata, providerOptionsNames: options.providerOptionsNames) {
        output = [
            "toolCall": .object([
                "toolType": .string(serverTool.type),
                "args": googleToolArguments(call.arguments),
                "id": .string(serverTool.id)
            ])
        ]
    } else {
        var functionCall: [String: JSONValue] = [
            "name": .string(call.name),
            "args": googleToolArguments(call.arguments)
        ]
        if options.includeFunctionCallIDs {
            functionCall["id"] = .string(call.id)
        }
        output = ["functionCall": .object(functionCall)]
    }
    output["thoughtSignature"] = effectiveSignature.map(JSONValue.string)
    return .object(output)
}

private func googleServerToolResponsePart(
    _ result: AIToolResult,
    serverTool: (id: String, type: String),
    providerOptionsNames: [String]
) -> JSONValue {
    let output = googleNormalizedToolOutput(result)
    var part: [String: JSONValue] = [
        "toolResponse": .object([
            "toolType": .string(serverTool.type),
            "response": output.type == "json" ? (output.value ?? .object([:])) : .object([:]),
            "id": .string(serverTool.id)
        ])
    ]
    let providerOptions = googlePartProviderOptions(result.providerMetadata, providerOptionsNames: providerOptionsNames)
    part["thoughtSignature"] = providerOptions?["thoughtSignature"].flatMap(googleSignatureString).map(JSONValue.string)
    return .object(part)
}

// MARK: - Tool results

/// The upstream `LanguageModelV4ToolResultOutput` view of a Swift tool result.
/// Typed outputs (`{type, value}`) are used as-is; raw tool return values map
/// to `text`/`json` (or `error-text`/`error-json` for failed executions) like
/// the AI SDK's `createToolModelOutput`.
struct GoogleToolOutput {
    var type: String
    var value: JSONValue?
    var reason: JSONValue?
}

private let googleTypedToolOutputTypes: Set<String> = ["text", "json", "error-text", "error-json", "execution-denied", "content"]

func googleNormalizedToolOutput(_ result: AIToolResult) -> GoogleToolOutput {
    let output = result.modelOutput ?? result.result
    if let object = output.objectValue,
       let type = object["type"]?.stringValue,
       googleTypedToolOutputTypes.contains(type) {
        return GoogleToolOutput(type: type, value: object["value"], reason: object["reason"])
    }
    if output.stringValue != nil {
        return GoogleToolOutput(type: result.isError ? "error-text" : "text", value: output)
    }
    return GoogleToolOutput(type: result.isError ? "error-json" : "json", value: output)
}

func googleGenerateContentToolResultParts(
    _ result: AIToolResult,
    options: GooglePromptConversionOptions
) -> [JSONValue] {
    let output = googleNormalizedToolOutput(result)
    if output.type == "content" {
        let content = output.value?.arrayValue ?? []
        return options.supportsFunctionResponseParts
            ? [googleFunctionResponseWithParts(result, content: content, options: options)]
            : googleLegacyFunctionResponseParts(result, content: content, options: options)
    }

    var response: [String: JSONValue] = ["name": .string(result.toolName)]
    switch output.type {
    case "execution-denied":
        response["error"] = output.reason.flatMap { $0 == .null ? nil : $0 } ?? .string("Tool call execution denied.")
    case "error-text", "error-json":
        response["error"] = googleSerializeFunctionResponseContent(output.value ?? .null)
    default:
        response["content"] = googleSerializeFunctionResponseContent(output.value ?? .null)
    }
    return [.object(["functionResponse": .object(googleFunctionResponseEnvelope(result, options: options, response: response))])]
}

private func googleFunctionResponseEnvelope(
    _ result: AIToolResult,
    options: GooglePromptConversionOptions,
    response: [String: JSONValue]
) -> [String: JSONValue] {
    var functionResponse: [String: JSONValue] = [
        "name": .string(result.toolName),
        "response": .object(response)
    ]
    if options.includeFunctionCallIDs {
        functionResponse["id"] = .string(result.toolCallID)
    }
    return functionResponse
}

/// Gemini 3+ multimodal function responses (`appendToolResultParts`).
private func googleFunctionResponseWithParts(
    _ result: AIToolResult,
    content: [JSONValue],
    options: GooglePromptConversionOptions
) -> JSONValue {
    var functionResponseParts: [JSONValue] = []
    var responseTextParts: [String] = []
    for item in content {
        switch item["type"]?.stringValue {
        case "text":
            responseTextParts.append(item["text"]?.stringValue ?? "")
        default:
            guard let file = GoogleToolResultFile(item) else {
                responseTextParts.append(googleJSONString(item) ?? "")
                continue
            }
            switch file.source {
            case let .base64(data):
                functionResponseParts.append(.object(["inlineData": .object([
                    "mimeType": .string(file.resolvedMediaType(base64: data)),
                    "data": .string(data)
                ])]))
            case let .url(url):
                if let dataURL = googleParseBase64DataURL(url) {
                    functionResponseParts.append(.object(["inlineData": .object([
                        "mimeType": .string(dataURL.mediaType),
                        "data": .string(dataURL.data)
                    ])]))
                } else if options.supportsGoogleCloudStorageToolResults,
                          isFullMediaType(file.mediaType),
                          url.hasPrefix("gs://"),
                          googleVertexSupportsCloudStorageToolResultMediaType(file.mediaType) {
                    functionResponseParts.append(.object(["fileData": .object([
                        "mimeType": .string(file.mediaType),
                        "fileUri": .string(url)
                    ])]))
                } else {
                    responseTextParts.append(googleJSONString(item) ?? "")
                }
            case .other:
                responseTextParts.append(googleJSONString(item) ?? "")
            }
        }
    }

    var functionResponse = googleFunctionResponseEnvelope(result, options: options, response: [
        "name": .string(result.toolName),
        "content": .string(responseTextParts.isEmpty ? "Tool executed successfully." : responseTextParts.joined(separator: "\n"))
    ])
    if !functionResponseParts.isEmpty {
        functionResponse["parts"] = .array(functionResponseParts)
    }
    return .object(["functionResponse": .object(functionResponse)])
}

/// Pre-Gemini 3 conversion (`appendLegacyToolResultParts`): each text part is
/// its own function response and inline files become top-level parts.
private func googleLegacyFunctionResponseParts(
    _ result: AIToolResult,
    content: [JSONValue],
    options: GooglePromptConversionOptions
) -> [JSONValue] {
    var parts: [JSONValue] = []
    for item in content {
        if item["type"]?.stringValue == "text" {
            let response: [String: JSONValue] = ["name": .string(result.toolName), "content": .string(item["text"]?.stringValue ?? "")]
            parts.append(.object(["functionResponse": .object(googleFunctionResponseEnvelope(result, options: options, response: response))]))
            continue
        }
        if let file = GoogleToolResultFile(item), case let .base64(data) = file.source {
            let topLevel = file.mediaType.split(separator: "/", maxSplits: 1).first.map(String.init) ?? file.mediaType
            parts.append(.object(["inlineData": .object([
                "mimeType": .string(file.resolvedMediaType(base64: data)),
                "data": .string(data)
            ])]))
            parts.append(.object(["text": .string("Tool executed successfully and returned this \(topLevel == "image" ? "image" : "file") as a response")]))
            continue
        }
        parts.append(.object(["text": .string(googleJSONString(item) ?? "")]))
    }
    return parts
}

/// A file entry of a `content` tool output. Accepts the V4 shape
/// (`{type: "file", mediaType, data: {type: "data" | "url", ...}}`) and the
/// legacy flat shapes (`image-data`, `file-data`, `image-url`, `file-url`).
private struct GoogleToolResultFile {
    enum Source {
        case base64(String)
        case url(String)
        case other
    }

    var mediaType: String
    var source: Source

    init?(_ item: JSONValue) {
        guard let type = item["type"]?.stringValue else { return nil }
        switch type {
        case "file", "media":
            guard let mediaType = item["mediaType"]?.stringValue else { return nil }
            self.mediaType = mediaType
            if let data = item["data"]?.stringValue {
                source = .base64(data)
            } else if let data = item["data"]?.objectValue {
                switch data["type"]?.stringValue {
                case "data", nil:
                    source = data["data"]?.stringValue.map(Source.base64) ?? .other
                case "url":
                    source = data["url"]?.stringValue.map(Source.url) ?? .other
                default:
                    source = .other
                }
            } else {
                return nil
            }
        case "image-data", "file-data":
            guard let data = item["data"]?.stringValue else { return nil }
            mediaType = item["mediaType"]?.stringValue ?? (type == "image-data" ? "image" : "application/octet-stream")
            source = .base64(data)
        case "image-url", "file-url":
            guard let url = item["url"]?.stringValue else { return nil }
            mediaType = item["mediaType"]?.stringValue ?? (type == "image-url" ? "image" : "application/octet-stream")
            source = .url(url)
        default:
            return nil
        }
    }

    func resolvedMediaType(base64 data: String) -> String {
        if let dataURL = googleParseBase64DataURL(data) { return dataURL.mediaType }
        guard !isFullMediaType(mediaType), let bytes = Data(base64Encoded: data) else { return mediaType }
        return (try? resolveFullMediaType(mediaType: mediaType, data: bytes)) ?? mediaType
    }
}

/// A Cloud Storage `fileData` part for a supported tool-result file URL.
func googleCloudStorageFileDataFromToolContent(_ contentPart: JSONValue) -> JSONValue? {
    guard let file = GoogleToolResultFile(contentPart),
          case let .url(url) = file.source,
          url.hasPrefix("gs://"),
          isFullMediaType(file.mediaType),
          googleVertexSupportsCloudStorageToolResultMediaType(file.mediaType) else {
        return nil
    }
    return .object(["fileData": .object(["mimeType": .string(file.mediaType), "fileUri": .string(url)])])
}

public func googleVertexSupportsCloudStorageToolResultMediaType(_ mediaType: String) -> Bool {
    ["image/png", "image/jpeg", "image/webp", "application/pdf", "text/plain"].contains(mediaType.lowercased())
}

func googleParseBase64DataURL(_ value: String) -> (mediaType: String, data: String)? {
    guard value.hasPrefix("data:"),
          let separator = value.range(of: ";base64,") else {
        return nil
    }
    let mediaType = String(value[value.index(value.startIndex, offsetBy: 5)..<separator.lowerBound])
    let data = String(value[separator.upperBound...])
    guard !mediaType.isEmpty, !mediaType.contains(","), !mediaType.contains(";"), !data.isEmpty else { return nil }
    return (mediaType, data)
}

func googleSerializeFunctionResponseContent(_ value: JSONValue) -> JSONValue {
    googleContainsJSONSchemaReference(value)
        ? .string(googleJSONString(value) ?? String(describing: value))
        : value
}

func googleContainsJSONSchemaReference(_ value: JSONValue) -> Bool {
    switch value {
    case let .array(values):
        return values.contains(where: googleContainsJSONSchemaReference)
    case let .object(values):
        return values.contains { key, nestedValue in
            key == "$ref" || googleContainsJSONSchemaReference(nestedValue)
        }
    case .string, .number, .bool, .null:
        return false
    }
}

public func googleMessageTargetIsGemini3(_ modelID: String) -> Bool {
    googleModelCapabilities(for: modelID).usesGemini3Features
}

func googleMissingThoughtSignatureWarning(toolNames: [String]) -> AIWarning? {
    guard !toolNames.isEmpty else { return nil }
    var unique: [String] = []
    for name in toolNames where !unique.contains(name) { unique.append(name) }
    return AIWarning(
        type: "other",
        message: "Replayed \(toolNames.count) `functionCall` part(s) for a Gemini 3 model without a `thoughtSignature` (tools: \(unique.map { "`\($0)`" }.joined(separator: ", "))). Injected the documented `skip_thought_signature_validator` sentinel to keep the request from failing with HTTP 400. The likely cause is application code that drops `providerOptions.google.thoughtSignature` when persisting or serializing assistant tool-call messages. See https://ai.google.dev/gemini-api/docs/thought-signatures."
    )
}

/// Upstream `readProviderOpts`: the first configured namespace wins as a whole,
/// then Vertex falls back to `google` and Google falls back to the Vertex keys.
func googlePartProviderOptions(_ metadata: [String: JSONValue], providerOptionsNames: [String]) -> JSONValue? {
    for name in providerOptionsNames {
        if let value = metadata[name], value != .null { return value }
    }
    let fallbacks = providerOptionsNames.contains("google")
        ? ["googleVertex", "vertex", "google.generative-ai"]
        : ["google", "google.generative-ai"]
    for name in fallbacks {
        if let value = metadata[name], value != .null { return value }
    }
    return nil
}

private func googleSignatureString(_ value: JSONValue) -> String? {
    switch value {
    case let .string(string): string
    case .null: nil
    case let .number(number): number.rounded() == number ? String(Int(number)) : String(number)
    case let .bool(bool): String(bool)
    case .array, .object: googleJSONString(value)
    }
}

func googleServerToolMetadata(from providerMetadata: [String: JSONValue], providerOptionsNames: [String]) -> (id: String, type: String)? {
    guard let options = googlePartProviderOptions(providerMetadata, providerOptionsNames: providerOptionsNames),
          let id = options["serverToolCallId"].flatMap(googleSignatureString),
          let type = options["serverToolType"].flatMap(googleSignatureString),
          !id.isEmpty, !type.isEmpty else {
        return nil
    }
    return (id, type)
}

func googleToolArguments(_ arguments: String) -> JSONValue {
    (try? decodeJSONBody(Data(arguments.utf8))) ?? .string(arguments)
}

/// `JSON.stringify` equivalent: compact and without escaped slashes. Keys are
/// sorted because `JSONValue` objects do not keep insertion order.
func googleJSONString(_ value: JSONValue) -> String? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(value) else { return nil }
    return String(data: data, encoding: .utf8)
}
