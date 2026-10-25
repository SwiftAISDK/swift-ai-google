import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public func googleInteractionsHeaders(_ requestHeaders: [String: String]) -> [String: String] {
    ["Api-Revision": "2026-05-20"].mergingHeaders(requestHeaders)
}

public struct GoogleInteractionsPreparedCall {
    public var body: [String: JSONValue]
    public var warnings: [AIWarning]
    public init(body: [String: JSONValue], warnings: [AIWarning]) {
        self.body = body
        self.warnings = warnings
    }

}

public func googleInteractionsPreparedCall(for request: LanguageModelRequest, modelID: String, agent: String?, stream: Bool) throws -> GoogleInteractionsPreparedCall {
    var options = googleGenerateContentOptions(from: request.extraBody)
    if let providerOptions = request.providerOptions["google"]?.objectValue {
        options.merge(providerOptions) { _, providerValue in providerValue }
    }
    let callResponseFormat = googleInteractionsResolvedCallResponseFormat(request: request, options: &options)
    let providerResponseFormat = options.removeValue(forKey: "responseFormat")
    let mediaResolution = try googleInteractionsMediaResolution(options["mediaResolution"])
    let optionSystemInstruction = try googleInteractionsSystemInstruction(options["systemInstruction"])
    let previousInteractionID = options["previousInteractionId"]?.stringValue
    let shouldPreserveFullHistory = options["store"]?.boolValue == false
    var conversionWarnings: [AIWarning] = []
    if previousInteractionID != nil, shouldPreserveFullHistory {
        conversionWarnings.append(AIWarning(
            type: "other",
            message: "google.interactions: providerOptions.google.previousInteractionId was set together with store: false. These are incoherent (the prior interaction cannot be referenced when nothing was stored on the server); the full history will be sent and previous_interaction_id will still be emitted."
        ))
    }
    let messages: [AIMessage]
    if let previousInteractionID, !shouldPreserveFullHistory {
        messages = googleInteractionsCompactMessages(
            request.messages,
            previousInteractionID: previousInteractionID
        )
    } else {
        messages = request.messages
    }
    let systemTexts = messages
        .filter { $0.role == .system }
        .map(\.combinedText)
    let convertedSystemInstruction = systemTexts.isEmpty ? nil : systemTexts.joined(separator: "\n\n")
    let input = try messages
        .filter { $0.role != .system }
        .flatMap {
            try googleInteractionsSteps(
                $0,
                mediaResolution: mediaResolution,
                warnings: &conversionWarnings
            )
        }
    let systemInstruction: String?
    if let convertedSystemInstruction, optionSystemInstruction != nil {
        conversionWarnings.append(AIWarning(
            type: "other",
            message: "google.interactions: both AI SDK system message and providerOptions.google.systemInstruction were set; using the AI SDK system message."
        ))
        systemInstruction = convertedSystemInstruction
    } else {
        systemInstruction = convertedSystemInstruction ?? optionSystemInstruction
    }

    var body: [String: JSONValue] = [
        agent == nil ? "model" : "agent": .string(agent ?? modelID),
        "input": .array(input)
    ]
    if stream, agent == nil {
        body["stream"] = true
    }
    if let systemInstruction {
        body["system_instruction"] = .string(systemInstruction)
    }
    if agent == nil {
        var generationConfig: [String: JSONValue] = [:]
        if let temperature = request.temperature { generationConfig["temperature"] = .number(temperature) }
        if let topP = request.topP { generationConfig["top_p"] = .number(topP) }
        if let topK = request.topK { generationConfig["top_k"] = .number(Double(topK)) }
        if let seed = request.seed { generationConfig["seed"] = .number(Double(seed)) }
        if let maxOutputTokens = request.maxOutputTokens { generationConfig["max_output_tokens"] = .number(Double(maxOutputTokens)) }
        if !request.stopSequences.isEmpty { generationConfig["stop_sequences"] = .array(request.stopSequences) }
        if let thinkingLevel = options["thinkingLevel"] { generationConfig["thinking_level"] = thinkingLevel }
        if let thinkingSummaries = options["thinkingSummaries"] { generationConfig["thinking_summaries"] = thinkingSummaries }
        if !generationConfig.isEmpty {
            body["generation_config"] = .object(generationConfig)
        }
    }
    body.merge(googleInteractionsOptions(from: options, callResponseFormat: callResponseFormat, providerResponseFormat: providerResponseFormat, isAgent: agent != nil)) { _, new in new }
    return GoogleInteractionsPreparedCall(
        body: body,
        warnings: conversionWarnings + googleInteractionsWarnings(
            request: request,
            options: options,
            callResponseFormat: callResponseFormat,
            isAgent: agent != nil
        )
    )
}

private func googleInteractionsMediaResolution(_ value: JSONValue?) throws -> String? {
    guard let value, value != .null else { return nil }
    guard let resolution = value.stringValue,
          ["low", "medium", "high", "ultra_high"].contains(resolution) else {
        throw AIError.invalidArgument(
            argument: "providerOptions.google.mediaResolution",
            message: "Google Interactions mediaResolution must be low, medium, high, or ultra_high."
        )
    }
    return resolution
}

private func googleInteractionsSystemInstruction(_ value: JSONValue?) throws -> String? {
    guard let value, value != .null else { return nil }
    guard let instruction = value.stringValue else {
        throw AIError.invalidArgument(
            argument: "providerOptions.google.systemInstruction",
            message: "Google Interactions systemInstruction must be a string."
        )
    }
    return instruction
}

private func googleInteractionsCompactMessages(
    _ messages: [AIMessage],
    previousInteractionID: String
) -> [AIMessage] {
    var output: [AIMessage] = []
    var droppedToolCallIDs = Set<String>()

    for message in messages {
        if message.role == .assistant {
            let matchesLinkedInteraction = message.content.contains { part in
                part.providerMetadata["google"]?["interactionId"]?.stringValue == previousInteractionID
            }
            if matchesLinkedInteraction {
                for part in message.content {
                    if case let .toolCall(call) = part {
                        droppedToolCallIDs.insert(call.id)
                    }
                }
                continue
            }
        } else if message.role == .tool {
            let remaining = message.content.filter { part in
                guard case let .toolResult(result) = part else { return true }
                return !droppedToolCallIDs.contains(result.toolCallID)
            }
            guard !remaining.isEmpty else { continue }
            var compacted = message
            compacted.content = remaining
            output.append(compacted)
            continue
        }
        output.append(message)
    }

    return output
}

func googleInteractionsSteps(
    _ message: AIMessage,
    mediaResolution: String? = nil,
    warnings: inout [AIWarning]
) throws -> [JSONValue] {
    switch message.role {
    case .user:
        let content = try googleInteractionsContent(
            message.content,
            mediaResolution: mediaResolution,
            warnings: &warnings
        )
        return content.isEmpty ? [] : [.object(["type": .string("user_input"), "content": .array(content)])]
    case .assistant:
        var steps: [JSONValue] = []
        var pendingModelOutput: [JSONValue] = []
        func flushModelOutput() {
            guard !pendingModelOutput.isEmpty else { return }
            steps.append(.object(["type": .string("model_output"), "content": .array(pendingModelOutput)]))
            pendingModelOutput.removeAll(keepingCapacity: true)
        }

        for part in message.content {
            switch part {
            case let .reasoning(text, providerMetadata):
                flushModelOutput()
                var step: [String: JSONValue] = ["type": .string("thought")]
                if let signature = googleInteractionsSignature(from: providerMetadata) {
                    step["signature"] = .string(signature)
                }
                if !text.isEmpty {
                    step["summary"] = .array([.object([
                        "type": .string("text"),
                        "text": .string(text)
                    ])])
                }
                steps.append(.object(step))
            case let .toolCall(call):
                flushModelOutput()
                var step: [String: JSONValue] = [
                    "type": .string("function_call"),
                    "id": .string(call.id),
                    "name": .string(call.name),
                    "arguments": googleInteractionsToolArguments(call.arguments)
                ]
                if let signature = googleInteractionsSignature(from: call.providerMetadata) {
                    step["signature"] = .string(signature)
                }
                steps.append(.object(step))
            case let .custom(value, providerMetadata):
                flushModelOutput()
                let kind = value["kind"]?.stringValue
                let google = providerMetadata["google"]?.objectValue ?? [:]
                let signature = google["signature"]?.stringValue
                if kind == "google.processing_call",
                   let processingID = google["processingId"]?.stringValue {
                    var step: [String: JSONValue] = [
                        "type": .string("processing_call"),
                        "id": .string(processingID)
                    ]
                    if let signature { step["signature"] = .string(signature) }
                    steps.append(.object(step))
                } else if kind == "google.processing_result",
                          let processingCallID = google["processingCallId"]?.stringValue {
                    var step: [String: JSONValue] = [
                        "type": .string("processing_result"),
                        "call_id": .string(processingCallID)
                    ]
                    if let signature { step["signature"] = .string(signature) }
                    steps.append(.object(step))
                } else {
                    warnings.append(AIWarning(
                        type: "other",
                        message: "google.interactions: unsupported or invalid custom assistant content part \"\(kind ?? "unknown")\"; part dropped."
                    ))
                }
            default:
                pendingModelOutput.append(contentsOf: try googleInteractionsContent(
                    [part],
                    mediaResolution: mediaResolution,
                    warnings: &warnings
                ))
            }
        }
        flushModelOutput()
        return steps
    case .tool:
        let content = try message.content.compactMap { part -> JSONValue? in
            guard case let .toolResult(result) = part else {
                warnings.append(AIWarning(
                    type: "other",
                    message: "google.interactions: unsupported tool message content part; part dropped."
                ))
                return nil
            }
            return try googleInteractionsFunctionResult(result, warnings: &warnings)
        }
        return content.isEmpty ? [] : [.object([
            "type": .string("user_input"),
            "content": .array(content)
        ])]
    case .system:
        return []
    }
}

private func googleInteractionsSignature(from providerMetadata: [String: JSONValue]) -> String? {
    providerMetadata["google"]?["signature"]?.stringValue
}

private func googleInteractionsToolArguments(_ arguments: String) -> JSONValue {
    guard let parsed = try? decodeJSONBody(Data(arguments.utf8)) else {
        return .object(["value": .string(arguments)])
    }
    if parsed.objectValue != nil {
        return parsed
    }
    return .object(["value": parsed])
}

private func googleInteractionsFunctionResult(
    _ result: AIToolResult,
    warnings: inout [AIWarning]
) throws -> JSONValue {
    let converted = try googleInteractionsFunctionResultValue(result, warnings: &warnings)
    var block: [String: JSONValue] = [
        "type": .string("function_result"),
        "call_id": .string(result.toolCallID),
        "name": .string(result.toolName),
        "result": converted.value
    ]
    if result.isError || converted.isError {
        block["is_error"] = .bool(true)
    }
    if let signature = googleInteractionsSignature(from: result.providerMetadata) {
        block["signature"] = .string(signature)
    }
    return .object(block)
}

private func googleInteractionsFunctionResultValue(
    _ result: AIToolResult,
    warnings: inout [AIWarning]
) throws -> (value: JSONValue, isError: Bool) {
    let output = result.modelOutput ?? result.result
    if let text = output.stringValue {
        return (.string(text), false)
    }
    guard let object = output.objectValue,
          let type = object["type"]?.stringValue else {
        return (.string(googleJSONString(output) ?? ""), false)
    }
    switch type {
    case "text", "error-text":
        return (.string(object["value"]?.stringValue ?? ""), type == "error-text")
    case "json", "error-json":
        return (.string(googleJSONString(object["value"] ?? .object([:])) ?? ""), type == "error-json")
    case "execution-denied":
        return (.string(object["reason"]?.stringValue ?? "Tool execution denied by user."), true)
    case "content":
        let blocks = try (object["value"]?.arrayValue ?? []).compactMap { item in
            try googleInteractionsFunctionResultContent(item, warnings: &warnings)
        }
        return (.array(blocks), false)
    default:
        return (.string(googleJSONString(object["value"] ?? output) ?? ""), false)
    }
}

private func googleInteractionsFunctionResultContent(
    _ item: JSONValue,
    warnings: inout [AIWarning]
) throws -> JSONValue? {
    switch item["type"]?.stringValue {
    case "text":
        return .object([
            "type": .string("text"),
            "text": .string(item["text"]?.stringValue ?? "")
        ])
    case "file":
        let mediaType = item["mediaType"]?.stringValue ?? item["media_type"]?.stringValue ?? ""
        guard mediaType.split(separator: "/").first == "image" else {
            warnings.append(AIWarning(
                type: "other",
                message: "google.interactions: tool-result file with mediaType \"\(mediaType)\" is not supported (Interactions `function_result.result` accepts only text and image content); part dropped."
            ))
            return nil
        }
        guard let data = item["data"]?.objectValue,
              let dataType = data["type"]?.stringValue else {
            warnings.append(AIWarning(
                type: "other",
                message: "google.interactions: malformed tool-result image file part; part dropped."
            ))
            return nil
        }
        switch dataType {
        case "data":
            guard let value = data["data"]?.stringValue else { return nil }
            return .object([
                "type": .string("image"),
                "data": .string(value),
                "mime_type": .string(mediaType)
            ])
        case "url":
            guard let url = data["url"]?.stringValue else { return nil }
            return .object([
                "type": .string("image"),
                "uri": .string(url),
                "mime_type": .string(mediaType)
            ])
        case "reference":
            let reference = (data["reference"]?.objectValue ?? [:]).compactMapValues(\.stringValue)
            guard !reference.isEmpty else { return nil }
            return .object([
                "type": .string("image"),
                "uri": .string(try resolveProviderReference(reference, provider: "google")),
                "mime_type": .string(mediaType)
            ])
        default:
            warnings.append(AIWarning(
                type: "other",
                message: "google.interactions: tool-result image part with unsupported data type \"\(dataType)\"; part dropped."
            ))
            return nil
        }
    default:
        warnings.append(AIWarning(
            type: "other",
            message: "google.interactions: unsupported tool-result content part; part dropped."
        ))
        return nil
    }
}

func googleInteractionsContent(
    _ content: [AIContentPart],
    mediaResolution: String? = nil,
    warnings: inout [AIWarning]
) throws -> [JSONValue] {
    try content.map { part in
        switch part {
        case let .text(text, _):
            return .object(["type": .string("text"), "text": .string(text)])
        case let .reasoning(text, _):
            return .object(["type": .string("text"), "text": .string(text)])
        case let .imageURL(url, _):
            var output: [String: JSONValue] = [
                "type": .string("image"),
                "uri": .string(url)
            ]
            if let mediaResolution {
                output["resolution"] = .string(mediaResolution)
            }
            return .object(output)
        case let .data(mimeType, data, providerMetadata),
             let .file(mimeType, data, _, providerMetadata):
            let resolvedMimeType = try resolveFullMediaType(mediaType: mimeType, data: data)
            let topLevel = resolvedMimeType.split(separator: "/").first.map(String.init) ?? "document"
            let type = ["image", "audio", "video"].contains(topLevel) ? topLevel : "document"
            var output: [String: JSONValue] = [
                "type": .string(type),
                "mime_type": .string(resolvedMimeType),
                "data": .string(data.base64EncodedString())
            ]
            if let mediaResolution, type == "image" || type == "video" {
                output["resolution"] = .string(mediaResolution)
            }
            if type == "video",
               let processing = googleInteractionsVideoProcessing(from: providerMetadata, warnings: &warnings) {
                output["processing"] = processing
            }
            return .object(output)
        case let .providerReference(mimeType, reference, _, providerMetadata):
            let topLevel = mimeType.split(separator: "/").first.map(String.init) ?? "document"
            let type = ["image", "audio", "video"].contains(topLevel) ? topLevel : "document"
            var output: [String: JSONValue] = [
                "type": .string(type),
                "uri": .string(try resolveProviderReference(reference, provider: "google"))
            ]
            if isFullMediaType(mimeType) {
                output["mime_type"] = .string(mimeType)
            }
            if let mediaResolution, type == "image" || type == "video" {
                output["resolution"] = .string(mediaResolution)
            }
            if type == "video",
               let processing = googleInteractionsVideoProcessing(from: providerMetadata, warnings: &warnings) {
                output["processing"] = processing
            }
            return .object(output)
        case let .toolCall(call):
            return .object([
                "type": .string("function_call"),
                "name": .string(call.name),
                "arguments": googleToolArguments(call.arguments)
            ])
        case let .toolResult(result):
            return .object([
                "type": .string("function_response"),
                "name": .string(result.toolName),
                "response": result.modelOutput ?? result.result
            ])
        case .reasoningFile, .custom, .toolApprovalRequest, .toolApprovalResponse:
            return .object(["type": .string("text"), "text": .string("")])
        }
    }
}

private func googleInteractionsVideoProcessing(
    from providerMetadata: [String: JSONValue],
    warnings: inout [AIWarning]
) -> JSONValue? {
    guard let processing = providerMetadata["google"]?["processing"] else { return nil }
    if let value = processing.stringValue,
       value == "agentic" || value == "static" {
        return .string(value)
    }
    if let object = processing.objectValue,
       object["type"]?.stringValue == "static" {
        var output: [String: JSONValue] = ["type": .string("static")]
        if let value = object["startOffset"]?.doubleValue { output["start_offset"] = .string(googleInteractionsDuration(value)) }
        if let value = object["endOffset"]?.doubleValue { output["end_offset"] = .string(googleInteractionsDuration(value)) }
        if let value = object["fps"]?.doubleValue { output["fps"] = .number(value) }
        return .object(output)
    }
    warnings.append(AIWarning(
        type: "other",
        message: "google.interactions: invalid providerOptions.google.processing on video file part; expected \"agentic\", \"static\", or a static processing configuration. Option dropped."
    ))
    return nil
}

private func googleInteractionsDuration(_ seconds: Double) -> String {
    let value = seconds.rounded(.towardZero) == seconds ? String(Int(seconds)) : String(seconds)
    return "\(value)s"
}

func googleInteractionsOptions(from extraBody: [String: JSONValue], callResponseFormat: JSONValue?, providerResponseFormat: JSONValue?, isAgent: Bool) -> [String: JSONValue] {
    var output: [String: JSONValue] = [:]
    if let previousInteractionId = extraBody["previousInteractionId"] { output["previous_interaction_id"] = previousInteractionId }
    if let serviceTier = extraBody["serviceTier"] { output["service_tier"] = serviceTier }
    if let store = extraBody["store"] { output["store"] = store }
    if let background = extraBody["background"] { output["background"] = background }
    if let responseModalities = extraBody["responseModalities"] { output["response_modalities"] = responseModalities }
    let responseFormat = googleInteractionsResponseFormat(callResponseFormat: callResponseFormat, providerResponseFormat: providerResponseFormat, isAgent: isAgent)
    if !responseFormat.isEmpty {
        output["response_format"] = .array(responseFormat)
    }
    if isAgent, let agentConfig = extraBody["agentConfig"] { output["agent_config"] = googleInteractionsSnakeCaseObject(agentConfig) }
    if isAgent, let environment = extraBody["environment"] { output["environment"] = googleInteractionsSnakeCaseObject(environment) }
    return output
}

func googleInteractionsResolvedCallResponseFormat(request: LanguageModelRequest, options: inout [String: JSONValue]) -> JSONValue? {
    if let responseFormat = request.responseFormat {
        if googleInteractionsIsCallResponseFormat(options["responseFormat"]) {
            options.removeValue(forKey: "responseFormat")
        }
        return googleInteractionsResponseFormatJSON(responseFormat)
    }
    guard googleInteractionsIsCallResponseFormat(options["responseFormat"]) else {
        return nil
    }
    return options.removeValue(forKey: "responseFormat")
}

func googleInteractionsIsCallResponseFormat(_ value: JSONValue?) -> Bool {
    guard let type = value?.objectValue?["type"]?.stringValue else { return false }
    return type == "json" || type == "text"
}

func googleInteractionsResponseFormatJSON(_ responseFormat: AIResponseFormat) -> JSONValue? {
    switch responseFormat {
    case .text:
        return nil
    case let .json(schema, name, description):
        return .object([
            "type": .string("json"),
            "schema": schema,
            "name": name.map(JSONValue.string),
            "description": description.map(JSONValue.string)
        ])
    }
}

func googleInteractionsResponseFormat(callResponseFormat: JSONValue?, providerResponseFormat: JSONValue?, isAgent: Bool) -> [JSONValue] {
    var entries: [JSONValue] = []
    if !isAgent, callResponseFormat?["type"]?.stringValue == "json" {
        var entry: [String: JSONValue] = [
            "type": .string("text"),
            "mime_type": .string("application/json")
        ]
        if let schema = callResponseFormat?["schema"] {
            entry["schema"] = schema
        }
        entries.append(.object(entry))
    }
    if let providerResponseFormat {
        if let providerEntries = providerResponseFormat.arrayValue {
            entries.append(contentsOf: providerEntries.map(googleInteractionsSnakeCaseObject))
        } else {
            entries.append(googleInteractionsSnakeCaseObject(providerResponseFormat))
        }
    }
    return entries
}

func googleInteractionsWarnings(request: LanguageModelRequest, options: [String: JSONValue], callResponseFormat: JSONValue?, isAgent: Bool) -> [AIWarning] {
    var warnings: [AIWarning] = []
    if !isAgent {
        if request.frequencyPenalty != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "frequencyPenalty"))
        }
        if request.presencePenalty != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "presencePenalty"))
        }
        return warnings
    }

    if callResponseFormat?["type"]?.stringValue == "json" {
        warnings.append(AIWarning(
            type: "other",
            message: "google.interactions: structured output (responseFormat) is not supported when an agent is set; responseFormat will be ignored."
        ))
    }

    var droppedFields: [String] = []
    if request.temperature != nil { droppedFields.append("temperature") }
    if request.topP != nil { droppedFields.append("topP") }
    if request.topK != nil { droppedFields.append("topK") }
    if request.frequencyPenalty != nil { droppedFields.append("frequencyPenalty") }
    if request.presencePenalty != nil { droppedFields.append("presencePenalty") }
    if request.seed != nil { droppedFields.append("seed") }
    if !request.stopSequences.isEmpty { droppedFields.append("stopSequences") }
    if request.maxOutputTokens != nil { droppedFields.append("maxOutputTokens") }
    if options["thinkingLevel"] != nil { droppedFields.append("thinkingLevel") }
    if options["thinkingSummaries"] != nil { droppedFields.append("thinkingSummaries") }
    if options["imageConfig"] != nil { droppedFields.append("imageConfig") }
    if !droppedFields.isEmpty {
        let verb = droppedFields.count == 1 ? "is" : "are"
        warnings.append(AIWarning(
            type: "other",
            message: "google.interactions: \(droppedFields.joined(separator: ", ")) \(verb) not supported when an agent is set; use providerOptions.google.agentConfig instead. Dropped from the request body."
        ))
    }
    return warnings
}

func googleInteractionsSnakeCaseObject(_ value: JSONValue) -> JSONValue {
    guard let object = value.objectValue else { return value }
    var converted: [String: JSONValue] = [:]
    for (key, value) in object {
        let mappedKey: String
        switch key {
        case "mimeType": mappedKey = "mime_type"
        case "aspectRatio": mappedKey = "aspect_ratio"
        case "imageSize": mappedKey = "image_size"
        case "gcsUri": mappedKey = "gcs_uri"
        case "startOffset": mappedKey = "start_offset"
        case "endOffset": mappedKey = "end_offset"
        case "thinkingSummaries": mappedKey = "thinking_summaries"
        case "collaborativePlanning": mappedKey = "collaborative_planning"
        default: mappedKey = key
        }
        converted[mappedKey] = googleInteractionsSnakeCaseObject(value)
    }
    return .object(converted)
}