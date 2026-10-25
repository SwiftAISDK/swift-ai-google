import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class GoogleGenerativeLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let config: GoogleModelConfiguration

    init(modelID: String, config: GoogleModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let jsonResponse = googlePrepareJSONResponseToolRequest(request, modelID: modelID)
        let prepared = try Self.generateContentBody(for: jsonResponse.request, modelID: modelID)
        let response = try await googleGenerateContentWithContinuation(body: prepared.body, abortSignal: request.abortSignal) { body in
            try await config.sendJSONResponse(
                path: "/models/\(modelID):generateContent",
                modelID: modelID,
                body: body,
                headers: request.headers.mergingHeaders(prepared.headers),
                abortSignal: request.abortSignal
            )
        }
        return try response.result(providerID: providerID, modelID: modelID, prepared: prepared, jsonResponseToolName: jsonResponse.name)
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let jsonResponse = googlePrepareJSONResponseToolRequest(request, modelID: modelID)
                    let prepared = try Self.generateContentBody(for: jsonResponse.request, modelID: modelID, isStreaming: true)
                    try await googleStreamWithContinuation(
                        body: prepared.body,
                        request: request,
                        modelID: modelID,
                        providerID: providerID,
                        prepared: prepared,
                        jsonResponseToolName: jsonResponse.name,
                        continuation: continuation,
                        errorHandler: { self.config.httpStatusError($0) }
                    ) { body in
                        let httpRequest = try config.request(
                            path: "/models/\(modelID):streamGenerateContent?alt=sse",
                            modelID: modelID,
                            body: body,
                            headers: request.headers.mergingHeaders(prepared.headers),
                            abortSignal: request.abortSignal
                        )
                        return (httpRequest, try await config.streamRequest(httpRequest))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    static func generateContentBody(for request: LanguageModelRequest, modelID: String, isStreaming: Bool = false) throws -> GoogleGenerateContentPreparedCall {
        let preparedOptions = try googlePrepareGenerateContentOptions(
            from: request,
            modelID: modelID,
            providerID: "google.generative-ai",
            isVertexProvider: false
        )
        var options = preparedOptions.options
        let responseFormat = googleResolvedResponseFormat(request: request, options: &options)
        var warnings = preparedOptions.warnings
        let systemText = request.messages
            .filter { $0.role == .system }
            .map(\.combinedText)
            .joined(separator: "\n")
        let rawContents = try request.messages
            .filter { $0.role != .system }
            .map { try googleGenerateContentMessageJSON($0, modelID: modelID, warnings: &warnings) }
        let preparedMessages = googleContentsWithSystemInstruction(systemText: systemText, contents: rawContents, modelID: modelID)

        var generationConfig: [String: JSONValue] = [:]
        googleApplyStandardGenerationSettings(request, to: &generationConfig)
        try googleApplyResponseFormat(responseFormat, options: options, to: &generationConfig)
        googleApplyProviderGenerationOptions(options, to: &generationConfig)

        var body: [String: JSONValue] = ["contents": .array(preparedMessages.contents)]
        if let systemInstruction = preparedMessages.systemInstruction {
            body["systemInstruction"] = systemInstruction
        }
        if !generationConfig.isEmpty { body["generationConfig"] = .object(generationConfig) }
        if let preparedTools = try googlePrepareTools(from: request.tools, toolChoice: options["toolChoice"], modelID: modelID, isVertexProvider: false) {
            warnings.append(contentsOf: preparedTools.warnings)
            if !preparedTools.tools.isEmpty {
                body["tools"] = .array(preparedTools.tools)
            }
            if let toolConfig = googleToolConfigWithProviderOptions(preparedTools.toolConfig, options: options, isStreaming: isStreaming, isVertexProvider: false) {
                body["toolConfig"] = toolConfig
            }
        } else if let toolConfig = googleToolConfigWithProviderOptions(nil, options: options, isStreaming: isStreaming, isVertexProvider: false) {
            body["toolConfig"] = toolConfig
        }
        body.merge(googleTopLevelGenerateContentOptions(options)) { _, new in new }
        body.merge(googleExtraBodyWithoutToolChoice(options)) { _, new in new }
        return GoogleGenerateContentPreparedCall(
            body: .object(body),
            warnings: warnings,
            headers: preparedOptions.headers,
            toolNameMapping: createToolNameMapping(
                tools: request.tools,
                providerToolNames: ["google.code_execution": "code_execution"]
            )
        )
    }

}

public struct GoogleGenerateContentPreparedCall {
    public var body: JSONValue
    public var warnings: [AIWarning]
    public var headers: [String: String]
    public var toolNameMapping: AIToolNameMapping
    public init(body: JSONValue, warnings: [AIWarning], headers: [String: String], toolNameMapping: AIToolNameMapping) {
        self.body = body
        self.warnings = warnings
        self.headers = headers
        self.toolNameMapping = toolNameMapping
    }

}

public func googlePrepareJSONResponseToolRequest(
    _ request: LanguageModelRequest,
    modelID: String
) -> (request: LanguageModelRequest, name: String?) {
    let choice = request.toolChoice ?? request.extraBody["toolChoice"]
    let choiceType = choice?["type"]?.stringValue ?? choice?.stringValue
    guard case let .json(schema?, _, description) = request.responseFormat,
          request.tools.contains(where: { name, schema in
              schema["type"]?.stringValue != "provider" && !(schema["id"]?.stringValue ?? name).hasPrefix("google.")
          }),
          modelID.range(of: #"(^|[/.])gemini-"#, options: [.regularExpression, .caseInsensitive]) != nil,
          !googleModelCapabilities(for: modelID).usesGemini3Features || choiceType == "required" || choiceType == "tool" else {
        return (request, nil)
    }
    var name = "json"
    var suffix = 1
    while request.tools[name] != nil {
        name = "json_\(suffix)"
        suffix += 1
    }
    var converted = request
    var tool = schema.objectValue ?? [:]
    tool["description"] = .string(description ?? "Respond with a JSON object.")
    converted.tools[name] = .object(tool)
    converted.responseFormat = nil
    converted.extraBody.removeValue(forKey: "responseFormat")
    if choiceType == "tool" {
        converted.toolChoice = choice
    } else if choiceType == "none" {
        converted.toolChoice = .object(["type": "tool", "toolName": .string(name)])
    } else {
        converted.toolChoice = .object(["type": "required"])
    }
    return (converted, name)
}

extension GoogleGenerativeLanguageModel {
    public static func imageGenerationContentBody(prompt: String, aspectRatio: String?, files: [ImageInputFile] = []) throws -> [String: JSONValue] {
        var generationConfig: [String: JSONValue] = ["responseModalities": .array(["IMAGE"])]
        if let aspectRatio {
            generationConfig["imageConfig"] = .object(["aspectRatio": .string(aspectRatio)])
        }
        var parts: [JSONValue] = []
        if !prompt.isEmpty {
            parts.append(.object(["text": .string(prompt)]))
        }
        for file in files {
            if let url = file.url {
                parts.append(.object([
                    "fileData": .object([
                        "fileUri": .string(url),
                        "mimeType": .string(file.mediaType ?? "image/*")
                    ])
                ]))
            } else if let data = file.data {
                parts.append(.object([
                    "inlineData": .object([
                        "mimeType": .string(try resolveFullMediaType(mediaType: file.mediaType ?? "image/*", data: data)),
                        "data": .string(data.base64EncodedString())
                    ])
                ]))
            }
        }
        return [
            "contents": .array([
                .object([
                    "role": .string("user"),
                    "parts": .array(parts.isEmpty ? [.object(["text": .string("")])] : parts)
                ])
            ]),
            "generationConfig": .object(generationConfig)
        ]
    }
}


let googleSkipThoughtSignatureValidator = "skip_thought_signature_validator"

public func googleGenerateContentMessageJSON(
    _ message: AIMessage,
    modelID: String,
    includeFunctionCallIDs: Bool = true,
    supportsGoogleCloudStorageToolResults: Bool = false,
    warnings: inout [AIWarning]
) throws -> JSONValue {
    let role = message.role == .assistant ? "model" : "user"
    var parts: [JSONValue] = []
    var modelResponseHasSignedFunctionCall = false
    for part in message.content {
        if case let .toolCall(call) = part {
            let thoughtSignature = googleThoughtSignature(from: call.providerMetadata)
            let isServerToolCall = googleServerToolMetadata(from: call.providerMetadata) != nil
            let shouldSkipMissingSignatureMitigation = message.role == .assistant
                && !isServerToolCall
                && thoughtSignature == nil
                && modelResponseHasSignedFunctionCall
            parts.append(googleGenerateContentToolCallPart(
                call,
                modelID: modelID,
                includeFunctionCallIDs: includeFunctionCallIDs,
                skipMissingSignatureMitigation: shouldSkipMissingSignatureMitigation,
                warnings: &warnings
            ))
            if message.role == .assistant, !isServerToolCall, thoughtSignature != nil {
                modelResponseHasSignedFunctionCall = true
            }
            continue
        }
        parts.append(contentsOf: try googleGenerateContentParts(
            part,
            modelID: modelID,
            includeFunctionCallIDs: includeFunctionCallIDs,
            supportsGoogleCloudStorageToolResults: supportsGoogleCloudStorageToolResults,
            warnings: &warnings
        ))
    }
    return .object(["role": .string(role), "parts": .array(parts)])
}

func googleGenerateContentParts(
    _ part: AIContentPart,
    modelID: String,
    includeFunctionCallIDs: Bool = true,
    supportsGoogleCloudStorageToolResults: Bool = false,
    warnings: inout [AIWarning]
) throws -> [JSONValue] {
    switch part {
    case let .text(text, _):
        return [.object(["text": .string(text)])]
    case let .reasoning(text, _):
        return [.object(["text": .string(text)])]
    case let .imageURL(url, _):
        return [.object(["fileData": .object(["fileUri": .string(url)])])]
    case let .data(mimeType, data, _), let .file(mimeType, data, _, _):
        let resolvedMimeType = try resolveFullMediaType(mediaType: mimeType, data: data)
        return [.object([
            "inlineData": .object([
                "mimeType": .string(resolvedMimeType),
                "data": .string(data.base64EncodedString())
            ])
        ])]
    case let .providerReference(_, reference, _, _):
        return [.object(["fileData": .object(["fileUri": .string((try? resolveProviderReference(reference, provider: "google")) ?? reference.values.first ?? "")])])]
    case let .toolCall(call):
        return [googleGenerateContentToolCallPart(
            call,
            modelID: modelID,
            includeFunctionCallIDs: includeFunctionCallIDs,
            warnings: &warnings
        )]
    case let .toolResult(result):
        return googleGenerateContentToolResultParts(
            result,
            modelID: modelID,
            includeFunctionCallIDs: includeFunctionCallIDs,
            supportsGoogleCloudStorageToolResults: supportsGoogleCloudStorageToolResults
        )
    case .reasoningFile, .custom, .toolApprovalRequest, .toolApprovalResponse:
        return [.object(["text": .string("")])]
    }
}

func googleGenerateContentToolCallPart(
    _ call: AIToolCall,
    modelID: String,
    includeFunctionCallIDs: Bool = true,
    skipMissingSignatureMitigation: Bool = false,
    warnings: inout [AIWarning]
) -> JSONValue {
    if call.providerExecuted, call.name == "code_execution" {
        return .object(["executableCode": googleToolArguments(call.arguments)])
    }

    let thoughtSignature = googleThoughtSignature(from: call.providerMetadata)
    let effectiveThoughtSignature: JSONValue?
    if thoughtSignature == nil,
       googleMessageTargetIsGemini3(modelID),
       !skipMissingSignatureMitigation {
        effectiveThoughtSignature = .string(googleSkipThoughtSignatureValidator)
        warnings.append(googleMissingThoughtSignatureWarning(toolName: call.name))
    } else {
        effectiveThoughtSignature = thoughtSignature
    }

    var output: [String: JSONValue]
    if let serverTool = googleServerToolMetadata(from: call.providerMetadata) {
        output = [
            "toolCall": .object([
                "toolType": .string(serverTool.type),
                "id": .string(serverTool.id),
                "args": googleToolArguments(call.arguments)
            ])
        ]
    } else {
        var functionCall: [String: JSONValue] = [
            "name": .string(call.name),
            "args": googleToolArguments(call.arguments)
        ]
        if includeFunctionCallIDs {
            functionCall["id"] = .string(call.id)
        }
        output = [
            "functionCall": .object(functionCall)
        ]
    }
    if let effectiveThoughtSignature {
        output["thoughtSignature"] = effectiveThoughtSignature
    }
    return .object(output)
}

func googleGenerateContentToolResultParts(
    _ result: AIToolResult,
    modelID: String,
    includeFunctionCallIDs: Bool = true,
    supportsGoogleCloudStorageToolResults: Bool = false
) -> [JSONValue] {
    let output = result.modelOutput ?? result.result
    let unwrappedOutput = output["type"]?.stringValue == "json"
        ? (output["value"] ?? output)
        : output
    if result.toolName == "code_execution",
       unwrappedOutput["outcome"] != nil {
        return [.object(["codeExecutionResult": unwrappedOutput])]
    }

    if let serverTool = googleServerToolMetadata(from: result.providerMetadata) {
        var output: [String: JSONValue] = [
            "toolResponse": .object([
                "toolType": .string(serverTool.type),
                "id": .string(serverTool.id),
                "response": unwrappedOutput
            ])
        ]
        if let thoughtSignature = googleThoughtSignature(from: result.providerMetadata) {
            output["thoughtSignature"] = thoughtSignature
        }
        return [.object(output)]
    }

    if output["type"]?.stringValue == "content",
       let value = output["value"]?.arrayValue {
        return googleGenerateContentToolResultContentParts(
            toolName: result.toolName,
            toolCallID: result.toolCallID,
            content: value,
            supportsFunctionResponseParts: googleMessageTargetIsGemini3(modelID),
            includeFunctionCallIDs: includeFunctionCallIDs,
            supportsGoogleCloudStorageToolResults: supportsGoogleCloudStorageToolResults
        )
    }

    let outputType = output["type"]?.stringValue
    let isError = outputType == "error-text" || outputType == "error-json" || outputType == "execution-denied"
    let errorValue = outputType == "execution-denied"
        ? (output["reason"] ?? .string("Tool call execution denied."))
        : (output["value"] ?? output)
    var functionResponse: [String: JSONValue] = [
        "name": .string(result.toolName),
        "response": isError
            ? .object(["name": .string(result.toolName), "error": googleSerializeFunctionResponseContent(errorValue)])
            : googleSerializeFunctionResponseContent(unwrappedOutput)
    ]
    if includeFunctionCallIDs {
        functionResponse["id"] = .string(result.toolCallID)
    }
    return [.object([
        "functionResponse": .object(functionResponse)
    ])]
}

func googleGenerateContentToolResultContentParts(
    toolName: String,
    toolCallID: String,
    content: [JSONValue],
    supportsFunctionResponseParts: Bool,
    includeFunctionCallIDs: Bool = true,
    supportsGoogleCloudStorageToolResults: Bool = false
) -> [JSONValue] {
    var textParts: [String] = []
    var inlineParts: [JSONValue] = []
    var legacyParts: [JSONValue] = []

    for contentPart in content {
        switch contentPart["type"]?.stringValue {
        case "text":
            textParts.append(contentPart["text"]?.stringValue ?? "")
        case "file", "image-data", "file-data":
            if supportsFunctionResponseParts,
               supportsGoogleCloudStorageToolResults,
               let fileData = googleCloudStorageFileDataFromToolContent(contentPart) {
                inlineParts.append(fileData)
            } else if let inlineData = googleInlineDataFromToolContent(contentPart) {
                if supportsFunctionResponseParts {
                    inlineParts.append(inlineData)
                } else {
                    legacyParts.append(.object(inlineData.objectValue ?? [:]))
                    legacyParts.append(.object(["text": .string("Tool executed successfully and returned this file as a response")]))
                }
            } else {
                textParts.append(googleJSONString(contentPart) ?? String(describing: contentPart))
            }
        default:
            textParts.append(googleJSONString(contentPart) ?? String(describing: contentPart))
        }
    }

    if !supportsFunctionResponseParts, !legacyParts.isEmpty {
        if !textParts.isEmpty {
            var functionResponse: [String: JSONValue] = [
                "name": .string(toolName),
                "response": .object(["name": .string(toolName), "content": .string(textParts.joined(separator: "\n"))])
            ]
            if includeFunctionCallIDs {
                functionResponse["id"] = .string(toolCallID)
            }
            legacyParts.insert(.object([
                "functionResponse": .object(functionResponse)
            ]), at: 0)
        }
        return legacyParts
    }

    var response: [String: JSONValue] = [
        "name": .string(toolName),
        "response": .object([
            "name": .string(toolName),
            "content": .string(textParts.isEmpty ? "Tool executed successfully." : textParts.joined(separator: "\n"))
        ])
    ]
    if includeFunctionCallIDs {
        response["id"] = .string(toolCallID)
    }
    if !inlineParts.isEmpty {
        response["parts"] = .array(inlineParts)
    }
    return [.object(["functionResponse": .object(response)])]
}

func googleInlineDataFromToolContent(_ contentPart: JSONValue) -> JSONValue? {
    if let mediaType = contentPart["mediaType"]?.stringValue,
       let rawData = contentPart["data"]?["data"]?.stringValue ?? contentPart["data"]?.stringValue {
        let dataURL = googleParseBase64DataURL(rawData)
        let base64 = dataURL?.data ?? rawData
        var resolvedMediaType = dataURL?.mediaType ?? mediaType
        if !isFullMediaType(resolvedMediaType),
           let bytes = Data(base64Encoded: base64),
           let fullMediaType = try? resolveFullMediaType(mediaType: resolvedMediaType, data: bytes) {
            resolvedMediaType = fullMediaType
        }
        return .object(["inlineData": .object(["mimeType": .string(resolvedMediaType), "data": .string(base64)])])
    }
    return nil
}

func googleCloudStorageFileDataFromToolContent(_ contentPart: JSONValue) -> JSONValue? {
    guard let mediaType = contentPart["mediaType"]?.stringValue,
          googleVertexSupportsCloudStorageToolResultMediaType(mediaType),
          contentPart["data"]?["type"]?.stringValue == "url",
          let url = contentPart["data"]?["url"]?.stringValue,
          url.hasPrefix("gs://") else {
        return nil
    }
    return .object(["fileData": .object(["mimeType": .string(mediaType), "fileUri": .string(url)])])
}

public func googleVertexSupportsCloudStorageToolResultMediaType(_ mediaType: String) -> Bool {
    ["image/png", "image/jpeg", "image/webp", "application/pdf", "text/plain"].contains(mediaType.lowercased())
}

private func googleParseBase64DataURL(_ value: String) -> (mediaType: String, data: String)? {
    guard value.hasPrefix("data:"),
          let separator = value.range(of: ";base64,") else {
        return nil
    }
    let mediaType = String(value[value.index(value.startIndex, offsetBy: 5)..<separator.lowerBound])
    let data = String(value[separator.upperBound...])
    guard !mediaType.isEmpty, !data.isEmpty else { return nil }
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

func googleMissingThoughtSignatureWarning(toolName: String) -> AIWarning {
    AIWarning(
        type: "other",
        message: "Replayed `functionCall` part for a Gemini 3 model without a `thoughtSignature` (tool: `\(toolName)`). Injected the documented `skip_thought_signature_validator` sentinel to keep the request from failing with HTTP 400. The likely cause is application code that drops `providerOptions.google.thoughtSignature` when persisting or serializing assistant tool-call messages. See https://ai.google.dev/gemini-api/docs/thought-signatures."
    )
}

func googleToolArguments(_ arguments: String) -> JSONValue {
    (try? decodeJSONBody(Data(arguments.utf8))) ?? .object([:])
}

func googleJSONString(_ value: JSONValue) -> String? {
    guard let data = try? encodeJSONBody(value) else { return nil }
    return String(data: data, encoding: .utf8)
}