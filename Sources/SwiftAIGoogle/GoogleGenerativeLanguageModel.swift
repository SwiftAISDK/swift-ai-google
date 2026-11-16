import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class GoogleGenerativeLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let config: GoogleModelConfiguration

    public var supportedURLs: [String: [AISupportedURLPattern]] {
        googleLanguageModelSupportedURLs(baseURL: config.baseURL, modelID: modelID)
    }

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
                path: "/\(googleModelPath(modelID)):generateContent",
                modelID: modelID,
                body: body,
                headers: request.headers.mergingHeaders(prepared.headers),
                abortSignal: request.abortSignal
            )
        }
        return try response.result(providerID: providerID, modelID: modelID, prepared: prepared, jsonResponseToolName: jsonResponse.name, requestMetadata: .init(body: prepared.body, headers: request.headers, url: try config.http.url(modelID, "/\(googleModelPath(modelID)):generateContent").absoluteString))
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
                            path: "/\(googleModelPath(modelID)):streamGenerateContent?alt=sse",
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
        try googleGenerateContentCall(
            for: request,
            modelID: modelID,
            providerID: "google.generative-ai",
            isVertexProvider: false,
            isStreaming: isStreaming
        )
    }

}

/// Upstream `getModelPath`: IDs that already contain a resource path
/// (`models/...`, `tunedModels/...`) are used as-is.
public func googleModelPath(_ modelID: String) -> String {
    modelID.contains("/") ? modelID : "models/\(modelID)"
}

/// Builds the `generateContent` request like upstream
/// `GoogleLanguageModel.prepareRequest`, shared by Gemini API and Vertex AI.
public func googleGenerateContentCall(
    for request: LanguageModelRequest,
    modelID: String,
    providerID: String,
    isVertexProvider: Bool,
    isStreaming: Bool = false,
    supportsGoogleCloudStorageToolResults: Bool = false
) throws -> GoogleGenerateContentPreparedCall {
    let preparedOptions = try googlePrepareGenerateContentOptions(
        from: request,
        modelID: modelID,
        providerID: providerID,
        isVertexProvider: isVertexProvider
    )
    var options = preparedOptions.options
    let responseFormat = googleResolvedResponseFormat(request: request, options: &options)
    var warnings = preparedOptions.requestWarnings
    let prompt = try googleGenerateContentPrompt(
        request.messages,
        options: GooglePromptConversionOptions(
            modelID: modelID,
            providerOptionsNames: isVertexProvider ? ["googleVertex", "vertex"] : ["google"],
            includeFunctionCallIDs: !isVertexProvider,
            supportsGoogleCloudStorageToolResults: supportsGoogleCloudStorageToolResults
        ),
        warnings: &warnings
    )
    warnings.append(contentsOf: preparedOptions.reasoningWarnings)

    var generationConfig: [String: JSONValue] = [:]
    googleApplyStandardGenerationSettings(request, to: &generationConfig)
    if googleDropsPenalties(modelID: modelID, isVertexProvider: isVertexProvider) {
        generationConfig.removeValue(forKey: "frequencyPenalty")
        generationConfig.removeValue(forKey: "presencePenalty")
    }
    try googleApplyResponseFormat(responseFormat, options: options, to: &generationConfig)
    googleApplyProviderGenerationOptions(options, to: &generationConfig)

    var body: [String: JSONValue] = [
        "generationConfig": .object(generationConfig),
        "contents": .array(prompt.contents)
    ]
    body["systemInstruction"] = prompt.systemInstruction
    var toolWarnings: [AIWarning] = []
    if let preparedTools = try googlePrepareTools(from: request.tools, toolChoice: options["toolChoice"], modelID: modelID, isVertexProvider: isVertexProvider) {
        toolWarnings = preparedTools.warnings
        if !preparedTools.tools.isEmpty {
            body["tools"] = .array(preparedTools.tools)
        }
        body["toolConfig"] = googleToolConfigWithProviderOptions(preparedTools.toolConfig, options: options, isStreaming: isStreaming, isVertexProvider: isVertexProvider)
    } else {
        body["toolConfig"] = googleToolConfigWithProviderOptions(nil, options: options, isStreaming: isStreaming, isVertexProvider: isVertexProvider)
    }
    body.merge(googleTopLevelGenerateContentOptions(options)) { _, new in new }
    body.merge(googleExtraBodyWithoutToolChoice(options)) { _, new in new }
    return GoogleGenerateContentPreparedCall(
        body: .object(body),
        warnings: warnings + toolWarnings,
        headers: preparedOptions.headers,
        toolNameMapping: createToolNameMapping(
            tools: request.tools,
            providerToolNames: ["google.code_execution": "code_execution"]
        )
    )
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
    var request = request
    if (choice?["type"]?.stringValue ?? choice?.stringValue) == "none" { request.tools = request.tools.filter { name, value in value["type"]?.stringValue != "provider" && !(value["id"]?.stringValue ?? name).hasPrefix("google.") } }
    let choiceType = choice?["type"]?.stringValue ?? choice?.stringValue
    guard case let .json(schema?, _, description) = request.responseFormat,
          request.tools.contains(where: { name, schema in
              schema["type"]?.stringValue != "provider" && !(schema["id"]?.stringValue ?? name).hasPrefix("google.")
          }),
          modelID.range(of: #"(^|/)gemini-"#, options: [.regularExpression, .caseInsensitive]) != nil,
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
    // Upstream `jsonResponseTool` keeps `responseFormat.schema` intact as its
    // `inputSchema`, separate from the tool description.
    converted.tools[name] = .object([
        "description": .string(description ?? "Respond with a JSON object."),
        "inputSchema": schema
    ])
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
