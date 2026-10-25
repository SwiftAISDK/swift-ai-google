import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


/// Google Generative Language implementation of the durable Batch V4 capability.
public final class GoogleBatchLanguageModel: BatchLanguageModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String

    private let languageModel: GoogleGenerativeLanguageModel
    private let config: GoogleModelConfiguration

    init(modelID: String, config: GoogleModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
        self.languageModel = GoogleGenerativeLanguageModel(modelID: modelID, config: config)
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        try await languageModel.generate(request)
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        languageModel.stream(request)
    }

    public func startBatch(
        _ options: AIBatchStartOptions<AILanguageModelBatchRequest>
    ) async throws -> AIBatchStartResult {
        try options.abortSignal?.throwIfAborted()

        let displayName = "ai-sdk-batch-\(UUID().uuidString.lowercased())"
        var inlineRequests: [JSONValue] = []
        let emptyInlineBody = googleBatchCreationBody(
            displayName: displayName,
            webhookURL: options.webhookURL,
            inputConfig: .object(["requests": .object(["requests": .array([])])])
        )
        var inlineBytes = try encodeJSONBody(emptyInlineBody).count
        var fileLines: [Data]?
        var warnings: [AIBatchWarning] = []

        for item in options.requests {
            try options.abortSignal?.throwIfAborted()
            let prepared = try GoogleGenerativeLanguageModel.generateContentBody(
                for: item.request,
                modelID: modelID
            )
            let inlineRequest: JSONValue = .object([
                "request": prepared.body,
                "metadata": .object(["key": .string(item.id)])
            ])
            let inlineRequestBytes = try encodeJSONBody(inlineRequest).count

            if fileLines == nil {
                let separatorBytes = inlineRequests.isEmpty ? 0 : 1
                if inlineBytes + inlineRequestBytes + separatorBytes < googleBatchInlineCreationMaxBytes {
                    inlineRequests.append(inlineRequest)
                    inlineBytes += inlineRequestBytes + separatorBytes
                } else {
                    fileLines = try inlineRequests.map { previous in
                        let line: JSONValue = .object([
                            "key": previous["metadata"]?["key"] ?? .string(""),
                            "request": previous["request"] ?? .object([:])
                        ])
                        return try googleBatchJSONLine(line)
                    }
                    inlineRequests.removeAll(keepingCapacity: false)
                    fileLines?.append(try googleBatchJSONLine(.object([
                        "key": .string(item.id),
                        "request": prepared.body
                    ])))
                }
            } else {
                fileLines?.append(try googleBatchJSONLine(.object([
                    "key": .string(item.id),
                    "request": prepared.body
                ])))
            }

            warnings.append(contentsOf: prepared.warnings.map {
                AIBatchWarning(requestID: item.id, warning: $0)
            })
        }

        let headers = googleBatchHeaders(options.headers, config: config)
        if let fileLines {
            let inputData = fileLines.reduce(into: Data()) { $0.append($1) }
            guard inputData.count <= googleBatchInputFileMaxBytes else {
                throw AIError.invalidArgument(
                    argument: "requests",
                    message: "Google batch input files must not exceed 2 GB."
                )
            }
            let uploaded = try await uploadBatchInput(
                inputData,
                displayName: displayName,
                headers: headers,
                abortSignal: options.abortSignal
            )
            let body = googleBatchCreationBody(
                displayName: displayName,
                webhookURL: options.webhookURL,
                inputConfig: .object(["fileName": .string(uploaded.name)])
            )
            let operation = try await sendJSON(
                url: try googleBatchCreateURL(),
                body: body,
                headers: headers,
                abortSignal: options.abortSignal
            )
            var googleMetadata: [String: JSONValue] = ["inputFileId": .string(uploaded.name)]
            if let expirationTime = uploaded.expirationTime {
                googleMetadata["inputFileExpiresAt"] = .string(expirationTime)
            }
            return AIBatchStartResult(
                batchID: try googleBatchOperationName(operation),
                status: googleBatchStatus(operation),
                warnings: warnings,
                providerMetadata: ["google": .object(googleMetadata)]
            )
        }

        let body = googleBatchCreationBody(
            displayName: displayName,
            webhookURL: options.webhookURL,
            inputConfig: .object([
                "requests": .object(["requests": .array(inlineRequests)])
            ])
        )
        let operation = try await sendJSON(
            url: try googleBatchCreateURL(),
            body: body,
            headers: headers,
            abortSignal: options.abortSignal
        )
        return AIBatchStartResult(
            batchID: try googleBatchOperationName(operation),
            status: googleBatchStatus(operation),
            warnings: warnings
        )
    }

    func startProviderBatch(
        _ options: AIBatchStartOptions<AIBatchRequest>
    ) async throws -> AIBatchStartResult {
        try options.abortSignal?.throwIfAborted()
        let commonModelID = try googleProviderBatchModelID(options.requests)

        let displayName = "ai-sdk-batch-\(UUID().uuidString.lowercased())"
        var inlineRequests: [JSONValue] = []
        let emptyInlineBody = googleBatchCreationBody(
            displayName: displayName,
            webhookURL: options.webhookURL,
            inputConfig: .object(["requests": .object(["requests": .array([])])])
        )
        var inlineBytes = try encodeJSONBody(emptyInlineBody).count
        var fileLines: [Data]?
        var warnings: [AIBatchWarning] = []

        for item in options.requests {
            try options.abortSignal?.throwIfAborted()
            let prepared: GoogleGenerateContentPreparedCall
            switch item {
            case let .text(text):
                prepared = try GoogleGenerativeLanguageModel.generateContentBody(
                    for: text.request,
                    modelID: commonModelID
                )
            case let .image(image):
                prepared = try googlePrepareImageBatchRequest(image.request, modelID: commonModelID)
            }
            let inlineRequest: JSONValue = .object([
                "request": prepared.body,
                "metadata": .object(["key": .string(item.id)])
            ])
            let inlineRequestBytes = try encodeJSONBody(inlineRequest).count

            if fileLines == nil {
                let separatorBytes = inlineRequests.isEmpty ? 0 : 1
                if inlineBytes + inlineRequestBytes + separatorBytes < googleBatchInlineCreationMaxBytes {
                    inlineRequests.append(inlineRequest)
                    inlineBytes += inlineRequestBytes + separatorBytes
                } else {
                    fileLines = try inlineRequests.map { previous in
                        try googleBatchJSONLine(.object([
                            "key": previous["metadata"]?["key"] ?? .string(""),
                            "request": previous["request"] ?? .object([:])
                        ]))
                    }
                    inlineRequests.removeAll(keepingCapacity: false)
                    fileLines?.append(try googleBatchJSONLine(.object([
                        "key": .string(item.id),
                        "request": prepared.body
                    ])))
                }
            } else {
                fileLines?.append(try googleBatchJSONLine(.object([
                    "key": .string(item.id),
                    "request": prepared.body
                ])))
            }

            warnings.append(contentsOf: prepared.warnings.map {
                AIBatchWarning(requestID: item.id, warning: $0)
            })
        }

        let headers = googleBatchHeaders(options.headers, config: config)
        if let fileLines {
            let inputData = fileLines.reduce(into: Data()) { $0.append($1) }
            guard inputData.count <= googleBatchInputFileMaxBytes else {
                throw AIError.invalidArgument(
                    argument: "requests",
                    message: "Google batch input files must not exceed 2 GB."
                )
            }
            let uploaded = try await uploadBatchInput(
                inputData,
                displayName: displayName,
                headers: headers,
                abortSignal: options.abortSignal
            )
            let body = googleBatchCreationBody(
                displayName: displayName,
                webhookURL: options.webhookURL,
                inputConfig: .object(["fileName": .string(uploaded.name)])
            )
            let operation = try await sendJSON(
                url: try googleBatchCreateURL(modelID: commonModelID),
                body: body,
                headers: headers,
                abortSignal: options.abortSignal
            )
            var googleMetadata: [String: JSONValue] = ["inputFileId": .string(uploaded.name)]
            if let expirationTime = uploaded.expirationTime {
                googleMetadata["inputFileExpiresAt"] = .string(expirationTime)
            }
            return AIBatchStartResult(
                batchID: try googleBatchOperationName(operation),
                status: googleBatchStatus(operation),
                warnings: warnings,
                providerMetadata: ["google": .object(googleMetadata)]
            )
        }

        let body = googleBatchCreationBody(
            displayName: displayName,
            webhookURL: options.webhookURL,
            inputConfig: .object([
                "requests": .object(["requests": .array(inlineRequests)])
            ])
        )
        let operation = try await sendJSON(
            url: try googleBatchCreateURL(modelID: commonModelID),
            body: body,
            headers: headers,
            abortSignal: options.abortSignal
        )
        return AIBatchStartResult(
            batchID: try googleBatchOperationName(operation),
            status: googleBatchStatus(operation),
            warnings: warnings
        )
    }

    public func getBatchStatus(_ options: AIBatchOperationOptions) async throws -> AIBatchStatus {
        googleBatchStatus(try await retrieveBatch(options))
    }

    public func getBatchResults(
        _ options: AIBatchOperationOptions
    ) async throws -> AsyncThrowingStream<AIBatchItemResult<TextGenerationResult>, Error> {
        let operation = try await retrieveBatch(options)
        let status = googleBatchStatus(operation)
        guard status.status != .pending else {
            throw AIError.invalidArgument(
                argument: "batchID",
                message: "Google batch \"\(options.batchID)\" is not complete."
            )
        }

        if let inline = operation["metadata"]?["output"]?["inlinedResponses"]?["inlinedResponses"]?.arrayValue
            ?? operation["response"]?["inlinedResponses"]?["inlinedResponses"]?.arrayValue {
            let lines = try inline.map { item in
                guard let key = item["metadata"]?["key"]?.stringValue,
                      !key.isEmpty else {
                    throw googleBatchInvalidResultKeyError(providerID: providerID)
                }
                return JSONValue.object([
                    "key": .string(key),
                    "response": item["response"] ?? .null,
                    "error": item["error"] ?? .null
                ])
            }
            return googleBatchResultStream(lines: lines, abortSignal: options.abortSignal)
        }

        let responsesFile = operation["metadata"]?["output"]?["responsesFile"]?.stringValue
            ?? operation["response"]?["responsesFile"]?.stringValue
        guard let responsesFile else {
            if status.status == .completed {
                throw AIError.invalidResponse(
                    provider: providerID,
                    message: "Google batch \"\(options.batchID)\" completed without batch output."
                )
            }
            return googleBatchResultStream(lines: [], abortSignal: options.abortSignal)
        }

        let encodedFile = responsesFile
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { googleBatchPathComponent(String($0)) }
            .joined(separator: "/")
        let outputURL = try requireURL(
            "\(googleBatchBaseOrigin())/download/v1beta/\(encodedFile):download?alt=media"
        )
        let request = AIHTTPRequest(
            method: "GET",
            url: outputURL,
            headers: googleBatchHeaders(options.headers, config: config),
            abortSignal: options.abortSignal,
            maxResponseBytes: googleBatchInputFileMaxBytes
        )
        let maxLineBytes = try aiBatchResultLineLimit(config.batchResultMaxLineBytes)
        let response = try await config.streamRequest(request)
        guard (200..<300).contains(response.statusCode) else {
            throw config.httpStatusError(try await bufferedHTTPResponse(from: response, request: request))
        }
        return googleBatchResultStream(body: response.body, abortSignal: options.abortSignal, url: request.url.absoluteString, maxLineBytes: maxLineBytes, cancelBody: response.cancelBody)
    }

    func getProviderBatchResults(
        _ options: AIBatchOperationOptions
    ) async throws -> AsyncThrowingStream<AIBatchV4ItemResult, Error> {
        let operation = try await retrieveBatch(options)
        let status = googleBatchStatus(operation)
        guard status.status != .pending else {
            throw AIError.invalidArgument(
                argument: "batchID",
                message: "Google batch \"\(options.batchID)\" is not complete."
            )
        }

        if let inline = operation["metadata"]?["output"]?["inlinedResponses"]?["inlinedResponses"]?.arrayValue
            ?? operation["response"]?["inlinedResponses"]?["inlinedResponses"]?.arrayValue {
            let lines = try inline.map { item in
                guard let key = item["metadata"]?["key"]?.stringValue,
                      !key.isEmpty else {
                    throw googleBatchInvalidResultKeyError(providerID: providerID)
                }
                return JSONValue.object([
                    "key": .string(key),
                    "response": item["response"] ?? .null,
                    "error": item["error"] ?? .null
                ])
            }
            return googleProviderBatchResultStream(lines: lines, abortSignal: options.abortSignal)
        }

        let responsesFile = operation["metadata"]?["output"]?["responsesFile"]?.stringValue
            ?? operation["response"]?["responsesFile"]?.stringValue
        guard let responsesFile else {
            if status.status == .completed {
                throw AIError.invalidResponse(
                    provider: providerID,
                    message: "Google batch \"\(options.batchID)\" completed without batch output."
                )
            }
            return googleProviderBatchResultStream(lines: [], abortSignal: options.abortSignal)
        }

        let encodedFile = responsesFile
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { googleBatchPathComponent(String($0)) }
            .joined(separator: "/")
        let outputURL = try requireURL(
            "\(googleBatchBaseOrigin())/download/v1beta/\(encodedFile):download?alt=media"
        )
        let request = AIHTTPRequest(
            method: "GET",
            url: outputURL,
            headers: googleBatchHeaders(options.headers, config: config),
            abortSignal: options.abortSignal,
            maxResponseBytes: googleBatchInputFileMaxBytes
        )
        let maxLineBytes = try aiBatchResultLineLimit(config.batchResultMaxLineBytes)
        let response = try await config.streamRequest(request)
        guard (200..<300).contains(response.statusCode) else {
            throw config.httpStatusError(try await bufferedHTTPResponse(from: response, request: request))
        }
        return googleProviderBatchResultStream(body: response.body, abortSignal: options.abortSignal, url: request.url.absoluteString, maxLineBytes: maxLineBytes, cancelBody: response.cancelBody)
    }

    func cancelProviderBatch(_ options: AIBatchOperationOptions) async throws -> AIBatchCancelResult {
        try options.abortSignal?.throwIfAborted()
        let encodedBatchID = options.batchID
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { googleBatchPathComponent(String($0)) }
            .joined(separator: "/")
        let value = try await sendJSON(
            url: try requireURL("\(config.baseURL)/\(encodedBatchID):cancel"),
            body: .object([:]),
            headers: googleBatchHeaders(options.headers, config: config),
            abortSignal: options.abortSignal
        )
        guard value.objectValue != nil else {
            throw AIError.invalidResponse(provider: providerID, message: "Invalid Google batch cancellation response.")
        }
        return AIBatchCancelResult()
    }

    func listProviderBatches(_ options: AIBatchListOptions) async throws -> AIBatchListResult {
        try options.abortSignal?.throwIfAborted()
        guard var components = URLComponents(string: "\(config.baseURL)/batches") else {
            throw AIError.invalidArgument(argument: "url", message: "Invalid Google batches list URL.")
        }
        var queryItems = components.queryItems ?? []
        if let limit = options.limit {
            queryItems.append(URLQueryItem(name: "pageSize", value: String(limit)))
        }
        if let cursor = options.cursor {
            queryItems.append(URLQueryItem(name: "pageToken", value: cursor))
        }
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let url = components.url else {
            throw AIError.invalidArgument(argument: "url", message: "Invalid Google batches list URL.")
        }
        let request = AIHTTPRequest(
            method: "GET",
            url: url,
            headers: googleBatchHeaders(options.headers, config: config),
            abortSignal: options.abortSignal
        )
        let response = try await config.transport.send(request)
        guard (200..<300).contains(response.statusCode) else {
            throw config.httpStatusError(response)
        }
        let page = try response.jsonValue()
        guard let object = page.objectValue,
              object["nextPageToken"] == nil
                || object["nextPageToken"] == .null
                || object["nextPageToken"]?.stringValue != nil else {
            throw AIError.invalidResponse(provider: providerID, message: "Invalid Google batches list response.")
        }
        let operations: [JSONValue]
        if let rawOperations = object["operations"], rawOperations != .null {
            guard let values = rawOperations.arrayValue else {
                throw AIError.invalidResponse(provider: providerID, message: "Invalid Google batches list response.")
            }
            operations = values
        } else {
            operations = []
        }
        let batches = try operations.map { operation in
            AIBatchListItem(
                batchID: try googleBatchOperationName(operation),
                status: googleBatchStatus(operation)
            )
        }
        return AIBatchListResult(
            batches: batches,
            nextCursor: object["nextPageToken"]?.stringValue
        )
    }

    private func retrieveBatch(_ options: AIBatchOperationOptions) async throws -> JSONValue {
        try options.abortSignal?.throwIfAborted()
        let encodedBatchID = options.batchID
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { googleBatchPathComponent(String($0)) }
            .joined(separator: "/")
        let request = AIHTTPRequest(
            method: "GET",
            url: try requireURL("\(config.baseURL)/\(encodedBatchID)"),
            headers: googleBatchHeaders(options.headers, config: config),
            abortSignal: options.abortSignal
        )
        let response = try await config.transport.send(request)
        guard (200..<300).contains(response.statusCode) else {
            throw config.httpStatusError(response)
        }
        let operation = try response.jsonValue()
        _ = try googleBatchOperationName(operation)
        return operation
    }

    private func uploadBatchInput(
        _ data: Data,
        displayName: String,
        headers: [String: String],
        abortSignal: AIAbortSignal?
    ) async throws -> (name: String, expirationTime: String?) {
        var startHeaders = headers
        startHeaders["X-Goog-Upload-Protocol"] = "resumable"
        startHeaders["X-Goog-Upload-Command"] = "start"
        startHeaders["X-Goog-Upload-Header-Content-Length"] = String(data.count)
        startHeaders["X-Goog-Upload-Header-Content-Type"] = "application/jsonl"
        startHeaders["content-type"] = startHeaders["content-type"] ?? "application/json"
        let startRequest = AIHTTPRequest(
            url: try requireURL("\(googleBatchBaseOrigin())/upload/v1beta/files"),
            headers: startHeaders,
            body: try encodeJSONBody(.object([
                "file": .object(["display_name": .string("\(displayName)-input")])
            ])),
            abortSignal: abortSignal
        )
        let startResponse = try await config.transport.send(startRequest)
        guard (200..<300).contains(startResponse.statusCode) else {
            throw config.httpStatusError(startResponse)
        }
        guard let uploadURLText = startResponse.headerValue("x-goog-upload-url") else {
            throw AIError.invalidResponse(
                provider: providerID,
                message: "Google did not return a resumable upload URL."
            )
        }

        let uploadRequest = AIHTTPRequest(
            url: try requireURL(uploadURLText),
            headers: [
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
                "Content-Type": "application/jsonl"
            ],
            body: data,
            abortSignal: abortSignal
        )
        let uploadResponse = try await config.transport.send(uploadRequest)
        guard (200..<300).contains(uploadResponse.statusCode) else {
            throw config.httpStatusError(uploadResponse)
        }
        let value = try uploadResponse.jsonValue()
        guard let name = value["file"]?["name"]?.stringValue else {
            throw AIError.invalidResponse(provider: providerID, message: "Invalid Google batch input file response.")
        }
        return (name, value["file"]?["expirationTime"]?.stringValue)
    }

    private func sendJSON(
        url: URL,
        body: JSONValue,
        headers: [String: String],
        abortSignal: AIAbortSignal?
    ) async throws -> JSONValue {
        var headers = headers
        headers["content-type"] = headers["content-type"] ?? "application/json"
        let request = AIHTTPRequest(
            url: url,
            headers: headers,
            body: try encodeJSONBody(body),
            abortSignal: abortSignal
        )
        let response = try await config.transport.send(request)
        guard (200..<300).contains(response.statusCode) else {
            throw config.httpStatusError(response)
        }
        return try response.jsonValue()
    }

    private func googleBatchCreateURL(modelID requestedModelID: String? = nil) throws -> URL {
        let selectedModelID = requestedModelID ?? modelID
        let modelPath = selectedModelID.contains("/") ? selectedModelID : "models/\(selectedModelID)"
        return try requireURL("\(config.baseURL)/\(modelPath):batchGenerateContent")
    }

    private func googleBatchBaseOrigin() -> String {
        config.baseURL.hasSuffix("/v1beta")
            ? String(config.baseURL.dropLast("/v1beta".count))
            : config.baseURL
    }

    private func googleBatchResultStream(
        lines: [JSONValue],
        abortSignal: AIAbortSignal?
    ) -> AsyncThrowingStream<AIBatchItemResult<TextGenerationResult>, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for line in lines {
                        try Task.checkCancellation()
                        try abortSignal?.throwIfAborted()
                        continuation.yield(try googleBatchItemResult(line, providerID: providerID))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func googleBatchResultStream(
        body: AsyncThrowingStream<Data, Error>,
        abortSignal: AIAbortSignal?,
        url: String,
        maxLineBytes: Int,
        cancelBody: @escaping @Sendable () -> Void
    ) -> AsyncThrowingStream<AIBatchItemResult<TextGenerationResult>, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await aiForEachBatchResultLine(body, url: url, maxLineBytes: maxLineBytes, abortSignal: abortSignal, cancelBody: cancelBody) { line in
                        if googleBatchHasNonWhitespace(line) {
                            continuation.yield(try googleBatchItemResult(decodeJSONBody(line), providerID: providerID))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func googleProviderBatchResultStream(
        lines: [JSONValue],
        abortSignal: AIAbortSignal?
    ) -> AsyncThrowingStream<AIBatchV4ItemResult, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for line in lines {
                        try Task.checkCancellation()
                        try abortSignal?.throwIfAborted()
                        continuation.yield(try googleProviderBatchItemResult(line, providerID: providerID))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func googleProviderBatchResultStream(
        body: AsyncThrowingStream<Data, Error>,
        abortSignal: AIAbortSignal?,
        url: String,
        maxLineBytes: Int,
        cancelBody: @escaping @Sendable () -> Void
    ) -> AsyncThrowingStream<AIBatchV4ItemResult, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await aiForEachBatchResultLine(body, url: url, maxLineBytes: maxLineBytes, abortSignal: abortSignal, cancelBody: cancelBody) { line in
                        if googleBatchHasNonWhitespace(line) {
                            continuation.yield(try googleProviderBatchItemResult(decodeJSONBody(line), providerID: providerID))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

/// Provider-owned Google Batch V4 service. Text and image requests may share a
/// batch only when every item selects the model embedded in the endpoint.
public final class GoogleBatchProvider: AIBatchProvider, @unchecked Sendable {
    public let providerID: String
    public let supportedURLs: [String: [AISupportedURLPattern]]
    private let config: GoogleModelConfiguration

    init(config: GoogleModelConfiguration) {
        self.providerID = googleBatchProviderID(from: config.providerID)
        self.supportedURLs = googleBatchSupportedURLs(baseURL: config.baseURL)
        self.config = config
    }

    public func startBatch(
        _ options: AIBatchStartOptions<AIBatchRequest>
    ) async throws -> AIBatchStartResult {
        try await model.startProviderBatch(options)
    }

    public func getBatchStatus(_ options: AIBatchOperationOptions) async throws -> AIBatchStatus {
        try await model.getBatchStatus(options)
    }

    public func getBatchResults(
        _ options: AIBatchOperationOptions
    ) async throws -> AsyncThrowingStream<AIBatchV4ItemResult, Error> {
        try await model.getProviderBatchResults(options)
    }

    public func cancelBatch(_ options: AIBatchOperationOptions) async throws -> AIBatchCancelResult {
        try await model.cancelProviderBatch(options)
    }

    public func listBatches(_ options: AIBatchListOptions) async throws -> AIBatchListResult {
        try await model.listProviderBatches(options)
    }

    private var model: GoogleBatchLanguageModel {
        GoogleBatchLanguageModel(modelID: "batch", config: config)
    }
}

private let googleBatchInputFileMaxBytes = 2 * 1024 * 1024 * 1024
private let googleBatchInlineCreationMaxBytes = 20_000_000

private func googleBatchProviderID(from providerID: String) -> String {
    if providerID.hasSuffix(".generative-ai") {
        return String(providerID.dropLast(".generative-ai".count)) + ".batch"
    }
    return providerID + ".batch"
}

private func googleBatchSupportedURLs(baseURL: String) -> [String: [AISupportedURLPattern]] {
    let defaultFilesPrefix = "https://generativelanguage.googleapis.com/v1beta/files/"
    let configuredFilesPrefix = "\(withoutTrailingSlash(baseURL).lowercased())/files/"
    return [
        "*": [
            AISupportedURLPattern { value in
                let value = value.lowercased()
                return value.hasPrefix(defaultFilesPrefix) || value.hasPrefix(configuredFilesPrefix)
            },
            AISupportedURLPattern(googleBatchSupportedYouTubeURL)
        ]
    ]
}

private func googleBatchSupportedYouTubeURL(_ value: String) -> Bool {
    value.range(
        of: #"^https://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+(?:&[A-Za-z0-9_=&.\-]*)?$"#,
        options: .regularExpression
    ) != nil || value.range(
        of: #"^https://youtu\.be/[A-Za-z0-9_-]+(?:\?[A-Za-z0-9_=&.\-]*)?$"#,
        options: .regularExpression
    ) != nil
}

private func googleProviderBatchModelID(_ requests: [AIBatchRequest]) throws -> String {
    guard let first = requests.first else {
        throw AIError.invalidArgument(
            argument: "requests",
            message: "Google batches require at least one request."
        )
    }
    guard let modelID = first.modelID, !modelID.isEmpty else {
        throw AIError.invalidArgument(
            argument: "requests",
            message: "Google batch request \"\(first.id)\" must specify a modelID."
        )
    }
    for request in requests {
        guard let requestModelID = request.modelID, !requestModelID.isEmpty else {
            throw AIError.invalidArgument(
                argument: "requests",
                message: "Google batch request \"\(request.id)\" must specify a modelID."
            )
        }
        guard requestModelID == modelID else {
            throw AIError.invalidArgument(
                argument: "requests",
                message: "Google batches require every request to use the same model because the model is part of the batch endpoint."
            )
        }
    }
    return modelID
}

private func googlePrepareImageBatchRequest(
    _ request: ImageGenerationRequest,
    modelID: String
) throws -> GoogleGenerateContentPreparedCall {
    if request.mask != nil {
        throw AIError.invalidArgument(
            argument: "mask",
            message: "Google batches do not support mask-based image editing."
        )
    }
    if let count = request.count, count > 1 {
        throw AIError.invalidArgument(
            argument: "count",
            message: "Google batches do not support multiple images per request."
        )
    }

    var warnings: [AIWarning] = []
    if request.size != nil {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "size",
            message: "This model does not support the `size` option. Use `aspectRatio` instead."
        ))
    }

    let options = googleImageProviderOptions(from: request)
    var body = try GoogleGenerativeLanguageModel.imageGenerationContentBody(
        prompt: request.prompt,
        aspectRatio: nil,
        files: request.files
    )
    var generationConfig = body["generationConfig"]?.objectValue ?? [:]
    googleApplyProviderGenerationOptions(options, to: &generationConfig)
    generationConfig["responseModalities"] = .array(["IMAGE"])
    var imageConfig = options["imageConfig"]?.objectValue ?? [:]
    if let aspectRatio = request.aspectRatio {
        imageConfig["aspectRatio"] = .string(aspectRatio)
    }
    if !imageConfig.isEmpty {
        generationConfig["imageConfig"] = .object(imageConfig)
    }
    if let seed = request.seed {
        generationConfig["seed"] = .number(Double(seed))
    }
    body["generationConfig"] = .object(generationConfig)
    body.merge(googleTopLevelGenerateContentOptions(options)) { _, new in new }
    body.merge(googleExtraBodyWithoutToolChoice(options).filter { $0.key != "googleSearch" }) { _, new in new }

    if let googleSearch = options["googleSearch"] {
        let preparedTools = try googlePrepareTools(
            from: [
                "google.google_search": GoogleTools.googleSearch(
                    searchTypes: googleSearch["searchTypes"],
                    timeRangeFilter: googleSearch["timeRangeFilter"]
                )
            ],
            toolChoice: nil,
            modelID: modelID,
            isVertexProvider: false
        )
        if let preparedTools, !preparedTools.tools.isEmpty {
            body["tools"] = .array(preparedTools.tools)
        }
        if let preparedTools {
            warnings.append(contentsOf: preparedTools.warnings)
        }
    }

    return GoogleGenerateContentPreparedCall(
        body: .object(body),
        warnings: warnings,
        headers: [:],
        toolNameMapping: createToolNameMapping(tools: [:], providerToolNames: [:])
    )
}

private func googleBatchCreationBody(
    displayName: String,
    webhookURL: String?,
    inputConfig: JSONValue
) -> JSONValue {
    var batch: [String: JSONValue] = [
        "displayName": .string(displayName),
        "inputConfig": inputConfig
    ]
    if let webhookURL {
        batch["webhookConfig"] = .object(["uris": .array([.string(webhookURL)])])
    }
    return .object(["batch": .object(batch)])
}

private func googleBatchJSONLine(_ value: JSONValue) throws -> Data {
    var data = try encodeJSONBody(value)
    data.append(0x0a)
    return data
}

private func googleBatchHeaders(
    _ operationHeaders: [String: String],
    config: GoogleModelConfiguration
) -> [String: String] {
    var headers = prepareHeaders(operationHeaders, defaultHeaders: config.headers)
    headers["user-agent"] = headers["user-agent"] ?? userAgent(config.providerID)
    return headers
}

private func googleBatchOperationName(_ operation: JSONValue) throws -> String {
    guard let name = operation["name"]?.stringValue, !name.isEmpty else {
        throw AIError.invalidResponse(provider: "google.generative-ai", message: "Invalid Google batch operation response.")
    }
    return name
}

private func googleBatchStatus(_ operation: JSONValue) -> AIBatchStatus {
    let rawStatus = operation["metadata"]?["state"]?.stringValue
    let errorValue = operation["error"]
    let error = googleBatchRPCError(errorValue, fallbackMessage: "Google batch failed.")
    let status: AIBatchLifecycleStatus
    if error != nil {
        status = .failed
    } else if let rawStatus {
        let normalized: String
        if rawStatus.hasPrefix("BATCH_STATE_") {
            normalized = String(rawStatus.dropFirst("BATCH_STATE_".count))
        } else if rawStatus.hasPrefix("JOB_STATE_") {
            normalized = String(rawStatus.dropFirst("JOB_STATE_".count))
        } else {
            normalized = rawStatus
        }
        switch normalized {
        case "SUCCEEDED": status = .completed
        case "FAILED", "CANCELLED", "EXPIRED": status = .failed
        default: status = .pending
        }
    } else {
        status = operation["done"]?.boolValue == true ? .completed : .pending
    }
    let stats = operation["metadata"]?["batchStats"]
    let total = googleBatchCount(stats?["requestCount"])
    let completed = googleBatchCount(stats?["successfulRequestCount"]) ?? 0
    let failed = googleBatchCount(stats?["failedRequestCount"]) ?? 0
    let pending = googleBatchCount(stats?["pendingRequestCount"]) ?? 0
    return AIBatchStatus(
        status: status,
        rawStatus: rawStatus,
        requestCounts: normalizedBatchRequestCounts(
            total: total,
            pending: pending,
            completed: completed,
            failed: failed
        ),
        error: error,
        createdAt: operation["metadata"]?["createTime"]?.stringValue
    )
}

private func googleBatchCount(_ value: JSONValue?) -> Int? {
    if let count = normalizedBatchJSONInteger(value) { return count }
    guard let text = value?.stringValue,
          !text.isEmpty,
          text.allSatisfy(\.isNumber),
          let count = Int(text),
          (0...aiBatchMaximumSafeInteger).contains(count) else {
        return nil
    }
    return count
}

private func googleBatchRPCError(_ value: JSONValue?, fallbackMessage: String) -> AIBatchError? {
    guard let value, value != .null else { return nil }
    let code: String?
    if let string = value["code"]?.stringValue {
        code = string
    } else if let number = value["code"]?.doubleValue {
        code = Int(exactly: number).map(String.init) ?? String(number)
    } else {
        code = nil
    }
    return AIBatchError(
        message: value["message"]?.stringValue ?? fallbackMessage,
        type: value["status"]?.stringValue,
        code: code
    )
}

private func googleBatchItemResult(
    _ line: JSONValue,
    providerID: String
) throws -> AIBatchItemResult<TextGenerationResult> {
    guard line.objectValue != nil,
          let id = line["key"]?.stringValue,
          !id.isEmpty else {
        throw googleBatchInvalidResultKeyError(providerID: providerID)
    }
    if let error = line["error"], error != .null {
        guard let object = error.objectValue,
              googleBatchRPCErrorFieldIsValid(object["code"], allowsNumber: true),
              googleBatchRPCErrorFieldIsValid(object["message"], allowsNumber: false),
              googleBatchRPCErrorFieldIsValid(object["status"], allowsNumber: false) else {
            throw AIError.invalidResponse(
                provider: providerID,
                message: "Invalid Google batch result row: error must match the Google RPC status shape."
            )
        }
    }
    if let error = googleBatchRPCError(line["error"], fallbackMessage: "Google batch request failed.") {
        if error.type == "CANCELLED" || error.code == "1" {
            return .cancelled(id: id, error: error)
        }
        return .failed(id: id, error: error)
    }
    guard let response = line["response"], response != .null else {
        return .failed(
            id: id,
            error: AIBatchError(
                message: "Google returned a batch result without a response or error.",
                code: "invalid_batch_result"
            )
        )
    }
    guard let candidates = response["candidates"]?.arrayValue, !candidates.isEmpty else {
        let feedback = response["promptFeedback"]
        let blockReason = feedback?["blockReason"]?.stringValue
        let metadata: [String: JSONValue] = feedback == nil
            ? [:]
            : ["google": .object(["promptFeedback": .object([
                "blockReason": blockReason.map(JSONValue.string) ?? .null
            ])])]
        return .failed(
            id: id,
            error: AIBatchError(
                message: blockReason.map { "Google blocked the batch request (\($0))." }
                    ?? "Google returned a batch response without any candidates.",
                type: blockReason,
                code: blockReason == nil ? "invalid_response" : "prompt_blocked"
            ),
            providerMetadata: metadata
        )
    }
    guard let parts = candidates[0]["content"]?["parts"]?.arrayValue else {
        return googleInvalidBatchResult(id: id)
    }

    var content: [AIResultContentPart] = []
    var lastCodeCallID: String?
    var lastServerCallID: String?
    for (index, part) in parts.enumerated() {
        if part["inlineData"] != nil || part["fileData"] != nil {
            return .failed(
                id: id,
                error: AIBatchError(
                    message: "Google returned a \"file\" content block, but that content is not supported in AI SDK text batches.",
                    code: "unsupported_content"
                )
            )
        }
        if let executableCode = part["executableCode"] {
            guard executableCode["code"]?.stringValue != nil else { return googleInvalidBatchResult(id: id) }
            let callID = "google-code-execution-\(index)"
            lastCodeCallID = callID
            content.append(.toolCall(AIToolCall(
                id: callID,
                name: "code_execution",
                arguments: googleGenerateContentArguments(executableCode),
                providerExecuted: true,
                rawValue: part
            )))
            continue
        }
        if let result = part["codeExecutionResult"] {
            guard let callID = lastCodeCallID else { return googleInvalidBatchResult(id: id) }
            content.append(.toolResult(AIToolResult(
                toolCallID: callID,
                toolName: "code_execution",
                result: googleCodeExecutionResultJSON(result),
                providerExecuted: true
            )))
            continue
        }
        if let textValue = part["text"] {
            guard let text = textValue.stringValue else { return googleInvalidBatchResult(id: id) }
            let metadata = googleThoughtSignatureProviderMetadata(from: part)
            if text.isEmpty {
                if !metadata.isEmpty, let lastIndex = content.indices.last {
                    content[lastIndex] = googleBatchMergingProviderMetadata(
                        metadata,
                        into: content[lastIndex]
                    )
                }
            } else {
                content.append(part["thought"]?.boolValue == true
                    ? .reasoning(text, providerMetadata: metadata)
                    : .text(text, providerMetadata: metadata))
            }
            continue
        }
        if let functionCall = part["functionCall"] {
            guard let name = functionCall["name"]?.stringValue else { return googleInvalidBatchResult(id: id) }
            content.append(.toolCall(AIToolCall(
                id: functionCall["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? "tool-call-\(index)",
                name: name,
                arguments: googleGenerateContentArguments(functionCall["args"]),
                providerMetadata: googleThoughtSignatureProviderMetadata(from: part),
                rawValue: part
            )))
            continue
        }
        if let toolCall = part["toolCall"] {
            guard let toolType = toolCall["toolType"]?.stringValue else { return googleInvalidBatchResult(id: id) }
            let callID = toolCall["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
                ?? "google-server-tool-\(index)"
            lastServerCallID = callID
            content.append(.toolCall(AIToolCall(
                id: callID,
                name: "server:\(toolType)",
                arguments: googleGenerateContentArguments(toolCall["args"]),
                providerExecuted: true,
                dynamic: true,
                providerMetadata: googleServerToolProviderMetadata(id: callID, type: toolType, part: part),
                rawValue: part
            )))
            continue
        }
        if let toolResponse = part["toolResponse"] {
            guard let toolType = toolResponse["toolType"]?.stringValue else { return googleInvalidBatchResult(id: id) }
            let callID = lastServerCallID
                ?? toolResponse["id"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
                ?? "google-server-tool-response-\(index)"
            content.append(.toolResult(AIToolResult(
                toolCallID: callID,
                toolName: "server:\(toolType)",
                result: toolResponse["response"] ?? .object([:]),
                dynamic: true,
                providerExecuted: true,
                providerMetadata: googleServerToolProviderMetadata(id: callID, type: toolType, part: part)
            )))
            lastServerCallID = nil
            continue
        }
        return googleInvalidBatchResult(id: id)
    }

    content.append(contentsOf: googleGenerateContentSources(from: response).map(AIResultContentPart.source))
    let text = content.compactMap { part -> String? in
        guard case let .text(value, _) = part else { return nil }
        return value
    }.joined()
    let reasoning = content.compactMap { part -> String? in
        guard case let .reasoning(value, _) = part else { return nil }
        return value
    }.joined()
    let hasClientToolCall = content.contains { part in
        guard case let .toolCall(call) = part else { return false }
        return !call.providerExecuted
    }
    return .succeeded(
        id: id,
        result: TextGenerationResult(
            text: text,
            content: content,
            reasoning: reasoning,
            finishReason: googleGenerateContentFinishReason(
                candidates[0]["finishReason"]?.stringValue,
                hasToolCalls: hasClientToolCall
            ),
            usage: googleGenerateContentUsage(from: response),
            providerMetadata: googleGenerateContentProviderMetadata(from: response),
            rawValue: response,
            responseMetadata: AIResponseMetadata(id: response["responseId"]?.stringValue)
        )
    )
}

private func googleProviderBatchItemResult(
    _ line: JSONValue,
    providerID: String
) throws -> AIBatchV4ItemResult {
    guard line.objectValue != nil,
          let id = line["key"]?.stringValue,
          !id.isEmpty else {
        throw googleBatchInvalidResultKeyError(providerID: providerID)
    }
    if let response = line["response"],
       response != .null,
       let imageResult = googleProviderBatchImageResult(response) {
        return .image(.succeeded(id: id, result: imageResult))
    }
    return .text(try googleBatchItemResult(line, providerID: providerID))
}

private func googleProviderBatchImageResult(_ response: JSONValue) -> ImageGenerationResult? {
    // GenerateContent normalization is defined in terms of the first candidate.
    // Thought-bearing inline data becomes a reasoning-file, not an image result.
    let images: [String] = (response["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? [])
        .compactMap { part -> String? in
            guard part["thought"]?.boolValue != true,
                  let inlineData = part["inlineData"]?.objectValue,
                  let mediaType = inlineData["mimeType"]?.stringValue,
                  mediaType.lowercased().hasPrefix("image/"),
                  let data = inlineData["data"]?.stringValue else {
                return nil
            }
            return data
        }
    guard !images.isEmpty else { return nil }

    let imageMetadata: [String: JSONValue] = [
        "google": .object(["images": .array(images.map { _ in JSONValue.object([:]) })])
    ]
    return ImageGenerationResult(
        urls: [],
        base64Images: images,
        rawValue: response,
        warnings: [],
        usage: googleGenerateContentUsage(from: response),
        providerMetadata: googleBatchMergedProviderMetadata(
            googleGenerateContentProviderMetadata(from: response),
            imageMetadata
        ),
        responseMetadata: AIResponseMetadata(
            id: response["responseId"]?.stringValue,
            timestamp: Date(),
            modelID: response["modelVersion"]?.stringValue
        )
    )
}

private func googleBatchInvalidResultKeyError(providerID: String) -> AIError {
    AIError.invalidResponse(
        provider: providerID,
        message: "Invalid Google batch result row: expected a non-empty string key."
    )
}

private func googleBatchRPCErrorFieldIsValid(_ value: JSONValue?, allowsNumber: Bool) -> Bool {
    guard let value else { return true }
    if value == .null || value.stringValue != nil { return true }
    return allowsNumber && value.doubleValue != nil
}

private func googleBatchMergingProviderMetadata(
    _ metadata: [String: JSONValue],
    into part: AIResultContentPart
) -> AIResultContentPart {
    switch part {
    case let .text(text, existing):
        return .text(text, providerMetadata: googleBatchMergedProviderMetadata(existing, metadata))
    case let .citedText(text, citations, existing):
        return .text(text, citations: citations, providerMetadata: googleBatchMergedProviderMetadata(existing, metadata))
    case let .reasoning(text, existing):
        return .reasoning(text, providerMetadata: googleBatchMergedProviderMetadata(existing, metadata))
    case let .source(source):
        var source = source
        source.providerMetadata = googleBatchMergedProviderMetadata(source.providerMetadata, metadata)
        return .source(source)
    case let .toolCall(call):
        var call = call
        call.providerMetadata = googleBatchMergedProviderMetadata(call.providerMetadata, metadata)
        return .toolCall(call)
    case let .toolResult(result):
        var result = result
        result.providerMetadata = googleBatchMergedProviderMetadata(result.providerMetadata, metadata)
        return .toolResult(result)
    case .file, .reasoningFile, .custom, .toolApprovalRequest, .toolApprovalResponse:
        return part
    }
}

private func googleBatchMergedProviderMetadata(
    _ existing: [String: JSONValue],
    _ additional: [String: JSONValue]
) -> [String: JSONValue] {
    var merged = existing
    for (namespace, value) in additional {
        if var existingNamespace = merged[namespace]?.objectValue,
           let additionalNamespace = value.objectValue {
            existingNamespace.merge(additionalNamespace) { _, new in new }
            merged[namespace] = .object(existingNamespace)
        } else {
            merged[namespace] = value
        }
    }
    return merged
}

private func googleInvalidBatchResult(id: String) -> AIBatchItemResult<TextGenerationResult> {
    .failed(
        id: id,
        error: AIBatchError(
            message: "Google returned an invalid GenerateContent batch result.",
            code: "invalid_response"
        )
    )
}

private func googleBatchPathComponent(_ value: String) -> String {
    let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'()")
    return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
}

private func googleBatchHasNonWhitespace(_ data: Data) -> Bool {
    data.contains { byte in
        byte != 0x20 && byte != 0x09 && byte != 0x0a && byte != 0x0d
    }
}