import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class GoogleEmbeddingModel: EmbeddingModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    public var maxEmbeddingsPerCall: Int? { 100 }
    public var providerOptionsTransformer: AIEmbeddingProviderOptionsTransformer? {
        { context in
            guard var google = context.providerOptions["google"]?.objectValue,
                  let content = google["content"]?.arrayValue else {
                return context.providerOptions
            }
            guard content.count == context.values.count else {
                throw AIError.invalidArgument(
                    argument: "providerOptions.google.content",
                    message: "The number of multimodal content entries (\(content.count)) must match the number of values (\(context.values.count))."
                )
            }
            guard context.startIndex >= 0,
                  context.endIndex >= context.startIndex,
                  context.endIndex <= content.count else {
                throw AIError.invalidArgument(
                    argument: "providerOptions.google.content",
                    message: "The automatic embedding batch range is outside the Google content array."
                )
            }
            var providerOptions = context.providerOptions
            google["content"] = .array(Array(content[context.startIndex..<context.endIndex]))
            providerOptions["google"] = .object(google)
            return providerOptions
        }
    }
    private let config: GoogleModelConfiguration

    init(modelID: String, config: GoogleModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
    }

    public func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResult {
        let embeddingLimit = maxEmbeddingsPerCall ?? 100
        guard request.values.count <= embeddingLimit else {
            throw AITooManyEmbeddingValuesForCallError(
                provider: providerID,
                modelID: modelID,
                maxEmbeddingsPerCall: embeddingLimit,
                values: request.values
            )
        }
        let options = googleEmbeddingProviderOptions(from: request)
        let multimodalContent = options["content"]?.arrayValue
        if let multimodalContent, multimodalContent.count != request.values.count {
            throw AIError.invalidArgument(
                argument: "providerOptions.google.content",
                message: "The number of multimodal content entries (\(multimodalContent.count)) must match the number of values (\(request.values.count))."
            )
        }
        let path = request.values.count == 1
            ? "/models/\(modelID):embedContent"
            : "/models/\(modelID):batchEmbedContents"
        let body: JSONValue
        if request.values.count == 1 {
            var object: [String: JSONValue] = [
                "model": .string("models/\(modelID)"),
                "content": .object(["parts": .array(googleEmbeddingParts(text: request.values[0], content: multimodalContent?.first))])
            ]
            if let outputDimensionality = options["outputDimensionality"] ?? request.dimensions.map({ JSONValue.number(Double($0)) }) {
                object["outputDimensionality"] = outputDimensionality
            }
            if let taskType = options["taskType"] {
                object["taskType"] = taskType
            }
            body = .object(object)
        } else {
            body = .object([
                "requests": .array(request.values.enumerated().map { index, value in
                    var object: [String: JSONValue] = [
                        "model": .string("models/\(modelID)"),
                        "content": .object([
                            "role": .string("user"),
                            "parts": .array(googleEmbeddingParts(text: value, content: multimodalContent?[index]))
                        ])
                    ]
                    if let outputDimensionality = options["outputDimensionality"] ?? request.dimensions.map({ JSONValue.number(Double($0)) }) {
                        object["outputDimensionality"] = outputDimensionality
                    }
                    if let taskType = options["taskType"] {
                        object["taskType"] = taskType
                    }
                    return .object(object)
                })
            ])
        }

        let response = try await config.sendJSONResponse(path: path, modelID: modelID, body: body, headers: request.headers, abortSignal: request.abortSignal)
        let raw = response.json
        let embeddings: [[Double]]
        if let values = raw["embedding"]?["values"]?.arrayValue {
            embeddings = [values.compactMap(\.doubleValue)]
        } else {
            embeddings = raw["embeddings"]?.arrayValue?.compactMap { item in
                item["values"]?.arrayValue?.compactMap(\.doubleValue)
            } ?? []
        }
        return EmbeddingResult(
            embeddings: embeddings,
            rawValue: raw,
            requestMetadata: AIRequestMetadata(body: body, headers: request.headers),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }
}

public final class GoogleImageGenerationModel: ImageModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let config: GoogleModelConfiguration

    init(modelID: String, config: GoogleModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
    }

    public func generateImage(_ request: ImageGenerationRequest) async throws -> ImageGenerationResult {
        if modelID.hasPrefix("gemini-") {
            return try await generateGeminiImage(request)
        }
        return try await generateImagen(request)
    }

    private func generateImagen(_ request: ImageGenerationRequest) async throws -> ImageGenerationResult {
        if !request.files.isEmpty {
            throw AIError.invalidArgument(argument: "files", message: "Google Generative AI does not support image editing with Imagen models. Use Google Vertex AI (@ai-sdk/google-vertex) for image editing capabilities.")
        }
        if request.mask != nil {
            throw AIError.invalidArgument(argument: "mask", message: "Google Generative AI does not support image editing with masks. Use Google Vertex AI (@ai-sdk/google-vertex) for image editing capabilities.")
        }
        var warnings: [AIWarning] = []
        if request.size != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "size",
                message: "This model does not support the `size` option. Use `aspectRatio` instead."
            ))
        }
        if request.seed != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "seed",
                message: "This model does not support the `seed` option through this provider."
            ))
        }
        let options = googleImageProviderOptions(from: request)
        if options["googleSearch"] != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "googleSearch",
                message: "Google Search grounding is only supported on Gemini image models."
            ))
        }
        let body = JSONValue.object([
            "instances": .array([.object(["prompt": .string(request.prompt)])]),
            "parameters": .object(googleImagenParameters(for: request, options: options))
        ])
        let response = try await config.sendJSONResponse(path: "/models/\(modelID):predict", modelID: modelID, body: body, headers: request.headers, abortSignal: request.abortSignal)
        let raw = response.json
        let images = raw["predictions"]?.arrayValue?.compactMap { $0["bytesBase64Encoded"]?.stringValue } ?? []
        guard !images.isEmpty else {
            throw AIError.invalidResponse(provider: providerID, message: "No Imagen predictions found.")
        }
        return ImageGenerationResult(
            urls: [],
            base64Images: images,
            rawValue: raw,
            warnings: warnings,
            providerMetadata: ["google": .object([
                "images": .array(images.map { _ in .object([:]) })
            ])],
            requestMetadata: imageGenerationRequestMetadata(request, body: body),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    private func editImagen(_ request: ImageGenerationRequest) async throws -> ImageGenerationResult {
        let options = googleImageProviderOptions(from: request.extraBody)
        let edit = options["edit"]?.objectValue ?? [:]
        var referenceImages: [JSONValue] = []

        for (index, file) in request.files.enumerated() {
            referenceImages.append(.object([
                "referenceType": .string("REFERENCE_TYPE_RAW"),
                "referenceId": .number(Double(index + 1)),
                "referenceImage": .object([
                    "bytesBase64Encoded": .string(try googleImageEditBase64(file))
                ])
            ]))
        }

        if let mask = request.mask {
            var maskImageConfig: [String: JSONValue] = [
                "maskMode": edit["maskMode"] ?? .string("MASK_MODE_USER_PROVIDED")
            ]
            if let dilation = edit["maskDilation"] {
                maskImageConfig["dilation"] = dilation
            }
            referenceImages.append(.object([
                "referenceType": .string("REFERENCE_TYPE_MASK"),
                "referenceId": .number(Double(request.files.count + 1)),
                "referenceImage": .object([
                    "bytesBase64Encoded": .string(try googleImageEditBase64(mask))
                ]),
                "maskImageConfig": .object(maskImageConfig)
            ]))
        }

        let body = JSONValue.object([
            "instances": .array([.object([
                "prompt": .string(request.prompt),
                "referenceImages": .array(referenceImages)
            ])]),
            "parameters": .object(googleImagenEditParameters(for: request, options: options, edit: edit))
        ])
        let response = try await config.sendJSONResponse(path: "/models/\(modelID):predict", modelID: modelID, body: body, headers: request.headers)
        let raw = response.json
        let images = raw["predictions"]?.arrayValue?.compactMap { $0["bytesBase64Encoded"]?.stringValue } ?? []
        guard !images.isEmpty else {
            throw AIError.invalidResponse(provider: providerID, message: "No Imagen edit predictions found.")
        }
        return ImageGenerationResult(
            urls: [],
            base64Images: images,
            rawValue: raw,
            requestMetadata: imageGenerationRequestMetadata(request, body: body),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    private func generateGeminiImage(_ request: ImageGenerationRequest) async throws -> ImageGenerationResult {
        if request.mask != nil {
            throw AIError.invalidArgument(argument: "mask", message: "Gemini image models do not support mask-based image editing.")
        }
        if let count = request.count, count > 1 {
            throw AIError.invalidArgument(argument: "count", message: "Gemini image models do not support generating a set number of images per call. Use n=1 or omit the n parameter.")
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
            aspectRatio: googleAspectRatio(from: request),
            files: request.files
        )
        var generationConfig = body["generationConfig"]?.objectValue ?? [:]
        googleApplyProviderGenerationOptions(options, to: &generationConfig)
        generationConfig["responseModalities"] = ["IMAGE"]
        if let seed = request.seed { generationConfig["seed"] = .number(Double(seed)) }
        if let aspectRatio = googleAspectRatio(from: request) {
            var imageConfig = generationConfig["imageConfig"]?.objectValue ?? [:]
            imageConfig["aspectRatio"] = .string(aspectRatio)
            generationConfig["imageConfig"] = .object(imageConfig)
        }
        body["generationConfig"] = .object(generationConfig)
        body.merge(googleTopLevelGenerateContentOptions(options)) { _, new in new }
        body.merge(googleExtraBodyWithoutToolChoice(options).filter { $0.key != "googleSearch" }) { _, new in new }
        if let googleSearch = options["googleSearch"] {
            let preparedTools = try googlePrepareTools(
                from: ["google.google_search": GoogleTools.googleSearch(searchTypes: googleSearch["searchTypes"], timeRangeFilter: googleSearch["timeRangeFilter"])],
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
        let response = try await config.sendJSONResponse(path: "/models/\(modelID):generateContent", modelID: modelID, body: .object(body), headers: request.headers, abortSignal: request.abortSignal)
        let raw = response.json
        let parts = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []
        let images = parts.compactMap { part in
            part["inlineData"]?["data"]?.stringValue
        }
        let finishReason = googleGenerateContentFinishReason(from: raw, hasToolCalls: false)
        guard !images.isEmpty || finishReason == "content-filter" else {
            throw AIError.invalidResponse(provider: providerID, message: "No Gemini image inlineData found.")
        }
        return ImageGenerationResult(
            urls: [],
            base64Images: images,
            rawValue: raw,
            warnings: warnings,
            usage: googleGenerateContentUsage(from: raw),
            providerMetadata: googleGenerateContentProviderMetadata(from: raw).merging(["google": .object([
                "finishReason": raw["candidates"]?[0]?["finishReason"] ?? .null,
                "images": .array(images.map { _ in .object([:]) })
            ])]) { old, new in
                var object = old.objectValue ?? [:]
                object.merge(new.objectValue ?? [:]) { _, incoming in incoming }
                return .object(object)
            },
            requestMetadata: imageGenerationRequestMetadata(request, body: .object(body)),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID),
            isRetryable: finishReason == "content-filter" ? false : nil
        )
    }
}

public final class GoogleSpeechModel: SpeechModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let config: GoogleModelConfiguration

    init(modelID: String, config: GoogleModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
    }

    public func speak(_ request: SpeechRequest) async throws -> SpeechResult {
        let prepared = try googlePrepareSpeechRequest(
            request,
            modelID: modelID,
            options: googleSpeechProviderOptions(from: request)
        )
        let response = try await config.sendJSONResponse(
            path: "/models/\(modelID):generateContent",
            modelID: modelID,
            body: prepared.body,
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let raw = response.json
        let inlineData = googleSpeechInlineData(from: raw)
        let mimeType = inlineData.mimeType
        let sampleRate = googleSpeechSampleRate(from: mimeType) ?? 24_000
        let bytes = inlineData.data.flatMap { Data(base64Encoded: $0) } ?? Data()
        let shouldWrapPCM = prepared.outputFormat == "AUDIO_WAV"
            && googleSpeechMimeTypeIsPCM(mimeType, usesStructuredSpeech: prepared.usesStructuredSpeech)
            && !bytes.isEmpty
        let audio = shouldWrapPCM ? googleSpeechWAVData(fromPCM: bytes, sampleRate: sampleRate) : bytes
        var warnings = prepared.warnings
        if prepared.outputFormat == "AUDIO_L16", !prepared.usesStructuredSpeech, !bytes.isEmpty {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "outputFormat",
                message: "Returning raw PCM audio (signed 16-bit little-endian, mono, \(sampleRate) Hz). These bytes have no container header and are not directly playable."
            ))
        }

        return SpeechResult(
            audio: audio,
            contentType: googleSpeechContentType(
                mimeType: mimeType,
                outputFormat: prepared.outputFormat,
                wrappedPCMAsWAV: shouldWrapPCM
            ),
            warnings: warnings,
            providerMetadata: ["google": .object([
                "sampleRate": .number(Double(sampleRate)),
                "mimeType": mimeType.map(JSONValue.string) ?? .null
            ])],
            requestMetadata: AIRequestMetadata(body: prepared.body, headers: request.headers),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID),
            usage: raw["usageMetadata"]?.objectValue
        )
    }
}

public final class GoogleVideoGenerationModel: VideoModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let config: GoogleModelConfiguration

    init(modelID: String, config: GoogleModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
    }

    public func generateVideo(_ request: VideoGenerationRequest) async throws -> VideoGenerationResult {
        let options = googleVideoProviderOptions(from: request)
        let preparedInstance = googleVideoInstance(for: request, options: options)
        let body = JSONValue.object([
            "instances": .array([.object(preparedInstance.instance)]),
            "parameters": .object(googleVideoParameters(for: request, options: options))
        ])
        let operation = try await config.sendJSON(path: "/models/\(modelID):predictLongRunning", modelID: modelID, body: body, headers: request.headers, abortSignal: request.abortSignal)
        guard let operationName = operation["name"]?.stringValue else {
            throw AIError.invalidResponse(provider: providerID, message: "Google video create response did not contain an operation name.")
        }

        let finalResponse = try await pollOperationResponse(named: operationName, headers: request.headers, timeoutNanoseconds: googlePollTimeout(options), intervalNanoseconds: googlePollInterval(options), abortSignal: request.abortSignal)
        let finalOperation = finalResponse.json
        let samples = finalOperation["response"]?["generateVideoResponse"]?["generatedSamples"]?.arrayValue ?? []
        let urls = samples.compactMap { sample in
            sample["video"]?["uri"]?.stringValue.map(googleVideoURLWithAPIKey)
        }
        guard !urls.isEmpty else {
            throw AIError.invalidResponse(provider: providerID, message: "Google video response did not contain generatedSamples video URIs.")
        }
        return VideoGenerationResult(
            urls: urls,
            operationID: operationName,
            rawValue: finalOperation,
            warnings: preparedInstance.warnings,
            providerMetadata: ["google": .object([
                "videos": .array(samples.compactMap { sample in
                    guard let uri = sample["video"]?["uri"]?.stringValue else { return nil }
                    return .object(["uri": .string(uri)])
                })
            ])],
            requestMetadata: videoGenerationRequestMetadata(request, body: body),
            responseMetadata: aiResponseMetadata(from: finalOperation, response: finalResponse.response, modelID: modelID)
        )
    }

    private func pollOperationResponse(named operationName: String, headers: [String: String], timeoutNanoseconds: UInt64, intervalNanoseconds: UInt64, abortSignal: AIAbortSignal?) async throws -> (json: JSONValue, response: AIHTTPResponse) {
        let started = DispatchTime.now().uptimeNanoseconds
        var latest: JSONValue?
        repeat {
            let response = try await config.transport.send(AIHTTPRequest(
                method: "GET",
                url: try requireURL("\(config.baseURL)/\(operationName)"),
                headers: config.headers.mergingHeaders(headers),
                abortSignal: abortSignal
            ))
            guard (200..<300).contains(response.statusCode) else {
                throw config.httpStatusError( response)
            }
            let raw = try response.jsonValue()
            latest = raw
            if raw["error"] != nil {
                throw AIError.invalidResponse(provider: providerID, message: raw["error"]?["message"]?.stringValue ?? "Google video generation failed.")
            }
            if raw["done"]?.boolValue == true {
                return (raw, response)
            }
            if intervalNanoseconds > 0 {
                try await sleepWithAbortSignal(nanoseconds: intervalNanoseconds, abortSignal: abortSignal)
            }
        } while DispatchTime.now().uptimeNanoseconds - started < timeoutNanoseconds

        throw AIError.invalidResponse(provider: providerID, message: "Google video generation timed out. Last response: \(latest.map(String.init(describing:)) ?? "none")")
    }

    private func googleVideoURLWithAPIKey(_ uri: String) -> String {
        guard let key = config.headers["x-goog-api-key"], !key.isEmpty else { return uri }
        guard isSameOrigin(uri, config.baseURL) else { return uri }
        return "\(uri)\(uri.contains("?") ? "&" : "?")key=\(key)"
    }
}

private enum GoogleSpeechOutputFormat {
    case wav
    case pcm
}

private func googleSpeechInlineData(from raw: JSONValue) -> (mimeType: String?, data: String?) {
    let parts = raw["candidates"]?[0]?["content"]?["parts"]?.arrayValue ?? []
    for part in parts {
        if let inlineData = part["inlineData"]?.objectValue {
            return (
                inlineData["mimeType"]?.stringValue,
                inlineData["data"]?.stringValue
            )
        }
    }
    return (nil, nil)
}

private func googleSpeechSampleRate(from mimeType: String?) -> Int? {
    guard let mimeType else { return nil }
    for component in mimeType.split(separator: ";") {
        let trimmed = component.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("rate=") else { continue }
        return Int(trimmed.dropFirst("rate=".count))
    }
    return nil
}

private func googleSpeechWAVData(fromPCM pcm: Data, sampleRate: Int) -> Data {
    let channelCount = 1
    let bitsPerSample = 16
    let byteRate = sampleRate * channelCount * bitsPerSample / 8
    let blockAlign = channelCount * bitsPerSample / 8
    var data = Data()
    data.append(Data("RIFF".utf8))
    googleSpeechAppendUInt32LE(UInt32(36 + pcm.count), to: &data)
    data.append(Data("WAVE".utf8))
    data.append(Data("fmt ".utf8))
    googleSpeechAppendUInt32LE(16, to: &data)
    googleSpeechAppendUInt16LE(1, to: &data)
    googleSpeechAppendUInt16LE(UInt16(channelCount), to: &data)
    googleSpeechAppendUInt32LE(UInt32(sampleRate), to: &data)
    googleSpeechAppendUInt32LE(UInt32(byteRate), to: &data)
    googleSpeechAppendUInt16LE(UInt16(blockAlign), to: &data)
    googleSpeechAppendUInt16LE(UInt16(bitsPerSample), to: &data)
    data.append(Data("data".utf8))
    googleSpeechAppendUInt32LE(UInt32(pcm.count), to: &data)
    data.append(pcm)
    return data
}

private func googleSpeechAppendUInt16LE(_ value: UInt16, to data: inout Data) {
    data.append(UInt8(value & 0xff))
    data.append(UInt8((value >> 8) & 0xff))
}

private func googleSpeechAppendUInt32LE(_ value: UInt32, to data: inout Data) {
    data.append(UInt8(value & 0xff))
    data.append(UInt8((value >> 8) & 0xff))
    data.append(UInt8((value >> 16) & 0xff))
    data.append(UInt8((value >> 24) & 0xff))
}