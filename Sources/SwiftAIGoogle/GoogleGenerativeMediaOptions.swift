import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleImagenParameters(for request: ImageGenerationRequest, options: [String: JSONValue]) -> [String: JSONValue] {
    var parameters: [String: JSONValue] = ["sampleCount": .number(Double(request.count ?? 1))]
    if let aspectRatio = request.aspectRatio ?? options["aspectRatio"]?.stringValue ?? request.extraBody["aspectRatio"]?.stringValue {
        parameters["aspectRatio"] = .string(aspectRatio)
    } else {
        parameters["aspectRatio"] = .string("1:1")
    }
    parameters.merge(googleOptionsWithoutPoll(options, excluding: ["googleSearch", "edit", "aspectRatio"])) { _, new in new }
    return parameters
}

func googleImagenEditParameters(for request: ImageGenerationRequest, options: [String: JSONValue], edit: [String: JSONValue]) -> [String: JSONValue] {
    var parameters = googleImagenParameters(for: request, options: options)
    parameters["editMode"] = edit["mode"] ?? .string("EDIT_MODE_INPAINT_INSERTION")
    if let baseSteps = edit["baseSteps"] {
        parameters["editConfig"] = .object(["baseSteps": baseSteps])
    }
    return parameters
}

func googleImageProviderOptions(from request: ImageGenerationRequest) -> [String: JSONValue] {
    var options = googleImageProviderOptions(from: request.extraBody)
    if let google = request.providerOptions["google"]?.objectValue {
        options.merge(google) { _, new in new }
    }
    return options
}

func googleImageProviderOptions(from extraBody: [String: JSONValue]) -> [String: JSONValue] {
    if let google = extraBody["google"]?.objectValue {
        return google
    }
    return extraBody.filter { $0.key != "google" && $0.key != "googleVertex" && $0.key != "vertex" }
}

func googleImageEditBase64(_ file: ImageInputFile) throws -> String {
    if file.url != nil {
        throw AIError.invalidArgument(argument: "files", message: "URL-based images are not supported for Google Vertex image editing. Provide image data directly.")
    }
    guard let data = file.data else {
        throw AIError.invalidArgument(argument: "files", message: "Image file must contain data for Google Vertex image editing.")
    }
    return data.base64EncodedString()
}

func googleEmbeddingProviderOptions(from request: EmbeddingRequest) -> [String: JSONValue] {
    request.providerOptions["google"]?.objectValue ?? [:]
}

func googleEmbeddingParts(text: String, content: JSONValue?) -> [JSONValue] {
    let textPart: [JSONValue] = text.isEmpty ? [] : [.object(["text": .string(text)])]
    guard let content else {
        return textPart.isEmpty ? [.object(["text": .string(text)])] : textPart
    }
    if content == .null {
        return textPart.isEmpty ? [.object(["text": .string(text)])] : textPart
    }
    if let parts = content.arrayValue {
        return textPart + parts
    }
    return textPart.isEmpty ? [.object(["text": .string(text)])] : textPart
}

func googleVideoProviderOptions(from request: VideoGenerationRequest) -> [String: JSONValue] {
    var options = request.extraBody
    if let google = request.providerOptions["google"]?.objectValue {
        options.merge(google) { _, new in new }
    }
    return options
}

func googleSpeechProviderOptions(from request: SpeechRequest) -> [String: JSONValue] {
    var options = request.extraBody
    if let google = request.extraBody["google"]?.objectValue {
        options = google
    }
    if let google = request.providerOptions["google"]?.objectValue {
        options.merge(google) { _, new in new }
    }
    return options.filter { $0.key != "google" && $0.key != "googleVertex" && $0.key != "vertex" }
}

public struct GooglePreparedSpeechRequest {
    public var body: JSONValue
    public var warnings: [AIWarning]
    public var outputFormat: String
    public var usesStructuredSpeech: Bool
    public init(body: JSONValue, warnings: [AIWarning], outputFormat: String, usesStructuredSpeech: Bool) {
        self.body = body
        self.warnings = warnings
        self.outputFormat = outputFormat
        self.usesStructuredSpeech = usesStructuredSpeech
    }

}

public func googlePrepareSpeechRequest(
    _ request: SpeechRequest,
    modelID: String,
    options: [String: JSONValue]
) throws -> GooglePreparedSpeechRequest {
    let usesStructuredSpeech = !modelID.hasPrefix("gemini-2.5-")
        && !modelID.hasPrefix("gemini-3.1-")
    var warnings: [AIWarning] = []

    if request.voice?.hasPrefix("voice_") == true || request.voice?.hasPrefix("voicekey_") == true {
        throw AIError.invalidArgument(
            argument: "voice",
            message: "Custom voices are not supported. Use a prebuilt voice instead."
        )
    }

    let multiSpeakerVoiceConfig = options["multiSpeakerVoiceConfig"]
    let speakerVoiceConfigs = multiSpeakerVoiceConfig?["speakerVoiceConfigs"]?.arrayValue ?? []
    if speakerVoiceConfigs.contains(where: { $0["voiceConfig"]?.objectValue?.keys.contains("voice") == true }) {
        throw AIError.invalidArgument(
            argument: "voice",
            message: "Custom voices are not supported. Use a prebuilt voice instead."
        )
    }

    if request.speed != nil {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "speed",
            message: "Google Gemini TTS models do not support the `speed` option. It was ignored."
        ))
    }
    if request.language != nil {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "language",
            message: "Google Gemini TTS models do not support the `language` option. Language is detected automatically from the input text."
        ))
    }

    var parts: [JSONValue]
    if usesStructuredSpeech {
        if options["turns"] != nil, options["speechMetadata"] != nil {
            throw AIError.invalidArgument(
                argument: "providerOptions",
                message: "Set speechMetadata on each turn when using turns."
            )
        }

        let turns = options["turns"]?.arrayValue
        if options["turns"] != nil, turns?.isEmpty != false {
            throw AIError.invalidArgument(
                argument: "providerOptions",
                message: "Google TTS turns must contain at least one turn."
            )
        }
        if turns != nil, !request.text.isEmpty {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "text",
                message: "Google TTS turns replace the top-level text."
            ))
        }

        let configuredSpeakers = Set(speakerVoiceConfigs.compactMap { $0["speaker"]?.stringValue })
        let sourceParts = turns ?? [
            .object([
                "text": .string(request.text),
                "speechMetadata": options["speechMetadata"] ?? .null
            ])
        ]
        parts = try sourceParts.map { part in
            guard let text = part["text"]?.stringValue else {
                throw AIError.invalidArgument(argument: "providerOptions", message: "Every Google TTS turn must contain text.")
            }
            let metadata = part["speechMetadata"]?.objectValue
            let speaker = metadata?["speaker"]?.stringValue
            let style = metadata?["style"]?.stringValue ?? request.instructions
            if multiSpeakerVoiceConfig != nil,
               (speaker == nil || !configuredSpeakers.contains(speaker!)) {
                throw AIError.invalidArgument(
                    argument: "speechMetadata.speaker",
                    message: "Every multi-speaker turn must specify a speechMetadata.speaker matching a configured speaker."
                )
            }
            var output: [String: JSONValue] = ["text": .string(text)]
            if style != nil || speaker != nil {
                output["speechMetadata"] = .object([
                    "style": style.map(JSONValue.string) ?? .null,
                    "speaker": speaker.map(JSONValue.string) ?? .null
                ].filter { $0.value != .null })
            }
            return .object(output)
        }
    } else {
        if options["turns"] != nil || options["speechMetadata"] != nil {
            throw AIError.invalidArgument(
                argument: "providerOptions",
                message: "Structured speech metadata and turns require Gemini 3.8 TTS."
            )
        }
        var promptText = request.text
        if let instructions = request.instructions {
            if multiSpeakerVoiceConfig != nil {
                warnings.append(AIWarning(
                    type: "unsupported",
                    feature: "instructions",
                    message: "Google Gemini TTS ignores `instructions` when `multiSpeakerVoiceConfig` is set, because prepending them would break multi-speaker transcript parsing."
                ))
            } else {
                promptText = "\(instructions): \(request.text)"
            }
        }
        parts = [.object(["text": .string(promptText)])]
    }

    let transcript = (options["turns"]?.arrayValue?.compactMap { $0["text"]?.stringValue }.joined()) ?? request.text
    guard !transcript.isEmpty else {
        throw AIError.invalidArgument(argument: "text", message: "Speech input must contain a non-empty transcript.")
    }

    let formats: [String: String] = usesStructuredSpeech
        ? [
            "wav": "AUDIO_WAV",
            "audio/wav": "AUDIO_WAV",
            "pcm": "AUDIO_L16",
            "audio/l16": "AUDIO_L16",
            "mulaw": "AUDIO_MULAW",
            "audio/mulaw": "AUDIO_MULAW",
            "alaw": "AUDIO_ALAW",
            "audio/alaw": "AUDIO_ALAW"
        ]
        : ["wav": "AUDIO_WAV", "pcm": "AUDIO_L16"]
    let requestedFormat = request.format?.lowercased()
    let outputFormat: String
    if let requestedFormat, let supported = formats[requestedFormat] {
        outputFormat = supported
    } else {
        outputFormat = "AUDIO_WAV"
        if let requestedFormat {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "outputFormat",
                message: "Unsupported output format: \(requestedFormat). Using wav instead."
            ))
        }
    }

    let speechConfig: JSONValue
    if let multiSpeakerVoiceConfig {
        speechConfig = .object(["multiSpeakerVoiceConfig": multiSpeakerVoiceConfig])
    } else {
        speechConfig = .object([
            "voiceConfig": .object([
                "prebuiltVoiceConfig": .object([
                    "voiceName": .string(request.voice ?? "Kore")
                ])
            ])
        ])
    }
    var generationConfig: [String: JSONValue] = [
        "responseModalities": .array([.string("AUDIO")]),
        "speechConfig": speechConfig
    ]
    if usesStructuredSpeech, request.format != nil {
        generationConfig["responseFormat"] = .object([
            "audio": .object(["mimeType": .string(outputFormat)])
        ])
    }
    return GooglePreparedSpeechRequest(
        body: .object([
            "contents": .array([.object([
                "role": .string("user"),
                "parts": .array(parts)
            ])]),
            "generationConfig": .object(generationConfig)
        ]),
        warnings: warnings,
        outputFormat: outputFormat,
        usesStructuredSpeech: usesStructuredSpeech
    )
}

public func googleSpeechContentType(
    mimeType: String?,
    outputFormat: String,
    wrappedPCMAsWAV: Bool
) -> String {
    if wrappedPCMAsWAV { return "audio/wav" }
    if let mimeType { return mimeType }
    switch outputFormat {
    case "AUDIO_L16": return "audio/L16"
    case "AUDIO_MULAW": return "audio/mulaw"
    case "AUDIO_ALAW": return "audio/alaw"
    default: return "audio/wav"
    }
}

public func googleSpeechMimeTypeIsPCM(_ mimeType: String?, usesStructuredSpeech: Bool) -> Bool {
    guard let mimeType else { return !usesStructuredSpeech }
    let lower = mimeType.lowercased()
    return lower.hasPrefix("audio/l16") || lower.hasPrefix("audio/pcm")
}

func googleVideoInstance(for request: VideoGenerationRequest, options: [String: JSONValue]) -> (instance: [String: JSONValue], warnings: [AIWarning]) {
    var instance: [String: JSONValue] = ["prompt": .string(request.prompt)]
    var warnings: [AIWarning] = []
    if let firstFrame = request.frameImages.first(where: { $0.frameType == .firstFrame }) {
        if let image = googleVideoInlineImage(firstFrame.image) {
            instance["image"] = image
        } else if firstFrame.image.url != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "URL-based image input",
                message: "Google Generative AI video models require base64-encoded images. URL will be ignored."
            ))
        }
    } else if let image = request.image {
        if image.url != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "URL-based image input",
                message: "Google Generative AI video models require base64-encoded images. URL will be ignored."
            ))
        } else if let image = googleVideoInlineImage(image) {
            instance["image"] = image
        }
    } else if let image = options["image"]?.objectValue {
        if let data = image["data"]?.stringValue {
            instance["image"] = .object([
                "bytesBase64Encoded": .string(data),
                "mimeType": image["mimeType"] ?? .string("image/png")
            ])
        }
    }
    if let lastFrame = request.frameImages.first(where: { $0.frameType == .lastFrame }), let image = googleVideoInlineImage(lastFrame.image) {
        instance["lastFrame"] = image
    }
    if !request.inputReferences.isEmpty {
        let referenceImages = request.inputReferences.compactMap(googleVideoReferenceImage)
        if !referenceImages.isEmpty {
            instance["referenceImages"] = .array(referenceImages)
        }
    } else if let referenceImages = options["referenceImages"]?.arrayValue {
        instance["referenceImages"] = .array(referenceImages.map { reference in
            guard let object = reference.objectValue else { return reference }
            if let bytesBase64Encoded = object["bytesBase64Encoded"]?.stringValue {
                return .object([
                    "bytesBase64Encoded": .string(bytesBase64Encoded),
                    "mimeType": object["mimeType"] ?? .string("image/png")
                ])
            }
            if let gcsUri = object["gcsUri"] {
                return .object(["gcsUri": gcsUri])
            }
            return reference
        })
    }
    return (instance, warnings)
}

func googleVideoInlineImage(_ image: ImageInputFile) -> JSONValue? {
    guard let data = image.data else { return nil }
    return .object([
        "bytesBase64Encoded": .string(data.base64EncodedString()),
        "mimeType": .string(image.mediaType ?? "image/png")
    ])
}

func googleVideoReferenceImage(_ image: ImageInputFile) -> JSONValue? {
    guard let prepared = googleVideoInlineImage(image) else { return nil }
    return .object([
        "image": prepared,
        "referenceType": .string("asset")
    ])
}

func googleVideoParameters(for request: VideoGenerationRequest, options: [String: JSONValue]) -> [String: JSONValue] {
    var parameters: [String: JSONValue] = [:]
    if let count = request.count {
        parameters["sampleCount"] = .number(Double(count))
    } else if let sampleCount = options["sampleCount"] ?? options["n"] {
        parameters["sampleCount"] = sampleCount
    }
    if let aspectRatio = request.aspectRatio {
        parameters["aspectRatio"] = .string(aspectRatio)
    }
    if let resolution = request.resolution ?? options["resolution"]?.stringValue {
        parameters["resolution"] = .string(googleVideoResolution(resolution))
    }
    if let duration = request.durationSeconds {
        parameters["durationSeconds"] = .number(duration)
    }
    if let seed = request.seed {
        parameters["seed"] = .number(Double(seed))
    } else if let seed = options["seed"] {
        parameters["seed"] = seed
    }
    parameters.merge(googleOptionsWithoutPoll(options, excluding: ["sampleCount", "n", "resolution", "seed", "pollIntervalMs", "pollTimeoutMs", "image", "referenceImages"])) { _, new in new }
    return parameters
}

func googleOptionsWithoutPoll(_ options: [String: JSONValue], excluding keys: Set<String>) -> [String: JSONValue] {
    options.filter { !keys.contains($0.key) }
}

func googleAspectRatio(from request: ImageGenerationRequest) -> String? {
    if let aspectRatio = request.aspectRatio { return aspectRatio }
    if let aspectRatio = request.extraBody["aspectRatio"]?.stringValue {
        return aspectRatio
    }
    if let size = request.size, size.contains(":") {
        return size
    }
    return nil
}

func googleVideoResolution(_ resolution: String) -> String {
    switch resolution {
    case "1280x720":
        return "720p"
    case "1920x1080":
        return "1080p"
    case "3840x2160":
        return "4k"
    default:
        return resolution
    }
}

func googlePollTimeout(_ extraBody: [String: JSONValue]) -> UInt64 {
    let milliseconds = extraBody["pollTimeoutMs"]?.intValue ?? 600_000
    return UInt64(milliseconds) * 1_000_000
}

func googlePollInterval(_ extraBody: [String: JSONValue]) -> UInt64 {
    let milliseconds = extraBody["pollIntervalMs"]?.intValue ?? 10_000
    return UInt64(milliseconds) * 1_000_000
}