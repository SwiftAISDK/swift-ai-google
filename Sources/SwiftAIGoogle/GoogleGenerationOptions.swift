import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleGenerateContentOptions(from extraBody: [String: JSONValue]) -> [String: JSONValue] {
    var output = extraBody
    if let nested = output.removeValue(forKey: "google")?.objectValue {
        output.merge(nested) { _, nested in nested }
    }
    return output
}

public func googlePrepareGenerateContentOptions(
    from request: LanguageModelRequest,
    modelID: String,
    providerID: String,
    isVertexProvider: Bool
) throws -> GooglePreparedGenerateContentOptions {
    var options = googleGenerateContentOptions(from: request.extraBody)
    if let toolChoice = request.toolChoice { options["toolChoice"] = toolChoice }
    let providerOptionsNames = isVertexProvider ? ["googleVertex", "vertex", "google"] : ["google"]
    for name in providerOptionsNames {
        guard let providerOptions = request.providerOptions[name]?.objectValue else { continue }
        options.merge(providerOptions.filter { googleKnownLanguageProviderOptionKeys.contains($0.key) }) { _, providerValue in providerValue }
        break
    }

    if let token = options["continuationToken"], token.stringValue == nil {
        throw AIError.invalidArgument(argument: "continuationToken", message: "Google continuationToken must be a string.")
    }

    var warnings: [AIWarning] = []
    var headers: [String: String] = [:]

    if options["streamFunctionCallArguments"]?.boolValue == true, !isVertexProvider {
        warnings.append(AIWarning(
            type: "other",
            message: "'streamFunctionCallArguments' is only supported on the Vertex AI API and will be ignored with the current Google provider (\(providerID)). See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#streaming-fc"
        ))
    }
    if options["serviceTier"] != nil, isVertexProvider {
        warnings.append(AIWarning(
            type: "other",
            message: "'serviceTier' is a Gemini API option and is not supported on Vertex AI. Use 'sharedRequestType' (and optionally 'requestType') instead. See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo"
        ))
    }
    if (options["sharedRequestType"] != nil || options["requestType"] != nil), !isVertexProvider {
        warnings.append(AIWarning(
            type: "other",
            message: "'sharedRequestType' and 'requestType' are Vertex AI options and are ignored with the current Google provider (\(providerID))."
        ))
    }

    if isVertexProvider {
        if let sharedRequestType = options["sharedRequestType"]?.stringValue {
            headers["X-Vertex-AI-LLM-Shared-Request-Type"] = sharedRequestType
        }
        if let requestType = options["requestType"]?.stringValue {
            headers["X-Vertex-AI-LLM-Request-Type"] = requestType
        }
    }

    let explicitThinking = options["thinkingConfig"]
    let hasExplicitThinking = explicitThinking?["thinkingBudget"]?.doubleValue != nil || explicitThinking?["thinkingLevel"]?.stringValue != nil
    let thinkingConfig = hasExplicitThinking ? nil : googleThinkingConfig(for: request.reasoning, modelID: modelID, warnings: &warnings)
    if var thinkingConfig {
        if let providerThinkingConfig = options["thinkingConfig"]?.objectValue {
            thinkingConfig.merge(providerThinkingConfig) { _, providerValue in providerValue }
        }
        options["thinkingConfig"] = .object(thinkingConfig)
    }

    if isVertexProvider {
        options.removeValue(forKey: "serviceTier")
    }
    options.removeValue(forKey: "sharedRequestType")
    options.removeValue(forKey: "requestType")

    return GooglePreparedGenerateContentOptions(options: options, warnings: warnings, headers: headers)
}

public func googleResolvedResponseFormat(request: LanguageModelRequest, options: inout [String: JSONValue]) -> JSONValue? {
    if let responseFormat = request.responseFormat {
        options.removeValue(forKey: "responseFormat")
        return googleResponseFormatJSON(responseFormat)
    }
    return options.removeValue(forKey: "responseFormat")
}

public func googleApplyResponseFormat(_ responseFormat: JSONValue?, options: [String: JSONValue], to generationConfig: inout [String: JSONValue]) throws {
    guard responseFormat?["type"]?.stringValue == "json" else { return }
    generationConfig["responseMimeType"] = .string("application/json")
    guard options["structuredOutputs"]?.boolValue != false,
          let schema = responseFormat?["schema"] else {
        return
    }
    generationConfig["responseJsonSchema"] = googleResponseJSONSchema(schema)
}

public func googleApplyStandardGenerationSettings(_ request: LanguageModelRequest, to generationConfig: inout [String: JSONValue]) {
    if let temperature = request.temperature { generationConfig["temperature"] = .number(temperature) }
    if let topP = request.topP { generationConfig["topP"] = .number(topP) }
    if let topK = request.topK { generationConfig["topK"] = .number(Double(topK)) }
    if let frequencyPenalty = request.frequencyPenalty { generationConfig["frequencyPenalty"] = .number(frequencyPenalty) }
    if let presencePenalty = request.presencePenalty { generationConfig["presencePenalty"] = .number(presencePenalty) }
    if let seed = request.seed { generationConfig["seed"] = .number(Double(seed)) }
    if let maxOutputTokens = request.maxOutputTokens { generationConfig["maxOutputTokens"] = .number(Double(maxOutputTokens)) }
    if !request.stopSequences.isEmpty { generationConfig["stopSequences"] = .array(request.stopSequences) }
}

public func googleApplyProviderGenerationOptions(_ options: [String: JSONValue], to generationConfig: inout [String: JSONValue]) {
    for key in ["responseModalities", "thinkingConfig", "mediaResolution", "imageConfig", "audioTimestamp"] {
        if let value = options[key] {
            generationConfig[key] = value
        }
    }
}

public func googleTopLevelGenerateContentOptions(_ options: [String: JSONValue]) -> [String: JSONValue] {
    var output: [String: JSONValue] = [:]
    for key in ["cachedContent", "labels", "serviceTier", "continuationToken"] {
        if let value = options[key] {
            output[key] = value
        }
    }
    if let safetySettings = options["safetySettings"] {
        output["safetySettings"] = safetySettings
    } else if let threshold = options["threshold"] {
        output["safetySettings"] = .array(googleConfigurableSafetySettingCategories.map { category in
            .object([
                "category": .string(category),
                "threshold": threshold
            ])
        })
    }
    return output
}

public func googleContentsWithSystemInstruction(systemText: String, contents: [JSONValue], modelID: String) -> (contents: [JSONValue], systemInstruction: JSONValue?) {
    guard !systemText.isEmpty else {
        return (contents, nil)
    }
    guard googleIsGemmaModel(modelID) else {
        return (
            contents,
            .object(["parts": .array([.object(["text": .string(systemText)])])])
        )
    }
    guard var first = contents.first?.objectValue,
          first["role"]?.stringValue == "user",
          var parts = first["parts"]?.arrayValue else {
        return (contents, nil)
    }

    parts.insert(.object(["text": .string(systemText + "\n\n")]), at: 0)
    first["parts"] = .array(parts)
    var adjusted = contents
    adjusted[0] = .object(first)
    return (adjusted, nil)
}

public func googleToolConfigWithProviderOptions(_ toolConfig: JSONValue?, options: [String: JSONValue], isStreaming: Bool, isVertexProvider: Bool) -> JSONValue? {
    let shouldStreamFunctionCallArguments = isStreaming && isVertexProvider && options["streamFunctionCallArguments"]?.boolValue == true
    let retrievalConfig = options["retrievalConfig"]
    guard toolConfig != nil || shouldStreamFunctionCallArguments || retrievalConfig != nil else { return nil }

    var output = toolConfig?.objectValue ?? [:]
    if shouldStreamFunctionCallArguments {
        var functionCallingConfig = output["functionCallingConfig"]?.objectValue ?? [:]
        functionCallingConfig["streamFunctionCallArguments"] = true
        output["functionCallingConfig"] = .object(functionCallingConfig)
    }
    if let retrievalConfig {
        output["retrievalConfig"] = retrievalConfig
    }
    return .object(output)
}

func googleThinkingConfig(for reasoning: String?, modelID: String, warnings: inout [AIWarning]) -> [String: JSONValue]? {
    guard let reasoning, reasoning != "provider-default" else { return nil }
    if googleIsGemini3OrNewer(modelID), !modelID.lowercased().contains("gemini-3-pro-image") {
        let minimumThinkingLevel = googleMinimumThinkingLevelForGemini3Model(modelID)
        if reasoning == "none" {
            return ["thinkingLevel": .string(minimumThinkingLevel)]
        }
        let effortMap = [
            "minimal": minimumThinkingLevel,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high"
        ]
        guard let mapped = effortMap[reasoning] else {
            warnings.append(AIWarning(type: "unsupported", feature: "reasoning", message: "reasoning \"\(reasoning)\" is not supported by this model."))
            return nil
        }
        if mapped != reasoning {
            warnings.append(AIWarning(
                type: "compatibility",
                feature: "reasoning",
                message: "reasoning \"\(reasoning)\" is not directly supported by this model. mapped to effort \"\(mapped)\"."
            ))
        }
        return ["thinkingLevel": .string(mapped)]
    }

    if reasoning == "none" {
        return ["thinkingBudget": .number(0)]
    }
    let budgetPercentages: [String: Double] = [
        "minimal": 0.02,
        "low": 0.1,
        "medium": 0.3,
        "high": 0.6,
        "xhigh": 0.9
    ]
    guard let percentage = budgetPercentages[reasoning] else {
        warnings.append(AIWarning(type: "unsupported", feature: "reasoning", message: "reasoning \"\(reasoning)\" is not supported by this model."))
        return nil
    }
    let maxThinkingTokens = googleMaxThinkingTokensForGemini25Model(modelID)
    let budget = min(maxThinkingTokens, max(0, Int((65536 * percentage).rounded())))
    return ["thinkingBudget": .number(Double(budget))]
}

private func googleMinimumThinkingLevelForGemini3Model(_ modelID: String) -> String {
    let normalized = modelID.lowercased()
    guard let range = normalized.range(of: #"(^|[/.])gemini-[^/]*$"#, options: .regularExpression) else { return "minimal" }
    let modelName = String(normalized[range]).trimmingCharacters(in: CharacterSet(charactersIn: "/."))
    if modelName == "gemini-flash-latest" {
        return "low"
    }

    if let match = modelName.range(of: #"^gemini-\d+(?:\.\d+)?-pro(?:$|-)"#, options: .regularExpression) {
        let version = modelName[match].dropFirst("gemini-".count).prefix(while: { $0.isNumber })
        if let major = Double(version), major >= 3 { return "low" }
    }

    let prefix = "gemini-"
    guard modelName.hasPrefix(prefix) else { return "minimal" }
    let afterPrefix = String(modelName.dropFirst(prefix.count))
    guard let flashRange = afterPrefix.range(of: "-flash") else { return "minimal" }
    let version = String(afterPrefix[..<flashRange.lowerBound])
    let suffix = String(afterPrefix[flashRange.upperBound...])
    guard suffix.isEmpty || suffix.hasPrefix("-") else { return "minimal" }
    if suffix == "-lite" || suffix.hasPrefix("-lite-") {
        return "minimal"
    }

    let versionComponents = version.split(separator: ".", omittingEmptySubsequences: false)
    guard versionComponents.count == 2,
          let majorVersion = Int(versionComponents[0]),
          let minorVersion = Int(versionComponents[1]) else {
        return "minimal"
    }
    return majorVersion > 3 || (majorVersion == 3 && minorVersion >= 7) ? "low" : "minimal"
}

func googleMaxThinkingTokensForGemini25Model(_ modelID: String) -> Int {
    let id = modelID.lowercased()
    if id.contains("2.5-pro") || id.contains("gemini-3-pro-image") {
        return 32768
    }
    return 24576
}

func googleIsGemmaModel(_ modelID: String) -> Bool {
    modelID.lowercased().hasPrefix("gemma-")
}

let googleKnownLanguageProviderOptionKeys: Set<String> = [
    "responseModalities",
    "thinkingConfig",
    "cachedContent",
    "structuredOutputs",
    "safetySettings",
    "threshold",
    "audioTimestamp",
    "labels",
    "mediaResolution",
    "imageConfig",
    "retrievalConfig",
    "streamFunctionCallArguments",
    "serviceTier",
    "continuationToken",
    "sharedRequestType",
    "requestType"
]

private let googleConfigurableSafetySettingCategories = [
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT"
]