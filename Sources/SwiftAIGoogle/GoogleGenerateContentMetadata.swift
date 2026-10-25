import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleThoughtSignatureProviderMetadata(from part: JSONValue) -> [String: JSONValue] {
    guard let thoughtSignature = part["thoughtSignature"] ?? part["thought_signature"],
          thoughtSignature.stringValue != nil else {
        return [:]
    }
    return ["google": .object(["thoughtSignature": thoughtSignature])]
}

func googleServerToolProviderMetadata(id: String, type: String, part: JSONValue) -> [String: JSONValue] {
    var google: [String: JSONValue] = [
        "serverToolCallId": .string(id),
        "serverToolType": .string(type)
    ]
    if let thoughtSignature = part["thoughtSignature"] ?? part["thought_signature"],
       thoughtSignature.stringValue != nil {
        google["thoughtSignature"] = thoughtSignature
    }
    return ["google": .object(google)]
}

func googleInlineDataProviderMetadata(from part: JSONValue) -> [String: JSONValue] {
    var google: [String: JSONValue] = [:]
    if part["thought"]?.boolValue == true {
        google["thought"] = .bool(true)
    }
    if let thoughtSignature = part["thoughtSignature"] ?? part["thought_signature"],
       thoughtSignature.stringValue != nil {
        google["thoughtSignature"] = thoughtSignature
    }
    return google.isEmpty ? [:] : ["google": .object(google)]
}

func googleThoughtSignature(from providerMetadata: [String: JSONValue]) -> JSONValue? {
    let value = providerMetadata["google"]?["thoughtSignature"]
        ?? providerMetadata["google.generative-ai"]?["thoughtSignature"]
        ?? providerMetadata["googleVertex"]?["thoughtSignature"]
        ?? providerMetadata["vertex"]?["thoughtSignature"]
    return value?.stringValue == nil ? nil : value
}

func googleServerToolMetadata(from providerMetadata: [String: JSONValue]) -> (id: String, type: String)? {
    let namespaces = ["google", "google.generative-ai", "googleVertex", "vertex"]
    for namespace in namespaces {
        guard let id = providerMetadata[namespace]?["serverToolCallId"]?.stringValue,
              let type = providerMetadata[namespace]?["serverToolType"]?.stringValue else {
            continue
        }
        return (id, type)
    }
    return nil
}

func googleCodeExecutionResultJSON(_ value: JSONValue) -> JSONValue {
    .object([
        "outcome": value["outcome"] ?? .null,
        "output": value["output"] ?? .string("")
    ])
}

func googleGenerateContentArguments(_ value: JSONValue?) -> String {
    let argumentValue = value ?? .object([:])
    guard let data = try? encodeJSONBody(argumentValue),
          let text = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return text
}

public func googleGenerateContentResponseMetadata(
    from raw: JSONValue,
    response: AIHTTPResponse,
    modelID: String?
) -> AIResponseMetadata {
    var metadata = aiResponseMetadata(from: raw, response: response, modelID: modelID)
    metadata.id = raw["responseId"]?.stringValue ?? metadata.id
    return metadata
}

func googlePartialArgValue(_ partialArg: JSONValue) -> JSONValue {
    if let value = partialArg["stringValue"]?.stringValue {
        return .string(value)
    }
    if let value = partialArg["numberValue"]?.doubleValue {
        return .number(value)
    }
    if let value = partialArg["boolValue"]?.boolValue {
        return .bool(value)
    }
    if let value = partialArg["value"] {
        return value
    }
    return .null
}

func googleSetPartialArgument(path: String, value: JSONValue, in arguments: inout [String: JSONValue]) {
    guard path.hasPrefix("$.") else { return }
    let key = String(path.dropFirst(2))
    guard !key.isEmpty, !key.contains(".") && !key.contains("[") else { return }

    if case let .string(newValue) = value,
       case let .string(existingValue) = arguments[key] {
        arguments[key] = .string(existingValue + newValue)
    } else {
        arguments[key] = value
    }
}