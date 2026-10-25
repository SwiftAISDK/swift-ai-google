import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public func googlePrepareTools(
    from tools: [String: JSONValue],
    toolChoice: JSONValue?,
    modelID: String,
    isVertexProvider: Bool
) throws -> GooglePreparedTools? {
    guard !tools.isEmpty else { return nil }

    let capabilities = googleModelCapabilities(for: modelID)
    var warnings: [AIWarning] = []
    let providerResults = tools.compactMap { name, schema -> GooglePreparedProviderTool? in
        let object = schema.objectValue
        let id = object?["id"]?.stringValue ?? name
        let isProviderTool = object?["type"]?.stringValue == "provider" || id.hasPrefix("google.")
        guard isProviderTool else { return nil }
        return googleProviderTool(name: name, schema: schema, modelID: modelID)
    }
    let providerTools = providerResults.compactMap(\.tool)
    warnings.append(contentsOf: providerResults.flatMap(\.warnings))
    let functionDeclarations = try googleFunctionDeclarations(from: tools)
    let hasFunctionTools = !functionDeclarations.isEmpty
    let hasProviderTools = !providerTools.isEmpty || tools.values.contains { schema in
        guard let object = schema.objectValue else { return false }
        return object["type"]?.stringValue == "provider" || object["id"]?.stringValue?.hasPrefix("google.") == true
    }

    if hasFunctionTools, hasProviderTools, !capabilities.usesGemini3Features {
        warnings.append(AIWarning(type: "unsupported", feature: "combination of function and provider-defined tools"))
    }
    if !isVertexProvider, tools.values.contains(where: { schema in
        let object = schema.objectValue
        return (object?["id"]?.stringValue ?? "") == "google.vertex_rag_store"
    }) {
        warnings.append(AIWarning(
            type: "other",
            message: "The 'vertex_rag_store' tool is only supported with the Google Vertex provider and might not be supported or could behave unexpectedly with the current Google provider."
        ))
    }

    if hasProviderTools {
        if hasFunctionTools, capabilities.usesGemini3Features, !providerTools.isEmpty {
            var prepared = providerTools
            prepared.append(.object(["functionDeclarations": .array(functionDeclarations)]))
            var config = googleToolConfig(from: toolChoice, hasStrictTools: googleHasStrictTools(tools), defaultMode: "VALIDATED")
                ?? .object(["functionCallingConfig": .object(["mode": .string("VALIDATED")])])
            if !isVertexProvider, var object = config.objectValue {
                object["includeServerSideToolInvocations"] = true
                config = .object(object)
            }
            return GooglePreparedTools(tools: prepared, toolConfig: config, warnings: warnings)
        }
        return GooglePreparedTools(tools: providerTools, toolConfig: nil, warnings: warnings)
    }

    guard hasFunctionTools else { return nil }
    return GooglePreparedTools(
        tools: [.object(["functionDeclarations": .array(functionDeclarations)])],
        toolConfig: googleToolConfig(from: toolChoice, hasStrictTools: googleHasStrictTools(tools), defaultMode: nil),
        warnings: warnings
    )
}

public func googleExtraBodyWithoutToolChoice(_ extraBody: [String: JSONValue]) -> [String: JSONValue] {
    var output = extraBody
    for key in [
        "toolChoice",
        "responseFormat",
        "structuredOutputs",
        "google",
        "googleVertex",
        "vertex",
        "responseModalities",
        "thinkingConfig",
        "mediaResolution",
        "imageConfig",
        "audioTimestamp",
        "safetySettings",
        "cachedContent",
        "labels",
        "serviceTier",
        "continuationToken",
        "sharedRequestType",
        "requestType",
        "streamFunctionCallArguments",
        "retrievalConfig"
    ] {
        output.removeValue(forKey: key)
    }
    return output
}

func googleResponseFormatJSON(_ responseFormat: AIResponseFormat) -> JSONValue? {
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

func googleFunctionDeclarations(from tools: [String: JSONValue]) throws -> [JSONValue] {
    tools.compactMap { name, schema in
        let object = schema.objectValue
        if object?["type"]?.stringValue == "provider" || object?["id"]?.stringValue?.hasPrefix("google.") == true {
            return nil
        }

        var declaration: [String: JSONValue] = [
            "name": .string(name),
            "description": .string(object?["description"]?.stringValue ?? "")
        ]
        declaration["parametersJsonSchema"] = schema
        return .object(declaration)
    }
}

func googleProviderTool(name: String, schema: JSONValue, modelID: String) -> GooglePreparedProviderTool {
    let object = schema.objectValue
    let id = object?["id"]?.stringValue ?? name
    let args = object?["args"]?.objectValue ?? [:]

    switch id {
    case "google.google_search":
        guard googleIsGemini2OrNewer(modelID) else {
            return googleUnsupportedProviderTool(id, details: "Google Search requires Gemini 2.0 or newer.")
        }
        return GooglePreparedProviderTool(tool: .object(["googleSearch": .object(args)]))
    case "google.enterprise_web_search":
        guard googleIsGemini2OrNewer(modelID) else {
            return googleUnsupportedProviderTool(id, details: "Enterprise Web Search requires Gemini 2.0 or newer.")
        }
        return GooglePreparedProviderTool(tool: .object(["enterpriseWebSearch": .object([:])]))
    case "google.url_context":
        guard googleIsGemini2OrNewer(modelID) else {
            return googleUnsupportedProviderTool(id, details: "The URL context tool is not supported with other Gemini models than Gemini 2.")
        }
        return GooglePreparedProviderTool(tool: .object(["urlContext": .object([:])]))
    case "google.code_execution":
        guard googleIsGemini2OrNewer(modelID) else {
            return googleUnsupportedProviderTool(id, details: "The code execution tool is not supported with other Gemini models than Gemini 2.")
        }
        return GooglePreparedProviderTool(tool: .object(["codeExecution": .object([:])]))
    case "google.file_search":
        guard googleModelCapabilities(for: modelID).supportsFileSearch else {
            return googleUnsupportedProviderTool(id, details: "The file search tool is only supported with Gemini 2.5 models and Gemini 3 models.")
        }
        return GooglePreparedProviderTool(tool: .object(["fileSearch": .object(args)]))
    case "google.vertex_rag_store":
        guard googleIsGemini2OrNewer(modelID) else {
            return googleUnsupportedProviderTool(id, details: "The RAG store tool is not supported with other Gemini models than Gemini 2.")
        }
        return GooglePreparedProviderTool(tool: .object([
            "retrieval": .object([
                "vertex_rag_store": .object([
                    "rag_resources": .object(["rag_corpus": args["ragCorpus"] ?? args["rag_corpus"]]),
                    "similarity_top_k": args["topK"] ?? args["top_k"]
                ])
            ])
        ]))
    case "google.google_maps":
        guard googleIsGemini2OrNewer(modelID) else {
            return googleUnsupportedProviderTool(id, details: "The Google Maps grounding tool is not supported with Gemini models other than Gemini 2 or newer.")
        }
        return GooglePreparedProviderTool(tool: .object(["googleMaps": .object([:])]))
    default:
        return googleUnsupportedProviderTool(id, details: nil)
    }
}

func googleUnsupportedProviderTool(_ id: String, details: String?) -> GooglePreparedProviderTool {
    GooglePreparedProviderTool(warnings: [
        AIWarning(
            type: "unsupported",
            feature: "provider-defined tool \(id)",
            message: details
        )
    ])
}

func googleToolConfig(from value: JSONValue?, hasStrictTools: Bool, defaultMode: String?) -> JSONValue? {
    guard let value else {
        guard hasStrictTools || defaultMode != nil else { return nil }
        return .object(["functionCallingConfig": .object(["mode": .string(defaultMode ?? "VALIDATED")])])
    }

    let type: String?
    let toolName: String?
    if let string = value.stringValue {
        type = string
        toolName = nil
    } else if let object = value.objectValue {
        type = object["type"]?.stringValue
        toolName = object["toolName"]?.stringValue
            ?? object["tool_name"]?.stringValue
            ?? object["function"]?["name"]?.stringValue
    } else {
        return nil
    }

    var config: [String: JSONValue]
    switch type {
    case "auto":
        config = ["mode": .string(hasStrictTools ? "VALIDATED" : "AUTO")]
    case "none":
        config = ["mode": .string("NONE")]
    case "required":
        config = ["mode": .string("ANY")]
    case "tool":
        guard let toolName else { return nil }
        config = [
            "mode": .string("ANY"),
            "allowedFunctionNames": .array([.string(toolName)])
        ]
    default:
        return nil
    }
    return .object(["functionCallingConfig": .object(config)])
}

func googleHasStrictTools(_ tools: [String: JSONValue]) -> Bool {
    tools.values.contains { $0["strict"]?.boolValue == true }
}

func googleIsGemini2OrNewer(_ modelID: String) -> Bool {
    googleModelCapabilities(for: modelID).supportsGemini2Tools
}

func googleIsGemini3OrNewer(_ modelID: String) -> Bool {
    googleModelCapabilities(for: modelID).usesGemini3Features
}