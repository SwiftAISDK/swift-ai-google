import SwiftAIProvider

func googleInteractionsPrepareNativeTools(_ tools: [String: JSONValue], choice: JSONValue?) -> (tools: [JSONValue], choice: JSONValue?, warnings: [AIWarning]) {
    let mode = choice?["type"]?.stringValue ?? choice?.stringValue
    if mode == "none" { return ([], nil, []) }
    var output: [JSONValue] = []
    var warnings: [AIWarning] = []
    for (name, tool) in tools.sorted(by: { $0.key < $1.key }) {
        if tool["type"]?.stringValue == "provider" || tool["id"]?.stringValue?.hasPrefix("google.") == true {
            let id = tool["id"]?.stringValue ?? name
            let args = tool["args"]?.objectValue ?? [:]
            let type = String(id.dropFirst("google.".count))
            var value: [String: JSONValue] = ["type": .string(type)]
            var mapping: [String: String] = [:]
            switch id {
            case "google.google_search":
                let search = args["searchTypes"]
                let types = [("webSearch", "web_search"), ("imageSearch", "image_search")].compactMap { key, wire in
                    search?[key] != nil && search?[key] != .null ? JSONValue.string(wire) : nil
                }
                if !types.isEmpty { value["search_types"] = .array(types) }
            case "google.code_execution", "google.url_context": break
            case "google.file_search": mapping = ["fileSearchStoreNames": "file_search_store_names", "topK": "top_k", "metadataFilter": "metadata_filter"]
            case "google.google_maps": mapping = ["latitude": "latitude", "longitude": "longitude", "enableWidget": "enable_widget"]
            case "google.computer_use":
                value["environment"] = args["environment"] ?? "browser"
                mapping = ["excludedPredefinedFunctions": "excludedPredefinedFunctions"]
            case "google.mcp_server": mapping = ["name": "name", "url": "url", "headers": "headers", "allowedTools": "allowed_tools"]
            case "google.retrieval":
                value["retrieval_types"] = args["retrievalTypes"] ?? .array(["vertex_ai_search"])
                mapping = ["vertexAiSearchConfig": "vertex_ai_search_config"]
            default:
                warnings.append(AIWarning(type: "unsupported", feature: "provider-defined tool \(id)", message: "provider-defined tool \(id) is not supported by google.interactions; tool dropped."))
                continue
            }
            for (key, wire) in mapping { if let field = args[key], field != .null { value[wire] = field } }
            output.append(.object(value))
        } else {
            var schema = tool["inputSchema"] ?? tool["parameters"] ?? tool
            if tool["inputSchema"] == nil && tool["parameters"] == nil, var object = schema.objectValue {
                for key in ["description", "providerOptions", "name"] { object.removeValue(forKey: key) }
                if object["type"] == "function" { object.removeValue(forKey: "type") }
                schema = .object(object)
            }
            output.append(.object(["type": "function", "name": .string(name), "description": tool["description"] ?? "", "parameters": schema]))
        }
    }
    let mappedChoice: JSONValue?
    if output.contains(where: { $0["type"] == "function" }) {
        switch mode {
        case "auto": mappedChoice = "auto"
        case "required": mappedChoice = "any"
        case "tool": mappedChoice = .object(["allowed_tools": .object(["mode": "validated", "tools": .array([choice?["toolName"] ?? choice?["name"] ?? ""])])])
        default: mappedChoice = nil
        }
    } else { mappedChoice = nil }
    return (output, mappedChoice, warnings)
}
