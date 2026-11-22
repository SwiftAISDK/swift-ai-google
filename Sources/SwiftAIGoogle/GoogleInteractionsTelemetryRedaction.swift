import SwiftAIProvider

@_spi(Vertex)
public func googleRedactInteractionsTelemetry(_ value: JSONValue) -> JSONValue {
    guard var object = value.objectValue else { return value }

    if object["id"] == "google.mcp_server" {
        object["args"] = object["args"].map(googleRedactMCPHeaders)
    }
    if object["type"] == "mcp_server" {
        return googleRedactMCPHeaders(.object(object))
    }
    if let tools = object["tools"]?.arrayValue {
        object["tools"] = .array(tools.map { tool in
            tool["type"] == "mcp_server" ? googleRedactMCPHeaders(tool) : tool
        })
    }
    return .object(object)
}

private func googleRedactMCPHeaders(_ value: JSONValue) -> JSONValue {
    guard var object = value.objectValue, let headers = object["headers"]?.objectValue else { return value }
    object["headers"] = .object(headers.mapValues { _ in "[redacted]" })
    return .object(object)
}
