import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleGroundingChunkSource(from chunk: JSONValue, index: Int) -> AISource? {
    if let web = chunk["web"], let uri = web["uri"]?.stringValue {
        return AISource(
            id: "grounding-\(index)",
            sourceType: "url",
            url: uri,
            title: web["title"]?.stringValue,
            rawValue: chunk
        )
    }

    if let image = chunk["image"], let sourceURI = image["sourceUri"]?.stringValue {
        return AISource(
            id: "grounding-\(index)",
            sourceType: "url",
            url: sourceURI,
            title: image["title"]?.stringValue,
            rawValue: chunk
        )
    }

    if let retrievedContext = chunk["retrievedContext"] {
        if let uri = retrievedContext["uri"]?.stringValue {
            if uri.hasPrefix("http://") || uri.hasPrefix("https://") {
                return AISource(
                    id: "grounding-\(index)",
                    sourceType: "url",
                    url: uri,
                    title: retrievedContext["title"]?.stringValue,
                    rawValue: chunk
                )
            }

            let filename = googleFilename(from: uri)
            return AISource(
                id: "grounding-\(index)",
                sourceType: "document",
                title: retrievedContext["title"]?.stringValue ?? "Unknown Document",
                mediaType: googleMediaType(for: filename),
                filename: filename,
                rawValue: chunk
            )
        }

        if let fileSearchStore = retrievedContext["fileSearchStore"]?.stringValue {
            return AISource(
                id: "grounding-\(index)",
                sourceType: "document",
                title: retrievedContext["title"]?.stringValue ?? "Unknown Document",
                mediaType: "application/octet-stream",
                filename: googleFilename(from: fileSearchStore),
                rawValue: chunk
            )
        }
    }

    if let maps = chunk["maps"], let uri = maps["uri"]?.stringValue {
        return AISource(
            id: "grounding-\(index)",
            sourceType: "url",
            url: uri,
            title: maps["title"]?.stringValue,
            rawValue: chunk
        )
    }

    return nil
}

func googleFilename(from uri: String) -> String? {
    uri.split(separator: "/").last.map(String.init)
}

func googleMediaType(for filename: String?) -> String {
    guard let filename = filename?.lowercased() else {
        return "application/octet-stream"
    }
    if filename.hasSuffix(".pdf") {
        return "application/pdf"
    }
    if filename.hasSuffix(".txt") {
        return "text/plain"
    }
    if filename.hasSuffix(".docx") {
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    if filename.hasSuffix(".doc") {
        return "application/msword"
    }
    if filename.hasSuffix(".md") || filename.hasSuffix(".markdown") {
        return "text/markdown"
    }
    return "application/octet-stream"
}

func googleSourceDeduplicationKey(_ source: AISource) -> String {
    if source.sourceType == "url", let url = source.url {
        return "url:\(url)"
    }
    return "document:\(source.filename ?? source.title ?? source.id)"
}

struct GoogleGenerateContentToolCallBuffer {
    var id: String?
    var name: String?
    var arguments: [String: JSONValue] = [:]
    var inputStarted = false
    var providerMetadata: [String: JSONValue] = [:]
    var rawValue: JSONValue?
    var responseArguments: String?
    var responseAccumulator: GoogleJSONResponseArgumentAccumulator?
}

struct GoogleGenerateContentStreamingToolCalls {
    private var buffers: [Int: GoogleGenerateContentToolCallBuffer] = [:]
    private var activeIndex: Int = 0
    private let jsonResponseToolName: String?

    init(jsonResponseToolName: String? = nil) {
        self.jsonResponseToolName = jsonResponseToolName
    }

    mutating func apply(functionCall: JSONValue, rawValue: JSONValue) -> [LanguageStreamPart] {
        if functionCall.objectValue?.isEmpty == true {
            return []
        }

        let index: Int
        if functionCall["name"]?.stringValue != nil {
            index = buffers.isEmpty ? 0 : activeIndex + (buffers[activeIndex]?.name == nil ? 0 : 1)
            activeIndex = index
        } else {
            index = activeIndex
        }

        var buffer = buffers[index] ?? GoogleGenerateContentToolCallBuffer()
        if let id = functionCall["id"]?.stringValue {
            buffer.id = id
        }
        if let name = functionCall["name"]?.stringValue {
            buffer.name = name
        }
        if let providerMetadata = googleThoughtSignatureProviderMetadata(from: rawValue)["google"] {
            buffer.providerMetadata["google"] = providerMetadata
        }
        buffer.rawValue = rawValue

        var emitted: [LanguageStreamPart] = []
        let id = buffer.id ?? "tool-call-\(index)"
        if !buffer.inputStarted, let name = buffer.name {
            emitted.append(.toolInputStart(id: id, name: name, providerMetadata: buffer.providerMetadata))
            buffer.inputStarted = true
        }
        if let args = functionCall["args"] {
            let arguments = buffer.name == jsonResponseToolName ? (args.stringValue ?? googleGenerateContentArguments(args)) : googleGenerateContentArguments(args)
            buffer.arguments = args.objectValue ?? [:]
            if buffer.name == jsonResponseToolName { buffer.responseArguments = arguments }
            emitted.append(.toolCallDelta(id: buffer.id, name: buffer.name, argumentsDelta: arguments, index: index))
            if buffer.inputStarted {
                emitted.append(.toolInputDelta(id: id, delta: arguments, providerMetadata: buffer.providerMetadata))
            }
        }
        if let partialArgs = functionCall["partialArgs"]?.arrayValue {
            if buffer.name == jsonResponseToolName {
                var accumulator = buffer.responseAccumulator ?? GoogleJSONResponseArgumentAccumulator()
                let delta = accumulator.apply(partialArgs)
                buffer.responseAccumulator = accumulator
                if buffer.inputStarted, !delta.isEmpty {
                    emitted.append(.toolInputDelta(id: id, delta: delta, providerMetadata: buffer.providerMetadata))
                }
            } else {
                var accumulator = buffer.responseAccumulator ?? GoogleJSONResponseArgumentAccumulator()
                for partialArg in partialArgs {
                    let textDelta = accumulator.apply([partialArg])
                    var snapshot = accumulator
                    _ = snapshot.finish()
                    if let parsed = try? decodeJSONBody(Data(snapshot.jsonText.utf8)), let arguments = parsed.objectValue {
                        buffer.arguments = arguments
                    }
                    let argumentsDelta = googleGenerateContentArguments(.object(buffer.arguments))
                    emitted.append(.toolCallDelta(id: buffer.id, name: buffer.name, argumentsDelta: argumentsDelta, index: index))
                    if buffer.inputStarted, !textDelta.isEmpty {
                        emitted.append(.toolInputDelta(id: id, delta: textDelta, providerMetadata: buffer.providerMetadata))
                    }
                }
                buffer.responseAccumulator = accumulator
            }
        }

        buffers[index] = buffer
        return emitted
    }

    mutating func finishedParts() -> [LanguageStreamPart] {
        var parts: [LanguageStreamPart] = []
        for index in buffers.keys.sorted() {
            guard var buffer = buffers[index], let name = buffer.name else { continue }
            let id = buffer.id ?? "tool-call-\(index)"
            if !buffer.inputStarted {
                parts.append(.toolInputStart(id: id, name: name, providerMetadata: buffer.providerMetadata))
                buffer.inputStarted = true
                buffers[index] = buffer
            }
            if var accumulator = buffer.responseAccumulator {
                let closing = accumulator.finish()
                if !closing.isEmpty {
                    parts.append(.toolInputDelta(id: id, delta: closing, providerMetadata: buffer.providerMetadata))
                }
                buffer.responseArguments = accumulator.jsonText
            }
            parts.append(.toolInputEnd(id: id, providerMetadata: buffer.providerMetadata))
            parts.append(.toolCall(AIToolCall(
                id: buffer.id ?? "tool-call-\(index)",
                name: name,
                arguments: buffer.responseArguments ?? googleGenerateContentArguments(.object(buffer.arguments)),
                providerMetadata: buffer.providerMetadata,
                rawValue: buffer.rawValue
            )))
        }
        return parts
    }
}

/// The response tool supplies JSON text directly; unlike the legacy tool-call
/// snapshots, its partial arguments must concatenate into a valid JSON object.
struct GoogleJSONResponseArgumentAccumulator {
    private struct Container {
        var segment: String
        var isArray: Bool
        var childCount = 0
    }
    private var containers: [Container] = []
    private var stringOpen = false
    private var stringPaths: Set<String> = []
    private var seenPaths: Set<String> = []
    private(set) var jsonText = ""

    mutating func apply(_ partialArgs: [JSONValue]) -> String {
        var delta = ""
        for arg in partialArgs {
            guard let rawPath = arg["jsonPath"]?.stringValue else { continue }
            let isArrayTerminator = rawPath.hasSuffix("[]")
            let path = isArrayTerminator ? String(rawPath.dropLast(2)) : rawPath
            guard path.hasPrefix("$."), path.count > 2 else { continue }
            let segments = Self.pathSegments(String(path.dropFirst(2)))
            guard let leaf = segments.last else { continue }
            if let string = arg["stringValue"]?.stringValue, stringPaths.contains(path) {
                let escaped = googleGenerateContentArguments(.string(string))
                delta += String(escaped.dropFirst().dropLast())
                continue
            }
            let value: JSONValue
            var isEmptyContainer = false
            if let text = arg["stringValue"]?.stringValue { value = .string(text) }
            else if let number = arg["numberValue"]?.doubleValue { value = .number(number) }
            else if let bool = arg["boolValue"]?.boolValue { value = .bool(bool) }
            else if arg.objectValue?["nullValue"] != nil { value = .null }
            else if let supplied = arg["value"] { value = supplied }
            else {
                guard !seenPaths.contains(segments.joined(separator: "/")), arg["willContinue"] != true else { continue }
                isEmptyContainer = true
                value = isArrayTerminator ? .array([JSONValue]()) : .object([:])
            }
            for depth in 1...segments.count { seenPaths.insert(segments.prefix(depth).joined(separator: "/")) }
            if stringOpen { delta += "\""; stringOpen = false }
            if containers.isEmpty { containers.append(Container(segment: "", isArray: false)); delta += "{" }
            let parents = Array(segments.dropLast())
            var commonDepth = 1
            while commonDepth < containers.count, commonDepth <= parents.count,
                  containers[commonDepth].segment == parents[commonDepth - 1] { commonDepth += 1 }
            while containers.count > commonDepth { delta += containers.removeLast().isArray ? "]" : "}" }
            for position in (containers.count - 1)..<parents.count {
                delta += startChild(parents[position])
                let child = position + 1 < parents.count ? parents[position + 1] : leaf
                let isArray = child.hasPrefix("[")
                delta += isArray ? "[" : "{"
                containers.append(Container(segment: parents[position], isArray: isArray))
            }
            delta += startChild(leaf)
            if isEmptyContainer {
                delta += isArrayTerminator ? "[" : "{"
                containers.append(Container(segment: leaf, isArray: isArrayTerminator))
                continue
            }
            let json = googleGenerateContentArguments(value)
            if value.stringValue != nil {
                stringPaths.insert(path)
            }
            if value.stringValue != nil, arg["willContinue"]?.boolValue == true {
                delta += String(json.dropLast())
                stringOpen = true
            } else {
                delta += json
            }
        }
        jsonText += delta
        return delta
    }

    mutating func finish() -> String {
        var delta = ""
        if stringOpen { delta += "\""; stringOpen = false }
        if containers.isEmpty { delta = "{}" }
        while !containers.isEmpty { delta += containers.removeLast().isArray ? "]" : "}" }
        jsonText += delta
        return delta
    }

    private mutating func startChild(_ segment: String) -> String {
        let position = containers.count - 1
        let comma = containers[position].childCount > 0 ? "," : ""
        containers[position].childCount += 1
        return comma + (segment.hasPrefix("[") ? "" : googleGenerateContentArguments(.string(segment)) + ":")
    }

    private static func pathSegments(_ path: String) -> [String] {
        guard let expression = try? NSRegularExpression(pattern: #"[^.\[\]]+|\[\d+\]"#) else { return [] }
        let string = path as NSString
        return expression.matches(in: path, range: NSRange(location: 0, length: string.length)).map { string.substring(with: $0.range) }
    }
}