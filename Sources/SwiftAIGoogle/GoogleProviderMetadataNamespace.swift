import Foundation
import SwiftAIProvider

/// Upstream writes Gemini provider metadata under every provider options name:
/// `google` for the Gemini API and `googleVertex` plus `vertex` for Vertex AI.
/// The shared parser writes `google`; hosting adapters remap it with these helpers.
public func googleProviderMetadata(_ metadata: [String: JSONValue], namespaces: [String]) -> [String: JSONValue] {
    guard namespaces != ["google"], let value = metadata["google"] else { return metadata }
    var output = metadata
    output.removeValue(forKey: "google")
    for namespace in namespaces { output[namespace] = value }
    return output
}

public func googleProviderMetadata(_ part: AIResultContentPart, namespaces: [String]) -> AIResultContentPart {
    func map(_ metadata: [String: JSONValue]) -> [String: JSONValue] { googleProviderMetadata(metadata, namespaces: namespaces) }
    switch part {
    case let .text(text, metadata): return .text(text, providerMetadata: map(metadata))
    case let .citedText(text, citations, metadata): return .citedText(text, citations: citations, providerMetadata: map(metadata))
    case let .reasoning(text, metadata): return .reasoning(text, providerMetadata: map(metadata))
    case var .source(source): source.providerMetadata = map(source.providerMetadata); return .source(source)
    case var .file(file): file.providerMetadata = map(file.providerMetadata); return .file(file)
    case var .reasoningFile(file): file.providerMetadata = map(file.providerMetadata); return .reasoningFile(file)
    case let .custom(value, metadata): return .custom(value, providerMetadata: map(metadata))
    case var .toolCall(call): call.providerMetadata = map(call.providerMetadata); return .toolCall(call)
    case var .toolResult(result): result.providerMetadata = map(result.providerMetadata); return .toolResult(result)
    case .toolApprovalRequest, .toolApprovalResponse: return part
    }
}

public func googleProviderMetadata(_ result: TextGenerationResult, namespaces: [String]) -> TextGenerationResult {
    guard namespaces != ["google"] else { return result }
    var output = result
    output.content = result.content.map { googleProviderMetadata($0, namespaces: namespaces) }
    output.toolCalls = resultToolCalls(from: output.content)
    output.toolResults = resultToolResults(from: output.content)
    output.sources = resultSources(from: output.content)
    output.files = output.content.compactMap { part in
        if case let .file(file) = part { return file }
        return nil
    }
    output.providerMetadata = googleProviderMetadata(result.providerMetadata, namespaces: namespaces)
    return output
}

public func googleProviderMetadata(_ part: LanguageStreamPart, namespaces: [String]) -> LanguageStreamPart {
    guard namespaces != ["google"] else { return part }
    func map(_ metadata: [String: JSONValue]) -> [String: JSONValue] { googleProviderMetadata(metadata, namespaces: namespaces) }
    switch part {
    case let .textStart(id, metadata): return .textStart(id: id, providerMetadata: map(metadata))
    case let .textDeltaPart(id, delta, metadata): return .textDeltaPart(id: id, delta: delta, providerMetadata: map(metadata))
    case let .textEnd(id, metadata): return .textEnd(id: id, providerMetadata: map(metadata))
    case let .textEndWithCitations(id, citations, metadata): return .textEndWithCitations(id: id, citations: citations, providerMetadata: map(metadata))
    case let .reasoningStart(id, metadata): return .reasoningStart(id: id, providerMetadata: map(metadata))
    case let .reasoningDeltaPart(id, delta, metadata): return .reasoningDeltaPart(id: id, delta: delta, providerMetadata: map(metadata))
    case let .reasoningEnd(id, metadata): return .reasoningEnd(id: id, providerMetadata: map(metadata))
    case let .toolInputStart(id, name, providerExecuted, dynamic, title, metadata):
        return .toolInputStart(id: id, name: name, providerExecuted: providerExecuted, dynamic: dynamic, title: title, providerMetadata: map(metadata))
    case let .toolInputStartText(id, name, providerExecuted, dynamic, title, metadata):
        return .toolInputStartText(id: id, name: name, providerExecuted: providerExecuted, dynamic: dynamic, title: title, providerMetadata: map(metadata))
    case let .toolInputDelta(id, delta, metadata): return .toolInputDelta(id: id, delta: delta, providerMetadata: map(metadata))
    case let .toolInputEnd(id, metadata): return .toolInputEnd(id: id, providerMetadata: map(metadata))
    case var .toolCall(call): call.providerMetadata = map(call.providerMetadata); return .toolCall(call)
    case var .toolResult(result): result.providerMetadata = map(result.providerMetadata); return .toolResult(result)
    case var .file(file): file.providerMetadata = map(file.providerMetadata); return .file(file)
    case var .reasoningFile(file): file.providerMetadata = map(file.providerMetadata); return .reasoningFile(file)
    case let .custom(value, metadata): return .custom(value, providerMetadata: map(metadata))
    case var .source(source): source.providerMetadata = map(source.providerMetadata); return .source(source)
    case let .finishMetadata(reason, usage, metadata): return .finishMetadata(reason: reason, usage: usage, providerMetadata: map(metadata))
    default: return part
    }
}
