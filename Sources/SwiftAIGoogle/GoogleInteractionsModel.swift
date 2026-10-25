import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class GoogleInteractionsLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let agent: String?
    private let config: GoogleModelConfiguration

    init(modelID: String, agent: String?, config: GoogleModelConfiguration) {
        self.providerID = "\(config.providerID).interactions"
        self.modelID = modelID
        self.agent = agent
        self.config = config
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try googleInteractionsPreparedCall(for: request, modelID: modelID, agent: agent, stream: false)
        let raw = try await sendInteractions(body: .object(prepared.body), headers: request.headers, abortSignal: request.abortSignal)
        let final = try await resolvedInteraction(raw, isBackground: prepared.body["background"]?.boolValue == true, requestHeaders: request.headers, abortSignal: request.abortSignal)
        let output = googleInteractionsParsedOutput(from: final)
        guard !output.content.isEmpty else {
            throw AIError.invalidResponse(provider: providerID, message: "No supported output found in Google Interactions response.")
        }
        return TextGenerationResult(
            text: output.text,
            content: output.content,
            reasoning: output.reasoning,
            finishReason: googleInteractionsFinishReason(status: final["status"]?.stringValue, hasFunctionCall: output.hasFunctionCall),
            usage: googleInteractionsUsage(from: final),
            sources: googleInteractionsSources(from: final),
            providerMetadata: googleInteractionsProviderMetadata(from: final),
            rawValue: final,
            warnings: prepared.warnings
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try googleInteractionsPreparedCall(for: request, modelID: modelID, agent: agent, stream: true)
                    let httpRequest = try config.request(
                        path: "/interactions",
                        modelID: modelID,
                        body: .object(prepared.body),
                        headers: googleInteractionsHeaders(request.headers),
                        abortSignal: request.abortSignal
                    )
                    let response = try await config.streamRequest(httpRequest)
                    guard (200..<300).contains(response.statusCode) else {
                        throw config.httpStatusError( try await bufferedHTTPResponse(from: response, request: httpRequest))
                    }

                    if !prepared.warnings.isEmpty {
                        continuation.yield(.streamStart(warnings: prepared.warnings))
                    }
                    var toolCalls = GoogleInteractionsStreamingToolCalls()
                    var content = GoogleInteractionsStreamingContent()
                    var hasFunctionCall = false
                    var sourceCounter = 0
                    var emittedSourceKeys: Set<String> = []
                    var didEmitFinish = false
                    var didReceiveStreamError = false
                    var interactionID: String?
                    var finishStatus: String?
                    for try await event in serverSentEvents(from: response.body) {
                        if event.data == "[DONE]" { break }
                        let raw: JSONValue
                        do { raw = try decodeJSONBody(Data(event.data.utf8)) }
                        catch {
                            didReceiveStreamError = true
                            continuation.yield(.error(message: error.localizedDescription))
                            continue
                        }
                        if request.includeRawChunks {
                            continuation.yield(.raw(raw))
                        }
                        if let providerError = googleInteractionsStreamProviderError(from: raw) {
                            didReceiveStreamError = true
                            continuation.yield(.providerError(providerError))
                            continue
                        }
                        let eventType = raw["event_type"]?.stringValue
                        if ["interaction.status_update", "interaction.in_progress", "interaction.requires_action"].contains(eventType ?? "") {
                            finishStatus = raw["status"]?.stringValue
                                ?? (eventType == "interaction.requires_action" ? "requires_action" : "in_progress")
                        }
                        let stepType = raw["step"]?["type"]?.stringValue
                        let defersBuiltinResultSources = eventType == "step.start"
                            && stepType.map { googleInteractionsBuiltinToolResultTypes.contains($0) } == true
                        if !defersBuiltinResultSources {
                            for source in googleInteractionsSources(from: raw, sourceCounter: &sourceCounter, emittedKeys: &emittedSourceKeys) {
                                continuation.yield(.source(source))
                            }
                        }
                        if let interaction = raw["interaction"] {
                            interactionID = interaction["id"]?.stringValue ?? interactionID
                            let metadata = googleInteractionsProviderMetadata(from: interaction)
                            if !metadata.isEmpty {
                                continuation.yield(.metadata(metadata))
                            }
                        }
                        if eventType == "step.start" {
                            for part in content.start(raw["step"], index: raw["index"]?.intValue, interactionID: interactionID) {
                                continuation.yield(part)
                            }
                        }
                        if eventType == "step.start",
                           raw["step"]?["type"]?.stringValue == "function_call" {
                            hasFunctionCall = true
                            for part in toolCalls.start(
                                step: raw["step"],
                                index: raw["index"]?.intValue,
                                interactionID: interactionID
                            ) {
                                continuation.yield(part)
                            }
                        }
                        if eventType == "step.delta", let delta = raw["delta"] {
                            for part in content.delta(delta, index: raw["index"]?.intValue, interactionID: interactionID) {
                                continuation.yield(part)
                            }
                            if delta["type"]?.stringValue == "arguments_delta" {
                                hasFunctionCall = true
                                for part in toolCalls.delta(
                                    delta,
                                    index: raw["index"]?.intValue,
                                    interactionID: interactionID
                                ) {
                                    continuation.yield(part)
                                }
                            }
                        }
                        if eventType == "step.stop" {
                            for part in content.stop(
                                index: raw["index"]?.intValue,
                                sourceCounter: &sourceCounter,
                                emittedSourceKeys: &emittedSourceKeys
                            ) {
                                continuation.yield(part)
                            }
                            for part in toolCalls.stop(index: raw["index"]?.intValue) {
                                continuation.yield(part)
                            }
                        }
                        if eventType == "interaction.completed" || eventType == "interaction.failed" || eventType == "interaction.incomplete" || eventType == "interaction.cancelled" {
                            let interaction = raw["interaction"] ?? raw
                            for part in content.finishParts(
                                sourceCounter: &sourceCounter,
                                emittedSourceKeys: &emittedSourceKeys
                            ) {
                                continuation.yield(part)
                            }
                            continuation.yield(.finishMetadata(
                                reason: googleInteractionsFinishReason(status: interaction["status"]?.stringValue ?? finishStatus, hasFunctionCall: hasFunctionCall),
                                usage: googleInteractionsUsage(from: interaction),
                                providerMetadata: googleInteractionsProviderMetadata(from: interaction)
                            ))
                            didEmitFinish = true
                            break
                        }
                    }
                    if !didEmitFinish {
                        for part in content.finishParts(
                            sourceCounter: &sourceCounter,
                            emittedSourceKeys: &emittedSourceKeys
                        ) {
                            continuation.yield(part)
                        }
                        continuation.yield(.finishMetadata(
                            reason: didReceiveStreamError ? "error" : finishStatus == nil || finishStatus == "in_progress"
                                ? "interrupted" : googleInteractionsFinishReason(status: finishStatus, hasFunctionCall: hasFunctionCall),
                            usage: nil,
                            providerMetadata: [:]
                        ))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func sendInteractions(body: JSONValue, headers: [String: String], abortSignal: AIAbortSignal?) async throws -> JSONValue {
        let response = try await config.transport.send(config.request(path: "/interactions", modelID: modelID, body: body, headers: googleInteractionsHeaders(headers), abortSignal: abortSignal))
        guard (200..<300).contains(response.statusCode) else {
            throw config.httpStatusError( response)
        }
        return try response.jsonValue()
    }

    private func resolvedInteraction(_ raw: JSONValue, isBackground: Bool, requestHeaders: [String: String], abortSignal: AIAbortSignal?) async throws -> JSONValue {
        guard agent != nil || isBackground,
              !googleInteractionsIsTerminal(raw["status"]?.stringValue),
              let id = raw["id"]?.stringValue else {
            return raw
        }
        return try await pollInteraction(id: id, requestHeaders: requestHeaders, timeoutNanoseconds: googleInteractionsPollTimeout(raw: raw), abortSignal: abortSignal)
    }

    private func pollInteraction(id: String, requestHeaders: [String: String], timeoutNanoseconds: UInt64, abortSignal: AIAbortSignal?) async throws -> JSONValue {
        let started = DispatchTime.now().uptimeNanoseconds
        repeat {
            let response = try await config.transport.send(AIHTTPRequest(
                method: "GET",
                url: try requireURL("\(config.baseURL)/interactions/\(id)"),
                headers: config.headers.mergingHeaders(googleInteractionsHeaders(requestHeaders)),
                abortSignal: abortSignal
            ))
            guard (200..<300).contains(response.statusCode) else {
                throw config.httpStatusError( response)
            }
            let raw = try response.jsonValue()
            if googleInteractionsIsTerminal(raw["status"]?.stringValue) {
                return raw
            }
            try await sleepWithAbortSignal(nanoseconds: 10_000_000, abortSignal: abortSignal)
        } while DispatchTime.now().uptimeNanoseconds - started < timeoutNanoseconds

        throw AIError.invalidResponse(provider: providerID, message: "Google Interactions polling timed out.")
    }
}