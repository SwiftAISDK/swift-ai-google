import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIGoogle

// @ai-sdk/google@4.0.121, exact release interactions tests: generation_config
// precedence, Hydrogen effort, compatibility fallback, agent dropping and stream-start.
@Suite struct Weekly20261115InteractionsReasoningTests {
    @Test(arguments: ["gemini-4-hydrogen", "gemini-4-argon", "gemini-3-flash-preview"])
    func mapsReasoningAndPreservesWarnings(modelID: String) async throws {
        let transport = RecordingTransport(response: try weeklyInteractionsJSONResponse())
        let provider = try AIProviders.google(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.interactionsModel(modelID).generate(.init(messages: [.user("Hello, how are you?")], reasoning: "xhigh"))
        let body = try decodeJSONBody(#require(await transport.requests().first?.body))
        #expect(!(await transport.requests()).first!.headers.keys.contains { $0.lowercased() == "api-revision" })
        #expect(body["generation_config"]?["thinking_level"] == .string(modelID == "gemini-4-hydrogen" ? "xhigh" : "high"))
        let expected: [AIWarning] = modelID == "gemini-4-hydrogen" ? [] : [
            AIWarning(type: "compatibility", feature: "reasoning", message: "reasoning \"xhigh\" is not directly supported by this model. mapped to effort \"high\".")
        ]
        #expect(result.warnings == expected)
    }

    @Test func explicitThinkingLevelOverridesReasoningWithoutWarnings() async throws {
        let transport = RecordingTransport(response: try weeklyInteractionsJSONResponse())
        let provider = try AIProviders.google(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.interactionsModel("gemini-3-flash-preview").generate(.init(
            messages: [.user("Hello, how are you?")], reasoning: "xhigh",
            providerOptions: ["google": ["thinkingLevel": "high", "thinkingSummaries": "auto"]]
        ))
        let body = try decodeJSONBody(#require(await transport.requests().first?.body))
        #expect(body["generation_config"]?["thinking_level"] == "high")
        #expect(body["generation_config"]?["thinking_summaries"] == "auto")
        #expect(result.warnings.isEmpty)
    }

    @Test func agentDropsReasoningAndEveryUnsupportedGenerationControl() async throws {
        let transport = RecordingTransport(response: try weeklyInteractionsJSONResponse())
        let provider = try AIProviders.google(configuration: .init(apiKey: "test-api-key", transport: transport))
        let result = try await provider.interactionsAgent("deep-research-pro-preview-12-2025").generate(.init(
            messages: [.user("Hello, how are you?")], temperature: 0.5, topP: 0.9, topK: 10,
            presencePenalty: 0.5, frequencyPenalty: 0.5, reasoning: "high",
            providerOptions: ["google": ["thinkingLevel": "high"]]
        ))
        let body = try decodeJSONBody(#require(await transport.requests().first?.body))
        #expect(body["generation_config"] == nil)
        let warning = try #require(result.warnings.first { $0.message?.contains("not supported when an agent is set") == true })
        for field in ["temperature", "topP", "topK", "frequencyPenalty", "presencePenalty", "reasoning", "thinkingLevel"] {
            #expect(warning.message?.contains(field) == true)
        }
    }

    @Test func streamingMapsReasoningAndStartsWithEmptyWarnings() async throws {
        let transport = RecordingTransport(response: try weeklyInteractionsStreamResponse())
        let provider = try AIProviders.google(configuration: .init(apiKey: "key", transport: transport))
        var parts: [LanguageStreamPart] = []
        for try await part in provider.interactionsModel("gemini-4-hydrogen").stream(.init(messages: [.user("Hello, how are you?")], reasoning: "xhigh")) { parts.append(part) }
        let body = try decodeJSONBody(#require(await transport.requests().first?.body))
        #expect(body["generation_config"]?["thinking_level"] == "xhigh")
        #expect(!(await transport.requests()).first!.headers.keys.contains { $0.lowercased() == "api-revision" })
        guard case let .streamStart(warnings) = parts.first else { Issue.record("Missing stream-start"); return }
        #expect(warnings.isEmpty)
        guard case let .finishMetadata(reason, usage, _) = parts.last else { Issue.record("Missing finish metadata"); return }
        #expect(reason == "stop")
        #expect(usage?.inputTokens == 7)
        #expect(usage?.totalTokens == 148)
    }
}

private func weeklyInteractionsJSONResponse() throws -> AIHTTPResponse {
    let url = try #require(Bundle.module.url(forResource: "interactions-basic", withExtension: "json"))
    return AIHTTPResponse(statusCode: 200, headers: ["content-type": "application/json"], body: try Data(contentsOf: url))
}

private func weeklyInteractionsStreamResponse() throws -> AIHTTPResponse {
    let url = try #require(Bundle.module.url(forResource: "interactions-basic.chunks", withExtension: "txt"))
    let chunks = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    return sseResponse(chunks.map { "data: \($0)\n\n" }.joined())
}
