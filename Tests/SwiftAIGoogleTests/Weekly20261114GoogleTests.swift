import Foundation
import Testing
@testable import SwiftAIGoogle
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

@Suite("Weekly20261114 Google native decisions and request metadata")
struct Weekly20261114GoogleTests {
    @Test func nativeDecisionsPreserveOrderedEvidenceAndProbabilities() async throws {
        let raw = #"{"answers":{"yes":{"type":"noul","noul":0.9,"confidence":0.8},"team":{"type":"choice","choice":"a","probabilities":{"a":0.7,"b":0.3}},"score":{"type":"score","score":0.5,"probabilities":{"0":0.5,"1":0.5},"legend":{"0":"low","1":"high"}}},"usage":{"total_input_tokens":193,"total_output_tokens":0,"input_tokens_by_modality":[{"modality":"text","tokens":193}]}}"#
        let transport = RecordingTransport(response: jsonResponse(raw))
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", baseURL: "https://example.com/v1beta", headers: ["x-provider": "configured"], transport: transport))
        let model = try provider.decisionModel("gemini-decision-1")
        #expect(model.providerID == "google.decision")
        let result = try await model.doDecide(.init(state: [.text("text"), .json(["data": 1]), .file(mediaType: "audio/wav", data: .data(Data([1,2]))), .file(mediaType: "application/pdf", data: .url(URL(string: "https://example.com/file.pdf")!))],
            questions: ["yes": .boolean(instructions: "Is it true?", criteria: ["true": "yes", "false": .null]),
                        "team": .choice(instructions: ["question": "Choose"], criteria: ["a": "One", "b": .null]),
                        "score": .score(instructions: "Rate", criteria: ["low", "high"])], headers: ["x-call": "forwarded"]))
        #expect(result.answers["yes"] == .boolean(probability: 0.9))
        #expect(result.answers["score"] == .score(score: 0.5, probabilities: ["0": 0.5, "1": 0.5]))
        #expect(result.usage?.inputTokens == 193); #expect(result.usage?.outputTokens == 0)
        #expect(result.providerMetadata["google"]?["confidence"]?["yes"] == 0.8)
        #expect(result.providerMetadata["google"]?["legend"]?["score"]?["1"] == "high")
        let request = try #require(await transport.requests().first)
        let body = try decodeJSONBody(#require(request.body))
        #expect(request.url.path == "/v1beta/decisions")
        #expect(request.headers["x-goog-api-key"] == "key"); #expect(request.headers["x-call"] == "forwarded")
        #expect(body["input"]?[0] == ["type": "text", "text": "text"])
        #expect(body["input"]?[2] == ["type": "audio", "mime_type": "audio/wav", "data": "AQI="])
        #expect(body["questions"]?["yes"]?["criteria"] == ["true_description": "yes"])
    }
    @Test func customNativeDecisionFactoryAndLegacyDecisionRoutingAreExplicit() throws {
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", name: "custom"))
        #expect(try provider.decisionModel("gemini-3.8-flash") is DecisionLanguageModel)
        #expect(provider.nativeDecisionModel("custom-decision").providerID == "custom.decision")
        #expect(try provider.extractionModel("gemini-3.8-flash").providerID == "custom.extraction")
    }
    @Test func extractionUsesSharedTypedContractWithGeminiStructuredOutput() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"{\"f0\":42}"}]},"finishReason":"STOP"}]}"#))
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.extractionModel("gemini-3.8-flash").doExtract(.init(state: [.text("Total: 42 EUR")], fields: ["total": .init(type: .number, instructions: "Invoice total", unit: "EUR")]))
        #expect(result.answers["total"]?.value == 42)
        let request = try #require(await transport.requests().first)
        #expect(request.url.path.hasSuffix(":generateContent"))
        let body = try decodeJSONBody(#require(request.body))
        #expect(body["generationConfig"]?["responseMimeType"] == "application/json")
    }

    @Test func safetyBlockReturnsRefusalsAndUnsupportedOptionsWarn() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"prompt_feedback":{"block_reason":"SAFETY","block_reason_message":"blocked"}}"#))
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.nativeDecisionModel("custom").doDecide(.init(state: [], questions: ["q": .boolean(instructions: "Question")], maxOutputTokens: 50, reasoning: "high", providerOptions: ["google": ["thinkingConfig": [:]]]))
        #expect(result.answers["q"] == .refusal)
        #expect(Set(result.warnings.compactMap(\.feature)) == ["maxOutputTokens", "reasoning", "providerOptions.google.thinkingConfig"])
    }
    @Test(arguments: [JSONValue.object(["answers": [:]]), ["answers": ["q": ["type": "noul", "noul": 1.1]]], ["answers": ["q": ["type": "choice", "choice": "unknown", "probabilities": [:]]]]])
    func invalidNativeAnswersFail(_ response: JSONValue) async throws {
        let transport = RecordingTransport(response: .init(statusCode: 200, body: try encodeJSONBody(response)))
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport))
        await #expect(throws: AIError.self) { try await provider.nativeDecisionModel("m").doDecide(.init(state: [], questions: ["q": .boolean(instructions: "Question")])) }
    }
    @Test func invalidQuestionsAndReferencesFailBeforeRequest() async throws {
        let transport = RecordingTransport(response: jsonResponse("{}"))
        let model = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport)).nativeDecisionModel("m")
        for questions: [String: AIDecisionQuestion] in [[:], ["q": .boolean()], ["q": .choice(instructions: "Q", criteria: ["a": "one"])]] {
            await #expect(throws: AIError.self) { try await model.doDecide(.init(state: [], questions: questions)) }
        }
        await #expect(throws: AIDecisionUnsupportedQuestionTypeError.self) { try await model.doDecide(.init(state: [], questions: ["q": .box(instructions: "Q")])) }
        await #expect(throws: AIDecisionUnsupportedFunctionalityError.self) { try await model.doDecide(.init(state: [.file(mediaType: "image/png", data: .reference("f"))], questions: ["q": .boolean(instructions: "Q")])) }
        #expect(await transport.requests().isEmpty)
    }
    @Test(arguments: ["gemini-4-hydrogen", "models/gemini-4-hydrogen-eap"])
    func hydrogenSupportsXhighAndLowMinimum(_ modelID: String) throws {
        var warnings: [AIWarning] = []
        #expect(googleThinkingConfig(for: "xhigh", modelID: modelID, warnings: &warnings) == ["thinkingLevel": "xhigh"])
        #expect(warnings.isEmpty)
        #expect(googleThinkingConfig(for: "none", modelID: modelID, warnings: &warnings) == ["thinkingLevel": "low"])
    }
    @Test func serviceTierHeaderFallbackHasBodyPrecedenceInGenerateAndStream() async throws {
        for tier: JSONValue in [.null, "priority"] {
            let raw: JSONValue = ["candidates": [["content": ["parts": [["text": "ok"]]], "finishReason": "STOP"]], "usageMetadata": ["serviceTier": tier]]
            let transport = RecordingTransport(response: .init(statusCode: 200, headers: ["x-gemini-service-tier": "flex"], body: try encodeJSONBody(raw)))
            let model = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("gemini-4-carbon")
            let expected: JSONValue = tier == .null ? "flex" : tier
            #expect(try await model.generate(.init(messages: [.user("Hi")])).providerMetadata["google"]?["serviceTier"] == expected)
            var state = GoogleGenerateContentStreamState(response: .init(statusCode: 200, headers: ["X-Gemini-Service-Tier": "flex"]), includeRawChunks: false, modelID: "m", warnings: [])
            _ = state.apply(raw)
            let finish = state.finish().compactMap { part -> [String: JSONValue]? in
                if case let .finishMetadata(_, _, metadata) = part { return metadata }; return nil
            }
            #expect(finish.first?["google"]?["serviceTier"] == expected)
        }
    }
    @Test func documentDedupUsesFullURIAndFileStoreTitle() {
        let contexts: [JSONValue] = [["uri": "gs://bucket/a/report.pdf"], ["uri": "gs://bucket/b/report.pdf"], ["fileSearchStore": "stores/a", "title": "One"], ["fileSearchStore": "stores/a", "title": "Two"]]
        let sources = contexts.enumerated().compactMap { googleGroundingChunkSource(from: ["retrievedContext": $0.element], index: $0.offset) }
        #expect(Set(sources.map(googleSourceDeduplicationKey)).count == 4)
    }
    @Test func documentSourcesStreamAndDeduplicateAcrossChunks() throws {
        var state = GoogleGenerateContentStreamState(response: .init(statusCode: 200), includeRawChunks: false, modelID: "m", warnings: [])
        let raw: JSONValue = ["candidates": [["groundingMetadata": ["groundingChunks": [
            ["retrievedContext": ["uri": "gs://bucket/a/report.pdf", "title": "A"]],
            ["retrievedContext": ["uri": "gs://bucket/b/report.pdf", "title": "B"]],
            ["retrievedContext": ["fileSearchStore": "stores/a", "title": "One"]]]]]]]
        let first = state.apply(raw).compactMap { part -> AISource? in if case let .source(source) = part { return source }; return nil }
        let second = state.apply(raw).compactMap { part -> AISource? in if case let .source(source) = part { return source }; return nil }
        #expect(first.map(\.sourceType) == ["document", "document", "document"])
        #expect(first.map(\.title) == ["A", "B", "One"])
        #expect(second.isEmpty)
    }

    @Test func dottedArrayIndexesRemainContiguousAndObjectDigitsStayKeys() {
        var accumulator = GoogleJSONResponseArgumentAccumulator()
        _ = accumulator.apply([["jsonPath": "$.rows[0][0]", "stringValue": "a", "willContinue": true],
            ["jsonPath": "$.rows.0.0", "stringValue": "b"], ["jsonPath": "$.rows.0.4", "stringValue": "ignored"],
            ["jsonPath": "$.rows.0.1", "stringValue": "c"], ["jsonPath": "$.rows.1[0]", "stringValue": "d"]])
        _ = accumulator.finish()
        #expect(accumulator.jsonText == #"{"rows":[["ab","c"],["d"]]}"#)
        var object = GoogleJSONResponseArgumentAccumulator()
        _ = object.apply([["jsonPath": "$.items.01.name", "stringValue": "a"]]); _ = object.finish()
        #expect(object.jsonText == #"{"items":{"01":{"name":"a"}}}"#)
    }
}
