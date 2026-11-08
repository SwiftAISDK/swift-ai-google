import Foundation
import Testing
@testable import SwiftAIGoogle
import SwiftAIProvider
import SwiftAIProviderUtilities
import SwiftAITransport

@Suite struct Weekly20261108GoogleTests {
    @Test func signedEmptyTextAndReasoningSurviveReplay() throws {
        var warnings: [AIWarning] = []
        let signature: [String: JSONValue] = ["google": ["thoughtSignature": "opaque"]]
        let body = try googleGenerateContentMessageJSON(.init(role: .assistant, content: [.text("", providerMetadata: signature), .reasoning("", providerMetadata: signature), .text("")]), modelID: "gemini-3.8-flash", warnings: &warnings)
        #expect(body["parts"] == [["text": "", "thoughtSignature": "opaque"], ["text": "", "thought": true, "thoughtSignature": "opaque"]])
    }

    @Test func toolInputStringsParseAsJSONOrRetainMalformedInput() {
        _ = GoogleContinuedGenerateContentResponse.result(providerID:modelID:prepared:jsonResponseToolName:)
        #expect(googleToolArguments(#"{"query":"swift"}"#) == ["query": "swift"])
        #expect(googleToolArguments("42") == 42)
        #expect(googleToolArguments("not json") == "not json")
    }

    @Test func sparseIndexesDoNotAllocateOrLeaveJSONGaps() throws {
        for args: [JSONValue] in [
            [["jsonPath": "$.items[20000000]", "numberValue": 1]],
            [["jsonPath": "$.items", "nullValue": .null], ["jsonPath": "$.items[20000000]", "numberValue": 1], ["jsonPath": "$.rows[0]", "nullValue": .null], ["jsonPath": "$.rows[0][3000000]", "numberValue": 2]],
            [["jsonPath": "$.items[999999999999999999999999]", "numberValue": 1]]
        ] {
            var accumulator = GoogleJSONResponseArgumentAccumulator()
            _ = accumulator.apply(args); _ = accumulator.finish()
            #expect(accumulator.jsonText.count < 100)
            _ = try decodeJSONBody(Data(accumulator.jsonText.utf8))
        }
        var accumulator = GoogleJSONResponseArgumentAccumulator()
        _ = accumulator.apply([["jsonPath": "$.items[0]", "stringValue": "a"], ["jsonPath": "$.items[5]", "stringValue": "b"], ["jsonPath": "$.items[1]", "stringValue": "c"]])
        _ = accumulator.finish()
        #expect(try decodeJSONBody(Data(accumulator.jsonText.utf8)) == ["items": ["a", "c"]])
        var nested = GoogleJSONResponseArgumentAccumulator()
        _ = nested.apply([["jsonPath": "$.rows[0][0]", "numberValue": 1], ["jsonPath": "$.rows[0][3000000]", "numberValue": 2], ["jsonPath": "$.rows[2000000].name", "stringValue": "x"]]); _ = nested.finish()
        #expect(try decodeJSONBody(Data(nested.jsonText.utf8)) == ["rows": [[1]]])
    }

    @Test func seedZeroAndSchemaPropertyOrderingReachProviderRequests() async throws {
        let transport = RecordingTransport(responses: [jsonResponse(#"{"name":"operations/zero"}"#), jsonResponse(#"{"name":"operations/zero","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"https://example.com/video"}}]}}}"#), jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"{}"}]},"finishReason":"STOP"}]}"#)])
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport))
        _ = try await provider.videoModel("veo-3.1-generate-preview").generateVideo(.init(prompt: "paint", seed: 0, providerOptions: ["google": ["pollIntervalMs": 0, "pollTimeoutMs": 1000]]))
        _ = try await provider.languageModel("gemini-2.5-flash").generate(.init(messages: [.user("json")], responseFormat: .json(schema: ["type": "object", "properties": ["value": ["type": "string"]], "propertyOrdering": ["value"]])))
        let requests = await transport.requests()
        #expect(try decodeJSONBody(#require(requests[0].body))["parameters"]?["seed"] == 0)
        #expect(try decodeJSONBody(#require(requests[2].body))["generationConfig"]?["responseJsonSchema"]?["propertyOrdering"] == ["value"])
    }

    @Test func noneOmitsProviderToolsAndKeepsDisabledFunctions() throws {
        let tools: [String: JSONValue] = ["search": GoogleTools.googleSearch(), "lookup": ["type": "object", "properties": [:]]]
        let prepared = try #require(try googlePrepareTools(from: tools, toolChoice: ["type": "none"], modelID: "gemini-3-pro", isVertexProvider: false))
        #expect(prepared.tools.count == 1)
        #expect(prepared.tools.first?["functionDeclarations"] != nil)
        #expect(prepared.toolConfig?["functionCallingConfig"]?["mode"] == "NONE")
        #expect(try googlePrepareTools(from: ["search": GoogleTools.googleSearch()], toolChoice: "none", modelID: "gemini-3-pro", isVertexProvider: false) == nil)
        let interactions = googleInteractionsPrepareNativeTools(tools, choice: "none")
        #expect(interactions.tools.isEmpty && interactions.choice == nil && interactions.warnings.isEmpty)
    }

    @Test func interactionsMapsFunctionsProviderToolsAndLabels() throws {
        let tools: [String: JSONValue] = ["lookup": ["type": "function", "inputSchema": ["type": "object"]], "search": GoogleTools.googleSearch(searchTypes: ["webSearch": [:], "imageSearch": [:]])]
        let request = LanguageModelRequest(messages: [.user("Find")], tools: tools, toolChoice: ["type": "tool", "toolName": "lookup"], providerOptions: ["google": ["labels": ["team": "docs"]]])
        let prepared = try googleInteractionsPreparedCall(for: request, modelID: "gemini-3-pro", agent: nil, stream: false)
        #expect(prepared.body["labels"] == ["team": "docs"])
        #expect(prepared.body["tool_choice"] == ["allowed_tools": ["mode": "validated", "tools": ["lookup"]]])
        #expect(prepared.body["tools"]?[0]?["parameters"] == ["type": "object"])
        #expect(prepared.body["tools"]?[1]?["search_types"] == ["web_search", "image_search"])
        #expect(throws: AIError.self) { _ = try googleInteractionsPreparedCall(for: .init(messages: [.user("x")], providerOptions: ["google": ["labels": ["bad": 4]]]), modelID: "gemini", agent: nil, stream: false) }
    }

    @Test func optionAliasesChooseFirstPresentNamespace() throws {
        for values: [String: JSONValue] in [["googleVertex": [:], "vertex": ["thinkingConfig": ["thinkingBudget": 5]]], ["googleVertex": .null, "vertex": [:], "google": ["thinkingConfig": ["thinkingBudget": 5]]]] {
            let call = try googlePrepareGenerateContentOptions(from: .init(messages: [.user("x")], providerOptions: values), modelID: "gemini", providerID: "google.vertex", isVertexProvider: true)
            #expect(call.options["thinkingConfig"] == nil)
        }
        #expect(throws: AIError.self) { _ = try googlePrepareGenerateContentOptions(from: .init(messages: [.user("x")], providerOptions: ["googleVertex": "bad", "google": [:]]), modelID: "gemini", providerID: "google.vertex", isVertexProvider: true) }
    }

    @Test func speechLanguageMapsToConfigAndAutoOmits() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/wav","data":"AQI="}}]}}]}"#))
        let provider = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport))
        for language in ["ja-JP", "auto"] {
            _ = try await provider.speechModel("gemini-tts").speak(.init(text: "hello", language: language))
        }
        let requests = await transport.requests()
        #expect(try decodeJSONBody(#require(requests[0].body))["generationConfig"]?["speechConfig"]?["languageCode"] == "ja-JP")
        #expect(try decodeJSONBody(#require(requests[1].body))["generationConfig"]?["speechConfig"]?["languageCode"] == nil)
    }

    @Test func completedCallsKeepTextOrderAndEmptySignatures() async throws {
        let raw: JSONValue = ["candidates": [["content": ["parts": [["text": "A"], ["functionCall": ["name": "lookup", "id": "call"]], ["text": "", "thoughtSignature": "opaque"], ["text": "B"]]], "finishReason": "STOP"]]]
        let transport = RecordingTransport(responses: [jsonResponse(String(decoding: try encodeJSONBody(raw), as: UTF8.self)), sseResponse("data: " + String(decoding: try encodeJSONBody(raw), as: UTF8.self))])
        let model = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", baseURL: "https://synthetic.invalid/v1beta", transport: transport)).languageModel("gemini-2.5-flash")
        let result = try await model.generate(.init(messages: [.user("x")]))
        #expect(result.content.compactMap { part -> String? in if case let .text(text, _) = part { return "text:\(text)" }; if case let .toolCall(call) = part { return "call:\(call.name)" }; return nil } == ["text:A", "call:lookup", "text:", "text:B"])
        #expect(result.requestMetadata.url == "https://synthetic.invalid/v1beta/models/gemini-2.5-flash:generateContent")
        var parts: [LanguageStreamPart] = []
        for try await part in model.stream(.init(messages: [.user("x")])) { parts.append(part) }
        #expect(parts.compactMap { part -> String? in if case let .textDeltaPart(_, text, _) = part { return "text:\(text)" }; if case let .toolCall(call) = part { return "call:\(call.name)" }; return nil } == ["text:A", "call:lookup", "text:", "text:B"])
        let callIndex = try #require(parts.firstIndex { if case .toolCall = $0 { return true }; return false })
        #expect(parts[(callIndex + 1)...].contains { if case let .textStart(_, metadata) = $0 { return metadata["google"]?["thoughtSignature"] == "opaque" }; return false })
        #expect(parts.contains { if case let .requestMetadata(metadata) = $0 { return metadata.url?.contains(":streamGenerateContent?alt=sse") == true }; return false })
    }

    @Test func partialTerminalCallClosesTextBeforeCompletion() async throws {
        let raw = #"{"candidates":[{"content":{"parts":[{"text":"A"},{"functionCall":{"id":"partial","name":"lookup","partialArgs":[{"jsonPath":"$.query","stringValue":"example"}],"willContinue":false}},{"text":"B"}]},"finishReason":"STOP"}]}"#
        let transport = RecordingTransport(response: sseResponse("data: " + raw))
        let model = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("gemini-2.5-flash")
        var projected: [String] = []
        for try await part in model.stream(.init(messages: [.user("x")])) {
            if case let .textDeltaPart(_, text, _) = part { projected.append(text) }
            if case let .toolCall(call) = part { projected.append(call.arguments) }
        }
        #expect(projected == ["A", #"{"query":"example"}"#, "B"])
    }

    @Test func RPCStatusMakesStreamErrorsRetryable() async throws {
        let transport = RecordingTransport(response: sseResponse(#"data: {"error":{"message":"busy","status":"RESOURCE_EXHAUSTED"}}"#))
        let model = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("gemini")
        var errors: [AIStreamProviderError] = []
        for try await part in model.stream(.init(messages: [.user("x")])) { if let error = part.streamProviderError { errors.append(error) } }
        #expect(errors.first?.statusCode == 429)
        #expect(errors.first?.isRetryable == true)
    }
}

@Suite struct Weekly20261108GoogleRealtimeTests {
    private func model(_ transport: RecordingTransport = RecordingTransport(response: jsonResponse("{}"))) throws -> GoogleRealtimeModel {
        try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key", baseURL: "https://example.com/prefix/v1beta", headers: ["x-custom": "yes"], transport: transport)).realtimeModel("gemini-3.8-live-extended-thinking")
    }

    @Test func serverConnectionAndClientSecretUseDistinctEndpoints() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"name":"auth_tokens/test","expireTime":"2030-01-01T00:00:00Z"}"#))
        let model = try model(transport)
        let server = try await model.getResolvedServerWebSocketConfig()
        #expect(server.url == "wss://example.com/prefix/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
        #expect(server.headers["x-goog-api-key"] == "key" && server.headers["x-custom"] == "yes")
        let secret = try await model.doCreateClientSecret(.init(expiresAfterSeconds: 120, sessionConfig: .init(instructions: "hello")))
        #expect(secret.url.hasSuffix("v1alpha.GenerativeService.BidiGenerateContentConstrained"))
        #expect(secret.expiresAt == 1_893_456_000)
        let request = try #require(await transport.requests().first)
        #expect(request.url.absoluteString == "https://example.com/prefix/v1alpha/auth_tokens?key=key")
        #expect(request.headers["x-goog-api-key"] == nil)
        let body = try decodeJSONBody(#require(request.body))
        #expect(body["uses"] == 0)
        #expect(body["bidiGenerateContentSetup"]?["generationConfig"]?["thinkingConfig"] == ["thinkingLevel": "low"])
        #expect(model.getWebSocketConfig(token: "a/b+c", url: secret.url).url.contains("access_token=a%2Fb%2Bc"))
        await #expect(throws: AIError.self) { _ = try await model.getResolvedServerWebSocketConfig(intent: .transcription) }
        await #expect(throws: AIError.self) { _ = try await model.serializeClientEvent(.inputAudioCommit, intent: .transcription) }
        #expect(throws: AIError.self) { _ = try model.buildSessionConfig(.init(), intent: .transcription) }
    }

    @Test func sessionOptionsPreserveTypedThinkingTranslationAndRawOverrides() async throws {
        let model = try model()
        let config = AIRealtimeSessionConfiguration(voice: "Puck", inputAudioFormat: .init(type: "audio/pcm", rate: 24_000), tools: [.init(name: "lookup", parameters: ["type": "object"])], providerOptions: ["google": ["thinkingConfig": ["thinkingBudget": 0], "translationConfig": ["targetLanguageCode": "ja"], "defaultToolBehavior": "NON_BLOCKING"], "generationConfig": ["temperature": 0.2]])
        let setup = model.buildSessionConfig(config)
        #expect(setup["generationConfig"] == ["temperature": 0.2, "thinkingConfig": ["thinkingBudget": 0], "translationConfig": ["targetLanguageCode": "ja"]])
        #expect(setup["tools"]?[0]?["functionDeclarations"]?[0]?["behavior"] == "NON_BLOCKING")
        _ = try await model.serializeClientEvent(.sessionUpdate(config))
        #expect(try await model.serializeClientEvent(.inputAudioAppend(audio: "AQI=")) == .json(["realtimeInput": ["audio": ["data": "AQI=", "mimeType": "audio/pcm;rate=24000"]]]))
        #expect(try await model.serializeClientEvent(.conversationItemCreate(.functionCallOutput(callID: "c", name: "lookup", output: "[1,2]"))) == .json(["toolResponse": ["functionResponses": [["id": "c", "name": "lookup", "response": ["output": [1, 2]]]]]]))
    }

    @Test func parsersIsolateTurnsAndUsageSnapshots() throws {
        let model = try model()
        let first = model.createServerEventParser(), second = model.createServerEventParser()
        let snapshot: JSONValue = ["usageMetadata": ["promptTokenCount": 4, "responseTokenCount": 0, "promptTokensDetails": [["modality": "audio", "tokenCount": 4]]]]
        #expect(first(snapshot).first?.usage?.inputAudioTokens == 4)
        _ = first(["serverContent": ["modelTurn": ["parts": [["text": "hello"]]], "generationComplete": true]])
        let done = first(["serverContent": ["turnComplete": true]])
        #expect(done.last == .responseDone(responseID: "google-resp-0", status: "completed", raw: ["serverContent": ["turnComplete": true]]))
        #expect(done.last?.usage == nil)
        let sameFrame: JSONValue = ["serverContent": ["turnComplete": true], "usageMetadata": ["responseTokenCount": 2, "thoughtsTokenCount": 1]]
        #expect(second(sameFrame).last?.usage?.outputTokens == 3)
        let other = second(["serverContent": ["modelTurn": ["parts": [["text": "new"]]]]])
        #expect(other.first == .textDelta(responseID: "google-resp-1", itemID: "google-item-1", delta: "new", raw: ["serverContent": ["modelTurn": ["parts": [["text": "new"]]]]]))
        let independent = model.createServerEventParser()(["serverContent": ["modelTurn": ["parts": [["text": "first"]]]]])
        #expect(independent.first == .textDelta(responseID: "google-resp-0", itemID: "google-item-0", delta: "first", raw: ["serverContent": ["modelTurn": ["parts": [["text": "first"]]]]]))
    }

    @Test func transcriptionFinishedBoundariesAndLateOutputKeepTheirIDs() throws {
        let parse = try model().createServerEventParser()
        _ = parse(["inputTranscription": ["text": "The quick", "finished": false]])
        _ = parse(["serverContent": ["turnComplete": true]])
        let late = parse(["inputTranscription": ["text": " fox", "finished": true]])
        #expect(late.first == .inputTranscriptionCompleted(itemID: "google-input-0", transcript: "The quick fox", raw: ["inputTranscription": ["text": " fox", "finished": true]]))
        let next = parse(["inputTranscription": ["text": "Next", "finished": false]])
        #expect(next.first == .inputTranscriptionCompleted(itemID: "google-input-1", transcript: "Next", raw: ["inputTranscription": ["text": "Next", "finished": false]]))
        let trailing = parse(["serverContent": ["outputTranscription": ["text": "late"]]])
        #expect(trailing.first == .audioTranscriptDelta(responseID: "google-resp-0", itemID: "google-item-0", delta: "late", raw: ["serverContent": ["outputTranscription": ["text": "late"]]]))
    }
}
