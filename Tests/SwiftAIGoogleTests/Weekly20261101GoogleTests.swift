import Foundation
import Testing
@testable import SwiftAIGoogle
import SwiftAIProvider
import SwiftAITransport

@Suite struct Weekly20261101GoogleTests {
    @Test func videoMetadataCoversInlineAndHostedFiles() throws {
        let metadata = GoogleVideoMetadata(startOffset: "0s", endOffset: "10.5s", fps: 0.5)
        let options: [String: JSONValue] = ["google": ["videoMetadata": metadata.jsonValue]]
        let parts: [AIContentPart] = [
            .file(mimeType: "video/mp4", data: Data([0, 1, 2, 3]), providerMetadata: options),
            .data(mimeType: "video/mp4", data: Data([0, 1, 2, 3]), providerMetadata: options),
            .providerReference(mimeType: "video/mp4", reference: ["google": "gs://VideoBucket/video.mp4"], providerMetadata: options),
            .imageURL("https://www.youtube.com/watch?v=example", providerMetadata: options)
        ]
        var warnings: [AIWarning] = []
        let wire = try googleGenerateContentMessageJSON(AIMessage(role: .user, content: parts), modelID: "gemini-3.8-flash", warnings: &warnings)
        let values = try #require(wire["parts"]?.arrayValue)
        #expect(values.allSatisfy { $0["videoMetadata"] == metadata.jsonValue })
        #expect(values[0]["inlineData"] == ["mimeType": "video/mp4", "data": "AAECAw=="])
        #expect(values[2]["fileData"] == ["mimeType": "video/mp4", "fileUri": "gs://VideoBucket/video.mp4"])
        #expect(values[3]["fileData"]?["fileUri"] == "https://www.youtube.com/watch?v=example")
    }

    @Test func videoMetadataUsesEndpointNamespaceAndStripsUnknownProperties() throws {
        let part = AIContentPart.file(mimeType: "video/mp4", data: Data([1]), providerMetadata: [
            "google": ["videoMetadata": ["fps": 1, "ignored": true]],
            "googleVertex": ["videoMetadata": ["fps": 2]],
            "vertex": ["videoMetadata": ["fps": 3]]
        ])
        for (names, fps) in [(["google"], 1.0), (["googleVertex", "vertex"], 2.0), (["vertex"], 3.0)] {
            var warnings: [AIWarning] = []
            let body = try googleGenerateContentMessageJSON(AIMessage(role: .user, content: [part]), modelID: "gemini-3.8-flash", providerOptionsNames: names, warnings: &warnings)
            #expect(body["parts"]?[0]?["videoMetadata"] == ["fps": .number(fps)])
        }
        for namespace in ["google", "googleVertex", "vertex"] {
            var warnings: [AIWarning] = []
            let body = try googleGenerateContentMessageJSON(AIMessage(role: .user, content: [
                .data(mimeType: "video/mp4", data: Data([1]), providerMetadata: [namespace: ["videoMetadata": ["fps": 2]]])
            ]), modelID: "gemini-3.8-flash", warnings: &warnings)
            #expect(body["parts"]?[0]?["videoMetadata"] == ["fps": 2])
        }
    }

    @Test func videoMetadataValidatesBoundariesAndAllowsOmission() throws {
        for value: JSONValue in [["fps": 0], ["fps": -1], ["fps": 24.1], ["fps": "2"], ["startOffset": 2], ["endOffset": .null], .array([JSONValue]())] {
            #expect(throws: AIError.self) {
                _ = try googleVideoMetadata(["google": ["videoMetadata": value]], providerOptionsNames: ["google"])
            }
        }
        #expect(try googleVideoMetadata(["google": ["videoMetadata": ["fps": 24]]], providerOptionsNames: ["google"]) == ["fps": 24])
        #expect(try googleVideoMetadata(["google": ["videoMetadata": .null]], providerOptionsNames: ["google"]) == nil)
        #expect(try googleVideoMetadata(["google": [:], "vertex": ["videoMetadata": ["fps": 2]]], providerOptionsNames: ["google"]) == nil)
    }

    @Test func continuedStringFinalChunksCloseBeforeContainerFinalization() throws {
        for ending: JSONValue in [["stringValue": "ton"], ["stringValue": ""], [:]] {
            var accumulator = GoogleJSONResponseArgumentAccumulator()
            #expect(accumulator.apply([["jsonPath": "$.location", "stringValue": "Bos", "willContinue": true]]) == "{\"location\":\"Bos")
            var final = try #require(ending.objectValue)
            final["jsonPath"] = "$.location"
            #expect(accumulator.apply([.object(final)]) == (ending["stringValue"]?.stringValue ?? "") + "\"")
            #expect(accumulator.finish() == "}")
            let decoded = try decodeJSONBody(Data(accumulator.jsonText.utf8))
            #expect(decoded["location"] == .string("Bos" + (ending["stringValue"]?.stringValue ?? "")))
        }
    }

    @Test func reasoningNoneUsesSupportedMinimumAndWarnings() throws {
        var warnings: [AIWarning] = []
        #expect(googleThinkingConfig(for: "none", modelID: "gemini-2.5-pro", warnings: &warnings) == ["thinkingBudget": 128])
        #expect(warnings == [AIWarning(type: "compatibility", feature: "reasoning", message: "reasoning \"none\" is not supported by this model. mapped to the minimum reasoning budget of 128 tokens.")])
        warnings = []
        #expect(googleThinkingConfig(for: "none", modelID: "gemini-2.5-flash-lite", warnings: &warnings) == ["thinkingBudget": 0])
        #expect(warnings.isEmpty)
        #expect(googleThinkingConfig(for: "none", modelID: "gemini-3-pro-preview", warnings: &warnings) == ["thinkingLevel": "low"])
        #expect(warnings.first?.type == "compatibility")
        warnings = []
        #expect(googleThinkingConfig(for: "max", modelID: "gemini-2.5-pro", warnings: &warnings) == ["thinkingBudget": 32768])
    }

    @Test func wireGenerationAndStreamingForwardVideoMetadataAndFreshCredentials() async throws {
        let keys = WeeklyGoogleKeys()
        let transport = RecordingTransport(responses: [
            jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}"#),
            sseResponse(#"data: {"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}"#)
        ])
        let provider = try GoogleGenerativeAIProvider(configuration: .init(transport: transport, apiKeyProvider: { await keys.next() }))
        #expect(await keys.count == 0)
        let model = try provider.languageModel("gemini-3.8-flash")
        let request = LanguageModelRequest(messages: [AIMessage(role: .user, content: [
            .file(mimeType: "video/mp4", data: Data([1]), providerMetadata: ["google": ["videoMetadata": ["fps": 0.5]]])
        ])])
        _ = try await model.generate(request)
        for try await _ in model.stream(request) {}
        let requests = await transport.requests()
        #expect(requests.map { $0.headers["x-goog-api-key"] } == ["key-1", "key-2"])
        for request in requests {
            let body = try decodeJSONBody(try #require(request.body))
            #expect(body["contents"]?[0]?["parts"]?[0]?["videoMetadata"] == ["fps": 0.5])
        }
    }

    @Test func fileUploadRefreshesStatusCredentialAndLeavesUploadURLUnauthenticated() async throws {
        let keys = WeeklyGoogleKeys()
        let transport = RecordingTransport(responses: [
            jsonResponse("{}", headers: ["x-goog-upload-url": "https://upload.example/video"]),
            jsonResponse(#"{"file":{"name":"files/video","state":"PROCESSING","uri":"https://generativelanguage.googleapis.com/v1beta/files/video"}}"#),
            jsonResponse(#"{"name":"files/video","state":"ACTIVE","uri":"https://generativelanguage.googleapis.com/v1beta/files/video"}"#)
        ])
        let provider = try GoogleGenerativeAIProvider(configuration: .init(transport: transport, apiKeyProvider: { await keys.next() }))
        _ = try await provider.files().uploadFile(FileUploadRequest(data: Data([1]), mediaType: "video/mp4", pollIntervalNanoseconds: 0))
        let requests = await transport.requests()
        #expect(requests.map { $0.headers["x-goog-api-key"] } == ["key-1", nil, "key-2"])
    }

    @Test func callerAPIKeyHeaderOverridesSupplierCaseInsensitively() async throws {
        let keys = WeeklyGoogleKeys()
        let transport = RecordingTransport(responses: [
            jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}"#)
        ])
        let provider = try GoogleGenerativeAIProvider(configuration: .init(
            headers: ["X-Goog-Api-Key": "caller-key"], transport: transport,
            apiKeyProvider: { await keys.next() }
        ))
        _ = try await provider.languageModel("gemini-3.8-flash").generate(.init(messages: [.user("hello")]))
        #expect(await keys.count == 0)
        #expect(await transport.requests().first?.headers["x-goog-api-key"] == "caller-key")
    }
}

private actor WeeklyGoogleKeys {
    private(set) var count = 0
    func next() -> String { count += 1; return "key-\(count)" }
}
