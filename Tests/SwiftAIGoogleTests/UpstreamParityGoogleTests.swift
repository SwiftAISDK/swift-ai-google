@testable import SwiftAIGoogle
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

/// Behavior-parity translations of @ai-sdk/google 4.0.121 matching-release tests
/// (convert-to-google-messages, google-prepare-tools, google-language-model,
/// get-model-path, google-provider) for divergences found in the parity audit.
@Suite struct UpstreamParityGoogleTests {
    private func model(_ id: String, transport: RecordingTransport, name: String? = nil) throws -> any LanguageModel {
        try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "key", transport: transport, name: name)).languageModel(id)
    }

    private func prompt(_ messages: [AIMessage], modelID: String = "gemini-2.5-flash", names: [String] = ["google"], warnings: inout [AIWarning]) throws -> (contents: [JSONValue], systemInstruction: JSONValue?) {
        try googleGenerateContentPrompt(messages, options: GooglePromptConversionOptions(modelID: modelID, providerOptionsNames: names), warnings: &warnings)
    }

    private func prompt(_ messages: [AIMessage], modelID: String = "gemini-2.5-flash", names: [String] = ["google"]) throws -> (contents: [JSONValue], systemInstruction: JSONValue?) {
        var warnings: [AIWarning] = []
        return try prompt(messages, modelID: modelID, names: names, warnings: &warnings)
    }

    private func prepared(_ request: LanguageModelRequest, modelID: String = "gemini-2.5-flash") throws -> GoogleGenerateContentPreparedCall {
        try GoogleGenerativeLanguageModel.generateContentBody(for: request, modelID: modelID)
    }

    // MARK: convert-to-google-messages.test.ts

    @Test func systemMessagesShouldStoreSystemMessageInSystemInstruction() throws {
        let result = try prompt([.system("Test"), .system("Second")])
        #expect(result.systemInstruction == ["parts": [["text": "Test"], ["text": "Second"]]])
        #expect(result.contents.isEmpty)
    }

    @Test func systemMessagesShouldThrowErrorWhenThereWasAlreadyAUserMessage() throws {
        #expect(throws: AIError.self) { try prompt([.user("Test"), .system("Test")]) }
    }

    @Test func gemmaShouldHandleMultipleSystemMessages() throws {
        let result = try prompt([.system("First"), .system("Second"), .user("Hello")], modelID: "gemma-3-12b-it")
        #expect(result.systemInstruction == nil)
        #expect(result.contents[0]["parts"] == [["text": "First\n\nSecond\n\n"], ["text": "Hello"]])
    }

    @Test func toolMessagesShouldConvertToolResultMessagesToFunctionResponses() throws {
        let result = try prompt([AIMessage(role: .tool, content: [.toolResult(AIToolResult(toolCallID: "testCallId", toolName: "testFunction", result: ["type": "json", "value": ["someData": "test result"]]))])])
        #expect(result.contents == [["role": "user", "parts": [["functionResponse": ["id": "testCallId", "name": "testFunction", "response": ["name": "testFunction", "content": ["someData": "test result"]]]]]]])
    }

    static let toolOutputStatusCases: [(JSONValue, JSONValue)] = [
        (["type": "text", "value": "deployed"], ["name": "deploy", "content": "deployed"]),
        (["type": "json", "value": ["version": "1.0"]], ["name": "deploy", "content": ["version": "1.0"]]),
        (["type": "error-text", "value": "migration failed"], ["name": "deploy", "error": "migration failed"]),
        (["type": "error-json", "value": ["code": "E_MIGRATION"]], ["name": "deploy", "error": ["code": "E_MIGRATION"]]),
        (["type": "error-json", "value": ["$ref": "#/$defs/Error"]], ["name": "deploy", "error": ##"{"$ref":"#/$defs/Error"}"##]),
        (["type": "execution-denied", "reason": "User declined."], ["name": "deploy", "error": "User declined."]),
        (["type": "execution-denied"], ["name": "deploy", "error": "Tool call execution denied."])
    ]

    @Test(arguments: toolOutputStatusCases)
    func toolMessagesShouldPreserveTheStatusOfToolOutput(_ output: JSONValue, _ response: JSONValue) throws {
        let result = try prompt([AIMessage(role: .tool, content: [.toolResult(AIToolResult(toolCallID: "call-deploy", toolName: "deploy", result: output))])])
        #expect(result.contents == [["role": "user", "parts": [["functionResponse": ["id": "call-deploy", "name": "deploy", "response": response]]]]])
    }

    @Test func rawSwiftToolResultsMapLikeCreateToolModelOutput() throws {
        let success = try prompt([AIMessage(role: .tool, content: [.toolResult(AIToolResult(toolCallID: "c", toolName: "lookup", result: ["answer": "known"]))])])
        #expect(success.contents[0]["parts"]?[0]?["functionResponse"]?["response"] == ["name": "lookup", "content": ["answer": "known"]])
        let failure = try prompt([AIMessage(role: .tool, content: [.toolResult(AIToolResult(toolCallID: "c", toolName: "lookup", result: "boom", isError: true))])])
        #expect(failure.contents[0]["parts"]?[0]?["functionResponse"]?["response"] == ["name": "lookup", "error": "boom"])
    }

    @Test func toolMessagesShouldUseLegacyToolResultConversionWhenFunctionResponsePartsAreUnsupported() throws {
        let output: JSONValue = ["type": "content", "value": [
            ["type": "text", "text": "Here is the generated image:"],
            ["type": "file", "data": ["type": "data", "data": "base64encodedimagedata"], "mediaType": "image/jpeg"],
            ["type": "file", "data": ["type": "data", "data": "base64pdfdata"], "mediaType": "application/pdf", "filename": "report.pdf"]
        ]]
        let result = try prompt([AIMessage(role: .tool, content: [.toolResult(AIToolResult(toolCallID: "testCallId", toolName: "imageGenerator", result: output))])], modelID: "gemini-2.5-flash")
        #expect(result.contents[0]["parts"] == [
            ["functionResponse": ["id": "testCallId", "name": "imageGenerator", "response": ["name": "imageGenerator", "content": "Here is the generated image:"]]],
            ["inlineData": ["mimeType": "image/jpeg", "data": "base64encodedimagedata"]],
            ["text": "Tool executed successfully and returned this image as a response"],
            ["inlineData": ["mimeType": "application/pdf", "data": "base64pdfdata"]],
            ["text": "Tool executed successfully and returned this file as a response"]
        ])
    }

    @Test func toolMessagesShouldConvertToolResultContentWithImageURLDataURLIntoFunctionResponseParts() throws {
        let output: JSONValue = ["type": "content", "value": [["type": "file", "data": ["type": "url", "url": "data:image/png;base64,base64pngdata"], "mediaType": "image"]]]
        let result = try prompt([AIMessage(role: .tool, content: [.toolResult(AIToolResult(toolCallID: "testCallId", toolName: "imageGenerator", result: output))])], modelID: "gemini-3-flash-preview")
        #expect(result.contents[0]["parts"]?[0] == ["functionResponse": [
            "id": "testCallId", "name": "imageGenerator",
            "response": ["name": "imageGenerator", "content": "Tool executed successfully."],
            "parts": [["inlineData": ["mimeType": "image/png", "data": "base64pngdata"]]]
        ]])
    }

    @Test func serverToolCombinationShouldConvertToolResultWithServerToolCallIdToToolResponseOnLastModelContent() throws {
        let serverMetadata: JSONValue = ["serverToolCallId": "server-id-1", "serverToolType": "GOOGLE_SEARCH_WEB"]
        let call = AIToolCall(id: "tc-1", name: "server:GOOGLE_SEARCH_WEB", arguments: #"{"query":"test"}"#, providerMetadata: ["google": serverMetadata])
        let result = AIToolResult(toolCallID: "tc-1", toolName: "server:GOOGLE_SEARCH_WEB", result: ["type": "json", "value": ["results": ["a"]]], providerMetadata: ["google": ["serverToolCallId": "server-id-1", "serverToolType": "GOOGLE_SEARCH_WEB", "thoughtSignature": "sig-resp"]])
        let converted = try prompt([.assistant(toolCalls: [call]), AIMessage(role: .tool, content: [.toolResult(result)])])
        #expect(converted.contents[0]["role"] == "model")
        #expect(converted.contents[0]["parts"] == [
            ["toolCall": ["toolType": "GOOGLE_SEARCH_WEB", "args": ["query": "test"], "id": "server-id-1"]],
            ["toolResponse": ["toolType": "GOOGLE_SEARCH_WEB", "response": ["results": ["a"]], "id": "server-id-1"], "thoughtSignature": "sig-resp"]
        ])
    }

    @Test func thoughtSignaturesShouldPreferVertexNamespaceOverGoogleNamespaceWhenBothArePresent() throws {
        let call = AIToolCall(id: "call1", name: "getWeather", arguments: #"{"location":"London"}"#, providerMetadata: ["vertex": ["thoughtSignature": "vertex_sig"], "google": ["thoughtSignature": "google_sig"]])
        let converted = try prompt([.assistant(toolCalls: [call])], names: ["googleVertex", "vertex"])
        #expect(converted.contents[0]["parts"]?[0]?["thoughtSignature"] == "vertex_sig")
    }

    @Test func gemini3MitigationEmitsOneWarningPerRequestListingEachAffectedToolName() throws {
        var warnings: [AIWarning] = []
        let calls = [
            AIToolCall(id: "tc_1", name: "weather", arguments: #"{"location":"SF"}"#),
            AIToolCall(id: "tc_2", name: "weather", arguments: #"{"location":"NYC"}"#),
            AIToolCall(id: "tc_3", name: "search", arguments: #"{"query":"q"}"#)
        ]
        _ = try prompt([.user("hi"), .assistant(toolCalls: calls)], modelID: "gemini-3-flash-preview", warnings: &warnings)
        #expect(warnings.count == 1)
        #expect(warnings[0].type == "other")
        #expect(warnings[0].message?.contains("3 ") == true)
        #expect(warnings[0].message?.contains("`weather`") == true)
        #expect(warnings[0].message?.contains("`search`") == true)
    }

    @Test func assistantMessagesShouldThrowErrorForURLFileData() throws {
        #expect(throws: AIError.self) { try prompt([AIMessage(role: .assistant, content: [.imageURL("https://example.com/image.png")])]) }
    }

    @Test func assistantMessagesShouldConvertReasoningFilePartsWithThoughtFlagAndSignature() throws {
        let file = AIStreamFile(mediaType: "image/png", data: Data([1, 2, 3]), providerMetadata: ["google": ["thoughtSignature": "sig_123"]])
        let converted = try prompt([AIMessage(role: .assistant, content: [.reasoningFile(file)])])
        #expect(converted.contents[0]["parts"] == [["inlineData": ["mimeType": "image/png", "data": "AQID"], "thought": true, "thoughtSignature": "sig_123"]])
    }

    @Test func userMessagesShouldThrowWhenProviderReferenceIsMissingGoogleKey() throws {
        #expect(throws: (any Error).self) {
            try prompt([AIMessage(role: .user, content: [.providerReference(mimeType: "application/pdf", reference: ["openai": "file-123"])])])
        }
    }

    // MARK: google-prepare-tools.test.ts

    @Test func shouldUseValidatedModeForCombinedToolsWithToolChoiceAutoOnGemini3() throws {
        let prepared = try googlePrepareTools(
            from: ["testFunction": ["type": "object", "properties": [:], "description": "A test function"], "google_search": ["type": "provider", "id": "google.google_search", "args": [:]]],
            toolChoice: ["type": "auto"], modelID: "gemini-3-flash-preview", isVertexProvider: false
        )
        #expect(prepared?.toolConfig == ["functionCallingConfig": ["mode": "VALIDATED"], "includeServerSideToolInvocations": true])
        #expect(prepared?.warnings == [])
    }

    @Test func shouldUseValidatedModeWhenAnyFunctionToolHasStrictTrueWithoutSendingStrictInTheSchema() throws {
        let prepared = try googlePrepareTools(from: ["testFunction": ["type": "object", "properties": [:], "strict": true]], toolChoice: nil, modelID: "gemini-2.5-flash", isVertexProvider: false)
        #expect(prepared?.toolConfig == ["functionCallingConfig": ["mode": "VALIDATED"]])
        #expect(prepared?.tools[0]["functionDeclarations"]?[0]?["parametersJsonSchema"]?["strict"] == nil)
    }

    // MARK: get-model-path.test.ts / google-provider.test.ts

    @Test func getModelPathPassesThroughResourcePaths() {
        #expect(googleModelPath("models/some-model") == "models/some-model")
        #expect(googleModelPath("tunedModels/some-model") == "tunedModels/some-model")
        #expect(googleModelPath("some-model") == "models/some-model")
    }

    @Test func languageModelUsesModelPathForTunedModels() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}"#))
        _ = try await model("tunedModels/my-model", transport: transport).generate(LanguageModelRequest(messages: [.user("hi")]))
        #expect(await transport.requests().first?.url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/tunedModels/my-model:generateContent")
    }

    @Test func customProviderNameKeepsTheSDKUserAgentSuffix() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}"#))
        _ = try await model("gemini-2.5-flash", transport: transport, name: "my-google").generate(LanguageModelRequest(messages: [.user("hi")]))
        #expect(await transport.requests().first?.headers["user-agent"] == "ai-sdk-google/4.0.121")
    }

    @Test func languageModelSupportedURLsMatchGetSupportedUrls() throws {
        let gemini = try model("gemini-2.5-flash", transport: RecordingTransport(response: jsonResponse("{}")))
        #expect(isURLSupported(mediaType: "video/mp4", url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ", supportedURLs: gemini.supportedURLs))
        #expect(isURLSupported(mediaType: "application/pdf", url: "https://generativelanguage.googleapis.com/v1beta/files/abc", supportedURLs: gemini.supportedURLs))
        #expect(isURLSupported(mediaType: "image/png", url: "https://example.com/a.png", supportedURLs: gemini.supportedURLs))
        let legacy = try model("gemini-2.0-flash", transport: RecordingTransport(response: jsonResponse("{}")))
        #expect(!isURLSupported(mediaType: "image/png", url: "https://example.com/a.png", supportedURLs: legacy.supportedURLs))
    }

    // MARK: google-language-model.test.ts (request)

    @Test func requestShouldWarnAndOmitPenaltiesForGemini25OnTheGeminiAPI() throws {
        let call = try prepared(LanguageModelRequest(messages: [.user("hi")], presencePenalty: 0.5, frequencyPenalty: 0.2))
        #expect(call.body["generationConfig"]?["presencePenalty"] == nil)
        #expect(call.body["generationConfig"]?["frequencyPenalty"] == nil)
        #expect(call.warnings == [AIWarning(type: "unsupported", feature: "frequencyPenalty"), AIWarning(type: "unsupported", feature: "presencePenalty")])
        let gemini3 = try prepared(LanguageModelRequest(messages: [.user("hi")], presencePenalty: 0.5), modelID: "gemini-3-flash-preview")
        #expect(gemini3.body["generationConfig"]?["presencePenalty"] == 0.5)
    }

    @Test func requestShouldDropVertexOnlyImageConfigFieldsWithAWarning() throws {
        let call = try prepared(LanguageModelRequest(messages: [.user("hi")], providerOptions: ["google": ["imageConfig": ["aspectRatio": "1:1", "personGeneration": "ALLOW_ALL", "prominentPeople": "ALLOW"]]]), modelID: "gemini-3-pro-image-preview")
        #expect(call.body["generationConfig"]?["imageConfig"] == ["aspectRatio": "1:1"])
        #expect(call.warnings == [AIWarning(type: "other", message: "'imageConfig.personGeneration', 'imageConfig.prominentPeople' are Vertex AI options and are ignored with the current Google provider (google.generative-ai).")])
    }

    @Test func requestShouldWarnFirstForVertexRagStoreOnTheGeminiAPI() throws {
        let call = try prepared(LanguageModelRequest(messages: [.user("hi")], presencePenalty: 1, tools: ["rag": ["type": "provider", "id": "google.vertex_rag_store", "args": ["ragCorpus": "corpus"]]]))
        #expect(call.warnings.first == AIWarning(type: "other", message: "The 'vertex_rag_store' tool is only supported with the Google Vertex provider and might not be supported or could behave unexpectedly with the current Google provider (google.generative-ai)."))
        #expect(call.warnings.dropFirst().first == AIWarning(type: "unsupported", feature: "presencePenalty"))
    }

    @Test func requestAlwaysSendsGenerationConfig() throws {
        #expect(try prepared(LanguageModelRequest(messages: [.user("Hello")]), modelID: "gemini-3-flash-preview").body["generationConfig"] == [:])
    }

    // MARK: google-language-model.test.ts (doGenerate)

    @Test func doGenerateShouldExtractReasoningAndTextAsSeparateParts() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"think","thought":true,"thoughtSignature":"sig"},{"text":"answer"}]},"finishReason":"STOP"}]}"#))
        let result = try await model("gemini-2.5-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("hi")]))
        #expect(result.text == "answer")
        #expect(result.reasoning == "think")
        #expect(result.content == [.reasoning("think", providerMetadata: ["google": ["thoughtSignature": "sig"]]), .text("answer")])
    }

    @Test func doGenerateShouldReturnEmptyContentInsteadOfThrowing() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[]}"#))
        let result = try await model("gemini-2.5-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("hi")]))
        #expect(result.content.isEmpty)
        #expect(result.finishReason == "other")
    }

    @Test func doGenerateShouldOnlyCountClientToolCallsForTheFinishReason() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}},{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}},{"text":"done"}]},"finishReason":"STOP"}]}"#))
        let result = try await model("gemini-2.5-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("hi")], tools: ["code_execution": ["type": "provider", "id": "google.code_execution", "args": [:]]]))
        #expect(result.finishReason == "stop")
    }

    @Test func doGenerateProviderMetadataHasNullDefaults() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}"#))
        let result = try await model("gemini-2.5-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("hi")]))
        #expect(result.providerMetadata["google"] == ["promptFeedback": .null, "groundingMetadata": .null, "urlContextMetadata": .null, "safetyRatings": .null, "usageMetadata": .null, "finishMessage": .null, "serviceTier": .null])
    }

    // MARK: google-language-model.test.ts (doStream)

    @Test func doStreamEmitsProviderExecutedCallsAsSingleToolCallAndKeepsStop() async throws {
        let transport = RecordingTransport(response: sseResponse(#"data: {"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}},{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}}]},"finishReason":"STOP"}]}"#))
        var parts: [LanguageStreamPart] = []
        for try await part in try model("gemini-2.5-flash", transport: transport).stream(LanguageModelRequest(messages: [.user("hi")], tools: ["code_execution": ["type": "provider", "id": "google.code_execution", "args": [:]]])) {
            parts.append(part)
        }
        #expect(!parts.contains { if case .toolInputStart = $0 { return true }; return false })
        #expect(parts.contains { if case let .toolCall(call) = $0 { return call.providerExecuted }; return false })
        guard case let .finishMetadata(reason, _, metadata)? = parts.last else { Issue.record("missing finish"); return }
        #expect(reason == "stop")
        #expect(metadata["google"]?["promptFeedback"] == .null)
        #expect(metadata["google"]?["safetyRatings"] == .null)
    }
}

extension UpstreamParityGoogleTests {
    // google-video-model.ts buildRequest: provider options are copied over the
    // standard parameters, so they take precedence.
    @Test func videoProviderOptionsOverrideStandardParameters() {
        let request = VideoGenerationRequest(prompt: "A cat", resolution: "1920x1080", seed: 1, count: 1, providerOptions: ["google": ["seed": 7, "resolution": "720p", "sampleCount": 2, "pollIntervalMs": 1]])
        let parameters = googleVideoParameters(for: request, options: googleVideoProviderOptions(from: request))
        #expect(parameters["seed"] == 7)
        #expect(parameters["resolution"] == "720p")
        #expect(parameters["sampleCount"] == 2)
        #expect(parameters["pollIntervalMs"] == nil)
    }

    /// google-prepare-tools `prepareFunctionDeclaration` and interactions
    /// `prepareGoogleInteractionsTools` send only `tool.inputSchema`.
    @Test func functionToolMetadataStaysOutOfParameterSchemas() throws {
        let tools: [String: JSONValue] = ["lookup": [
            "type": "object", "properties": ["q": ["type": "string"]],
            "description": "Look up", "strict": true,
            "providerOptions": ["google": ["flag": true]], "inputExamples": [["input": ["q": "a"]]]
        ]]
        let schema: JSONValue = ["type": "object", "properties": ["q": ["type": "string"]]]
        #expect(try googleFunctionDeclarations(from: tools) == [["name": "lookup", "description": "Look up", "parametersJsonSchema": schema]])
        #expect(googleInteractionsPrepareNativeTools(tools, choice: nil).tools == [["type": "function", "name": "lookup", "description": "Look up", "parameters": schema]])
    }

    /// google-language-model `jsonResponseTool.inputSchema` is `responseFormat.schema` as-is.
    @Test func jsonResponseToolKeepsTheResponseSchemaIntact() throws {
        let schema: JSONValue = ["type": "object", "description": "Answer", "properties": ["a": ["type": "string"]]]
        let request = LanguageModelRequest(messages: [.user("x")], responseFormat: .json(schema: schema, name: nil, description: "Respond"),
                                           tools: ["lookup": ["type": "object", "properties": [:]]])
        let prepared = googlePrepareJSONResponseToolRequest(request, modelID: "gemini-2.5-flash")
        let name = try #require(prepared.name)
        let declarations = try googleFunctionDeclarations(from: prepared.request.tools)
        let json = try #require(declarations.first { $0["name"] == .string(name) })
        #expect(json == ["name": .string(name), "description": "Respond", "parametersJsonSchema": schema])
    }
}
