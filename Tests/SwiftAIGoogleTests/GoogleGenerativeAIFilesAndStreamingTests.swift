@testable import SwiftAIGoogle
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func googleFilesUploadUsesResumableUploadFlow() async throws {
    let transport = RecordingTransport(responses: [
        AIHTTPResponse(statusCode: 200, headers: ["x-goog-upload-url": "https://upload.example.com/session"], body: Data()),
        jsonResponse("""
        {"file":{"name":"files/abc","displayName":"Clip","mimeType":"video/mp4","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc","state":"ACTIVE"}}
        """)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let imageModel = try provider.imageModel("imagen-3.0-generate-002")
    #expect(imageModel.providerID == "google.generative-ai")
    let files = provider.files()
    #expect(files.providerID == "google.generative-ai")
    let result = try await files.uploadFile(FileUploadRequest(data: Data("video".utf8), mediaType: "video/mp4", displayName: "Clip"))

    #expect(result.providerReference["google"] == "https://generativelanguage.googleapis.com/v1beta/files/abc")
    let requests = await transport.requests()
    #expect(requests.count == 2)
    #expect(requests[0].url.absoluteString == "https://generativelanguage.googleapis.com/upload/v1beta/files")
    #expect(requests[0].headers["X-Goog-Upload-Protocol"] == "resumable")
    #expect(requests[0].headers["X-Goog-Upload-Header-Content-Length"] == "5")
    let startBody = try decodeJSONBody(try #require(requests[0].body))
    #expect(startBody["file"]?["display_name"]?.stringValue == "Clip")
    #expect(requests[1].url.absoluteString == "https://upload.example.com/session")
    #expect(requests[1].headers["X-Goog-Upload-Command"] == "upload, finalize")
    #expect(requests[1].headers["Content-Length"] == nil)
    #expect(requests[1].body == Data("video".utf8))
}

@Test func googleFilesThreadsPerCallHeadersAndAbortThroughStartUploadAndStatus() async throws {
    let transport = RecordingTransport(responses: [
        AIHTTPResponse(statusCode: 200, headers: ["x-goog-upload-url": "https://upload.example.com/session"], body: Data()),
        jsonResponse(#"{"file":{"name":"files/abc","mimeType":"application/pdf","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc","state":"PROCESSING"}}"#),
        jsonResponse(#"{"name":"files/abc","mimeType":"application/pdf","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc","state":"ACTIVE"}"#)
    ])
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(
        apiKey: "gemini-key",
        headers: ["Provider-Header": "provider"],
        transport: transport
    ))
    let controller = AIAbortController()

    _ = try await provider.files().uploadFile(FileUploadRequest(
        data: Data([1, 2, 3]),
        mediaType: "application/pdf",
        pollIntervalNanoseconds: 0,
        headers: ["x-request-id": "req-1"],
        abortSignal: controller.signal
    ))

    let requests = await transport.requests()
    #expect(requests.count == 3)
    #expect(requests.allSatisfy { $0.abortSignal === controller.signal })
    #expect(normalizeHeaders(requests[0].headers)["x-request-id"] == "req-1")
    #expect(normalizeHeaders(requests[0].headers)["provider-header"] == "provider")
    #expect(normalizeHeaders(requests[2].headers)["x-request-id"] == "req-1")
    #expect(normalizeHeaders(requests[2].headers)["provider-header"] == "provider")
    #expect(normalizeHeaders(requests[1].headers)["x-request-id"] == nil)
}
@Test func googleLanguageStreamsGenerateContentEvents() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"candidates":[{"content":{"parts":[{"text":"gem"}],"role":"model"},"index":0,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://source.example.com","title":"Source Title"}}]}}]}

    data: {"candidates":[{"content":{"parts":[{"text":"ini"}],"role":"model"},"finishReason":"STOP","index":0,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://source.example.com","title":"Source Title"}}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    var deltas: [String] = []
    var sources: [AISource] = []
    var usage: TokenUsage?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Ping")])) {
        switch part {
        case let .textDeltaPart(_, delta, _):
            deltas.append(delta)
        case let .source(source):
            sources.append(source)
        case let .finishMetadata(_, value, _):
            usage = value
        default:
            break
        }
    }

    #expect(deltas == ["gem", "ini"])
    #expect(sources.count == 1)
    #expect(sources[0].id == "grounding-0")
    #expect(sources[0].url == "https://source.example.com")
    #expect(sources[0].title == "Source Title")
    #expect(usage?.totalTokens == 3)
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse")
    #expect(request.headers["x-goog-api-key"] == "gemini-key")
}

@Test func googleLanguageCanonicalLifecycleTransitionsBetweenReasoningAndText() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"candidates":[{"content":{"parts":[{"text":"think","thought":true},{"text":"answer"}],"role":"model"},"index":0}]}

    data: {"candidates":[{"content":{"parts":[{"text":" more"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":3,"totalTokenCount":4}}

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    var lifecycle: [String] = []
    var legacyDeltaCount = 0
    var terminalCount = 0
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Ping")])) {
        switch part {
        case let .reasoningStart(id, _):
            lifecycle.append("reasoning-start:\(id)")
        case let .reasoningDeltaPart(id, delta, _):
            lifecycle.append("reasoning-delta:\(id):\(delta)")
        case let .reasoningEnd(id, _):
            lifecycle.append("reasoning-end:\(id)")
        case let .textStart(id, _):
            lifecycle.append("text-start:\(id)")
        case let .textDeltaPart(id, delta, _):
            lifecycle.append("text-delta:\(id):\(delta)")
        case let .textEnd(id, _), let .textEndWithCitations(id, _, _):
            lifecycle.append("text-end:\(id)")
        case .textDelta, .reasoningDelta:
            legacyDeltaCount += 1
        case .finishMetadata:
            terminalCount += 1
        default:
            break
        }
    }

    #expect(lifecycle == [
        "reasoning-start:0",
        "reasoning-delta:0:think",
        "reasoning-end:0",
        "text-start:1",
        "text-delta:1:answer",
        "text-delta:1: more",
        "text-end:1"
    ])
    #expect(legacyDeltaCount == 0)
    #expect(terminalCount == 1)
}
@Test func googleLanguageStreamSurfacesRepeatedProviderResponseIDOnce() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"responseId":"google-stream-response-123","candidates":[{"content":{"parts":[{"text":"gem"}],"role":"model"},"index":0}]}

    data: {"responseId":"google-stream-response-123","candidates":[{"content":{"parts":[{"text":"ini"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}

    """, headers: ["x-request-id": "http-request-123"]))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-3.6-flash")

    var responseMetadata: [AIResponseMetadata] = []
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Ping")])) {
        if case let .responseMetadata(metadata) = part {
            responseMetadata.append(metadata)
        }
    }

    #expect(responseMetadata.count == 1)
    #expect(responseMetadata[0].id == "google-stream-response-123")
    #expect(responseMetadata[0].headers["x-request-id"] == "http-request-123")
}
@Test func googleLanguageStreamPreservesFinishProviderMetadataAcrossChunks() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"candidates":[{"content":{"parts":[{"text":"hello"}],"role":"model"},"index":0,"groundingMetadata":{"webSearchQueries":["super bowl"],"groundingChunks":[{"web":{"uri":"https://example.com/superbowl","title":"Super Bowl"}}]},"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.com/page","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"}]}}]}

    data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP","finishMessage":"finished","index":0,"safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"NEGLIGIBLE"}]}],"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3,"serviceTier":"priority"}}

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    var providerMetadata: [String: JSONValue] = [:]
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Ping")])) {
        if case let .finishMetadata(_, _, metadata) = part {
            providerMetadata = metadata
        }
    }

    let google = try #require(providerMetadata["google"])
    #expect(google["groundingMetadata"]?["webSearchQueries"]?[0]?.stringValue == "super bowl")
    #expect(google["urlContextMetadata"]?["urlMetadata"]?[0]?["retrievedUrl"]?.stringValue == "https://example.com/page")
    #expect(google["safetyRatings"]?[0]?["category"]?.stringValue == "HARM_CATEGORY_DANGEROUS_CONTENT")
    #expect(google["promptFeedback"]?["blockReason"]?.stringValue == "SAFETY")
    #expect(google["finishMessage"]?.stringValue == "finished")
    #expect(google["serviceTier"]?.stringValue == "priority")
}
@Test func googleLanguageStreamsFunctionCallPartialArguments() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"weather","willContinue":true}}]},"index":0}]}

    data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.location","stringValue":"San ","willContinue":true}],"willContinue":true}}]},"index":0}]}

    data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.location","stringValue":"Francisco","willContinue":true}],"willContinue":true}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":29,"candidatesTokenCount":15,"totalTokenCount":44}}

    """))
    let provider = try AIProviders.google(configuration: GoogleProviderSettings(apiKey: "gemini-key", transport: transport))
    let model = try provider.languageModel("gemini-2.5-flash")

    var inputLifecycle: [String] = []
    var finalCall: AIToolCall?
    var finishReason: String?
    var totalTokens: Int?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Weather?")])) {
        switch part {
        case let .toolInputStart(id, name, _, _, _, _):
            inputLifecycle.append("start:\(id):\(name)")
        case let .toolInputDelta(id, delta, _):
            inputLifecycle.append("delta:\(id):\(delta)")
        case let .toolInputEnd(id, _):
            inputLifecycle.append("end:\(id)")
        case let .toolCall(call):
            finalCall = call
        case let .finishMetadata(reason, usage, _):
            finishReason = reason
            totalTokens = usage?.totalTokens
        default:
            break
        }
    }

    let call = try #require(finalCall)
    #expect(inputLifecycle == [
        "start:tool-call-0:weather",
        #"delta:tool-call-0:{"location":"San "#,
        "delta:tool-call-0:Francisco",
        #"delta:tool-call-0:"}"#,
        "end:tool-call-0"
    ])
    #expect(call.id == "tool-call-0")
    #expect(call.name == "weather")
    #expect(try decodeJSONBody(Data(call.arguments.utf8))["location"]?.stringValue == "San Francisco")
    #expect(finishReason == "tool-calls")
    #expect(totalTokens == 44)
}
