import Foundation
import Testing
import SwiftAI
import SwiftAIProvider
import SwiftAIProviderUtilities
import SwiftAITransport
@testable import SwiftAIGoogle

@Suite struct Weekly20261108GoogleNativeSessionTests {
    @Test func serverConnectionComposesSetupInputUsageAndClosure() async throws {
        let socket = WeeklyRealtimeFixture()
        let model = try GoogleGenerativeAIProvider(configuration: .init(apiKey: "key")).realtimeModel("gemini-3.8-live")
        #expect(model.providerID == "google.realtime")
        let session = try await AIRealtimeSession.connectServer(model: model, sessionConfiguration: .init(instructions: "native", inputAudioFormat: .init(type: "audio/pcm", rate: 24_000)), webSocketTransport: socket)
        #expect(session.providerID == "google.realtime")
        let recorded = WeeklyRealtimeEvents()
        let consumer = Task { for try await event in session { await recorded.record(event) } }
        socket.connection.open()
        try await socket.connection.waitForMessages(1)
        #expect(socket.connection.messages()[0]["setup"]?["model"] == "models/gemini-3.8-live")
        #expect(socket.connection.messages()[0]["setup"]?["systemInstruction"]?["parts"]?[0]?["text"] == "native")
        let acknowledgement: JSONValue = ["setupComplete": [:]]
        socket.connection.json(acknowledgement)
        try await session.appendAudio(Data([1, 2]))
        try await session.commitAudio()
        try await socket.connection.waitForMessages(3)
        #expect(socket.connection.messages()[1] == ["realtimeInput": ["audio": ["data": "AQI=", "mimeType": "audio/pcm;rate=24000"]]])
        #expect(socket.connection.messages()[2] == ["realtimeInput": ["audioStreamEnd": true]])
        let completion: JSONValue = ["serverContent": ["turnComplete": true], "usageMetadata": ["promptTokenCount": 0, "responseTokenCount": 2]]
        socket.connection.json(completion)
        try await recorded.waitForUsage()
        await session.close(reason: "verified")
        try await consumer.value
        let events = await recorded.values
        #expect(events.contains(.server(.sessionCreated(sessionID: nil, raw: acknowledgement))))
        let usage = try #require(events.compactMap { if case let .server(event) = $0 { return event.usage }; return nil }.first)
        #expect(usage.inputTokens == 0 && usage.outputTokens == 2)
        #expect(events.contains(.closed(.init(code: 1000, reason: "verified"))))
        #expect(socket.connection.closes() == [1000])
        let request = try #require(socket.requests().first)
        #expect(request.headers["x-goog-api-key"] == "key")
        #expect(request.url.path.hasSuffix("v1beta.GenerativeService.BidiGenerateContent"))
    }
}

private final class WeeklyRealtimeFixture: AIDuplexWebSocketTransport, @unchecked Sendable {
    let connection = WeeklyRealtimeConnection()
    private let lock = NSLock()
    private var recorded: [AIDuplexWebSocketRequest] = []
    func connect(_ request: AIDuplexWebSocketRequest) async throws -> any AIDuplexWebSocketConnection { lock.withLock { recorded.append(request) }; return connection }
    func requests() -> [AIDuplexWebSocketRequest] { lock.withLock { recorded } }
}

private final class WeeklyRealtimeConnection: AIDuplexWebSocketConnection, @unchecked Sendable {
    let events: AsyncThrowingStream<AIDuplexWebSocketEvent, Error>
    private let continuation: AsyncThrowingStream<AIDuplexWebSocketEvent, Error>.Continuation
    private let lock = NSLock()
    private var sent: [JSONValue] = []
    private var closed: [Int] = []
    init() { let pair = AsyncThrowingStream<AIDuplexWebSocketEvent, Error>.makeStream(); events = pair.stream; continuation = pair.continuation }
    func open() { continuation.yield(.opened(protocol: nil)) }
    func json(_ value: JSONValue) { continuation.yield(.message(.text(String(decoding: (try? encodeJSONBody(value)) ?? Data(), as: UTF8.self)))) }
    func send(_ message: AIDuplexWebSocketMessage) async throws { if case let .text(text) = message { let value = try secureJSONParse(text); lock.withLock { sent.append(value) } } }
    func close(code: Int, reason: Data?) async { lock.withLock { closed.append(code) }; continuation.finish() }
    func messages() -> [JSONValue] { lock.withLock { sent } }
    func closes() -> [Int] { lock.withLock { closed } }
    func waitForMessages(_ count: Int) async throws {
        for _ in 0..<1000 { if messages().count >= count { return }; try await Task.sleep(nanoseconds: 1_000_000) }
        throw AIError.invalidResponse(provider: "test", message: "Realtime send timeout")
    }
}

private actor WeeklyRealtimeEvents {
    var values: [AIRealtimeSessionEvent] = []
    func record(_ event: AIRealtimeSessionEvent) { values.append(event) }
    func waitForUsage() async throws {
        for _ in 0..<1000 {
            if values.contains(where: { if case let .server(event) = $0 { return event.usage != nil }; return false }) { return }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        throw AIError.invalidResponse(provider: "test", message: "Realtime usage forwarding timeout")
    }
}
