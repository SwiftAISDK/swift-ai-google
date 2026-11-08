import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

/// Gemini Live codec supporting ephemeral client connections and trusted server connections.
public final class GoogleRealtimeModel: AIRealtimeModelV4, @unchecked Sendable {
    public let specificationVersion = "v4"
    public var providerID: String {
        let base = config.providerID.hasSuffix(".generative-ai") ? String(config.providerID.dropLast(".generative-ai".count)) : config.providerID
        return base + ".realtime"
    }
    public let modelID: String
    public let capabilities: AIRealtimeModelCapabilities? = .init(conversation: .turnBased, transports: [.webSocket], connections: [.clientSecretWebSocket, .serverWebSocket])
    private let config: GoogleModelConfiguration
    private let mapper = GoogleRealtimeMapper()

    init(modelID: String, config: GoogleModelConfiguration) { self.modelID = modelID; self.config = config }

    public func doCreateClientSecret(_ options: AIRealtimeClientSecretOptions) async throws -> AIRealtimeClientSecretResult {
        try options.abortSignal?.throwIfAborted()
        let headers = normalizeHeaders(try await config.http.resolvedHeaders())
        guard let apiKey = headers["x-goog-api-key"], !apiKey.isEmpty else {
            throw AIError.invalidArgument(argument: "apiKey", message: "Google Generative AI API key is required for realtime token creation.")
        }
        let now = Date()
        let window = Double(options.expiresAfterSeconds ?? 60)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let body: JSONValue = .object([
            "uses": 0,
            "newSessionExpireTime": .string(formatter.string(from: now.addingTimeInterval(window))),
            "expireTime": .string(formatter.string(from: now.addingTimeInterval(window + 1_800))),
            "bidiGenerateContentSetup": googleRealtimeSessionConfig(options.sessionConfig, modelID: modelID)
        ])
        var components = try googleRealtimeURL(config.baseURL, path: "/v1alpha/auth_tokens", websocket: false)
        components.percentEncodedQuery = "key=" + googleRealtimeQueryValue(apiKey)
        guard let url = components.url else { throw AIError.invalidURL(config.baseURL) }
        let response = try await config.transport.send(AIHTTPRequest(method: "POST", url: url, headers: ["content-type": "application/json"], body: try encodeJSONBody(body), abortSignal: options.abortSignal))
        guard (200..<300).contains(response.statusCode) else { throw config.httpStatusError(response) }
        let raw = try response.jsonValue()
        guard let token = raw["name"]?.stringValue, !token.isEmpty else { throw AIError.invalidResponse(provider: providerID, message: "Google realtime auth token response did not contain a name.") }
        let expiry = raw["expireTime"]?.stringValue.flatMap { formatter.date(from: $0) ?? ISO8601DateFormatter().date(from: $0) }.map { Int($0.timeIntervalSince1970) }
        return .init(token: token, url: try googleRealtimeURLString(config.baseURL, server: false), expiresAt: expiry)
    }

    public func getWebSocketConfig(token: String, url: String) -> AIRealtimeWebSocketConfiguration {
        guard var components = URLComponents(string: url) else { return .init(url: url) }
        components.percentEncodedQuery = [components.percentEncodedQuery, "access_token=" + googleRealtimeQueryValue(token)].compactMap { $0 }.joined(separator: "&")
        return .init(url: components.url?.absoluteString ?? url)
    }

    public func getServerWebSocketConfig() throws -> AIRealtimeWebSocketConfiguration {
        .init(url: try googleRealtimeURLString(config.baseURL, server: true), headers: config.headers)
    }
    public func getResolvedServerWebSocketConfig() async throws -> AIRealtimeWebSocketConfiguration {
        .init(url: try googleRealtimeURLString(config.baseURL, server: true), headers: try await config.http.resolvedHeaders())
    }
    public func parseServerEvent(_ raw: JSONValue) -> [AIRealtimeServerEvent] { mapper.parse(raw) }
    public func createServerEventParser() -> AIRealtimeServerEventParser {
        let parser = GoogleRealtimeMapper()
        return { parser.parse($0) }
    }
    public func serializeClientEvent(_ event: AIRealtimeClientEvent) async throws -> AIRealtimeWireMessage? {
        mapper.serialize(event, modelID: modelID)
    }
    public func buildSessionConfig(_ config: AIRealtimeSessionConfiguration) -> JSONValue { googleRealtimeSessionConfig(config, modelID: modelID) }
}

private func googleRealtimeQueryValue(_ value: String) -> String {
    var allowed = CharacterSet.alphanumerics
    allowed.insert(charactersIn: "-._~")
    return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
}

private func googleRealtimeURL(_ baseURL: String, path: String, websocket: Bool) throws -> URLComponents {
    guard var components = URLComponents(string: baseURL), let scheme = components.scheme, ["http", "https"].contains(scheme), components.host != nil else { throw AIError.invalidURL(baseURL) }
    var prefix = withoutTrailingSlash(components.path)
    if prefix.hasSuffix("/v1alpha") || prefix.hasSuffix("/v1beta") { prefix = String(prefix[..<prefix.lastIndex(of: "/")!]) }
    components.path = prefix + path
    if websocket { components.scheme = scheme == "https" ? "wss" : "ws" }
    return components
}

private func googleRealtimeURLString(_ baseURL: String, server: Bool) throws -> String {
    let path = server
        ? "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        : "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained"
    guard let url = try googleRealtimeURL(baseURL, path: "/ws/" + path, websocket: true).url else { throw AIError.invalidURL(baseURL) }
    return url.absoluteString
}

func googleRealtimeSessionConfig(_ config: AIRealtimeSessionConfiguration?, modelID: String) -> JSONValue {
    var generation: [String: JSONValue] = ["responseModalities": .array(config?.outputModalities?.map { .string($0.rawValue.uppercased()) } ?? ["AUDIO"])]
    if let voice = config?.voice { generation["speechConfig"] = .object(["voiceConfig": .object(["prebuiltVoiceConfig": .object(["voiceName": .string(voice)])])]) }
    var setup: [String: JSONValue] = ["model": .string(modelID.contains("/") ? modelID : "models/" + modelID), "generationConfig": .object(generation)]
    var options = config?.providerOptions ?? [:]
    let google = options.removeValue(forKey: "google")?.objectValue ?? [:]
    if let instructions = config?.instructions { setup["systemInstruction"] = .object(["parts": .array([.object(["text": .string(instructions)])])]) }
    if let tools = config?.tools, !tools.isEmpty {
        setup["tools"] = .array([.object(["functionDeclarations": .array(tools.map { tool in
            var declaration: [String: JSONValue] = ["name": .string(tool.name), "parametersJsonSchema": tool.parameters]
            declaration["description"] = tool.description.map(JSONValue.string)
            declaration["behavior"] = google["defaultToolBehavior"]
            return .object(declaration)
        })])])
    }
    if config?.inputAudioTranscription != nil { setup["inputAudioTranscription"] = .object([:]) }
    if config?.outputAudioTranscription != nil { setup["outputAudioTranscription"] = .object([:]) }
    let modelName = modelID.split(separator: "/").last.map(String.init)?.lowercased() ?? ""
    let thinkingModel = modelName.range(of: #"^gemini-\d+\.\d+-live\b.*thinking"#, options: .regularExpression) != nil
    let explicit = google["thinkingConfig"] ?? options["generationConfig"]?["thinkingConfig"]
    var thinking = google["thinkingConfig"]
    if thinkingModel, explicit?["thinkingLevel"] == nil || explicit?["thinkingLevel"] == .null,
       explicit?["thinkingBudget"] == nil || explicit?["thinkingBudget"] == .null {
        var value = explicit?.objectValue ?? [:]; value["thinkingLevel"] = "low"; thinking = .object(value)
    }
    setup.merge(options) { _, raw in raw }
    if google["translationConfig"] != nil || thinking != nil {
        generation = setup["generationConfig"]?.objectValue ?? generation
        if let translation = google["translationConfig"] { generation["translationConfig"] = translation }
        if let thinking { generation["thinkingConfig"] = thinking }
        setup["generationConfig"] = .object(generation)
    }
    return .object(setup)
}

private final class GoogleRealtimeMapper: @unchecked Sendable {
    private let lock = NSLock()
    private var turn = 0
    private var input = 0
    private var inputText = ""
    private enum Boundary { case none, interrupted, complete }
    private var boundary = Boundary.none
    private var preserveInput = false
    private var closed = false
    private var hasAudio = false
    private var hasText = false
    private var hasTranscript = false
    private var inputRate = 16_000
    private var responseID: String { "google-resp-\(turn)" }
    private var itemID: String { "google-item-\(turn)" }

    private func beginTurn() {
        guard closed else { return }
        turn += 1; closed = false; hasAudio = false; hasText = false; hasTranscript = false; preserveInput = false
    }
    private func finishInput() { if !inputText.isEmpty { input += 1; inputText = "" } }
    private func transcription(_ value: JSONValue, raw: JSONValue) -> AIRealtimeServerEvent? {
        guard let text = value["text"]?.stringValue, !text.isEmpty else { if value["finished"] == true { finishInput() }; return nil }
        let hasFinishedSignal = value["finished"] != nil && value["finished"] != .null
        if !inputText.isEmpty && (boundary == .interrupted || (boundary == .complete && !hasFinishedSignal)) { finishInput() }
        boundary = .none; inputText += text
        let event = AIRealtimeServerEvent.inputTranscriptionCompleted(itemID: "google-input-\(input)", transcript: inputText, raw: raw)
        if value["finished"] == true { finishInput() }
        return event
    }

    func parse(_ raw: JSONValue) -> [AIRealtimeServerEvent] {
        lock.withLock {
            var events = parseEvent(raw)
            let update = AIRealtimeServerEvent.custom(rawType: "usageMetadata", raw: raw)
            if update.usage != nil, !events.contains(where: { if case .responseDone = $0 { return true }; return false }),
               !(events.count == 1 && events.first == update) { events.append(update) }
            return events
        }
    }
    private func parseEvent(_ raw: JSONValue) -> [AIRealtimeServerEvent] {
        if let setup = raw["setupComplete"], setup != .null { return [.sessionCreated(sessionID: nil, raw: raw)] }
        if let call = raw["toolCall"], call != .null {
            beginTurn()
            return (call["functionCalls"]?.arrayValue ?? []).flatMap { (function: JSONValue) -> [AIRealtimeServerEvent] in
                let arguments = (try? encodeJSONBody(function["args"] ?? .object([:]))).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                let id = function["id"]?.stringValue ?? ""
                return [.functionCallArgumentsDelta(responseID: responseID, itemID: itemID, callID: id, delta: arguments, raw: raw), .functionCallArgumentsDone(responseID: responseID, itemID: itemID, callID: id, name: function["name"]?.stringValue ?? "", arguments: arguments, raw: raw)]
            }
        }
        for type in ["toolCallCancellation", "goAway", "sessionResumptionUpdate"] { if let value = raw[type], value != .null { return [.custom(rawType: type, raw: raw)] } }
        guard let content = raw["serverContent"], content != .null else {
            if let value = raw["inputTranscription"], let event = transcription(value, raw: raw) { return [event] }
            return [.custom(rawType: raw.objectValue?.keys.sorted().first ?? "unknown", raw: raw)]
        }
        var events: [AIRealtimeServerEvent] = []
        if content["interrupted"] == true { boundary = .interrupted; preserveInput = true; closed = true; events.append(.speechStarted(itemID: nil, raw: raw)) }
        if let parts = content["modelTurn"]?["parts"]?.arrayValue {
            beginTurn()
            for part in parts {
                if let audio = part["inlineData"]?["data"]?.stringValue, !audio.isEmpty { hasAudio = true; events.append(.audioDelta(responseID: responseID, itemID: itemID, delta: audio, raw: raw)) }
                if let text = part["text"]?.stringValue, !text.isEmpty { hasText = true; events.append(.textDelta(responseID: responseID, itemID: itemID, delta: text, raw: raw)) }
            }
        }
        if let text = content["outputTranscription"]?["text"]?.stringValue, !text.isEmpty { hasTranscript = true; events.append(.audioTranscriptDelta(responseID: responseID, itemID: itemID, delta: text, raw: raw)) }
        if let value = content["inputTranscription"], let event = transcription(value, raw: raw) { events.append(event) }
        if content["generationComplete"] == true { events.append(.custom(rawType: "generationComplete", raw: raw)) }
        if let status = content["interactionStatus"], status != .null { events.append(.custom(rawType: "interactionStatus", raw: raw)) }
        if content["waitingForInput"] == true { events.append(.custom(rawType: "waitingForInput", raw: raw)) }
        if content["turnComplete"] == true {
            if hasAudio { events.append(.audioDone(responseID: responseID, itemID: itemID, raw: raw)) }
            if hasText { events.append(.textDone(responseID: responseID, itemID: itemID, text: nil, raw: raw)) }
            if hasTranscript { events.append(.audioTranscriptDone(responseID: responseID, itemID: itemID, transcript: nil, raw: raw)) }
            events.append(.responseDone(responseID: responseID, status: "completed", raw: raw))
            closed = true
            if preserveInput { preserveInput = false } else if boundary != .interrupted { boundary = .complete }
        }
        return events.isEmpty ? [.custom(rawType: "serverContent", raw: raw)] : events
    }

    func serialize(_ event: AIRealtimeClientEvent, modelID: String) -> AIRealtimeWireMessage? {
        lock.withLock {
            switch event {
            case let .sessionUpdate(config), let .sessionUpdateWithEventID(config, _):
                if let rate = config.inputAudioFormat?.rate { inputRate = rate }
                return .json(.object(["setup": googleRealtimeSessionConfig(config, modelID: modelID)]))
            case let .inputAudioAppend(audio), let .inputAudioAppendWithEventID(audio, _):
                return .json(.object(["realtimeInput": .object(["audio": .object(["data": .string(audio), "mimeType": .string("audio/pcm;rate=\(inputRate)")])])]))
            case .inputAudioCommit: return .json(.object(["realtimeInput": .object(["audioStreamEnd": true])]))
            case let .conversationItemCreate(item):
                switch item {
                case let .textMessage(text): return .json(.object(["realtimeInput": .object(["text": .string(text)])]))
                case let .functionCallOutput(callID, name, output):
                    let parsed = (try? decodeJSONBody(Data(output.utf8))) ?? .string(output)
                    let response = parsed.objectValue != nil ? parsed : .object(["output": parsed])
                    var function: [String: JSONValue] = ["id": .string(callID), "response": response]
                    function["name"] = name.map(JSONValue.string)
                    return .json(.object(["toolResponse": .object(["functionResponses": .array([.object(function)])])]))
                case .audioMessage: return nil
                }
            default: return nil
            }
        }
    }
}
