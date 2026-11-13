import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

/// Gemini's native Decisions endpoint. Structured-generation decisions remain available for legacy model IDs.
public final class GoogleDecisionModel: AIDecisionModelV4, Sendable {
    public let specificationVersion = "v4"
    public let supportedQuestionTypes: [AIDecisionQuestionType] = [.choice, .score, .boolean]
    public let providerID: String
    public let modelID: String
    private let config: GoogleModelConfiguration
    public init(modelID: String, config: GoogleModelConfiguration) {
        self.modelID = modelID; self.config = config
        let root = config.providerID.hasSuffix(".generative-ai") ? String(config.providerID.dropLast(".generative-ai".count)) : config.providerID
        self.providerID = root + ".decision"
    }
    public func doDecide(_ options: AIDecisionModelV4CallOptions) async throws -> AIDecisionModelV4Result {
        try options.abortSignal?.throwIfAborted()
        guard !options.questions.isEmpty else { throw invalidArgument("questions", "Google Decisions requires at least one question.") }
        var questions: [String: JSONValue] = [:]
        for (id, question) in options.questions {
            if case .box = question { throw AIDecisionUnsupportedQuestionTypeError(questionID: id, questionType: .box, providerID: providerID, modelID: modelID) }
            let instructions = question.instructions
            guard instructions != .null else { throw invalidArgument("questions.\(id).instructions", "Google Decisions requires instructions for every question.") }
            let criteria: JSONValue?
            let type: String
            switch question {
            case let .choice(_, values):
                guard (2...255).contains(values.count) else { throw invalidArgument("questions.\(id).criteria", "Google Decisions requires 2 to 255 options or score levels.") }
                type = "choice"; criteria = .object(try values.mapValues { $0 == .null ? .null : .string(try text($0)) })
            case let .score(_, values):
                guard (2...255).contains(values.count), !values.contains(where: { $0 == .null || $0 == "" }) else {
                    throw invalidArgument("questions.\(id).criteria", "Google Decisions requires 2 to 255 non-empty score level descriptions.")
                }
                type = "score"; criteria = .array(try values.map { .string(try text($0)) })
            case let .boolean(_, values):
                type = "noul"
                criteria = try values.map { .object(["true_description": try $0["true"].flatMap { $0 == .null ? nil : .string(try text($0)) },
                                                    "false_description": try $0["false"].flatMap { $0 == .null ? nil : .string(try text($0)) }]) }
            case .box: preconditionFailure("Box rejected above")
            }
            questions[id] = .object(["type": .string(type), "instructions": .string(try text(instructions)), "criteria": criteria])
        }
        let input: [JSONValue] = try options.state.map { part in
            switch part {
            case let .text(value): return ["type": "text", "text": .string(value)]
            case let .json(value): return ["type": "text", "text": .string(String(decoding: try encodeJSONBody(value), as: UTF8.self))]
            case let .file(mediaType, data, _, _):
                let type = mediaType == "application/pdf" ? "document" : topLevelMediaType(mediaType)
                guard ["image", "audio", "video", "document"].contains(type) else {
                    throw AIDecisionUnsupportedFunctionalityError(functionality: "Google decision file media type: \(mediaType)")
                }
                var result: [String: JSONValue] = ["type": .string(type), "mime_type": .string(mediaType)]
                switch data {
                case let .data(bytes): result["data"] = .string(bytes.base64EncodedString())
                case let .base64(value): result["data"] = .string(value)
                case let .url(url): result["uri"] = .string(url.absoluteString)
                case .reference: throw AIDecisionUnsupportedFunctionalityError(functionality: "Google decision file input: reference")
                }
                return .object(result)
            }
        }
        let result = try await config.sendJSONResponse(path: "/decisions", modelID: modelID,
            body: ["model": .string(modelID), "input": .array(input), "questions": .object(questions)], headers: options.headers, abortSignal: options.abortSignal)
        try options.abortSignal?.throwIfAborted()
        let raw = result.json
        let blocked = raw["prompt_feedback"]?["block_reason"]?.stringValue != nil
        let native = raw["answers"]?.objectValue ?? [:]
        guard blocked || Set(native.keys) == Set(options.questions.keys) else { throw invalid("Google Decisions must return exactly one answer for every question.") }
        var answers: [String: AIDecisionAnswer] = [:], confidence: [String: JSONValue] = [:], legend: [String: JSONValue] = [:]
        for (id, answer) in native {
            if let value = answer["confidence"], value != .null { confidence[id] = .number(try probability(value)) }
            if let value = answer["legend"], value != .null {
                guard let map = value.objectValue, map.values.allSatisfy({ $0.stringValue != nil }) else { throw invalid("Invalid score legend.") }
                legend[id] = value
            }
        }
        for (id, question) in options.questions {
            if blocked { answers[id] = .refusal; continue }
            guard let answer = native[id] else { throw invalid("Missing answer.") }
            switch (question, answer["type"]?.stringValue) {
            case (.boolean, "noul"): answers[id] = .boolean(probability: try probability(answer["noul"]))
            case let (.choice(_, criteria), "choice"):
                guard let choice = answer["choice"]?.stringValue, criteria[choice] != nil else { throw invalid("Invalid choice for \(id).") }
                answers[id] = .choice(choice: choice, probabilities: try probabilities(answer["probabilities"]))
            case let (.score(_, criteria), "score"):
                guard let score = answer["score"]?.doubleValue, score.isFinite, score >= 0, score <= Double(criteria.count - 1) else { throw invalid("Invalid score for \(id).") }
                answers[id] = .score(score: score, probabilities: try probabilities(answer["probabilities"]))
            default: throw invalid("Google Decisions returned an invalid answer for question \(id).")
            }
        }
        var metadata: [String: JSONValue] = ["confidence": .object(confidence), "legend": .object(legend)]
        metadata["usage"] = raw["usage"].flatMap { value -> JSONValue? in value == .null ? nil : value }
        metadata["promptFeedback"] = raw["prompt_feedback"].flatMap { value -> JSONValue? in value == .null ? nil : value }
        var warnings: [AIWarning] = []
        if options.maxOutputTokens != nil { warnings.append(.init(type: "unsupported", feature: "maxOutputTokens")) }
        if options.reasoning != nil { warnings.append(.init(type: "unsupported", feature: "reasoning")) }
        warnings += (options.providerOptions["google"]?.objectValue ?? [:]).keys.sorted().map { .init(type: "unsupported", feature: "providerOptions.google.\($0)") }
        let usage: AIDecisionModelUsage?
        if let rawUsage = raw["usage"], rawUsage != .null {
            guard rawUsage.objectValue != nil else { throw invalid("Invalid decision usage.") }
            for key in ["total_input_tokens", "total_output_tokens", "total_tokens"] {
                if let value = rawUsage[key], value != .null, value.doubleValue == nil { throw invalid("Invalid decision usage.") }
            }
            usage = .init(inputTokens: rawUsage["total_input_tokens"]?.intValue, outputTokens: rawUsage["total_output_tokens"]?.intValue)
        } else { usage = nil }
        return .init(answers: answers, usage: usage, warnings: warnings, providerMetadata: ["google": .object(metadata)],
                     response: .init(modelID: modelID, headers: result.response.headers, body: raw))
    }
    private func text(_ value: JSONValue) throws -> String {
        if let text = value.stringValue { return text }
        return String(decoding: try encodeJSONBody(value), as: UTF8.self)
    }
    private func probability(_ value: JSONValue?) throws -> Double {
        guard let value = value?.doubleValue, value.isFinite, (0...1).contains(value) else { throw invalid("Invalid probability.") }; return value
    }
    private func probabilities(_ value: JSONValue?) throws -> [String: Double] {
        guard let values = value?.objectValue else { throw invalid("Invalid probability distribution.") }; return try values.mapValues { try probability($0) }
    }
    private func invalid(_ message: String) -> AIError { .invalidResponse(provider: providerID, message: message) }
    private func invalidArgument(_ argument: String, _ message: String) -> AIError { .invalidArgument(argument: argument, message: message) }
}
