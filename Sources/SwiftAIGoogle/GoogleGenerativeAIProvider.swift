import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class GoogleGenerativeAIProvider: AIProvider, AIEvaluationProvider, AIDecisionProvider, @unchecked Sendable {
    public let providerID: String
    public let supportedCapabilities: Set<ModelCapability> = [.language, .embedding, .image, .speech, .video, .evaluation, .decision]
    private let config: GoogleModelConfiguration

    public init(configuration settings: GoogleProviderSettings) throws {
        providerID = settings.name ?? "google.generative-ai"
        let names = ["GOOGLE_GENERATIVE_AI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
        guard let key = settings.apiKey ?? environmentValue(names) else {
            throw AIError.missingAPIKey(provider: providerID, environmentVariables: names)
        }
        var headers = settings.headers
        headers["x-goog-api-key"] = headers["x-goog-api-key"] ?? key
        if providerID == "google.generative-ai" {
            headers = withUserAgentSuffix(headers, "ai-sdk-google/4.0.105")
        }
        let baseURL = withoutTrailingSlash(settings.baseURL ?? "https://generativelanguage.googleapis.com/v1beta")
        config = GoogleModelConfiguration(
            http: AIProviderHTTPClient(providerID: providerID, headers: headers, transport: settings.transport,
                url: { _, path in try providerURL("\(baseURL)\(path)", queryParams: settings.queryParams) }),
            baseURL: baseURL, transformRequestBody: settings.transformRequestBody,
            batchResultMaxLineBytes: settings.batchResultDownloads?.maxLineBytes
        )
    }

    /// Supplies routing and response policy for a hosting or compatibility adapter.
    public init(modelConfiguration: GoogleModelConfiguration) {
        providerID = modelConfiguration.providerID
        config = modelConfiguration
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        GoogleBatchLanguageModel(modelID: modelID, config: config)
    }

    public func batchLanguageModel(_ modelID: String) -> any BatchLanguageModel {
        GoogleBatchLanguageModel(modelID: modelID, config: config)
    }

    /// Creates an experimental Choice/Score/Boolean evaluation model using Gemini.
    public func evaluationModel(_ modelID: String) throws -> any AIEvaluationModelV4 {
        EvaluationLanguageModel(
            model: GoogleBatchLanguageModel(modelID: modelID, config: config),
            providerID: googleEvaluationProviderID(from: providerID)
        )
    }

    public func decisionModel(_ modelID: String) throws -> any AIDecisionModelV4 {
        DecisionLanguageModel(
            model: try languageModel(modelID),
            providerID: String(googleEvaluationProviderID(from: providerID).dropLast(".evaluation".count)) + ".decision"
        )
    }

    public func experimentalDecisionModel(_ modelID: String) throws -> any AIDecisionModelV4 {
        try decisionModel(modelID)
    }

    /// Provider-owned Batch V4 service. The model-bound factory above remains
    /// available as a source-compatible text-batch shim.
    public func experimentalBatch() -> any AIBatchProvider {
        GoogleBatchProvider(config: config)
    }

    public func interactionsModel(_ modelID: String) -> any LanguageModel {
        GoogleInteractionsLanguageModel(modelID: modelID, agent: nil, config: config)
    }

    public func interactionsAgent(_ agentName: String) -> any LanguageModel {
        GoogleInteractionsLanguageModel(modelID: agentName, agent: agentName, config: config)
    }

    public func files() -> any AIFileClient {
        GoogleFileClient(providerID: config.providerID, config: config)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        GoogleEmbeddingModel(modelID: modelID, config: config)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        GoogleImageGenerationModel(modelID: modelID, config: config)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        GoogleVideoGenerationModel(modelID: modelID, config: config)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        GoogleSpeechModel(modelID: modelID, config: config.withProviderID("\(providerID).speech"))
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .reranking, modelID: modelID)
    }
}

private func googleEvaluationProviderID(from languageProviderID: String) -> String {
    if languageProviderID.hasSuffix(".generative-ai") {
        return String(languageProviderID.dropLast(".generative-ai".count)) + ".evaluation"
    }
    return languageProviderID + ".evaluation"
}
public extension AIProviders {
    static func google(configuration: GoogleProviderSettings) throws -> GoogleGenerativeAIProvider {
        try GoogleGenerativeAIProvider(configuration: configuration)
    }
}
