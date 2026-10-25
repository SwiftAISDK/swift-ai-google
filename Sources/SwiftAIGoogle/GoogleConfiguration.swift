import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public struct GoogleProviderSettings: Sendable {
    public var apiKey: String?
    public var baseURL: String?
    public var headers: [String: String]
    public var queryParams: [String: String]
    public var transport: any AITransport
    public var transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])?
    public var batchResultDownloads: AIBatchResultDownloadSettings?
    public var name: String?

    public init(
        apiKey: String? = nil,
        baseURL: String? = nil,
        headers: [String: String] = [:],
        queryParams: [String: String] = [:],
        transport: any AITransport = URLSessionTransport.shared,
        transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])? = nil,
        batchResultDownloads: AIBatchResultDownloadSettings? = nil,
        name: String? = nil
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.headers = headers
        self.queryParams = queryParams
        self.transport = transport
        self.transformRequestBody = transformRequestBody
        self.batchResultDownloads = batchResultDownloads
        self.name = name
    }
}

/// HTTP routing and batching policy shared by Gemini and hosting adapters.
public struct GoogleModelConfiguration: AIProviderHTTPConfiguration {
    public var http: AIProviderHTTPClient
    public var baseURL: String
    public var transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])?
    public var batchResultMaxLineBytes: Int?
    public var errorHandler: (@Sendable (String, AIHTTPResponse) -> AIError)?

    public init(http: AIProviderHTTPClient, baseURL: String,
                transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])? = nil,
                batchResultMaxLineBytes: Int? = nil) {
        self.http = http; self.baseURL = withoutTrailingSlash(baseURL)
        self.transformRequestBody = transformRequestBody
        self.batchResultMaxLineBytes = batchResultMaxLineBytes
    }
    public func httpStatusError(_ response: AIHTTPResponse) -> AIError { errorHandler?(providerID, response) ?? apiCallError(provider: providerID, response: response) }
}
