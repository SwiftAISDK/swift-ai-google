import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public final class GoogleFileClient: AIFileClient, @unchecked Sendable {
    public let providerID: String
    private let config: GoogleModelConfiguration

    init(providerID: String, config: GoogleModelConfiguration) {
        self.providerID = providerID
        self.config = config
    }

    public func uploadFile(_ request: FileUploadRequest) async throws -> FileUploadResult {
        if request.fileData.isStream {
            await request.fileData.cancelStream()
            throw unsupportedStreamingFileUpload(providerID: providerID)
        }
        try request.abortSignal?.throwIfAborted()
        let origin = config.baseURL.hasSuffix("/v1beta") ? String(config.baseURL.dropLast("/v1beta".count)) : config.baseURL
        var startHeaders = config.headers.mergingHeaders(request.headers)
        startHeaders["X-Goog-Upload-Protocol"] = "resumable"
        startHeaders["X-Goog-Upload-Command"] = "start"
        startHeaders["X-Goog-Upload-Header-Content-Length"] = String(request.data.count)
        startHeaders["X-Goog-Upload-Header-Content-Type"] = request.mediaType
        startHeaders["Content-Type"] = "application/json"

        let initBody: JSONValue = .object([
            "file": .object(request.displayName.map { ["display_name": .string($0)] } ?? [:])
        ])
        let startResponse = try await config.transport.send(AIHTTPRequest(
            method: "POST",
            url: try requireURL("\(origin)/upload/v1beta/files"),
            headers: startHeaders,
            body: try encodeJSONBody(initBody),
            abortSignal: request.abortSignal
        ))
        guard (200..<300).contains(startResponse.statusCode) else {
            throw config.httpStatusError( startResponse)
        }
        guard let uploadURL = headerValue(startResponse.headers, "x-goog-upload-url") else {
            throw AIError.invalidResponse(provider: providerID, message: "No x-goog-upload-url returned from resumable upload start.")
        }

        let uploadResponse = try await config.transport.send(AIHTTPRequest(
            method: "POST",
            url: try requireURL(uploadURL),
            headers: [
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize"
            ],
            body: request.data,
            abortSignal: request.abortSignal
        ))
        guard (200..<300).contains(uploadResponse.statusCode) else {
            throw config.httpStatusError( uploadResponse)
        }
        var raw = try uploadResponse.jsonValue()
        var file = raw["file"] ?? raw
        var metadataResponse = uploadResponse

        let started = DispatchTime.now().uptimeNanoseconds
        while file["state"]?.stringValue == "PROCESSING" {
            if DispatchTime.now().uptimeNanoseconds - started > request.pollTimeoutNanoseconds {
                throw AIError.invalidResponse(provider: providerID, message: "Google file processing timed out for \(file["name"]?.stringValue ?? "unknown file").")
            }
            try await sleepWithAbortSignal(nanoseconds: request.pollIntervalNanoseconds, abortSignal: request.abortSignal)
            guard let name = file["name"]?.stringValue else { break }
            let statusResponse = try await config.transport.send(AIHTTPRequest(
                method: "GET",
                url: try requireURL("\(config.baseURL)/\(name)"),
                headers: config.headers.mergingHeaders(request.headers),
                abortSignal: request.abortSignal
            ))
            guard (200..<300).contains(statusResponse.statusCode) else {
                throw config.httpStatusError( statusResponse)
            }
            file = try statusResponse.jsonValue()
            raw = file
            metadataResponse = statusResponse
        }
        if file["state"]?.stringValue == "FAILED" {
            throw AIError.invalidResponse(provider: providerID, message: "Google file processing failed for \(file["name"]?.stringValue ?? "unknown file").")
        }

        return FileUploadResult(
            providerReference: ["google": file["uri"]?.stringValue ?? ""],
            filename: request.filename,
            mediaType: file["mimeType"]?.stringValue ?? request.mediaType,
            metadata: fileMetadata(from: file),
            rawValue: raw,
            requestMetadata: googleFileUploadRequestMetadata(request),
            responseMetadata: aiResponseMetadata(from: raw, response: metadataResponse)
        )
    }
}

private func googleFileUploadRequestMetadata(_ request: FileUploadRequest) -> AIRequestMetadata {
    var file: [String: JSONValue] = fileUploadMetadata(request, defaultFilename: "file")
    if let displayName = request.displayName {
        file["displayName"] = .string(displayName)
    }
    return AIRequestMetadata(body: .object(["file": .object(file)]), headers: request.headers)
}

private func headerValue(_ headers: [String: String], _ name: String) -> String? {
    headers.first { $0.key.caseInsensitiveCompare(name) == .orderedSame }?.value
}