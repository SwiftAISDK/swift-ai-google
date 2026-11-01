import Foundation
import SwiftAIProvider

/// Preprocessing controls for a video file part in a Gemini prompt.
/// Place `jsonValue` under the part's `providerMetadata["google"]["videoMetadata"]`.
public struct GoogleVideoMetadata: Codable, Equatable, Sendable {
    public var startOffset: String?
    public var endOffset: String?
    /// Frames per second in (0, 24]. The service defaults to 1 when omitted.
    public var fps: Double?

    public init(startOffset: String? = nil, endOffset: String? = nil, fps: Double? = nil) {
        self.startOffset = startOffset
        self.endOffset = endOffset
        self.fps = fps
    }

    public var jsonValue: JSONValue {
        var value: [String: JSONValue] = [:]
        value["startOffset"] = startOffset.map(JSONValue.string)
        value["endOffset"] = endOffset.map(JSONValue.string)
        value["fps"] = fps.map(JSONValue.number)
        return .object(value)
    }
}

func googleVideoMetadata(_ metadata: [String: JSONValue], providerOptionsNames: [String]) throws -> JSONValue? {
    let names = providerOptionsNames + ["google", "googleVertex", "vertex"].filter { !providerOptionsNames.contains($0) }
    guard let options = names.compactMap({ metadata[$0].flatMap { $0 == .null ? nil : $0 } }).first,
          let raw = options["videoMetadata"], raw != .null else { return nil }
    guard let object = raw.objectValue else {
        throw AIError.invalidArgument(argument: "videoMetadata", message: "Google videoMetadata must be an object.")
    }
    var result: [String: JSONValue] = [:]
    for key in ["startOffset", "endOffset"] {
        if let value = object[key] {
            guard value.stringValue != nil else {
                throw AIError.invalidArgument(argument: "videoMetadata.\(key)", message: "Google video \(key) must be a string.")
            }
            result[key] = value
        }
    }
    if let value = object["fps"] {
        guard let fps = value.doubleValue, fps.isFinite, fps > 0, fps <= 24 else {
            throw AIError.invalidArgument(argument: "videoMetadata.fps", message: "Google video fps must be greater than 0 and at most 24.")
        }
        result["fps"] = value
    }
    return .object(result)
}
