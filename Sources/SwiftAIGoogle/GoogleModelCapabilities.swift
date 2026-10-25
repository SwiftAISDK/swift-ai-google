import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


struct GoogleModelCapabilities: Equatable, Sendable {
    var supportsGemini2Tools: Bool
    var supportsFileSearch: Bool
    var usesGemini3Features: Bool
}

func googleModelCapabilities(for modelID: String) -> GoogleModelCapabilities {
    let normalizedModelID = modelID.lowercased()
    let geminiModel = normalizedModelID.range(of: #"(^|[/.])gemini-"#, options: .regularExpression).map { range in
        String(normalizedModelID[range.lowerBound...]).trimmingCharacters(in: CharacterSet(charactersIn: "/."))
    }

    let isGeminiModel = geminiModel != nil
    let isGemini2Model = geminiModel.map {
        googleModelFamily($0, matches: "gemini-2")
    } ?? false
    let isGemini25Model = geminiModel.map {
        googleModelFamily($0, matches: "gemini-2.5")
    } ?? false
    let isKnownPreGemini2Model = geminiModel.map { model in
        googleModelFamily(model, matches: "gemini-1")
            || model == "gemini-pro"
            || model == "gemini-pro-vision"
            || googleModelFamily(model, matches: "gemini-robotics-er-1.5")
    } ?? false
    let usesGemini3Features = isGeminiModel
        && !isKnownPreGemini2Model
        && !isGemini2Model

    return GoogleModelCapabilities(
        supportsGemini2Tools: (isGeminiModel && !isKnownPreGemini2Model)
            || normalizedModelID.contains("nano-banana"),
        supportsFileSearch: isGemini25Model || usesGemini3Features,
        usesGemini3Features: usesGemini3Features
    )
}

private func googleModelFamily(_ modelID: String, matches family: String) -> Bool {
    guard modelID.hasPrefix(family) else { return false }
    guard modelID.count > family.count else { return true }
    let boundary = modelID.index(modelID.startIndex, offsetBy: family.count)
    return modelID[boundary] == "." || modelID[boundary] == "-"
}