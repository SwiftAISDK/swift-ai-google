import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public extension GoogleImageGenerationModel {
    var supportsFileInputs: Bool? {
        if ["gemini-2.5-flash-image", "gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"].contains(modelID) { return true }
        return nil
    }

    var supportsMaskInputs: Bool? {
        return supportsFileInputs == nil ? nil : false
    }
}
