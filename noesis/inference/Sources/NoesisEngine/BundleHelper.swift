import Foundation

extension Bundle {
    /// A bundle for the NoesisEngine module
    /// This provides Bundle.module-like functionality
    static var noesisEngine: Bundle {
        // Try to find the resource bundle in various locations
        let candidates = [
            // SPM resource bundle location
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("NoesisEngine_NoesisEngine.bundle"),
            // Alternative SPM location
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("../NoesisEngine_NoesisEngine.bundle"),
            // If we're in a framework
            Bundle(for: NoesisEngine.self).bundleURL,
            // Main bundle as fallback
            Bundle.main.bundleURL
        ]
        
        for candidate in candidates {
            if let bundle = Bundle(url: candidate) {
                return bundle
            }
        }
        
        // Final fallback
        return Bundle.main
    }
}