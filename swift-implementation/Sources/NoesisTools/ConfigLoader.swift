import Foundation

/// Simplified configuration for Noesis CLI tools
public struct NoesisConfig: Codable {
    public struct Debug: Codable {
        public let verbose: Bool
        public let showModelPaths: Bool
        public let logTokenization: Bool
    }
    
    public struct Generation: Codable {
        public let defaultTemperature: Float
        public let defaultMaxTokens: Int
        public let defaultTopP: Float
    }
    
    public struct Chat: Codable {
        public let systemPrompt: String
        public let reasoningLevel: String
        public let showStats: Bool
    }
    
    public let debug: Debug
    public let generation: Generation
    public let chat: Chat
}

public class ConfigLoader {
    private static let configFileName = "noesis.config.json"
    private static let templateFileName = "noesis.config.template.json"
    
    /// Load configuration from various locations
    public static func load() -> NoesisConfig? {
        // Search paths in order of priority
        let searchPaths = [
            // 1. Current directory
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(configFileName),
            
            // 2. User's home directory ~/.noesis/
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".noesis")
                .appendingPathComponent(configFileName),
            
            // 3. Next to the executable
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent(configFileName),
            
            // 4. In the Swift package directory (for development)
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("noesis.config.json")
        ]
        
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path.path) {
                do {
                    let data = try Data(contentsOf: path)
                    let config = try JSONDecoder().decode(NoesisConfig.self, from: data)
                    return config
                } catch {
                    print("Warning: Failed to load config from \(path.path): \(error)")
                }
            }
        }
        
        return nil
    }
    
    /// Get model path using automatic discovery or provided path
    public static func resolveModelPath(_ providedPath: String?) -> String? {
        if let providedPath = providedPath {
            // If a path is explicitly provided, use it
            return providedPath
        }
        
        // Use automatic discovery from standard location
        return discoverModelPath()
    }
    
    /// Automatically discover model from standard ~/.noesis/models location
    public static func discoverModelPath() -> String? {
        let fileManager = FileManager.default
        let homeDir = fileManager.homeDirectoryForCurrentUser
        
        // Standard model locations in priority order
        let modelPaths = [
            homeDir.appendingPathComponent(".noesis/models/gpt-oss-20b/metal/model.bin"),
            homeDir.appendingPathComponent(".noesis/models/gpt-oss-120b/metal/model.bin")
        ]
        
        for modelPath in modelPaths {
            if fileManager.fileExists(atPath: modelPath.path) {
                // Verify it's a valid GPT-OSS model
                do {
                    let attributes = try fileManager.attributesOfItem(atPath: modelPath.path)
                    if let fileSize = attributes[.size] as? Int, fileSize > 10_000_000 {
                        let data = try Data(contentsOf: modelPath, options: [.mappedIfSafe])
                        if data.prefix(12) == Data("GPT-OSS v1.0".utf8) {
                            return modelPath.path
                        }
                    }
                } catch {
                    continue // Try next path
                }
            }
        }
        
        return nil
    }
    
    /// Create simplified default config file  
    public static func createDefaultConfig() throws {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let noesisDir = homeDir.appendingPathComponent(".noesis")
        let configPath = noesisDir.appendingPathComponent(configFileName)
        
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: noesisDir,
            withIntermediateDirectories: true
        )
        
        // Create simplified default config
        let defaultConfig = NoesisConfig(
            debug: NoesisConfig.Debug(
                verbose: false,
                showModelPaths: false,
                logTokenization: false
            ),
            generation: NoesisConfig.Generation(
                defaultTemperature: 0.7,
                defaultMaxTokens: 500,
                defaultTopP: 0.9
            ),
            chat: NoesisConfig.Chat(
                systemPrompt: "You are a helpful AI assistant.",
                reasoningLevel: "medium",
                showStats: false
            )
        )
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(defaultConfig)
        try data.write(to: configPath)
        
        print("Created simplified config file at: \(configPath.path)")
    }
    
    /// Convenience method to check if verbose debug mode is enabled
    public static func isVerboseMode() -> Bool {
        return load()?.debug.verbose ?? false
    }
    
    /// Convenience method to check if model paths should be shown
    public static func shouldShowModelPaths() -> Bool {
        return load()?.debug.showModelPaths ?? false
    }
}