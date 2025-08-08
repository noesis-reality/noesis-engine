import Foundation
import ArgumentParser
import NoesisTools

/// Command-line tool to export models to GPT-OSS format
struct ExportModel: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-model",
        abstract: "Export a Hugging Face checkpoint to GPT-OSS Metal format"
    )
    
    @Argument(help: "Path to checkpoint directory containing config.json and model.safetensors")
    var source: String
    
    @Argument(help: "Path to output model.bin file")
    var destination: String
    
    @Flag(help: "Skip Q/K scaling (not recommended)")
    var noScaling = false
    
    @Flag(help: "Download checkpoint from Hugging Face if not present")
    var download = false
    
    @Option(help: "Hugging Face repository ID (e.g., 'gpt-oss/gpt-oss-20b')")
    var repo: String?
    
    mutating func run() async throws {
        let sourceURL = URL(fileURLWithPath: source)
        let destURL = URL(fileURLWithPath: destination)
        
        // Check if we need to download first
        if download, let repoId = repo {
            print("📥 Downloading checkpoint from \(repoId)...")
            let downloader = HuggingFaceDownloader()
            try await downloader.downloadCheckpoint(repoId: repoId, localDir: sourceURL)
        }
        
        // Verify checkpoint exists
        let configPath = sourceURL.appendingPathComponent("config.json")
        let safetensorsPath = sourceURL.appendingPathComponent("model.safetensors")
        
        guard FileManager.default.fileExists(atPath: configPath.path) else {
            throw ValidationError.fileNotFound("config.json not found at \(configPath.path)")
        }
        
        guard FileManager.default.fileExists(atPath: safetensorsPath.path) else {
            throw ValidationError.fileNotFound("model.safetensors not found at \(safetensorsPath.path)")
        }
        
        // Create output directory if needed
        let outputDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        
        // Export the model
        print("🔄 Exporting model to GPT-OSS format...")
        let exporter = ModelExporter()
        let config = ModelExporter.ExportConfig(
            checkpointDir: sourceURL,
            outputPath: destURL,
            applyQKScaling: !noScaling
        )
        
        try exporter.export(config: config)
        
        // Verify output file
        let outputAttrs = try FileManager.default.attributesOfItem(atPath: destURL.path)
        let fileSize = outputAttrs[.size] as? Int ?? 0
        let sizeGB = Double(fileSize) / (1024 * 1024 * 1024)
        
        print("\n✅ Model exported successfully!")
        print("  Output: \(destURL.path)")
        print("  Size: \(String(format: "%.2f", sizeGB)) GB")
    }
}

// MARK: - Errors

enum ValidationError: LocalizedError {
    case fileNotFound(String)
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound(let msg):
            return msg
        }
    }
}

// MARK: - Main

@main
struct ExportModelCLI {
    static func main() {
        ExportModel.main()
    }
}