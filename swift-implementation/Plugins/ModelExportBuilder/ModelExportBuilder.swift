import Foundation
import PackagePlugin

/// Native Swift build plugin that downloads pre-converted Metal weights from HuggingFace
/// Uses FileManager for directory operations and minimal external commands
@main
struct ModelExportBuilder: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        var commands: [Command] = []
        
        // Use standard ~/.noesis/models location matching config expectations
        let homeURL = URL(fileURLWithPath: NSHomeDirectory())
        let noesisURL = homeURL.appendingPathComponent(".noesis/models")
        let modelDirURL = noesisURL.appendingPathComponent("gpt-oss-20b") // Base download directory 
        let metalModelURL = modelDirURL.appendingPathComponent("metal")    // HF creates this subdirectory
        let modelBinURL = metalModelURL.appendingPathComponent("model.bin") // Final model location
        
        // Convert URLs to Paths for Swift Package Manager compatibility
        let baseModelDir = Path(modelDirURL.path)  // Directory to download TO
        let modelBinPath = Path(modelBinURL.path)   // Final model.bin location
        
        print("🏗️  ModelExportBuilder: Setting up pre-converted Metal model")
        print("   🤗 HuggingFace model: openai/gpt-oss-20b (metal/* pre-converted)")
        print("   📁 Target: \(modelBinPath.string)")
        
        // Check if model already exists and is valid using native Swift
        let fileManager = FileManager.default
        
        if fileManager.fileExists(atPath: modelBinPath.string) {
            do {
                let attributes = try fileManager.attributesOfItem(atPath: modelBinPath.string)
                if let fileSize = attributes[.size] as? Int, fileSize > 10_000_000 {
                    // Check GPT-OSS magic header
                    let data = try Data(contentsOf: modelBinURL, options: [.mappedIfSafe])
                    let magicHeader = Data("GPT-OSS v1.0".utf8)
                    if data.prefix(12) == magicHeader {
                        print("✅ Valid pre-converted Metal model found (\(fileSize.formatted()) bytes)")
                        return [] // No commands needed, model is ready
                    }
                }
            } catch {
                print("⚠️  Model file exists but couldn't verify, will re-download")
            }
        }
        
        // Add command to create directory and download model using shell
        commands.append(.buildCommand(
            displayName: "Setup model directory and download",
            executable: Path("/bin/sh"),
            arguments: [
                "-c",
                """
                # Create directory structure
                mkdir -p '\(baseModelDir.string)' || {
                    echo "❌ Cannot create ~/.noesis directory due to build sandbox restrictions"
                    echo "   Please run manually: mkdir -p \(baseModelDir.string)"
                    echo "   Then re-run: swift build --product noesis-generate"
                    exit 1
                }
                echo "✅ Created directory: \(baseModelDir.string)"
                
                # Check if model already exists and is valid
                MODEL_BIN='\(modelBinPath.string)'
                if [ -f "$MODEL_BIN" ] && [ $(stat -f%z "$MODEL_BIN" 2>/dev/null || echo 0) -gt 10000000 ]; then
                    if head -c 12 "$MODEL_BIN" | grep -q "GPT-OSS v1.0"; then
                        echo "✅ Valid model already exists: $MODEL_BIN"
                        exit 0
                    fi
                fi
                
                # Download model
                echo "📥 Downloading pre-converted Metal model..."
                export PATH="$HOME/.pyenv/shims:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
                
                if command -v hf >/dev/null 2>&1; then
                    hf download openai/gpt-oss-20b --include "metal/*" --local-dir '\(baseModelDir.string)' || {
                        echo "⚠️  Download failed - continuing build"
                        echo "   Manual download: hf download openai/gpt-oss-20b --include \"metal/*\" --local-dir \(baseModelDir.string)/"
                        exit 0
                    }
                    
                    if [ -f "$MODEL_BIN" ]; then
                        SIZE=$(stat -f%z "$MODEL_BIN" 2>/dev/null || echo 0)
                        echo "✅ Download successful: $SIZE bytes at $MODEL_BIN"
                    else
                        echo "⚠️  Download completed but model.bin not found"
                    fi
                else
                    echo "⚠️  hf command not found - model download skipped"
                    echo "   Install: pip install huggingface-hub[cli]"
                    echo "   Manual: hf download openai/gpt-oss-20b --include \"metal/*\" --local-dir \(baseModelDir.string)/"
                fi
                """
            ],
            environment: [:],
            inputFiles: [],
            outputFiles: []
        ))
        
        return commands
    }
}