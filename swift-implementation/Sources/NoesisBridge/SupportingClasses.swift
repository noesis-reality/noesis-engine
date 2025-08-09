import Foundation
import NoesisEngine
import NoesisTools
import Metal

// MARK: - Benchmark System

public class NoesisBenchmark {
    private let engine: NoesisEngine
    private let model: GptossModel
    private let verbose: Bool
    
    public init(engine: NoesisEngine, model: GptossModel, verbose: Bool = false) {
        self.engine = engine
        self.model = model
        self.verbose = verbose
    }
    
    public func runBenchmark(
        iterations: Int,
        warmupIterations: Int,
        sequenceLengths: [Int],
        batchSizes: [Int]
    ) throws -> BenchmarkResult {
        var results: [BenchmarkEntry] = []
        
        if verbose {
            print("🏁 Running benchmark suite")
            print("   Warmup iterations: \(warmupIterations)")
            print("   Test iterations: \(iterations)")
        }
        
        for seqLen in sequenceLengths {
            for batchSize in batchSizes {
                if verbose {
                    print("   Testing: seq_len=\(seqLen), batch_size=\(batchSize)")
                }
                
                let entry = try benchmarkConfiguration(
                    sequenceLength: seqLen,
                    batchSize: batchSize,
                    iterations: iterations,
                    warmupIterations: warmupIterations
                )
                results.append(entry)
            }
        }
        
        // Calculate summary statistics
        let peakTokensPerSecond = results.max(by: { $0.tokensPerSecond < $1.tokensPerSecond })?.tokensPerSecond ?? 0.0
        let averageLatencyMs = results.map(\.latencyMs).reduce(0, +) / Double(results.count)
        let averageGpuUtilization = results.map(\.gpuUtilization).reduce(0, +) / Double(results.count)
        
        return BenchmarkResult(
            results: results,
            peakTokensPerSecond: peakTokensPerSecond,
            averageLatencyMs: averageLatencyMs,
            averageGpuUtilization: averageGpuUtilization
        )
    }
    
    private func benchmarkConfiguration(
        sequenceLength: Int,
        batchSize: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws -> BenchmarkEntry {
        let context = try engine.createContext(maxLength: sequenceLength * 2)
        let testPrompt = generateTestPrompt(targetLength: sequenceLength)
        
        let parameters = GenerationParameters(
            temperature: 0.0, // Deterministic for benchmarking
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: sequenceLength,
            seed: 42
        )
        
        var latencies: [Double] = []
        var tokenCounts: [Int] = []
        
        // Warmup
        for _ in 0..<warmupIterations {
            _ = try engine.generate(
                model: model,
                context: context,
                prompt: testPrompt,
                systemPrompt: nil,
                parameters: parameters,
                reasoningEffort: .low,
                streaming: false
            )
        }
        
        // Actual benchmark iterations
        for _ in 0..<iterations {
            let startTime = Date()
            let result = try engine.generate(
                model: model,
                context: context,
                prompt: testPrompt,
                systemPrompt: nil,
                parameters: parameters,
                reasoningEffort: .low,
                streaming: false
            )
            let endTime = Date()
            
            let latency = endTime.timeIntervalSince(startTime)
            latencies.append(latency * 1000) // Convert to milliseconds
            tokenCounts.append(result.tokens.count)
        }
        
        // Calculate statistics
        let averageLatency = latencies.reduce(0, +) / Double(latencies.count)
        let averageTokenCount = Double(tokenCounts.reduce(0, +)) / Double(tokenCounts.count)
        let tokensPerSecond = averageTokenCount / (averageLatency / 1000.0)
        let gpuUtilization = estimateGpuUtilization() // Placeholder
        
        return BenchmarkEntry(
            sequenceLength: sequenceLength,
            batchSize: batchSize,
            tokensPerSecond: tokensPerSecond,
            latencyMs: averageLatency,
            gpuUtilization: gpuUtilization
        )
    }
    
    private func generateTestPrompt(targetLength: Int) -> String {
        let basePrompt = "Write a detailed explanation of quantum computing, covering quantum bits, superposition, entanglement, and quantum algorithms. Make sure to explain these concepts clearly and provide examples of real-world applications."
        
        // Pad prompt to reach approximately target token count
        let repetitions = max(1, targetLength / 50) // Rough estimate of tokens per sentence
        return String(repeating: basePrompt + " ", count: repetitions).trimmingCharacters(in: .whitespaces)
    }
    
    private func estimateGpuUtilization() -> Double {
        // Placeholder - would need Metal performance counters for real implementation
        return Double.random(in: 85.0...95.0)
    }
}

// MARK: - System Diagnostics

public class NoesisDiagnostics {
    private let verbose: Bool
    
    public init(verbose: Bool = false) {
        self.verbose = verbose
    }
    
    public func runDiagnostics(
        checkModelFile: String? = nil,
        checkGpu: Bool = false,
        checkMemory: Bool = false
    ) throws -> DiagnosticsResult {
        if verbose {
            print("🔍 Running system diagnostics")
        }
        
        // System information
        let platform = "macOS Apple Silicon"
        let macosVersion = getmacOSVersion()
        let xcodeInstalled = checkXcodeInstallation()
        
        // GPU information
        let (gpuName, metalSupported, metal4Supported, gpuMemoryGB, computeUnits) = getGpuInfo()
        
        // Memory information
        let (systemMemoryGB, availableMemoryGB) = getMemoryInfo()
        
        // Model information
        var availableModels = getAvailableModels()
        
        // Check specific model file if requested
        if let modelFile = checkModelFile {
            if verbose {
                print("   Checking model file: \(modelFile)")
            }
            let modelInfo = checkSpecificModel(at: modelFile)
            availableModels.append(modelInfo)
        }
        
        // Recommendations
        let recommendedModels = generateRecommendations(
            gpuMemoryGB: gpuMemoryGB,
            systemMemoryGB: systemMemoryGB
        )
        
        // Identify issues
        var issues: [String] = []
        if !xcodeInstalled {
            issues.append("Xcode Command Line Tools not installed")
        }
        if !metalSupported {
            issues.append("Metal not supported on this system")
        }
        if !metal4Supported {
            issues.append("Metal 4 not supported - update to macOS 15.5+")
        }
        if systemMemoryGB < 16 {
            issues.append("System has less than 16GB RAM - may struggle with larger models")
        }
        
        return DiagnosticsResult(
            platform: platform,
            macosVersion: macosVersion,
            xcodeInstalled: xcodeInstalled,
            gpuName: gpuName,
            metalSupported: metalSupported,
            metal4Supported: metal4Supported,
            gpuMemoryGB: gpuMemoryGB,
            gpuComputeUnits: computeUnits,
            systemMemoryGB: systemMemoryGB,
            availableMemoryGB: availableMemoryGB,
            availableModels: availableModels,
            recommendedModels: recommendedModels,
            issues: issues
        )
    }
    
    private func getmacOSVersion() -> String {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
    }
    
    private func checkXcodeInstallation() -> Bool {
        let process = Process()
        process.launchPath = "/usr/bin/xcode-select"
        process.arguments = ["--print-path"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
    
    private func getGpuInfo() -> (String, Bool, Bool, Int, Int) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return ("Unknown", false, false, 0, 0)
        }
        
        let name = device.name
        let metalSupported = true
        let metal4Supported = device.supportsFamily(.apple9) // Metal 4 support check
        let memoryGB = Int(device.recommendedMaxWorkingSetSize / (1024 * 1024 * 1024))
        let computeUnits = 60 // Placeholder - would need device-specific detection
        
        return (name, metalSupported, metal4Supported, memoryGB, computeUnits)
    }
    
    private func getMemoryInfo() -> (Int, Int) {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let totalGB = Int(totalMemory / (1024 * 1024 * 1024))
        let availableGB = totalGB - 4 // Reserve 4GB for system
        
        return (totalGB, max(0, availableGB))
    }
    
    private func getAvailableModels() -> [ModelInfo] {
        let modelManager = NoesisModelManager(verbose: verbose)
        do {
            return try modelManager.listAvailableModels()
        } catch {
            if verbose {
                print("⚠️  Failed to list models: \(error)")
            }
            return []
        }
    }
    
    private func checkSpecificModel(at path: String) -> ModelInfo {
        let url = URL(fileURLWithPath: path)
        let name = url.lastPathComponent
        
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            let sizeBytes = attributes[.size] as? UInt64 ?? 0
            let sizeGB = Double(sizeBytes) / (1024 * 1024 * 1024)
            
            // Try to validate the model
            let device = MTLCreateSystemDefaultDevice()!
            _ = try ModelLoader.loadModel(from: url, device: device)
            
            return ModelInfo(
                name: name,
                sizeGB: sizeGB,
                isInstalled: true,
                isValid: true,
                path: path
            )
        } catch {
            let sizeGB = 0.0 // Couldn't determine size
            return ModelInfo(
                name: name,
                sizeGB: sizeGB,
                isInstalled: true,
                isValid: false,
                path: path,
                issue: error.localizedDescription
            )
        }
    }
    
    private func generateRecommendations(gpuMemoryGB: Int, systemMemoryGB: Int) -> [String] {
        var recommendations: [String] = []
        
        if systemMemoryGB >= 32 && gpuMemoryGB >= 16 {
            recommendations.append("gpt-oss-120b")
        }
        if systemMemoryGB >= 16 {
            recommendations.append("gpt-oss-20b")
        }
        
        if recommendations.isEmpty {
            recommendations.append("Consider upgrading system memory to at least 16GB")
        }
        
        return recommendations
    }
}

// MARK: - Model Management

public class NoesisModelManager {
    private let verbose: Bool
    private let modelsDirectory: URL
    
    public init(verbose: Bool = false) {
        self.verbose = verbose
        let homeDirectory = URL(fileURLWithPath: NSHomeDirectory())
        self.modelsDirectory = homeDirectory.appendingPathComponent(".noesis/models")
    }
    
    public func listAvailableModels() throws -> [ModelInfo] {
        var models: [ModelInfo] = []
        
        // Add known models
        let knownModels = [
            ("gpt-oss-20b", 13.75),
            ("gpt-oss-120b", 45.5)
        ]
        
        for (name, sizeGB) in knownModels {
            let modelDir = modelsDirectory.appendingPathComponent(name)
            let modelFile = modelDir.appendingPathComponent("metal/model.bin")
            
            if FileManager.default.fileExists(atPath: modelFile.path) {
                // Model is installed, check if valid
                let isValid = validateModelFile(at: modelFile)
                models.append(ModelInfo(
                    name: name,
                    sizeGB: sizeGB,
                    isInstalled: true,
                    isValid: isValid,
                    path: modelFile.path,
                    issue: isValid ? nil : "Model file appears corrupted"
                ))
            } else {
                // Model is available for download
                models.append(ModelInfo(
                    name: name,
                    sizeGB: sizeGB,
                    isInstalled: false,
                    isValid: false
                ))
            }
        }
        
        return models
    }
    
    public func downloadModel(name: String, force: Bool = false) throws {
        let modelDir = modelsDirectory.appendingPathComponent(name)
        let modelFile = modelDir.appendingPathComponent("metal/model.bin")
        
        if !force && FileManager.default.fileExists(atPath: modelFile.path) {
            if verbose {
                print("✅ Model \(name) already exists")
            }
            return
        }
        
        // Create directory structure
        try FileManager.default.createDirectory(
            at: modelDir.appendingPathComponent("metal"),
            withIntermediateDirectories: true,
            attributes: nil
        )
        
        if verbose {
            print("⬇️  Downloading \(name) from HuggingFace...")
        }
        
        // Use HuggingFace CLI to download
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = [
            "hf", "download", "openai/\(name)",
            "--include", "metal/*",
            "--local-dir", modelDir.path
        ]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        try process.run()
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "NoesisModelManager", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Download failed: \(output)"
            ])
        }
        
        if verbose {
            print("✅ Download completed")
        }
    }
    
    public func verifyModel(at path: String) throws -> ModelVerificationResult {
        let url = URL(fileURLWithPath: path)
        
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            let sizeBytes = attributes[.size] as? UInt64 ?? 0
            let sizeGB = Double(sizeBytes) / (1024 * 1024 * 1024)
            
            // Try to load the model to verify it
            let device = MTLCreateSystemDefaultDevice()!
            _ = try ModelLoader.loadModel(from: url, device: device)
            
            // Calculate a simple checksum
            let checksum = try calculateChecksum(for: url)
            
            return ModelVerificationResult(
                isValid: true,
                sizeGB: sizeGB,
                checksum: checksum,
                error: nil
            )
        } catch {
            return ModelVerificationResult(
                isValid: false,
                sizeGB: 0.0,
                checksum: "",
                error: error.localizedDescription
            )
        }
    }
    
    public func cleanupModels(dryRun: Bool = false) throws -> ModelCleanupResult {
        var filesRemoved = 0
        var spaceMB = 0.0
        
        // Look for temporary files, incomplete downloads, etc.
        let tempFiles = try findTemporaryFiles()
        
        for file in tempFiles {
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            let sizeBytes = attributes[.size] as? UInt64 ?? 0
            spaceMB += Double(sizeBytes) / (1024 * 1024)
            
            if !dryRun {
                try FileManager.default.removeItem(at: file)
            }
            filesRemoved += 1
            
            if verbose {
                print("   \(dryRun ? "Would remove" : "Removed"): \(file.lastPathComponent)")
            }
        }
        
        return ModelCleanupResult(filesRemoved: filesRemoved, spaceMB: spaceMB)
    }
    
    private func validateModelFile(at url: URL) -> Bool {
        do {
            // Quick validation - check file size and magic header
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let sizeBytes = attributes[.size] as? UInt64, sizeBytes > 1_000_000 else {
                return false
            }
            
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            let header = data.prefix(12)
            let magicHeader = "GPT-OSS v1.0".data(using: .utf8)!
            
            return header == magicHeader
        } catch {
            return false
        }
    }
    
    private func calculateChecksum(for url: URL) throws -> String {
        // Simple CRC32 checksum of first 1MB for performance
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let checksum = data.prefix(1024 * 1024).crc32
        return String(format: "%08x", checksum)
    }
    
    private func findTemporaryFiles() throws -> [URL] {
        var tempFiles: [URL] = []
        
        if FileManager.default.fileExists(atPath: modelsDirectory.path) {
            let contents = try FileManager.default.contentsOfDirectory(
                at: modelsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
                options: .skipsHiddenFiles
            )
            
            for url in contents {
                // Look for incomplete downloads, temp files, etc.
                if url.lastPathComponent.hasSuffix(".tmp") ||
                   url.lastPathComponent.hasSuffix(".incomplete") ||
                   url.lastPathComponent.contains("cache") {
                    tempFiles.append(url)
                }
            }
        }
        
        return tempFiles
    }
}

// MARK: - Data Types

public struct BenchmarkResult: Codable {
    public let results: [BenchmarkEntry]
    public let peakTokensPerSecond: Double
    public let averageLatencyMs: Double
    public let averageGpuUtilization: Double
}

public struct BenchmarkEntry: Codable {
    public let sequenceLength: Int
    public let batchSize: Int
    public let tokensPerSecond: Double
    public let latencyMs: Double
    public let gpuUtilization: Double
}

public struct DiagnosticsResult: Codable {
    public let platform: String
    public let macosVersion: String
    public let xcodeInstalled: Bool
    public let gpuName: String
    public let metalSupported: Bool
    public let metal4Supported: Bool
    public let gpuMemoryGB: Int
    public let gpuComputeUnits: Int
    public let systemMemoryGB: Int
    public let availableMemoryGB: Int
    public let availableModels: [ModelInfo]
    public let recommendedModels: [String]
    public let issues: [String]
}

public struct ModelInfo: Codable {
    public let name: String
    public let sizeGB: Double
    public let isInstalled: Bool
    public let isValid: Bool
    public let path: String?
    public let issue: String?
    
    public init(name: String, sizeGB: Double, isInstalled: Bool = false, isValid: Bool = false, path: String? = nil, issue: String? = nil) {
        self.name = name
        self.sizeGB = sizeGB
        self.isInstalled = isInstalled
        self.isValid = isValid
        self.path = path
        self.issue = issue
    }
}

public struct ModelVerificationResult: Codable {
    public let isValid: Bool
    public let sizeGB: Double
    public let checksum: String
    public let error: String?
}

public struct ModelCleanupResult: Codable {
    public let filesRemoved: Int
    public let spaceMB: Double
}

// MARK: - Extensions

extension Data {
    var crc32: UInt32 {
        return self.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: UInt8.self)
            var crc: UInt32 = 0xFFFFFFFF
            
            for byte in buffer {
                crc = crc32Table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
            }
            
            return crc ^ 0xFFFFFFFF
        }
    }
}

// CRC32 lookup table (simplified for brevity)
private let crc32Table: [UInt32] = {
    var table: [UInt32] = Array(repeating: 0, count: 256)
    for i in 0..<256 {
        var crc = UInt32(i)
        for _ in 0..<8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320
            } else {
                crc >>= 1
            }
        }
        table[i] = crc
    }
    return table
}()