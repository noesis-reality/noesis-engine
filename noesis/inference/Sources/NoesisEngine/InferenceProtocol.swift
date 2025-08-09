// Sources/NoesisEngine/InferenceProtocol.swift

import Foundation

/// Unified interface for all inference engines (Noesis, llama.cpp, GPT-OSS)
/// Enables fair benchmarking and comparison across implementations
public protocol InferenceEngine {
    
    /// Initialize and load the model from the specified path
    /// - Parameter path: Path to the model file or directory
    func loadModel(path: String) async throws
    
    /// Generate text using the loaded model
    /// - Parameters:
    ///   - prompt: Input text prompt
    ///   - maxTokens: Maximum number of tokens to generate  
    ///   - temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
    /// - Returns: Inference result with generated text and performance metrics
    func generate(prompt: String, maxTokens: Int, temperature: Float) async throws -> InferenceResult
    
    /// Warm up the engine (load shaders, allocate buffers, etc.)
    /// Should be called before benchmarking to ensure fair timing
    func warmup() async throws
    
    /// Clean up resources and reset state
    /// Called between benchmark runs to ensure clean state
    func cleanup() throws
    
    /// Get human-readable display name for this engine
    /// - Returns: Display name (e.g., "Noesis Engine (Metal 4)", "llama.cpp (GPU)")
    func getDisplayName() -> String
    
    /// Check if the engine is available and ready to use
    /// - Returns: True if the engine can be used for inference
    func isAvailable() -> Bool
}

/// Result of an inference operation with performance metrics
public struct InferenceResult {
    /// Generated text output
    public let text: String
    
    /// Number of tokens actually generated
    public let tokensGenerated: Int
    
    /// Total inference time in milliseconds
    public let timeMs: Double
    
    /// Generation speed (tokens per second)
    public let tokensPerSecond: Double
    
    /// Memory usage in megabytes
    public let memoryUsageMB: Int
    
    /// GPU memory usage in megabytes (if applicable)
    public let gpuMemoryUsageMB: Int?
    
    /// Additional engine-specific metadata
    public let metadata: [String: Any]
    
    public init(
        text: String,
        tokensGenerated: Int,
        timeMs: Double,
        tokensPerSecond: Double,
        memoryUsageMB: Int,
        gpuMemoryUsageMB: Int? = nil,
        metadata: [String: Any] = [:]
    ) {
        self.text = text
        self.tokensGenerated = tokensGenerated
        self.timeMs = timeMs
        self.tokensPerSecond = tokensPerSecond
        self.memoryUsageMB = memoryUsageMB
        self.gpuMemoryUsageMB = gpuMemoryUsageMB
        self.metadata = metadata
    }
}

/// Standard error types for inference engines
public enum InferenceError: Error, LocalizedError {
    case modelNotLoaded
    case modelLoadFailed(String)
    case generationFailed(String)
    case engineNotAvailable(String)
    case invalidParameters(String)
    case resourceExhausted(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .generationFailed(let reason):
            return "Text generation failed: \(reason)"
        case .engineNotAvailable(let reason):
            return "Inference engine not available: \(reason)"
        case .invalidParameters(let reason):
            return "Invalid parameters: \(reason)"
        case .resourceExhausted(let reason):
            return "Resource exhausted: \(reason)"
        }
    }
}