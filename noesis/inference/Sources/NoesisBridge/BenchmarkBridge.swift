// Sources/NoesisBridge/BenchmarkBridge.swift

import Foundation
import NoesisEngine

/// JNI bridge structure for passing inference results back to Kotlin
public struct InferenceBridgeResult {
    public let success: Bool
    public let errorMessage: String?
    public let generatedText: String?
    public let tokensGenerated: Int32
    public let inferenceTimeMs: Double
    public let tokensPerSecond: Double
    public let memoryUsageMB: Int32
    public let gpuMemoryUsageMB: Int32
    
    public init(
        success: Bool,
        errorMessage: String? = nil,
        generatedText: String? = nil,
        tokensGenerated: Int32 = 0,
        inferenceTimeMs: Double = 0.0,
        tokensPerSecond: Double = 0.0,
        memoryUsageMB: Int32 = 0,
        gpuMemoryUsageMB: Int32 = 0
    ) {
        self.success = success
        self.errorMessage = errorMessage
        self.generatedText = generatedText
        self.tokensGenerated = tokensGenerated
        self.inferenceTimeMs = inferenceTimeMs
        self.tokensPerSecond = tokensPerSecond
        self.memoryUsageMB = memoryUsageMB
        self.gpuMemoryUsageMB = gpuMemoryUsageMB
    }
    
    public static func fromInferenceResult(_ result: InferenceResult) -> InferenceBridgeResult {
        return InferenceBridgeResult(
            success: true,
            generatedText: result.text,
            tokensGenerated: Int32(result.tokensGenerated),
            inferenceTimeMs: result.timeMs,
            tokensPerSecond: result.tokensPerSecond,
            memoryUsageMB: Int32(result.memoryUsageMB),
            gpuMemoryUsageMB: Int32(result.gpuMemoryUsageMB ?? 0)
        )
    }
    
    public static func fromError(_ error: Error) -> InferenceBridgeResult {
        return InferenceBridgeResult(
            success: false,
            errorMessage: error.localizedDescription
        )
    }
}

// MARK: - Global Engine Instances

/// Global llama.cpp engine instance
private var globalLlamaEngine: LlamaInferenceEngine?

// MARK: - C-Compatible Bridge Functions

/// Initialize llama.cpp engine with model path
/// - Parameters:
///   - modelPath: Path to the .gguf model file
///   - maxContextLength: Maximum context window size
///   - useGpu: Whether to use GPU acceleration
///   - threads: Number of CPU threads (0 = auto-detect)
/// - Returns: InferenceBridgeResult indicating success/failure
@_cdecl("llamaEngineInit")
public func llamaEngineInit(
    _ modelPath: UnsafePointer<CChar>,
    _ maxContextLength: Int32,
    _ useGpu: Bool,
    _ threads: Int32
) -> InferenceBridgeResult {
    
    do {
        let path = String(cString: modelPath)
        let threadCount = threads > 0 ? threads : nil
        
        // Create new llama engine instance
        globalLlamaEngine = LlamaInferenceEngine(
            maxContextLength: maxContextLength,
            useGpu: useGpu,
            threads: threadCount
        )
        
        guard let engine = globalLlamaEngine else {
            return InferenceBridgeResult(
                success: false,
                errorMessage: "Failed to create llama.cpp engine"
            )
        }
        
        let group = DispatchGroup()
        var loadResult: Result<Void, Error>?
        
        group.enter()
        Task {
            do {
                try await engine.loadModel(path: path)
                loadResult = .success(())
            } catch {
                loadResult = .failure(error)
            }
            group.leave()
        }
        
        group.wait()
        
        switch loadResult {
        case .success:
            return InferenceBridgeResult(success: true)
        case .failure(let error):
            return InferenceBridgeResult.fromError(error)
        case .none:
            return InferenceBridgeResult(
                success: false,
                errorMessage: "Model loading timeout"
            )
        }
        
    } catch {
        return InferenceBridgeResult.fromError(error)
    }
}

/// Run warmup on llama.cpp engine
@_cdecl("llamaEngineWarmup")
public func llamaEngineWarmup() -> InferenceBridgeResult {
    guard let engine = globalLlamaEngine else {
        return InferenceBridgeResult(
            success: false,
            errorMessage: "llama.cpp engine not initialized"
        )
    }
    
    let group = DispatchGroup()
    var warmupResult: Result<Void, Error>?
    
    group.enter()
    Task {
        do {
            try await engine.warmup()
            warmupResult = .success(())
        } catch {
            warmupResult = .failure(error)
        }
        group.leave()
    }
    
    group.wait()
    
    switch warmupResult {
    case .success:
        return InferenceBridgeResult(success: true)
    case .failure(let error):
        return InferenceBridgeResult.fromError(error)
    case .none:
        return InferenceBridgeResult(
            success: false,
            errorMessage: "Warmup timeout"
        )
    }
}

/// Run inference on llama.cpp engine
/// - Parameters:
///   - prompt: Input text prompt (C string)
///   - maxTokens: Maximum tokens to generate
///   - temperature: Sampling temperature
/// - Returns: InferenceBridgeResult with generated text and metrics
@_cdecl("llamaEngineGenerate")
public func llamaEngineGenerate(
    _ prompt: UnsafePointer<CChar>,
    _ maxTokens: Int32,
    _ temperature: Float
) -> InferenceBridgeResult {
    
    guard let engine = globalLlamaEngine else {
        return InferenceBridgeResult(
            success: false,
            errorMessage: "llama.cpp engine not initialized"
        )
    }
    
    let promptString = String(cString: prompt)
    
    let group = DispatchGroup()
    var generateResult: Result<InferenceResult, Error>?
    
    group.enter()
    Task {
        do {
            let result = try await engine.generate(
                prompt: promptString,
                maxTokens: Int(maxTokens),
                temperature: temperature
            )
            generateResult = .success(result)
        } catch {
            generateResult = .failure(error)
        }
        group.leave()
    }
    
    group.wait()
    
    switch generateResult {
    case .success(let result):
        return InferenceBridgeResult.fromInferenceResult(result)
    case .failure(let error):
        return InferenceBridgeResult.fromError(error)
    case .none:
        return InferenceBridgeResult(
            success: false,
            errorMessage: "Generation timeout"
        )
    }
}

/// Clean up llama.cpp engine resources
@_cdecl("llamaEngineCleanup")
public func llamaEngineCleanup() -> InferenceBridgeResult {
    guard let engine = globalLlamaEngine else {
        return InferenceBridgeResult(success: true) // Already cleaned up
    }
    
    do {
        engine.cleanup()
        globalLlamaEngine = nil
        return InferenceBridgeResult(success: true)
    } catch {
        return InferenceBridgeResult.fromError(error)
    }
}

/// Check if llama.cpp engine is available
@_cdecl("llamaEngineIsAvailable")
public func llamaEngineIsAvailable() -> Bool {
    return true
}

/// Get display name for llama.cpp engine
@_cdecl("llamaEngineGetDisplayName")
public func llamaEngineGetDisplayName() -> UnsafePointer<CChar>? {
    let displayName = "llama.cpp XCFramework"
    return UnsafePointer<CChar>(strdup(displayName))
}

// MARK: - Memory Management Helper

/// Free C string allocated by Swift
@_cdecl("llamaEngineFreeString")
public func llamaEngineFreeString(_ cString: UnsafePointer<CChar>?) {
    if let cString = cString {
        free(UnsafeMutablePointer(mutating: cString))
    }
}