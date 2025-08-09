import Foundation
import NoesisEngine
import NoesisTools
import Metal

/**
 * Simplified Swift JNI Bridge for Noesis Engine - INFERENCE ONLY
 * 
 * This bridge provides ONLY the core inference operations needed by the Kotlin layer.
 * All user-layer logic (chat sessions, diagnostics, model management, benchmarking)
 * is handled in Kotlin. This Swift bridge is purely for LLM inference performance.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */

// MARK: - Global State Management (Minimal)

private var globalEngine: NoesisEngine?
private var globalModel: GptossModel?
private var isVerbose: Bool = false

// MARK: - JNI Export Functions (Inference Only)

/**
 * Initialize the Swift inference engine
 */
@_cdecl("Java_ai_noesisreality_engine_NoesisInferenceEngine_nativeInitialize")
public func jniInitialize(env: UnsafeMutablePointer<JNIEnv>, obj: jobject, modelPathJString: jstring?, verbose: jboolean) -> jstring? {
    do {
        isVerbose = (verbose == JNI_TRUE)
        
        // Initialize Metal engine
        globalEngine = try NoesisEngine()
        
        // Load model if path provided
        if let modelPathJString = modelPathJString {
            let modelPath = String.fromJString(env: env, jstr: modelPathJString)
            try loadModel(at: modelPath)
        }
        
        if isVerbose {
            print("🚀 Noesis Swift inference engine initialized")
        }
        
        return "Inference engine initialized successfully".toJString(env: env)
        
    } catch {
        let errorMessage = "Failed to initialize inference engine: \(error)"
        print("❌ \(errorMessage)")
        return errorMessage.toJString(env: env)
    }
}

/**
 * Core text generation - the primary function of this bridge
 */
@_cdecl("Java_ai_noesisreality_engine_NoesisInferenceEngine_nativeGenerate")
public func jniGenerate(env: UnsafeMutablePointer<JNIEnv>, obj: jobject, requestJString: jstring) -> jstring? {
    do {
        let requestString = String.fromJString(env: env, jstr: requestJString)
        let requestData = requestString.data(using: .utf8)!
        let request = try JSONDecoder().decode(InferenceRequest.self, from: requestData)
        
        guard let engine = globalEngine, let model = globalModel else {
            throw InferenceError.engineNotReady
        }
        
        if isVerbose {
            print("🎯 Swift inference: \"\(String(request.prompt.prefix(50)))...\"")
        }
        
        let startTime = Date()
        
        // Create generation context
        let context = try engine.createContext(maxLength: request.maxTokens * 2)
        
        // Set up generation parameters
        let generationParams = GenerationParameters(
            temperature: request.temperature,
            topP: request.topP,
            repetitionPenalty: request.repetitionPenalty,
            maxTokens: request.maxTokens,
            seed: request.seed
        )
        
        // Generate text using Swift Metal performance
        let result = try engine.generate(
            model: model,
            context: context,
            prompt: request.prompt,
            systemPrompt: request.systemPrompt,
            parameters: generationParams,
            reasoningEffort: ReasoningEffort(rawValue: request.reasoningEffort) ?? .medium,
            streaming: false // Kotlin handles streaming
        )
        
        let endTime = Date()
        let timeMs = Int64((endTime.timeIntervalSince(startTime)) * 1000)
        let tokensPerSecond = Double(result.tokens.count) / endTime.timeIntervalSince(startTime)
        
        // Create minimal result focused on inference data
        let inferenceResult = InferenceResult(
            text: result.text,
            tokens: result.tokens,
            tokensGenerated: result.tokens.count,
            timeMs: timeMs,
            tokensPerSecond: tokensPerSecond,
            gpuMemoryMB: getCurrentGPUMemoryUsageMB()
        )
        
        // Serialize result to JSON
        let resultData = try JSONEncoder().encode(inferenceResult)
        let resultString = String(data: resultData, encoding: .utf8)!
        
        if isVerbose {
            print("✅ Swift inference: \(result.tokens.count) tokens in \(timeMs)ms")
        }
        
        return resultString.toJString(env: env)
        
    } catch {
        let errorResult = InferenceResult(
            text: "",
            tokens: [],
            tokensGenerated: 0,
            timeMs: 0,
            tokensPerSecond: 0.0,
            gpuMemoryMB: 0,
            error: "Inference failed: \(error)"
        )
        
        do {
            let errorData = try JSONEncoder().encode(errorResult)
            let errorString = String(data: errorData, encoding: .utf8)!
            return errorString.toJString(env: env)
        } catch {
            return "Fatal inference error".toJString(env: env)
        }
    }
}

/**
 * Get basic model information (only what's needed for inference)
 */
@_cdecl("Java_ai_noesisreality_engine_NoesisInferenceEngine_nativeGetModelInfo")
public func jniGetModelInfo(env: UnsafeMutablePointer<JNIEnv>, obj: jobject) -> jstring? {
    do {
        guard let model = globalModel else {
            throw InferenceError.modelNotLoaded
        }
        
        let modelInfo = ModelInfo(
            name: model.name ?? "Unknown",
            parameters: Int64(model.parameterCount ?? 0),
            contextLength: model.contextLength ?? 8192,
            vocabularySize: model.vocabularySize ?? 50257,
            embeddingDim: model.embeddingDim ?? 4096,
            numBlocks: model.numBlocks ?? 32,
            isLoaded: true
        )
        
        let resultData = try JSONEncoder().encode(modelInfo)
        let resultString = String(data: resultData, encoding: .utf8)!
        
        return resultString.toJString(env: env)
        
    } catch {
        let errorMessage = "Failed to get model info: \(error)"
        print("❌ \(errorMessage)")
        return errorMessage.toJString(env: env)
    }
}

/**
 * Shutdown the inference engine
 */
@_cdecl("Java_ai_noesisreality_engine_NoesisInferenceEngine_nativeShutdown")
public func jniShutdown(env: UnsafeMutablePointer<JNIEnv>, obj: jobject) {
    if isVerbose {
        print("🛑 Shutting down Swift inference engine")
    }
    
    globalModel = nil
    globalEngine = nil
}

// MARK: - Helper Functions (Minimal)

private func loadModel(at path: String) throws {
    guard let engine = globalEngine else {
        throw InferenceError.engineNotReady
    }
    
    if isVerbose {
        print("📦 Loading model: \(path)")
    }
    
    let modelURL = URL(fileURLWithPath: path)
    let device = MTLCreateSystemDefaultDevice()!
    globalModel = try ModelLoader.loadModel(from: modelURL, device: device)
    
    if isVerbose {
        print("✅ Model loaded for inference")
    }
}

private func getCurrentGPUMemoryUsageMB() -> Int {
    // Basic Metal memory usage estimation
    guard let device = MTLCreateSystemDefaultDevice() else { return 0 }
    
    // This is a simplified estimation - in a real implementation,
    // you'd query Metal performance counters
    return Int(device.recommendedMaxWorkingSetSize / (1024 * 1024))
}

// MARK: - Data Structures (Minimal, Inference-Focused)

private struct InferenceRequest: Codable {
    let prompt: String
    let systemPrompt: String?
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let repetitionPenalty: Float
    let reasoningEffort: String
    let seed: Int?
}

private struct InferenceResult: Codable {
    let text: String
    let tokens: [Int]
    let tokensGenerated: Int
    let timeMs: Int64
    let tokensPerSecond: Double
    let gpuMemoryMB: Int
    let error: String?
    
    init(text: String, tokens: [Int], tokensGenerated: Int, timeMs: Int64, tokensPerSecond: Double, gpuMemoryMB: Int, error: String? = nil) {
        self.text = text
        self.tokens = tokens
        self.tokensGenerated = tokensGenerated
        self.timeMs = timeMs
        self.tokensPerSecond = tokensPerSecond
        self.gpuMemoryMB = gpuMemoryMB
        self.error = error
    }
}

private struct ModelInfo: Codable {
    let name: String
    let parameters: Int64
    let contextLength: Int
    let vocabularySize: Int
    let embeddingDim: Int
    let numBlocks: Int
    let isLoaded: Bool
}

private enum InferenceError: Error {
    case engineNotReady
    case modelNotLoaded
    case generationFailed(String)
}

// MARK: - JNI String Helpers (Reused)

extension String {
    static func fromJString(env: UnsafeMutablePointer<JNIEnv>, jstr: jstring) -> String {
        let ptr = env.pointee.pointee.GetStringUTFChars(env, jstr, nil)
        let str = String(cString: ptr!)
        env.pointee.pointee.ReleaseStringUTFChars(env, jstr, ptr)
        return str
    }
    
    func toJString(env: UnsafeMutablePointer<JNIEnv>) -> jstring {
        return env.pointee.pointee.NewStringUTF(env, self)
    }
}