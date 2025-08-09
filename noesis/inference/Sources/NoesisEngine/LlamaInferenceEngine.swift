// Sources/NoesisEngine/LlamaInferenceEngine.swift

import Foundation
import LlamaFramework

/// llama.cpp inference engine implementation using XCFramework
public class LlamaInferenceEngine: InferenceEngine {
    
    // MARK: - Private Properties
    
    private var context: OpaquePointer?
    private var model: OpaquePointer?
    private var isModelLoaded = false
    private let maxContextLength: Int32
    private let useGpu: Bool
    private let threads: Int32
    
    // MARK: - Initialization
    
    public init(
        maxContextLength: Int32 = 2048,
        useGpu: Bool = true,
        threads: Int32? = nil
    ) {
        self.maxContextLength = maxContextLength
        self.useGpu = useGpu
        
        if let threads = threads {
            self.threads = threads
        } else {
            self.threads = Int32(ProcessInfo.processInfo.processorCount / 2)
        }
        
        // Initialize llama.cpp backend
        llama_backend_init()
    }
    
    deinit {
        cleanup()
        llama_backend_free()
    }
    
    // MARK: - InferenceEngine Protocol Implementation
    
    public func loadModel(path: String) async throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw InferenceError.modelLoadFailed("Model file not found: \(path)")
        }
        
        // Clean up any existing model
        cleanup()
        
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // Set up model parameters
                    var modelParams = llama_model_default_params()
                    modelParams.n_gpu_layers = self.useGpu ? -1 : 0
                    modelParams.use_mmap = true
                    modelParams.use_mlock = false
                    
                    // Load the model
                    self.model = llama_load_model_from_file(path.cString(using: .utf8), modelParams)
                    guard self.model != nil else {
                        throw InferenceError.modelLoadFailed("Failed to load model from: \(path)")
                    }
                    
                    // Set up context parameters
                    var contextParams = llama_context_default_params()
                    contextParams.n_ctx = self.maxContextLength
                    contextParams.n_threads = self.threads
                    contextParams.n_threads_batch = self.threads
                    contextParams.seed = UInt32.random(in: 0...UInt32.max)
                    contextParams.f16_kv = true
                    contextParams.logits_all = false
                    
                    // Create context
                    self.context = llama_new_context_with_model(self.model!, contextParams)
                    guard self.context != nil else {
                        throw InferenceError.modelLoadFailed("Failed to create llama.cpp context")
                    }
                    
                    self.isModelLoaded = true
                    continuation.resume(returning: ())
                    
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    public func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float
    ) async throws -> InferenceResult {
        guard isModelLoaded, let context = context, let model = model else {
            throw InferenceError.modelNotLoaded
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let startMemory = self.getCurrentMemoryUsage()
                    
                    // Tokenize input prompt
                    let promptTokens = self.tokenizePrompt(prompt, model: model)
                    
                    // Prepare generation parameters
                    let maxNewTokens = min(maxTokens, Int(self.maxContextLength) - promptTokens.count)
                    guard maxNewTokens > 0 else {
                        throw InferenceError.invalidParameters("Prompt too long for context window")
                    }
                    
                    // Evaluate prompt tokens
                    let promptResult = llama_decode(context, llama_batch_get_one(promptTokens, Int32(promptTokens.count), 0, 0))
                    guard promptResult >= 0 else {
                        throw InferenceError.generationFailed("Failed to evaluate prompt tokens")
                    }
                    
                    // Generate tokens one by one
                    var generatedTokens: [llama_token] = []
                    var generatedText = ""
                    
                    for _ in 0..<maxNewTokens {
                        // Sample next token
                        let nextToken = self.sampleNextToken(context: context, temperature: temperature)
                        
                        // Check for end-of-sequence
                        if llama_token_is_eog(model, nextToken) {
                            break
                        }
                        
                        generatedTokens.append(nextToken)
                        
                        // Convert token to text
                        if let tokenText = self.tokenToString(nextToken, model: model) {
                            generatedText += tokenText
                        }
                        
                        // Evaluate next token
                        let evalResult = llama_decode(context, llama_batch_get_one([nextToken], 1, Int32(promptTokens.count + generatedTokens.count - 1), 0))
                        guard evalResult >= 0 else {
                            throw InferenceError.generationFailed("Failed to evaluate generated token")
                        }
                    }
                    
                    let endTime = CFAbsoluteTimeGetCurrent()
                    let endMemory = self.getCurrentMemoryUsage()
                    
                    let inferenceTimeMs = (endTime - startTime) * 1000.0
                    let tokensPerSecond = Double(generatedTokens.count) / (inferenceTimeMs / 1000.0)
                    let memoryUsageMB = max(endMemory - startMemory, 0)
                    
                    let result = InferenceResult(
                        text: generatedText,
                        tokensGenerated: generatedTokens.count,
                        timeMs: inferenceTimeMs,
                        tokensPerSecond: tokensPerSecond,
                        memoryUsageMB: memoryUsageMB,
                        gpuMemoryUsageMB: self.useGpu ? self.getGpuMemoryUsage() : nil,
                        metadata: [
                            "engine": "llama.cpp",
                            "version": "b6121",
                            "gpu_layers": self.useGpu ? -1 : 0,
                            "threads": self.threads,
                            "context_length": self.maxContextLength,
                            "prompt_tokens": promptTokens.count
                        ]
                    )
                    
                    continuation.resume(returning: result)
                    
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    public func warmup() async throws {
        guard isModelLoaded else {
            throw InferenceError.modelNotLoaded
        }
        
        _ = try await generate(prompt: "Hello", maxTokens: 1, temperature: 0.0)
    }
    
    public func cleanup() {
        if let context = context {
            llama_free(context)
            self.context = nil
        }
        
        if let model = model {
            llama_free_model(model)
            self.model = nil
        }
        
        isModelLoaded = false
    }
    
    public func getDisplayName() -> String {
        return "llama.cpp\(useGpu ? " (GPU)" : " (CPU)") XCFramework"
    }
    
    public func isAvailable() -> Bool {
        return true
    }
    
    // MARK: - Private Helper Methods
    
    private func tokenizePrompt(_ prompt: String, model: OpaquePointer) -> [llama_token] {
        let cPrompt = prompt.cString(using: .utf8)!
        let maxTokens = Int32(prompt.count + 100)
        var tokens = Array<llama_token>(repeating: 0, count: Int(maxTokens))
        
        let tokenCount = llama_tokenize(
            model,
            cPrompt,
            Int32(cPrompt.count - 1),
            &tokens,
            maxTokens,
            true,
            false
        )
        
        guard tokenCount >= 0 else {
            return []
        }
        
        return Array(tokens.prefix(Int(tokenCount)))
    }
    
    private func sampleNextToken(context: OpaquePointer, temperature: Float) -> llama_token {
        let logits = llama_get_logits(context)
        let nVocab = llama_n_vocab(llama_get_model(context))
        
        // Create candidates array
        var candidates = Array<llama_token_data>(repeating: llama_token_data(), count: Int(nVocab))
        for i in 0..<Int(nVocab) {
            candidates[i].id = llama_token(i)
            candidates[i].logit = logits![i]
            candidates[i].p = 0.0
        }
        
        // Create candidates pointer
        var candidatesP = llama_token_data_array(
            data: &candidates,
            size: Int(nVocab),
            sorted: false
        )
        
        // Apply temperature sampling
        llama_sample_softmax(context, &candidatesP)
        
        if temperature <= 0.0 {
            return llama_sample_token_greedy(context, &candidatesP)
        } else {
            llama_sample_temp(context, &candidatesP, temperature)
            return llama_sample_token(context, &candidatesP)
        }
    }
    
    private func tokenToString(_ token: llama_token, model: OpaquePointer) -> String? {
        let bufferSize = 256
        var buffer = Array<CChar>(repeating: 0, count: bufferSize)
        
        let length = llama_token_to_piece(model, token, &buffer, Int32(bufferSize), false)
        guard length > 0 else {
            return nil
        }
        
        return String(cString: buffer)
    }
    
    private func getCurrentMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            return Int(info.resident_size) / (1024 * 1024)
        }
        
        return 0
    }
    
    private func getGpuMemoryUsage() -> Int {
        return 0
    }
}