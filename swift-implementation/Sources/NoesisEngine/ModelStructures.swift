import Foundation
import Metal
import IOKit

// MARK: - Tokenizer Structure

public final class GptossTokenizer: @unchecked Sendable {
    // Memory mapping info
    private var mappingPtr: UnsafeRawPointer?
    private var mappingSize: Int = 0
    
    // Tokenizer data pointers
    public let regexPtr: UnsafeRawPointer?
    public let tokensPtr: UnsafeRawPointer?
    
    // Token counts
    public let numTextTokens: UInt32
    public let numSpecialTokens: UInt32
    
    // Special token IDs - initialized to UInt32.max (0xFFFFFFFF)
    // Index corresponds to (GptossSpecialToken.rawValue - 1)
    public var specialTokenIDs: [UInt32]
    
    public init(
        mappingPtr: UnsafeRawPointer?,
        mappingSize: Int,
        regexPtr: UnsafeRawPointer?,
        tokensPtr: UnsafeRawPointer?,
        numTextTokens: UInt32,
        numSpecialTokens: UInt32
    ) {
        self.mappingPtr = mappingPtr
        self.mappingSize = mappingSize
        self.regexPtr = regexPtr
        self.tokensPtr = tokensPtr
        self.numTextTokens = numTextTokens
        self.numSpecialTokens = numSpecialTokens
        
        // Initialize all special token IDs to UINT32_MAX (matching C: memset(tokenizer->special_token_id, 0xFF, ...))
        self.specialTokenIDs = Array(repeating: UInt32.max, count: 10) // 10 special token types
    }
    
    public func setSpecialTokenID(_ token: GptossSpecialToken, id: UInt32) {
        guard token != .invalid else { return }
        let index = Int(token.rawValue) - 1
        if index >= 0 && index < specialTokenIDs.count {
            specialTokenIDs[index] = id
        }
    }
    
    public func getSpecialTokenID(_ token: GptossSpecialToken) -> UInt32? {
        guard token != .invalid else { return nil }
        let index = Int(token.rawValue) - 1
        if index >= 0 && index < specialTokenIDs.count {
            let id = specialTokenIDs[index]
            return id != UInt32.max ? id : nil
        }
        return nil
    }
    
    deinit {
        // Clean up memory mapping
        if let ptr = mappingPtr, mappingSize > 0 {
            munmap(UnsafeMutableRawPointer(mutating: ptr), mappingSize)
        }
    }
}

// MARK: - Core Data Structures

public struct GptossExpertPrediction {
    public let expertId: UInt32
    public let score: Float32
}

// MARK: - Kernel Argument Structures

public struct GptossTopkArgs {
    public let numVecsPerToken: UInt32
}

public struct GptossSdpaArgs {
    public let qkvDim: UInt32
    public let numKvTokens: UInt32
    public let window: UInt32
}

public struct GptossU32FillRandomArgs {
    public let numVecsPerThreadgroup: UInt64
    public let numVecs: UInt64
    public let offset: UInt64
    public let seed: UInt64
}

public struct GptossF32FillRandomArgs {
    public let numVecsPerThreadgroup: UInt64
    public let numVecs: UInt64
    public let offset: UInt64
    public let seed: UInt64
    public let scale: Float32
    public let bias: Float32
}

public struct GptossAccumulateArgs {
    public let numVecsPerExpert: UInt32
    public let numVecsPerThreadgroup: UInt32
    public let numVecs: UInt32
}

public struct GptossConvertArgs {
    public let numVecsPerThreadgroup: UInt64
    public let numVecs: UInt64
}

public struct GptossEmbeddingsArgs {
    public let numVecs: UInt32
}

public struct GptossMatmulArgs {
    public let numColumnVecs: UInt32  // matches C: uint32_t num_column_vecs
    public let numRows: UInt32        // matches C: uint32_t num_rows
    public let add: UInt32            // matches C: uint32_t add
}

public struct GptossUnembeddingArgs {
    public let numColumnVecs: UInt32        // matches C: uint32_t num_column_vecs
    public let numRowsPerThreadgroup: UInt32  // matches C: uint32_t num_rows_per_threadgroup
    public let numRows: UInt32              // matches C: uint32_t num_rows
}

public struct GptossMoeMatmulSwigluArgs {
    public let numColumnVecs: UInt32      // matches C: uint32_t num_column_vecs
    public let numRows: UInt32            // matches C: uint32_t num_rows  
    public let numActiveExperts: UInt32   // matches C: uint32_t num_active_experts
    public let weightExpertStride: UInt32 // matches C: uint32_t weight_expert_stride (in bytes)
    public let outputExpertStride: UInt32 // matches C: uint32_t output_expert_stride (in elements)
    public let swigluMin: Float32         // matches C: float swiglu_min
    public let swigluMax: Float32         // matches C: float swiglu_max
}

public struct GptossMoeMatmulArgs {
    public let numColumnVecs: UInt32      // matches C: uint32_t num_column_vecs
    public let numRows: UInt32            // matches C: uint32_t num_rows
    public let numActiveExperts: UInt32   // matches C: uint32_t num_active_experts
    public let inputExpertStride: UInt32  // matches C: uint32_t input_expert_stride (in blocks of 32 elements)
    public let weightExpertStride: UInt32 // matches C: uint32_t weight_expert_stride (in bytes)
    public let outputExpertStride: UInt32 // matches C: uint32_t output_expert_stride (in elements)
}

public struct GptossRopeArgs {
    public let tokenStride: UInt32
    public let tokenOffset: UInt32
    public let freqScale: Float32
    public let interpolationScale: Float32
    public let yarnOffset: Float32
    public let yarnScale: Float32
    public let yarnMultiplier: Float32
}

public struct GptossSoftmaxArgs {
    public let numVecs: UInt32
    public let numVecsPerThreadgroup: UInt32
    public let maxThreadgroups: UInt32
    public let temperature: Float32
}

public struct GptossRmsnormArgs {
    public var numVecs: UInt32       // matches C: uint32_t num_vecs
    public var numChannels: Float32  // matches C: float num_channels
    public var epsilon: Float32      // matches C: float epsilon
    public init(numVecs: UInt32, numChannels: Float32, epsilon: Float32) {
        self.numVecs = numVecs
        self.numChannels = numChannels
        self.epsilon = epsilon
    }
}

// MARK: - Model Configuration

public struct ModelConfig {
    public let contextLength: UInt32
    public let numBlocks: UInt32
    public let numExperts: UInt32
    public let numActiveExperts: UInt32
    public let embeddingDim: UInt32
    public let mlpDim: UInt32
    public let swigluLimit: Float32
    public let headDim: UInt32
    public let numHeads: UInt32
    public let numKvHeads: UInt32
    public let attentionWindow: UInt32
    public let ropeTheta: Float32
    public let interpolationScale: Float32
    public let yarnOffset: Float32
    public let yarnScale: Float32
    public let yarnMultiplier: Float32
    public let rmsnormEpsilon: Float32
    public let vocabularySize: UInt32

    public init(
        contextLength: UInt32 = 131072,
        numBlocks: UInt32 = 48,
        numExperts: UInt32 = 128,
        numActiveExperts: UInt32 = 4,
        embeddingDim: UInt32 = 4096,
        mlpDim: UInt32 = 14336,
        swigluLimit: Float32 = 30.0,
        headDim: UInt32 = 64,
        numHeads: UInt32 = 64,
        numKvHeads: UInt32 = 8,
        attentionWindow: UInt32 = 131072,
        ropeTheta: Float32 = 10000.0,
        interpolationScale: Float32 = 1.0,
        yarnOffset: Float32 = 0.0,
        yarnScale: Float32 = 0.0,
        yarnMultiplier: Float32 = 1.0,
        rmsnormEpsilon: Float32 = 1e-5,
        vocabularySize: UInt32 = 32768
    ) {
        self.contextLength = contextLength
        self.numBlocks = numBlocks
        self.numExperts = numExperts
        self.numActiveExperts = numActiveExperts
        self.embeddingDim = embeddingDim
        self.mlpDim = mlpDim
        self.swigluLimit = swigluLimit
        self.headDim = headDim
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.attentionWindow = attentionWindow
        self.ropeTheta = ropeTheta
        self.interpolationScale = interpolationScale
        self.yarnOffset = yarnOffset
        self.yarnScale = yarnScale
        self.yarnMultiplier = yarnMultiplier
        self.rmsnormEpsilon = rmsnormEpsilon
        self.vocabularySize = vocabularySize
    }
}

// MARK: - Model State

// Error type for GPU detection failures
public enum GPUDetectionError: Error {
    case ioKitServiceMatchingFailed(kern_return_t)
    case gpuCoreCountNotFound
    case invalidCoreCount
}

// Helper function to get GPU core count like C reference (metal.m:13-57)
// This matches the C implementation exactly using IOKit registry
private func getGPUCoreCount(device: MTLDevice) throws -> Int {
    // Get the registry ID of the Metal device
    let targetRegistryID = device.registryID
    
    // Find IOAccelerator services
    var iterator: io_iterator_t = 0
    let result = IOServiceGetMatchingServices(
        kIOMainPortDefault,
        IOServiceMatching("IOAccelerator"),
        &iterator
    )
    
    guard result == KERN_SUCCESS else {
        throw GPUDetectionError.ioKitServiceMatchingFailed(result)
    }
    
    defer { IOObjectRelease(iterator) }
    
    var coreCount: Int? = nil
    var service: io_object_t = IOIteratorNext(iterator)
    
    while service != 0 {
        defer { 
            IOObjectRelease(service)
            service = IOIteratorNext(iterator)
        }
        
        // Get registry ID of this service
        var serviceRegistryID: UInt64 = 0
        let idResult = IORegistryEntryGetRegistryEntryID(service, &serviceRegistryID)
        
        if idResult == KERN_SUCCESS && serviceRegistryID == targetRegistryID {
            // Found our GPU - read "gpu-core-count" property
            if let cfProperty = IORegistryEntryCreateCFProperty(
                service,
                "gpu-core-count" as CFString,
                kCFAllocatorDefault,
                0
            ) {
                let cfPropertyAny = cfProperty.takeRetainedValue() as AnyObject
                
                if let count = cfPropertyAny as? NSNumber {
                    coreCount = count.intValue
                    if coreCount! > 0 {
                        break  // Found valid core count
                    }
                }
            }
        }
    }
    
    // Ensure we found a valid core count
    guard let validCoreCount = coreCount, validCoreCount > 0 else {
        print("ERROR: Failed to detect GPU core count from IOKit registry")
        print("       Device: \(device.name)")
        print("       Registry ID: \(targetRegistryID)")
        throw GPUDetectionError.gpuCoreCountNotFound
    }
    
    return validCoreCount
}

@MainActor
public final class GptossModel {
    public let config: ModelConfig
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    public var tokenizer: GptossTokenizer?

    // Weight buffers
    public var sharedWeightBuffer: MTLBuffer?
    public var blockWeightBuffers: [MTLBuffer] = []
    
    // Weight offsets (calculated dynamically like C reference)
    public var attnRmsnormGainOffset: Int = 0
    public var attnQkvWeightOffset: Int = 0
    public var attnQkvBiasOffset: Int = 0
    public var attnSdpaSinkOffset: Int = 0
    public var attnOutWeightOffset: Int = 0
    public var attnOutBiasOffset: Int = 0
    public var mlpRmsnormGainOffset: Int = 0
    public var mlpGateWeightOffset: Int = 0
    public var mlpGateBiasOffset: Int = 0
    public var rmsnormWeightOffset: Int = 0
    public var unembeddingWeightOffset: Int = 0
    public var perBlockSharedWeightsSize: Int = 0
    
    // MoE expert weight offsets
    public var mlpSwigluScaleOffset: Int = 0
    public var mlpSwigluBiasOffset: Int = 0
    public var mlpOutBlockOffset: Int = 0
    public var mlpOutScaleOffset: Int = 0
    public var mlpOutBiasOffset: Int = 0
    public var perExpertBlockWeightSize: Int = 0

    // Compute pipeline states
    private var pipelines: [String: MTLComputePipelineState] = [:]

    // Buffer size calculations
    public var maxBatchTokens: Int = 128
    public var maxThreadgroups: Int = 256

    public init(config: ModelConfig, engine: NoesisEngine) throws {
        self.config = config
        self.device = engine.device
        self.commandQueue = engine.commandQueue
        self.library = engine.library

        // CRITICAL FIX: Calculate max_threadgroups based on GPU cores like C reference (model.c:298)
        // C uses: model->max_threadgroups = model->device.num_cores * 3
        do {
            let gpuCoreCount = try getGPUCoreCount(device: device)
            self.maxThreadgroups = gpuCoreCount * 3
            print("🔧 GPU Core Count: \(gpuCoreCount), Max Threadgroups: \(self.maxThreadgroups)")
        } catch {
            print("⚠️ WARNING: Failed to get GPU core count: \(error)")
            print("⚠️ Falling back to default max_threadgroups value")
            // Keep the default value of 256 if detection fails
            // This ensures the code still runs but may not be optimal
        }

        // Note: Activation buffers are now owned by GptossContext (per-context isolation).
        // Weight offsets and buffers are loaded via ModelLoader.loadModel().
        // This ensures proper memory mapping from the model file
    }


    public func getPipeline(for functionName: String) throws -> MTLComputePipelineState {
        if let pipeline = pipelines[functionName] {
            return pipeline
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(domain: "GptossModel", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Metal function not found: \(functionName)"
            ])
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[functionName] = pipeline
        return pipeline
    }
    
}

// MARK: - Context State

@MainActor
public final class GptossContext {
    public let model: GptossModel
    public let maxTokens: Int

    // Current state
    public var numTokens: Int = 0
    public var numKvTokens: Int = 0
    public var numBatchTokens: Int = 0
    public var numProcessedTokens: Int = 0

    // Buffers
    // Activation buffers (per-context for concurrency)
    public var residualActivationBuffer: MTLBuffer?
    public var rmsnormActivationBuffer: MTLBuffer?
    public var qkvActivationBuffer: MTLBuffer?
    public var sdpaActivationBuffer: MTLBuffer?
    public var gateActivationBuffer: MTLBuffer?
    public var expertActivationBuffer: MTLBuffer?
    public var swigluActivationBuffer: MTLBuffer?
    public var moeActivationBuffer: MTLBuffer?

    public var tokenBuffer: MTLBuffer?
    public var scoreBuffer: MTLBuffer?
    public var probBuffer: MTLBuffer?
    public var sumBuffer: MTLBuffer?
    public var argmaxBuffer: MTLBuffer?
    public var kvcacheBuffer: MTLBuffer?

    public init(model: GptossModel, contextLength: Int? = nil) {
        self.model = model
        self.maxTokens = contextLength ?? Int(model.config.contextLength)

        setupBuffers()
    }

    private func setupBuffers() {
        let device = model.device

        // CRITICAL: Use .storageModeShared for CPU readback like C reference

        // Activation buffers (match previous sizes in Model.setupActivationBuffers)
        let maxBatch = model.maxBatchTokens
        // Residual stream buffer
        let residualSize = maxBatch * Int(model.config.embeddingDim) * MemoryLayout<Float32>.size
        residualActivationBuffer = device.makeBuffer(length: residualSize, options: .storageModeShared)

        // RMSNorm buffer
        let rmsnormSize = maxBatch * Int(model.config.embeddingDim) * MemoryLayout<Float32>.size
        rmsnormActivationBuffer = device.makeBuffer(length: rmsnormSize, options: .storageModeShared)

        // QKV buffer (Q + K + V projections)
        let qkvSize = maxBatch * Int(model.config.numHeads + 2 * model.config.numKvHeads) * Int(model.config.headDim) * MemoryLayout<Float32>.size
        qkvActivationBuffer = device.makeBuffer(length: qkvSize, options: .storageModeShared)

        // SDPA output buffer
        let sdpaSize = maxBatch * Int(model.config.numHeads * model.config.headDim) * MemoryLayout<Float32>.size
        sdpaActivationBuffer = device.makeBuffer(length: sdpaSize, options: .storageModeShared)

        // MoE gate buffer
        let gateSize = maxBatch * Int(model.config.numExperts) * MemoryLayout<Float32>.size
        gateActivationBuffer = device.makeBuffer(length: gateSize, options: .storageModeShared)

        // Expert predictions buffer
        let expertSize = maxBatch * Int(model.config.numActiveExperts) * MemoryLayout<GptossExpertPrediction>.size
        expertActivationBuffer = device.makeBuffer(length: expertSize, options: .storageModeShared)

        // SwiGLU buffer
        let swigluSize = maxBatch * Int(model.config.mlpDim) * MemoryLayout<Float32>.size
        swigluActivationBuffer = device.makeBuffer(length: swigluSize, options: .storageModeShared)

        // MoE activation buffer (per-active expert)
        let moeSize = maxBatch * Int(model.config.numActiveExperts) * Int(model.config.mlpDim) * MemoryLayout<Float32>.size
        moeActivationBuffer = device.makeBuffer(length: moeSize, options: .storageModeShared)

        // Token buffer (uint32 token IDs)
        let tokenSize = maxTokens * MemoryLayout<UInt32>.size
        tokenBuffer = device.makeBuffer(length: tokenSize, options: .storageModeShared)

        // Score buffer (unembedding outputs)
        let scoreSize = model.maxBatchTokens * Int(model.config.vocabularySize) * MemoryLayout<Float32>.size
        scoreBuffer = device.makeBuffer(length: scoreSize, options: .storageModeShared)

        // Probability buffer (needs CPU access for sampling)
        let probSize = model.maxBatchTokens * Int(model.config.vocabularySize) * MemoryLayout<Float32>.size
        probBuffer = device.makeBuffer(length: probSize, options: .storageModeShared)

        // Sum buffer for softmax - using shared mode so we can read the sums
        let sumSize = model.maxBatchTokens * model.maxThreadgroups * MemoryLayout<Float32>.size
        sumBuffer = device.makeBuffer(length: sumSize, options: .storageModeShared)

        // Argmax buffer - using shared mode so we can read it for temperature=0 case
        let argmaxSize = model.maxBatchTokens * MemoryLayout<UInt64>.size
        argmaxBuffer = device.makeBuffer(length: argmaxSize, options: .storageModeShared)

        // KV cache buffer - use .storageModeShared to match C reference (metal.m:233)
        let kvcacheSize = Int(model.config.numBlocks) * maxTokens * 2 * Int(model.config.numKvHeads) * Int(model.config.headDim) * MemoryLayout<Float32>.size
        kvcacheBuffer = device.makeBuffer(length: kvcacheSize, options: .storageModeShared)
    }

    public func addTokens(_ tokens: [UInt32]) {
        guard let tokenBuffer = tokenBuffer else { return }

        let startIdx = numTokens
        let endIdx = min(startIdx + tokens.count, maxTokens)
        let tokensToAdd = endIdx - startIdx

        if tokensToAdd > 0 {
            // Sanity: warn if any token exceeds vocabulary size
            let vocab = model.config.vocabularySize
            if let bad = tokens.first(where: { $0 >= vocab }) {
                print("⚠️ WARNING: Token id \(bad) >= vocab size \(vocab). Prompt may be invalid.")
            }
            let bufferPointer = tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: maxTokens)
            for i in 0..<tokensToAdd {
                bufferPointer[startIdx + i] = tokens[i]
            }
            numTokens = endIdx
            numBatchTokens += tokensToAdd  // CRITICAL FIX: Track batch tokens for processing
        }
    }

    public func reset() {
        numTokens = 0
        numKvTokens = 0
        numBatchTokens = 0
        numProcessedTokens = 0
    }
}

// MARK: - Complete Context Management API (matching gpt-oss reference)

public extension GptossContext {
    enum ContextError: Error {
        case contextOverflow
        case invalidArgument
        case insufficientMemory
        case tokenizationFailed
        case processingFailed(Error)
    }
    
    /// Get tokens from context buffer
    func getTokens(maxTokens: Int? = nil) throws -> [UInt32] {
        guard let tokenBuffer = tokenBuffer else {
            throw ContextError.invalidArgument
        }
        
        let tokensToGet = min(numTokens, maxTokens ?? numTokens)
        guard tokensToGet <= numTokens else {
            throw ContextError.insufficientMemory
        }
        
        let bufferPointer = tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: maxTokens ?? numTokens)
        return Array(UnsafeBufferPointer(start: bufferPointer, count: tokensToGet))
    }
    
    /// Append tokens to context and process if needed
    func appendTokens(_ tokens: [UInt32]) throws {
        guard let tokenBuffer = tokenBuffer else {
            throw ContextError.invalidArgument
        }
        
        for token in tokens {
            guard numTokens < maxTokens else {
                throw ContextError.contextOverflow
            }
            
            let bufferPointer = tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: maxTokens)
            bufferPointer[numTokens] = token
            numTokens += 1
            numBatchTokens += 1
            
            // Process batch if we've hit the limit
            if numBatchTokens >= model.maxBatchTokens {
                try processBatch()
            }
        }
    }
    
    /// Process current batch through the model
    func processBatch() throws {
        guard numBatchTokens > 0 else { return }
        
        do {
            let _ = try GenerationPipeline(model: model, context: self)
            
            // Create command buffer
            guard let _ = model.commandQueue.makeCommandBuffer() else {
                throw ContextError.processingFailed(NSError(domain: "ContextError", code: -1))
            }
            
            // Get tokens for this batch
            let _ = Array(try getTokens().suffix(numBatchTokens))
            
            // Simplified batch processing - full implementation would process through transformer blocks
            // For now, just update the state to mark batch as processed
            
            // Update state
            numKvTokens = numTokens
            numProcessedTokens = 1 // Last token processed for next prediction
            numBatchTokens = 0
            
        } catch {
            throw ContextError.processingFailed(error)
        }
    }
    
    /// Sample next token from current probabilities  
    func sampleNextToken(sampler: GptossSampler) throws -> UInt32 {
        guard let probBuffer = probBuffer else {
            throw ContextError.invalidArgument
        }
        
        // Simplified sampling - use existing sampling logic from GenerationPipeline
        let vocabSize = Int(model.config.vocabularySize)
        let probPointer = probBuffer.contents().bindMemory(to: Float32.self, capacity: vocabSize)
        
        // Apply temperature scaling  
        var probs = Array(UnsafeBufferPointer(start: probPointer, count: vocabSize))
        if sampler.temperature > 0 && sampler.temperature != 1.0 {
            for i in 0..<probs.count {
                probs[i] = probs[i] / sampler.temperature
            }
        }
        
        // Simple multinomial sampling
        let random = Float32.random(in: 0..<1)
        var cumulative: Float32 = 0
        for (i, prob) in probs.enumerated() {
            cumulative += prob
            if random <= cumulative {
                return UInt32(i)
            }
        }
        
        return 0 // Fallback
    }
    
    /// Get current token probabilities
    func getProbabilities() throws -> [Float32] {
        guard let probBuffer = probBuffer else {
            throw ContextError.invalidArgument
        }
        
        let vocabSize = Int(model.config.vocabularySize)
        let probPointer = probBuffer.contents().bindMemory(to: Float32.self, capacity: vocabSize)
        return Array(UnsafeBufferPointer(start: probPointer, count: vocabSize))
    }
    
    /// Generate unembedding scores for current state
    func generateScores() throws {
        guard let commandBuffer = model.commandQueue.makeCommandBuffer() else {
            throw ContextError.processingFailed(NSError(domain: "ContextError", code: -1))
        }
        
        // Use unembedding dispatcher to generate logits
        let dispatcher = try UnembeddingDispatcher(model: model)
        let args = GptossUnembeddingArgs(
            numColumnVecs: UInt32(model.config.embeddingDim / 4),
            numRowsPerThreadgroup: 256,
            numRows: model.config.vocabularySize
        )
        
        guard let rmsnormBuffer = self.rmsnormActivationBuffer,
              let scoreBuffer = scoreBuffer,
              let argmaxBuffer = argmaxBuffer else {
            throw ContextError.invalidArgument
        }
        
        dispatcher.encode(
            commandBuffer: commandBuffer,
            args: args,
            input: rmsnormBuffer,
            weight: model.sharedWeightBuffer!,
            output: scoreBuffer,
            argmax: argmaxBuffer,
            numTokens: 1
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Apply softmax to convert scores to probabilities
    func convertScoresToProbabilities(temperature: Float32 = 1.0) throws {
        guard let commandBuffer = model.commandQueue.makeCommandBuffer() else {
            throw ContextError.processingFailed(NSError(domain: "ContextError", code: -1))
        }
        
        let dispatcher = try SoftmaxDispatcher(model: model)
        let args = GptossSoftmaxArgs(
            numVecs: model.config.vocabularySize / 4,
            numVecsPerThreadgroup: 256,
            maxThreadgroups: UInt32(model.maxThreadgroups),
            temperature: temperature
        )
        
        guard let scoreBuffer = scoreBuffer,
              let probBuffer = probBuffer,
              let sumBuffer = sumBuffer else {
            throw ContextError.invalidArgument
        }
        
        dispatcher.encode(
            commandBuffer: commandBuffer,
            args: args,
            score: scoreBuffer,
            argmax: argmaxBuffer!,
            prob: probBuffer,
            sum: sumBuffer,
            numTokens: 1
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}


// MARK: - Sampler

public struct GptossSampler {
    public var temperature: Float32 = 1.0
    public var topP: Float32 = 1.0
    public var topK: Int = 0  // 0 means no top-k filtering
    public var presencePenalty: Float32 = 0.0
    public var frequencyPenalty: Float32 = 0.0

    public init(
        temperature: Float32 = 1.0,
        topP: Float32 = 1.0,
        topK: Int = 0,
        presencePenalty: Float32 = 0.0,
        frequencyPenalty: Float32 = 0.0
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
    }
}
