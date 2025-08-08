import Foundation
import Metal

// MARK: - Generation Pipeline

@MainActor
public final class GenerationPipeline {
    private let model: GptossModel
    private let context: GptossContext
    
    // Token frequency tracking for penalties
    private var tokenFrequencies: [UInt32: Int] = [:]
    private var usedTokens: Set<UInt32> = []
    
    // Kernel dispatchers
    private let embeddingsDispatcher: EmbeddingsDispatcher
    private let rmsnormDispatcher: RmsnormDispatcher
    private let matmulDispatcher: MatmulDispatcher
    private let unembeddingDispatcher: UnembeddingDispatcher
    private let ropeDispatcher: RopeDispatcher
    private let sdpaDispatcher: SdpaDispatcher
    private let topkDispatcher: TopKDispatcher
    private let moeMatmulDispatcher: MoeMatmulDispatcher
    private let accumulateDispatcher: AccumulateDispatcher
    private let softmaxDispatcher: SoftmaxDispatcher
    
    public init(model: GptossModel, context: GptossContext) throws {
        self.model = model
        self.context = context
        
        // Initialize dispatchers
        self.embeddingsDispatcher = try EmbeddingsDispatcher(model: model)
        self.rmsnormDispatcher = try RmsnormDispatcher(model: model)
        self.matmulDispatcher = try MatmulDispatcher(model: model)
        self.unembeddingDispatcher = try UnembeddingDispatcher(model: model)
        self.ropeDispatcher = try RopeDispatcher(model: model)
        self.sdpaDispatcher = try SdpaDispatcher(model: model)
        self.topkDispatcher = try TopKDispatcher(model: model)
        self.moeMatmulDispatcher = try MoeMatmulDispatcher(model: model)
        self.accumulateDispatcher = try AccumulateDispatcher(model: model)
        self.softmaxDispatcher = try SoftmaxDispatcher(model: model)
    }
    
    // MARK: - Generation Methods
    
    public func generateTokens(
        prompt: [UInt32],
        maxTokens: Int,
        sampler: GptossSampler = GptossSampler(),
        onToken: ((UInt32) -> Bool)? = nil
    ) throws -> [UInt32] {
        // Reset token tracking
        resetTokenTracking()
        
        // Only reset context if we have a new prompt
        if !prompt.isEmpty {
            context.reset()
            // Add prompt tokens to context and track their frequencies
            context.addTokens(prompt)
            // Track prompt tokens for frequency/presence penalties
            for token in prompt {
                trackToken(token)
            }
        } else {
            // Track existing tokens in context for frequency/presence penalties
            if let existingTokens = try? context.getTokens() {
                for token in existingTokens {
                    trackToken(token)
                }
            }
        }
        
        var generatedTokens: [UInt32] = []
        
        // Generate new tokens
        for _ in 0..<maxTokens {
            // CRITICAL: Process any pending tokens BEFORE generating
            // This matches the C reference implementation
            if context.numBatchTokens > 0 {
                if ProcessInfo.processInfo.environment["NOESIS_DEBUG"] == "1" {
                    print("   [dbg] Processing batch of \(context.numBatchTokens) tokens before generation")
                }
                try processBatch()
            }
            
            // Generate next token from the processed state
            let nextToken = try generateNextToken(sampler: sampler)
            generatedTokens.append(nextToken)
            
            // Track the generated token for penalties
            trackToken(nextToken)
            
            // Add to context for next iteration
            context.addTokens([nextToken])
            
            // Call token callback
            if let onToken = onToken {
                if !onToken(nextToken) {
                    break
                }
            }
            
            // Check for EOS token
            if nextToken == 0 {  // Assuming 0 is EOS
                break
            }
        }
        
        return generatedTokens
    }
    
    private func resetTokenTracking() {
        tokenFrequencies.removeAll()
        usedTokens.removeAll()
    }
    
    private func trackToken(_ token: UInt32) {
        tokenFrequencies[token, default: 0] += 1
        usedTokens.insert(token)
    }
    
    private func processBatch() throws {
        if ProcessInfo.processInfo.environment["NOESIS_DEBUG"] == "1" {
            print("   [dbg] processBatch: numTokens=\(context.numTokens), numBatchTokens=\(context.numBatchTokens), numKvTokens=\(context.numKvTokens)")
        }
        guard let commandBuffer = model.commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "GenerationPipeline", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create command buffer"
            ])
        }
        
        let batchStart = context.numProcessedTokens
        let batchEnd = context.numTokens
        let batchSize = batchEnd - batchStart
        
        if batchSize == 0 { return }
        
        // 1. EMBEDDINGS: Convert tokens to embeddings first
        guard let sharedWeights = model.sharedWeightBuffer,
              let residual = context.residualActivationBuffer,
              let tokenBuffer = context.tokenBuffer else {
            throw NSError(domain: "GenerationPipeline", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Missing required buffers for embeddings"
            ])
        }
        
        let embeddingsArgs = GptossEmbeddingsArgs(
            numVecs: model.config.embeddingDim / 4
        )
        
        // Calculate token offset: (context->num_tokens - context->num_batch_tokens) * sizeof(uint32_t)
        let tokenOffsetInBytes = (context.numTokens - context.numBatchTokens) * MemoryLayout<UInt32>.size
        
        
        embeddingsDispatcher.encode(
            commandBuffer: commandBuffer,
            args: embeddingsArgs,
            tokens: tokenBuffer,
            tokenOffset: tokenOffsetInBytes,  // FIX: Use proper token offset matching C implementation
            weights: sharedWeights,  // Embedding weights (need proper offset)
            output: residual,  // Output to residual stream
            numTokens: batchSize
        )
        
        // 2. Process each transformer block
        for blockIdx in 0..<Int(model.config.numBlocks) {
            try processBlock(
                commandBuffer: commandBuffer,
                blockIdx: blockIdx,
                tokenStart: batchStart,
                tokenEnd: batchEnd
            )
        }
        
        // Commit and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Update context state to match C reference (context.c:446-448)
        context.numKvTokens = context.numTokens  // All tokens are now in KV cache
        context.numProcessedTokens = 1  // CRITICAL FIX: Last block always processes 1 token (C reference line 447)
        context.numBatchTokens = 0  // Batch has been processed
        
        if ProcessInfo.processInfo.environment["NOESIS_DEBUG"] == "1" {
            print("   [dbg] After processBatch: numKvTokens=\(context.numKvTokens), numProcessedTokens=\(context.numProcessedTokens)")
        }
    }
    
    private func processBlock(
        commandBuffer: MTLCommandBuffer,
        blockIdx: Int,
        tokenStart: Int,
        tokenEnd: Int
    ) throws {
        // CRITICAL FIX: Last block should only process 1 token (the last token)
        // This matches C implementation: last_block ? 1 : context->num_batch_tokens
        let isLastBlock = blockIdx == Int(model.config.numBlocks) - 1
        let batchSize = tokenEnd - tokenStart
        let numOutputTokens = isLastBlock ? 1 : batchSize
        let config = model.config
        
        // Calculate input offset matching C reference:
        // offset = embedding_dim * (num_batch_tokens - num_output_tokens) * sizeof(float)
        let inputOffset = Int(config.embeddingDim) * (batchSize - numOutputTokens) * MemoryLayout<Float32>.size
        
        // Get weight buffer for this block
        guard blockIdx < model.blockWeightBuffers.count else { return }
        let blockWeights = model.blockWeightBuffers[blockIdx]
        guard let sharedWeights = model.sharedWeightBuffer,
              let residual = context.residualActivationBuffer,
              let rmsnorm = context.rmsnormActivationBuffer,
              let qkv = context.qkvActivationBuffer,
              let sdpa = context.sdpaActivationBuffer,
              let gate = context.gateActivationBuffer,
              let expert = context.expertActivationBuffer,
              let swiglu = context.swigluActivationBuffer,
              let moe = context.moeActivationBuffer,
              let kvcache = context.kvcacheBuffer else {
            return
        }
        
        // 1. Attention RMSNorm
        let attnRmsnormArgs = GptossRmsnormArgs(
            numVecs: config.embeddingDim / 4,
            numChannels: Float(config.embeddingDim),
            epsilon: config.rmsnormEpsilon
        )
        rmsnormDispatcher.encode(
            commandBuffer: commandBuffer,
            args: attnRmsnormArgs,
            input: residual,
            inputOffset: inputOffset,  // FIX: Use offset for last block
            weights: sharedWeights,  // offset to attention rmsnorm weights
            weightsOffset: model.attnRmsnormGainOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            output: rmsnorm,
            outputOffset: 0,
            tokens: numOutputTokens
        )
        
        // 2. QKV Projection
        let qkvArgs = GptossMatmulArgs(
            numColumnVecs: config.embeddingDim / 4,
            numRows: (config.numHeads + 2 * config.numKvHeads) * config.headDim,
            add: 0
        )
        matmulDispatcher.encode(
            commandBuffer: commandBuffer,
            args: qkvArgs,
            input: rmsnorm,
            inputOffset: 0,  // RMSNorm output starts at 0
            weight: sharedWeights,  // offset to QKV weights
            weightOffset: model.attnQkvWeightOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            bias: sharedWeights,    // offset to QKV bias
            biasOffset: model.attnQkvBiasOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            output: qkv,
            outputOffset: 0,
            numTokens: numOutputTokens
        )
        
        // 3. RoPE (Rotary Position Embeddings)
        let ropeArgs = GptossRopeArgs(
            tokenStride: (config.numHeads + 2 * config.numKvHeads) * config.headDim / 2,
            tokenOffset: UInt32(tokenStart + (batchSize - numOutputTokens)),  // Adjust token offset for last block
            freqScale: -log(config.ropeTheta) / Float(config.headDim),
            interpolationScale: config.interpolationScale,
            yarnOffset: config.yarnOffset,
            yarnScale: config.yarnScale,
            yarnMultiplier: config.yarnMultiplier
        )
        ropeDispatcher.encode(
            commandBuffer: commandBuffer,
            args: ropeArgs,
            activations: qkv,
            numHeadPairs: Int((config.numHeads + 2 * config.numKvHeads) * config.headDim / 2),
            numTokens: numOutputTokens
        )
        
        // 3.5. Copy KV values to cache (CRITICAL: matches C reference lines 209-221)
        // This must happen AFTER RoPE but BEFORE SDPA
        let attnQkvDim = Int(config.numHeads + 2 * config.numKvHeads) * Int(config.headDim)
        for t in 0..<batchSize {
            // Copy K and V from QKV buffer to KV cache
            // Source: QKV buffer at offset for K/V (after Q)
            let kvSourceOffset = (t * attnQkvDim + Int(config.numHeads * config.headDim)) * MemoryLayout<Float32>.size
            
            // Destination: KV cache for this block and token
            let kvDestOffset = (blockIdx * context.maxTokens + context.numKvTokens + t) * 2 * Int(config.numKvHeads) * Int(config.headDim) * MemoryLayout<Float32>.size
            
            // Size: K and V for all KV heads
            let kvCopySize = 2 * Int(config.numKvHeads) * Int(config.headDim) * MemoryLayout<Float32>.size
            
            // Use blit encoder for buffer copy
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                blitEncoder.copy(
                    from: qkv,
                    sourceOffset: kvSourceOffset,
                    to: kvcache,
                    destinationOffset: kvDestOffset,
                    size: kvCopySize
                )
                blitEncoder.endEncoding()
            }
        }
        
        // 4. SDPA (Scaled Dot-Product Attention)
        // Calculate Q offset for SDPA (matches C reference line 227)
        // attnQkvDim already calculated above
        let qOffset = attnQkvDim * (batchSize - numOutputTokens) * MemoryLayout<Float32>.size
        
        // Calculate KV cache offsets (per block)
        let kvCacheOffset = blockIdx * context.maxTokens * 2 * Int(config.numKvHeads) * Int(config.headDim) * MemoryLayout<Float32>.size
        let kCacheOffset = kvCacheOffset
        let vCacheOffset = kvCacheOffset + context.maxTokens * Int(config.numKvHeads) * Int(config.headDim) * MemoryLayout<Float32>.size
        
        // CRITICAL FIX: Alternate window between attentionWindow and UINT32_MAX
        // This matches C reference line 235: n % 2 == 0 ? model->attention_window : UINT32_MAX
        let window = (blockIdx % 2 == 0) ? config.attentionWindow : UInt32.max
        
        let sdpaArgs = GptossSdpaArgs(
            qkvDim: (config.numHeads + 2 * config.numKvHeads) * config.headDim,
            numKvTokens: UInt32(context.numKvTokens + (batchSize - numOutputTokens)),  // Adjust KV token count
            window: window
        )
        sdpaDispatcher.encode(
            commandBuffer: commandBuffer,
            args: sdpaArgs,
            q: qkv,
            qOffset: qOffset,  // CRITICAL FIX: Use correct Q offset for last block
            k: kvcache,  // K cache
            kOffset: kCacheOffset,  // CRITICAL FIX: Per-block KV cache offset
            v: kvcache,  // V cache  
            vOffset: vCacheOffset,  // CRITICAL FIX: Per-block KV cache offset
            s: sharedWeights,  // sink tokens
            sOffset: model.attnSdpaSinkOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            output: sdpa,
            outputOffset: 0,
            numTokens: numOutputTokens,  // CRITICAL FIX: Use numOutputTokens
            numKvHeads: Int(config.numKvHeads)
        )
        
        // 5. Attention Output Projection
        let attnOutArgs = GptossMatmulArgs(
            numColumnVecs: config.numHeads * config.headDim / 4,
            numRows: config.embeddingDim,
            add: 1  // Add to residual
        )
        matmulDispatcher.encode(
            commandBuffer: commandBuffer,
            args: attnOutArgs,
            input: sdpa,
            inputOffset: 0,
            weight: sharedWeights,  // offset to attention output weights
            weightOffset: model.attnOutWeightOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            bias: sharedWeights,    // offset to attention output bias
            biasOffset: model.attnOutBiasOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            output: residual,
            outputOffset: inputOffset,  // FIX: Write back to same offset we read from
            numTokens: numOutputTokens
        )
        
        // 6. MLP RMSNorm
        let mlpRmsnormArgs = GptossRmsnormArgs(
            numVecs: config.embeddingDim / 4,
            numChannels: Float(config.embeddingDim),
            epsilon: config.rmsnormEpsilon
        )
        rmsnormDispatcher.encode(
            commandBuffer: commandBuffer,
            args: mlpRmsnormArgs,
            input: residual,
            inputOffset: inputOffset,  // FIX: Read from same offset
            weights: sharedWeights,  // offset to MLP rmsnorm weights
            weightsOffset: model.mlpRmsnormGainOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            output: rmsnorm,
            outputOffset: 0,
            tokens: numOutputTokens
        )
        
        // 7. MoE Gating
        let gateArgs = GptossMatmulArgs(
            numColumnVecs: config.embeddingDim / 4,
            numRows: config.numExperts,
            add: 0
        )
        matmulDispatcher.encode(
            commandBuffer: commandBuffer,
            args: gateArgs,
            input: rmsnorm,
            inputOffset: 0,
            weight: sharedWeights,  // offset to gate weights
            weightOffset: model.mlpGateWeightOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            bias: sharedWeights,    // offset to gate bias
            biasOffset: model.mlpGateBiasOffset + model.perBlockSharedWeightsSize * blockIdx,  // CRITICAL FIX: Per-block offset
            output: gate,
            outputOffset: 0,
            numTokens: numOutputTokens
        )
        
        // 8. TopK Expert Selection
        let topkArgs = GptossTopkArgs(
            numVecsPerToken: config.numExperts / 4
        )
        topkDispatcher.encode(
            commandBuffer: commandBuffer,
            args: topkArgs,
            input: gate,
            inputOffset: 0,
            output: expert,
            outputOffset: 0,
            numTokens: numOutputTokens,
            numExperts: Int(config.numExperts)
        )
        
        // 9. MoE MLP with SwiGLU
        let moeSwigluArgs = GptossMoeMatmulSwigluArgs(
            numColumnVecs: config.embeddingDim / 32,  // MF4 quantization
            numRows: config.mlpDim,
            numActiveExperts: config.numActiveExperts,
            weightExpertStride: UInt32(MemoryLayout<Float>.size * Int(config.mlpDim * config.embeddingDim / 8)),
            outputExpertStride: config.mlpDim,
            swigluMin: -config.swigluLimit,
            swigluMax: config.swigluLimit
        )
        moeMatmulDispatcher.encodeSwiglu(
            commandBuffer: commandBuffer,
            args: moeSwigluArgs,
            input: rmsnorm,
            inputOffset: 0,
            expert: expert,
            expertOffset: 0,
            weightBlocks: blockWeights,
            weightBlocksOffset: 0,  // Block weights are per-expert, no block multiplier needed
            weightScales: blockWeights,  // scales are interleaved
            weightScalesOffset: model.mlpSwigluScaleOffset,  // Expert buffer offset (no block multiplier)
            bias: blockWeights,          // bias offset
            biasOffset: model.mlpSwigluBiasOffset,  // Expert buffer offset (no block multiplier)
            output: swiglu,
            outputOffset: 0,
            numTokens: numOutputTokens
        )
        
        // 10. MoE Output Projection
        let moeOutArgs = GptossMoeMatmulArgs(
            numColumnVecs: config.mlpDim / 32,  // MF4 quantization
            numRows: config.embeddingDim,
            numActiveExperts: config.numActiveExperts,
            inputExpertStride: config.mlpDim / 32,
            weightExpertStride: UInt32(MemoryLayout<Float>.size * Int(config.embeddingDim * config.mlpDim / 8)),
            outputExpertStride: config.embeddingDim
        )
        moeMatmulDispatcher.encodeMatmul(
            commandBuffer: commandBuffer,
            args: moeOutArgs,
            input: swiglu,
            inputOffset: 0,
            expert: expert,
            expertOffset: 0,
            weightBlocks: blockWeights,
            weightBlocksOffset: model.mlpOutBlockOffset,  // Expert buffer offset (no block multiplier)
            weightScales: blockWeights,
            weightScalesOffset: model.mlpOutScaleOffset,  // Expert buffer offset (no block multiplier)
            bias: blockWeights,
            biasOffset: model.mlpOutBiasOffset,  // Expert buffer offset (no block multiplier)
            output: moe,
            outputOffset: 0,
            numTokens: numOutputTokens
        )
        
        // 11. Accumulate Expert Outputs
        let accumulateArgs = GptossAccumulateArgs(
            numVecsPerExpert: config.embeddingDim / 4,
            numVecsPerThreadgroup: 256,
            numVecs: config.embeddingDim / 4
        )
        accumulateDispatcher.encode(
            commandBuffer: commandBuffer,
            args: accumulateArgs,
            input: moe,
            inputOffset: 0,
            expert: expert,
            expertOffset: 0,
            output: residual,
            outputOffset: inputOffset,  // FIX: Accumulate back to same offset we read from
            numTokens: numOutputTokens
        )
    }
    
    private func generateNextToken(sampler: GptossSampler) throws -> UInt32 {
        guard let commandBuffer = model.commandQueue.makeCommandBuffer(),
              let sharedWeights = model.sharedWeightBuffer,
              let residual = context.residualActivationBuffer,
              let rmsnorm = context.rmsnormActivationBuffer,
              let scores = context.scoreBuffer,
              let probs = context.probBuffer,
              let argmax = context.argmaxBuffer,
              let sum = context.sumBuffer else {
            throw NSError(domain: "GenerationPipeline", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Missing required buffers"
            ])
        }
        
        let config = model.config
        
        // After processBatch, numProcessedTokens is always 1 (from last block)
        // The final processed token is always at offset 0 in rmsnorm_activation_buffer
        // because the last block only processes 1 token starting from offset 0
        // C reference: input_offset = embedding_dim * (num_batch_tokens - 1) * sizeof(float)
        // With numBatchTokens=1 during generation, this gives offset = 0
        let lastTokenOffset = 0
        
        // Final RMSNorm - read from the last token position
        let finalRmsnormArgs = GptossRmsnormArgs(
            numVecs: config.embeddingDim / 4,
            numChannels: Float(config.embeddingDim),
            epsilon: config.rmsnormEpsilon
        )
        rmsnormDispatcher.encode(
            commandBuffer: commandBuffer,
            args: finalRmsnormArgs,
            input: residual,
            inputOffset: lastTokenOffset,  // FIX: Read from last token position
            weights: sharedWeights,
            weightsOffset: model.rmsnormWeightOffset,  // Final RMSNorm - no block multiplier
            output: rmsnorm,
            outputOffset: 0,
            tokens: 1
        )
        
        // Initialize argmax buffer to 0xFF (required for atomic_min operations)
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            let pattern = UInt8(0xFF)
            // Fill just the first uint64 that we'll use for this single token generation
            blitEncoder.fill(buffer: argmax, range: 0..<8, value: pattern)
            blitEncoder.endEncoding()
        }
        
        // Unembedding (logits computation)
        let unembeddingArgs = GptossUnembeddingArgs(
            numColumnVecs: config.embeddingDim / 4,
            numRowsPerThreadgroup: 256,
            numRows: config.vocabularySize
        )
        unembeddingDispatcher.encode(
            commandBuffer: commandBuffer,
            args: unembeddingArgs,
            input: rmsnorm,
            inputOffset: 0,  // RMSNorm output starts at 0
            weight: sharedWeights,
            weightOffset: model.unembeddingWeightOffset,  // Final unembedding - no block multiplier
            output: scores,
            outputOffset: 0,
            argmax: argmax,
            argmaxOffset: 0,
            numTokens: 1
        )
        
        // C reference logic: temperature=0 uses argmax directly, temperature>0 runs softmax
        if sampler.temperature == 0.0 {
            // Commit and wait for unembedding to complete (argmax computed there)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // Optional debug: dump a few rmsnorm values and top-5 scores
            if ProcessInfo.processInfo.environment["NOESIS_DEBUG"] == "1" {
                let embDim = Int(config.embeddingDim)
                let rmsPtr = rmsnorm.contents().bindMemory(to: Float32.self, capacity: embDim)
                let rmsPreview = Array(UnsafeBufferPointer(start: rmsPtr, count: min(8, embDim)))
                print("   [dbg] rmsnorm[0..7]: \(rmsPreview)")

                let vocabSize = Int(config.vocabularySize)
                let scorePtr = scores.contents().bindMemory(to: Float32.self, capacity: vocabSize)
                var top: [(Int, Float32)] = []
                top.reserveCapacity(8)
                for i in 0..<vocabSize {
                    let v = scorePtr[i]
                    if top.count < 8 {
                        top.append((i, v))
                        top.sort { $0.1 > $1.1 }
                    } else if v > top.last!.1 {
                        top.removeLast()
                        top.append((i, v))
                        top.sort { $0.1 > $1.1 }
                    }
                }
                print("   [dbg] top-8 logits: \(top.map { "\($0.0):\(String(format: "%.3f", $0.1))" }.joined(separator: ", "))")
            }

            // Read directly from argmax buffer like C reference
            // The Metal kernel stores argmax as uint2 {row_index, score_bits} cast to ulong
            // We need to extract the row_index (first 32 bits) as the token ID
            
            // First check if buffer was actually updated
            let argmax64Ptr = argmax.contents().bindMemory(to: UInt64.self, capacity: 1)
            let raw64Value = argmax64Ptr[0]
            
            
            if raw64Value == 0xFFFFFFFFFFFFFFFF {
                print("🚨 WARNING: Argmax buffer unchanged (0xFFFFFFFFFFFFFFFF)")
                print("   This means the unembedding kernel likely failed to compute any results")
                // Fallback: compute argmax on CPU from scores buffer if available
                let vocabSize = Int(config.vocabularySize)
                let scorePtr = scores.contents().bindMemory(to: Float32.self, capacity: vocabSize)
                let scoresArr = Array(UnsafeBufferPointer(start: scorePtr, count: vocabSize))
                if let (idx, _) = scoresArr.enumerated().max(by: { $0.element < $1.element }) {
                    print("   Fallback CPU argmax -> token \(idx)")
                    return UInt32(idx)
                }
                // As a last resort, return <|endoftext|>
                return 199999
            }
            
            let argmaxPtr = argmax.contents().bindMemory(to: UInt32.self, capacity: 2)
            let tokenId = argmaxPtr[0]  // First 32 bits = row index = token ID
            let scoreBits = argmaxPtr[1]  // Second 32 bits = score bits (for comparison)
            
            
            // Bounds check: token ID should be within vocabulary size
            guard tokenId < config.vocabularySize else {
                print("🚨 ERROR: Token ID \(tokenId) exceeds vocabulary size \(config.vocabularySize)")
                print("🚨 Score bits: \(scoreBits)")
                print("🚨 Raw 64-bit value: 0x\(String(raw64Value, radix: 16))")
                // Fallback to <|endoftext|> token
                return 199999
            }
            
            return tokenId
        } else {
            // Temperature > 0: run softmax and sample probabilistically
            let threadgroupSize: UInt32 = 256
            let numVecs = config.vocabularySize
            let maxTg = UInt32(model.maxThreadgroups)
            // math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size
            let numVecsPerThreadgroup = ((numVecs + (maxTg * threadgroupSize) - 1) / (maxTg * threadgroupSize)) * threadgroupSize
            
            let softmaxArgs = GptossSoftmaxArgs(
                numVecs: config.vocabularySize,
                numVecsPerThreadgroup: numVecsPerThreadgroup,
                maxThreadgroups: maxTg,
                temperature: sampler.temperature
            )
            softmaxDispatcher.encode(
                commandBuffer: commandBuffer,
                args: softmaxArgs,
                score: scores,
                argmax: argmax,
                prob: probs,
                sum: sum,
                numTokens: 1
            )
            
            // Commit and wait
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // Optional debug for temperature>0 path
            if ProcessInfo.processInfo.environment["NOESIS_DEBUG"] == "1" {
                let embDim = Int(config.embeddingDim)
                let rmsPtr = rmsnorm.contents().bindMemory(to: Float32.self, capacity: embDim)
                let rmsPreview = Array(UnsafeBufferPointer(start: rmsPtr, count: min(8, embDim)))
                print("   [dbg] rmsnorm[0..7]: \(rmsPreview)")
            }

            // Sample using C reference algorithm
            return sampleTokenCReference(from: probs, sampler: sampler, sum: sum, argmax: argmax)
        }
    }
    
    /// C reference-compatible sampling
    private func sampleTokenCReference(from probBuffer: MTLBuffer, sampler: GptossSampler, sum sumBuffer: MTLBuffer, argmax argmaxBuffer: MTLBuffer) -> UInt32 {
        let config = model.config
        
        // Calculate sampling parameters like C reference
        let (numVecsPerThreadgroup, actualThreadgroups) = CReferenceSampler.calculateSamplingParams(
            vocabularySize: config.vocabularySize,
            maxThreadgroups: UInt32(model.maxThreadgroups)
        )
        
        return CReferenceSampler.sampleToken(
            probBuffer: probBuffer,
            sumBuffer: sumBuffer,
            argmaxBuffer: argmaxBuffer,
            numTokens: UInt32(context.numTokens),
            vocabularySize: config.vocabularySize,
            numThreadgroups: actualThreadgroups,
            numVecsPerThreadgroup: numVecsPerThreadgroup,
            temperature: sampler.temperature,
            seed: 0  // Use deterministic seed for now
        )
    }
    
    private func sampleToken(from probBuffer: MTLBuffer, sampler: GptossSampler) -> UInt32 {
        let vocabSize = Int(model.config.vocabularySize)
        let probPointer = probBuffer.contents().bindMemory(to: Float32.self, capacity: vocabSize)
        
        // Convert to array for easier manipulation
        var probs = Array(UnsafeBufferPointer(start: probPointer, count: vocabSize))
        
        // Check for NaN values and handle them
        let hasNaN = probs.contains { $0.isNaN }
        if hasNaN {
            print("⚠️ Warning: Softmax produced NaN probabilities. Falling back to argmax from scores.")
            // If we have NaN probabilities, use the scores buffer directly
            if let scoreBuffer = context.scoreBuffer {
                let scorePointer = scoreBuffer.contents().bindMemory(to: Float32.self, capacity: vocabSize)
                let scores = Array(UnsafeBufferPointer(start: scorePointer, count: vocabSize))
                // Find argmax of scores
                let maxIndex = scores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
                return UInt32(maxIndex)
            }
            return 0 // Ultimate fallback
        }
        
        // The probs buffer contains unnormalized exp(x - max) values from softmax kernel
        // We need to normalize them by dividing by the sum
        if let sumBuffer = context.sumBuffer {
            // Calculate actual number of threadgroups used (same calculation as in softmax dispatch)
            let threadgroupSize: UInt32 = 256
            let numVecs = model.config.vocabularySize
            let maxTg = UInt32(model.maxThreadgroups)
            let numVecsPerThreadgroup = ((numVecs + (maxTg * threadgroupSize) - 1) / (maxTg * threadgroupSize)) * threadgroupSize
            let actualThreadgroups = min(Int(maxTg), Int((numVecs + numVecsPerThreadgroup - 1) / numVecsPerThreadgroup))
            
            let sumPointer = sumBuffer.contents().bindMemory(to: Float32.self, capacity: actualThreadgroups)
            var totalSum: Float32 = 0
            for i in 0..<actualThreadgroups {
                totalSum += sumPointer[i]
            }
            
            // Normalize probabilities
            if totalSum > 0 {
                for i in 0..<probs.count {
                    probs[i] /= totalSum
                }
            } else {
                print("⚠️ WARNING: Softmax sum is 0 or negative!")
            }
        }
        
        var logits = probs // Continue with the normalized probs
        
        // Apply frequency and presence penalties BEFORE temperature scaling
        applyFrequencyPenalties(to: &logits, sampler: sampler)
        
        // Apply temperature scaling
        if sampler.temperature > 0 && sampler.temperature != 1.0 {
            for i in 0..<logits.count {
                logits[i] = logits[i] / sampler.temperature
            }
        }
        
        // Apply softmax to convert logits to probabilities
        applySoftmax(to: &logits)
        
        // Apply top-k filtering if specified
        if sampler.topK > 0 {
            applyTopKSampling(to: &logits, k: sampler.topK)
        }
        
        // Apply top-p (nucleus) sampling if specified
        if sampler.topP < 1.0 {
            applyTopPSampling(to: &logits, p: sampler.topP)
        }
        
        // Handle special case: temperature = 0 means deterministic (argmax)
        if sampler.temperature == 0.0 {
            // Read the argmax directly from the argmax buffer (computed by unembedding kernel)
            if let argmaxBuffer = context.argmaxBuffer {
                // CRITICAL FIX: Read argmax as uint2 {token_id, score_bits}
                // Metal kernel writes this format, not UInt64
                let argmaxPtr = argmaxBuffer.contents().bindMemory(to: UInt32.self, capacity: 2)
                let tokenId = argmaxPtr[0]  // First 32 bits = token ID
                // let scoreBits = argmaxPtr[1]  // Second 32 bits = score bits (unused for now)
                return tokenId
            }
            // Fallback if argmax buffer not available
            return UInt32(logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
        }
        
        // Multinomial sampling from probability distribution
        return multinomialSample(from: logits)
    }
    
    private func applyFrequencyPenalties(to logits: inout [Float32], sampler: GptossSampler) {
        // Apply frequency penalty (reduces probability of tokens based on how often they appear)
        if sampler.frequencyPenalty != 0.0 {
            for (token, frequency) in tokenFrequencies {
                if Int(token) < logits.count {
                    logits[Int(token)] -= sampler.frequencyPenalty * Float32(frequency)
                }
            }
        }
        
        // Apply presence penalty (reduces probability of tokens that have already appeared)  
        if sampler.presencePenalty != 0.0 {
            for token in usedTokens {
                if Int(token) < logits.count {
                    logits[Int(token)] -= sampler.presencePenalty
                }
            }
        }
    }
    
    private func applyTopKSampling(to probs: inout [Float32], k: Int) {
        // Find the k-th largest probability
        let sortedProbs = probs.sorted(by: >)
        let threshold = sortedProbs[min(k - 1, sortedProbs.count - 1)]
        
        // Zero out probabilities below the k-th largest
        var keptProbs: Float32 = 0
        for i in 0..<probs.count {
            if probs[i] < threshold {
                probs[i] = 0
            } else {
                keptProbs += probs[i]
            }
        }
        
        // Renormalize remaining probabilities
        if keptProbs > 0 {
            for i in 0..<probs.count {
                if probs[i] > 0 {
                    probs[i] = probs[i] / keptProbs
                }
            }
        }
    }
    
    private func applySoftmax(to probs: inout [Float32]) {
        // Find max for numerical stability
        let maxLogit = probs.max() ?? 0
        
        // Subtract max and exponentiate
        for i in 0..<probs.count {
            probs[i] = exp(probs[i] - maxLogit)
        }
        
        // Normalize
        let sum = probs.reduce(0, +)
        if sum > 0 {
            for i in 0..<probs.count {
                probs[i] = probs[i] / sum
            }
        }
    }
    
    private func applyTopPSampling(to probs: inout [Float32], p: Float32) {
        // Create indexed probability pairs and sort by probability descending
        let indexedProbs = probs.enumerated().sorted { $0.element > $1.element }
        
        // Calculate cumulative probabilities and find cutoff
        var cumulative: Float32 = 0
        var cutoffIndex = probs.count
        
        for (i, (_, prob)) in indexedProbs.enumerated() {
            cumulative += prob
            if cumulative >= p {
                cutoffIndex = i + 1
                break
            }
        }
        
        // Zero out probabilities below cutoff
        var newProbs = [Float32](repeating: 0, count: probs.count)
        var newSum: Float32 = 0
        
        for i in 0..<cutoffIndex {
            let (originalIndex, prob) = indexedProbs[i]
            newProbs[originalIndex] = prob
            newSum += prob
        }
        
        // Renormalize remaining probabilities
        if newSum > 0 {
            for i in 0..<newProbs.count {
                newProbs[i] = newProbs[i] / newSum
            }
        }
        
        probs = newProbs
    }
    
    private func multinomialSample(from probs: [Float32]) -> UInt32 {
        let random = Float32.random(in: 0..<1)
        var cumulative: Float32 = 0
        
        for (i, prob) in probs.enumerated() {
            cumulative += prob
            if random <= cumulative {
                return UInt32(i)
            }
        }
        
        // Fallback to last non-zero probability
        for (i, prob) in probs.enumerated().reversed() {
            if prob > 0 {
                return UInt32(i)
            }
        }
        
        return 0 // Ultimate fallback
    }
}

// MARK: - Batch Processing

public extension GenerationPipeline {
    func processBatchedPrompts(
        prompts: [[UInt32]],
        maxTokens: Int,
        sampler: GptossSampler = GptossSampler()
    ) throws -> [[UInt32]] {
        var results: [[UInt32]] = []
        
        for prompt in prompts {
            let generated = try generateTokens(
                prompt: prompt,
                maxTokens: maxTokens,
                sampler: sampler
            )
            results.append(generated)
        }
        
        return results
    }
}
