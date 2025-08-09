import Foundation
import Metal

// MARK: - C Reference Sampler

/// Implements sampling algorithm matching the C reference implementation
public struct CReferenceSampler {
    
    /// Calculate sampling parameters matching C reference
    public static func calculateSamplingParams(
        vocabularySize: UInt32,
        maxThreadgroups: UInt32
    ) -> (numVecsPerThreadgroup: UInt32, actualThreadgroups: UInt32) {
        let threadgroupSize: UInt32 = 256
        
        // Match C reference: math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size
        let numVecsPerThreadgroup = ((vocabularySize + (maxThreadgroups * threadgroupSize) - 1) / 
                                     (maxThreadgroups * threadgroupSize)) * threadgroupSize
        
        // Calculate actual threadgroups needed
        let actualThreadgroups = min(maxThreadgroups, 
                                     (vocabularySize + numVecsPerThreadgroup - 1) / numVecsPerThreadgroup)
        
        return (numVecsPerThreadgroup, actualThreadgroups)
    }
    
    /// Sample token from probability buffer using C reference algorithm
    public static func sampleToken(
        probBuffer: MTLBuffer,
        sumBuffer: MTLBuffer,
        argmaxBuffer: MTLBuffer,
        numTokens: UInt32,
        vocabularySize: UInt32,
        numThreadgroups: UInt32,
        numVecsPerThreadgroup: UInt32,
        temperature: Float,
        seed: UInt64
    ) -> UInt32 {
        
        if temperature == 0.0 {
            // Temperature 0: return argmax directly
            // The argmax buffer contains uint2 {token_id, score_bits}
            let argmaxPtr = argmaxBuffer.contents().bindMemory(to: UInt32.self, capacity: 2)
            return argmaxPtr[0]  // First 32 bits = token ID
        }
        
        // Temperature > 0: probabilistic sampling
        // First, get the sum from all threadgroups
        let sumPtr = sumBuffer.contents().bindMemory(to: Float32.self, capacity: Int(numThreadgroups))
        var totalSum: Float32 = 0
        for i in 0..<Int(numThreadgroups) {
            totalSum += sumPtr[i]
        }
        
        // Get probabilities and normalize
        let vocabSize = Int(vocabularySize)
        let probPtr = probBuffer.contents().bindMemory(to: Float32.self, capacity: vocabSize)
        
        // Generate random value for sampling
        let random = generateRandom(seed: seed, tokenIndex: numTokens)
        let threshold = random * totalSum
        
        // Cumulative sampling
        var cumulative: Float32 = 0
        for i in 0..<vocabSize {
            cumulative += probPtr[i]
            if cumulative >= threshold {
                return UInt32(i)
            }
        }
        
        // Fallback to last token
        return vocabularySize - 1
    }
    
    /// Generate deterministic random number based on seed and token index
    private static func generateRandom(seed: UInt64, tokenIndex: UInt32) -> Float32 {
        // Simple PRNG matching C reference behavior
        var state = seed &+ UInt64(tokenIndex)
        
        // XorShift64
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        state &*= 0x2545F4914F6CDD1D
        
        // Convert to float in [0, 1)
        return Float32(state >> 32) / Float32(UInt32.max)
    }
}

// Note: GptossSampler is defined in ModelStructures.swift

// MARK: - Sampling Utilities

extension GptossSampler {
    
    /// Apply top-k filtering to logits
    public func applyTopK(to logits: inout [Float32], k: Int) {
        guard k > 0 && k < logits.count else { return }
        
        // Find k-th largest value
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        let threshold = sorted[min(k - 1, sorted.count - 1)].element
        
        // Zero out values below threshold
        for i in 0..<logits.count {
            if logits[i] < threshold {
                logits[i] = -Float.infinity
            }
        }
    }
    
    /// Apply nucleus (top-p) sampling
    public func applyTopP(to logits: inout [Float32], p: Float32) {
        guard p < 1.0 else { return }
        
        // Sort by probability descending
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        
        // Apply softmax to get probabilities
        let maxLogit = logits.max() ?? 0
        var probs = logits.map { exp($0 - maxLogit) }
        let sum = probs.reduce(0, +)
        probs = probs.map { $0 / sum }
        
        // Find cumulative cutoff
        var cumulative: Float32 = 0
        var cutoffIndex = probs.count
        
        for (i, idx) in sorted.enumerated() {
            cumulative += probs[idx.offset]
            if cumulative >= p {
                cutoffIndex = i + 1
                break
            }
        }
        
        // Zero out probabilities outside nucleus
        for i in cutoffIndex..<sorted.count {
            logits[sorted[i].offset] = -Float.infinity
        }
    }
    
    /// Apply repetition penalties
    public func applyPenalties(
        to logits: inout [Float32],
        usedTokens: Set<UInt32>,
        tokenFrequencies: [UInt32: Int]
    ) {
        // Apply frequency penalty
        if frequencyPenalty != 0 {
            for (token, count) in tokenFrequencies {
                if Int(token) < logits.count {
                    logits[Int(token)] -= frequencyPenalty * Float32(count)
                }
            }
        }
        
        // Apply presence penalty
        if presencePenalty != 0 {
            for token in usedTokens {
                if Int(token) < logits.count {
                    logits[Int(token)] -= presencePenalty
                }
            }
        }
    }
}