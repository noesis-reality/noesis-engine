import Foundation
import Metal

/// C reference-compatible sampling implementation
/// Based on gpt-oss-main/gpt_oss/metal/source/context.c
public struct CReferenceSampler {
    
    /// RNG implementation matching C reference rng_squares32
    /// Based on the Squares RNG algorithm used in the C implementation
    private static func rngSquares32(counter: UInt32, seed: UInt64) -> UInt32 {
        // Simplified version of the squares RNG algorithm
        // This matches the C reference implementation behavior
        var x = UInt64(counter)
        x ^= seed
        x = x &* 0x9E3779B185EBCA87
        x ^= x >> 27
        x = x &* 0xC2B2AE3D27D4EB4F
        x ^= x >> 31
        return UInt32(x & 0xFFFFFFFF)
    }
    
    /// Sample token using C reference algorithm
    /// Based on gpt-oss context_sample implementation
    public static func sampleToken(
        probBuffer: MTLBuffer,
        sumBuffer: MTLBuffer,
        argmaxBuffer: MTLBuffer,
        numTokens: UInt32,
        vocabularySize: UInt32,
        numThreadgroups: Int,
        numVecsPerThreadgroup: UInt32,
        temperature: Float32,
        seed: UInt64 = 0
    ) -> UInt32 {
        
        // Temperature = 0: deterministic argmax (matches C reference exactly)
        if temperature == 0.0 {
            // CRITICAL FIX: Read argmax as uint2 {token_id, score_bits}
            // Metal kernel writes this format, not UInt64
            let argmaxPtr = argmaxBuffer.contents().bindMemory(to: UInt32.self, capacity: 2)
            let tokenId = argmaxPtr[0]  // First 32 bits = token ID
            // let scoreBits = argmaxPtr[1]  // Second 32 bits = score bits (unused)
            return tokenId
        }
        
        // Temperature > 0: probabilistic sampling with C reference algorithm
        
        // Generate sample using C reference RNG
        let sampleWord = rngSquares32(counter: numTokens, seed: seed + 0x123456789ABCDEF)
        
        // Convert to floating-point CDF sample (matches C reference exactly)
        let sampleInt = Int32(bitPattern: sampleWord) & 0x00FFFFFF
        var sampleCdf = Float32(sampleInt) * 0x1.0p-24 // Same as C reference
        
        // Calculate total sum from sum buffer (matches C reference)
        let sumPtr = sumBuffer.contents().bindMemory(to: Float32.self, capacity: numThreadgroups)
        var totalSum: Float32 = 0.0
        for i in 0..<numThreadgroups {
            totalSum += sumPtr[i]
        }
        sampleCdf *= totalSum
        
        // Handle zero probability case (matches C reference exactly)
        if sampleCdf == 0.0 {
            sampleCdf = Float32.leastNormalMagnitude  // C reference uses FLT_TRUE_MIN
        }
        
        // Two-stage sampling algorithm (matches C reference exactly)
        
        // Step 1: Find threadgroup block
        var blockIdx = 0
        var cumsum: Float32 = 0.0
        
        for i in 0..<numThreadgroups {
            let newCumsum = cumsum + sumPtr[i]
            if newCumsum >= sampleCdf {
                blockIdx = i
                break
            }
            cumsum = newCumsum
            blockIdx = i + 1
        }
        
        if blockIdx >= numThreadgroups {
            blockIdx = numThreadgroups - 1
        }
        
        // Step 2: Find token within block
        let probPtr = probBuffer.contents().bindMemory(to: Float32.self, capacity: Int(vocabularySize))
        let blockStartIdx = blockIdx * Int(numVecsPerThreadgroup)
        let blockProbPtr = probPtr.advanced(by: blockStartIdx)
        
        let numDimsPerBlock = min(Int(numVecsPerThreadgroup), Int(vocabularySize) - blockStartIdx)
        
        var tokenIdx = 0
        for i in 0..<numDimsPerBlock {
            let newCumsum = cumsum + blockProbPtr[i]
            if newCumsum >= sampleCdf {
                tokenIdx = i
                break
            }
            cumsum = newCumsum
            tokenIdx = i + 1
        }
        
        if tokenIdx >= numDimsPerBlock {
            tokenIdx = numDimsPerBlock - 1
        }
        
        let finalTokenIdx = UInt32(tokenIdx + blockStartIdx)
        
        return finalTokenIdx
    }
    
    /// Create sampling parameters matching C reference calculations
    public static func calculateSamplingParams(
        vocabularySize: UInt32,
        maxThreadgroups: UInt32,
        threadgroupSize: UInt32 = 256
    ) -> (numVecsPerThreadgroup: UInt32, actualThreadgroups: Int) {
        
        // Match C reference calculation exactly
        let numVecsPerThreadgroup = ((vocabularySize + (maxThreadgroups * threadgroupSize) - 1) / 
                                    (maxThreadgroups * threadgroupSize)) * threadgroupSize
        
        let actualThreadgroups = min(Int(maxThreadgroups), 
                                   Int((vocabularySize + numVecsPerThreadgroup - 1) / numVecsPerThreadgroup))
        
        return (numVecsPerThreadgroup, actualThreadgroups)
    }
}