import Foundation

/// MXFP4 (Microscaling FP4) quantization support for MoE weights
public struct MXFP4 {
    
    /// FP4 value lookup table
    static let fp4Values: [Float] = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ]
    
    /// UE8 offset bias for scales
    static let ue8Offset: UInt8 = 14
    
    /// Bytes per MXFP4 block (32 FP4 numbers packed in 16 bytes)
    static let bytesPerBlock = 16
    
    /// Pack float weights into MXFP4 format
    public static func quantize(weights: [Float], blockSize: Int = 32) -> (blocks: Data, scales: Data) {
        var blocks = Data()
        var scales = Data()
        
        // Process weights in blocks
        for blockStart in stride(from: 0, to: weights.count, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, weights.count)
            let block = Array(weights[blockStart..<blockEnd])
            
            // Find scale for this block
            let maxAbs = block.map { abs($0) }.max() ?? 0
            let scale = findBestScale(for: maxAbs)
            
            // Quantize block values
            var packedBytes = Data()
            for i in stride(from: 0, to: block.count, by: 2) {
                let val1 = quantizeValue(block[i], scale: scale)
                let val2 = i + 1 < block.count ? quantizeValue(block[i + 1], scale: scale) : 0
                
                // Pack two 4-bit values into one byte
                let packed = (val2 << 4) | (val1 & 0x0F)
                packedBytes.append(packed)
            }
            
            blocks.append(packedBytes)
            
            // Store scale with UE8 offset
            let scaleWithOffset = UInt8(min(254, max(0, scale + Int(ue8Offset))))
            scales.append(scaleWithOffset)
        }
        
        return (blocks, scales)
    }
    
    /// Dequantize MXFP4 format back to floats
    public static func dequantize(blocks: Data, scales: Data, outputCount: Int) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(outputCount)
        
        var blockIndex = 0
        var scaleIndex = 0
        
        while result.count < outputCount && blockIndex < blocks.count {
            guard scaleIndex < scales.count else { break }
            
            // Get scale for this block (subtract UE8 offset)
            let scaleWithOffset = scales[scaleIndex]
            let scale = Int(scaleWithOffset) - Int(ue8Offset)
            let scaleFactor = pow(2.0, Float(scale))
            
            // Process packed bytes in this block
            let blockEnd = min(blockIndex + bytesPerBlock, blocks.count)
            for byteIdx in blockIndex..<blockEnd {
                let packed = blocks[byteIdx]
                
                // Extract low and high nibbles
                let lowNibble = packed & 0x0F
                let highNibble = (packed >> 4) & 0x0F
                
                // Lookup FP4 values and apply scale
                let val1 = fp4Values[Int(lowNibble)] * scaleFactor
                let val2 = fp4Values[Int(highNibble)] * scaleFactor
                
                result.append(val1)
                if result.count < outputCount {
                    result.append(val2)
                }
            }
            
            blockIndex += bytesPerBlock
            scaleIndex += 1
        }
        
        return result
    }
    
    /// Convert MXFP4 to BFloat16 for Metal
    public static func dequantizeToBFloat16(blocks: Data, scales: Data, outputCount: Int) -> Data {
        let floats = dequantize(blocks: blocks, scales: scales, outputCount: outputCount)
        
        var bf16Data = Data(capacity: floats.count * 2)
        for float in floats {
            let bf16 = float.bfloat16
            bf16Data.append(bf16.littleEndianData)
        }
        
        return bf16Data
    }
    
    // MARK: - Private Helpers
    
    private static func findBestScale(for maxValue: Float) -> Int {
        // Find appropriate power of 2 scale
        guard maxValue > 0 else { return -127 }
        
        let log2Value = log2(maxValue / 6.0) // 6.0 is max FP4 value
        return Int(ceil(log2Value))
    }
    
    private static func quantizeValue(_ value: Float, scale: Int) -> UInt8 {
        let scaleFactor = pow(2.0, Float(scale))
        let scaled = value / scaleFactor
        
        // Find closest FP4 value
        var bestIdx = 0
        var bestDiff = Float.infinity
        
        for (idx, fp4Val) in fp4Values.enumerated() {
            let diff = abs(scaled - fp4Val)
            if diff < bestDiff {
                bestDiff = diff
                bestIdx = idx
            }
        }
        
        return UInt8(bestIdx)
    }
}

