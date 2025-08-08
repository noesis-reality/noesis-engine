import Foundation
import Metal

// MARK: - GPT-OSS File Format Structures

public struct GptossFileHeader {
    public static let validMagic: [UInt8] = [
        0x47, 0x50, 0x54, 0x2D,  // "GPT-"
        0x4F, 0x53, 0x53, 0x20,  // "OSS "
        0x76, 0x31, 0x2E, 0x30   // "v1.0"
    ]

    public let magic: [UInt8] // 12 bytes
    public let zero: UInt32

    public init(magic: [UInt8], zero: UInt32 = 0) {
        self.magic = magic
        self.zero = zero
    }

    public var isValid: Bool {
        return magic == Self.validMagic && zero == 0
    }

    public static let size = 16  // 12 bytes magic + 4 bytes zero
}

public struct GptossModelHeader {
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

    public static let size = 17 * 4  // 17 32-bit values
}

public struct GptossTiktokenTokenizerHeader {
    public let numSpecialTokens: UInt32
    public let numTextTokens: UInt32
    public let regexSize: UInt32
    public let tokensSize: UInt32

    public static let size = 4 * 4  // 4 32-bit values
}

// MARK: - UUID Support

public struct GptossUUID: Sendable, Hashable {
    public let bytes: [UInt8] // 16 bytes

    public init(_ bytes: [UInt8]) {
        assert(bytes.count == 16)
        self.bytes = bytes
    }

    // UUIDs matching C reference implementation exactly
    public static let gptossModelUUID = GptossUUID([
        0xDF, 0x52, 0xDC, 0x86, 0x17, 0x89, 0x4E, 0xD0,
        0xA2, 0x95, 0x66, 0xF1, 0x05, 0x08, 0x14, 0x5B
    ])

    public static let appleGPULayoutUUID = GptossUUID([
        0x22, 0x91, 0x77, 0xA8, 0x57, 0x75, 0x42, 0x68,
        0xBF, 0xD8, 0xD5, 0x88, 0xB3, 0x51, 0xC5, 0x6D
    ])

    public static let tiktokenTokenizerUUID = GptossUUID([
        0x74, 0x01, 0xAD, 0xED, 0x2A, 0x95, 0x40, 0xCB,
        0xB7, 0x82, 0x9C, 0xCE, 0xBA, 0xAF, 0xE7, 0x2B
    ])

    public func equals(_ other: GptossUUID) -> Bool {
        return bytes == other.bytes
    }
}

// MARK: - Weight Layout Calculations

public struct WeightLayout {
    public let embeddingWeightSize: Int
    public let rmsnormWeightSize: Int
    public let attnQkvWeightSize: Int
    public let attnQkvBiasSize: Int
    public let attnSinkWeightSize: Int
    public let attnOutWeightSize: Int
    public let attnOutBiasSize: Int
    public let mlpGateWeightSize: Int
    public let mlpGateBiasSize: Int
    public let unembeddingWeightSize: Int
    public let perBlockSharedWeightsSize: Int
    public let sharedWeightsSize: Int

    // Offsets within shared weights buffer
    public let attnRmsnormGainOffset: Int
    public let attnQkvWeightOffset: Int
    public let attnQkvBiasOffset: Int
    public let attnSdpaSinkOffset: Int
    public let attnOutWeightOffset: Int
    public let attnOutBiasOffset: Int
    public let mlpRmsnormGainOffset: Int
    public let mlpGateWeightOffset: Int
    public let mlpGateBiasOffset: Int
    public let rmsnormWeightOffset: Int
    public let unembeddingWeightOffset: Int

    // MoE per-expert weights (matching C reference exactly)
    public let mlpSwigluWeightBlockSize: Int
    public let mlpSwigluWeightScaleSize: Int  // CRITICAL: Was missing!
    public let mlpSwigluBiasSize: Int
    public let mlpOutWeightBlockSize: Int
    public let mlpOutWeightScaleSize: Int      // CRITICAL: Was missing!
    public let mlpOutBiasSize: Int
    public let perExpertBlockWeightSize: Int

    public init(config: ModelConfig) {
        // Helper function to round up to 16-byte alignment
        func roundUp16(_ size: Int) -> Int {
            return (size + 15) & ~15
        }

        // Helper function to round up to page size
        func roundUpToPageSize(_ size: Int) -> Int {
            let pageSize = 16384  // 16KB pages on Apple Silicon
            return (size + pageSize - 1) & ~(pageSize - 1)
        }

        // Calculate sizes
        embeddingWeightSize = roundUp16(Int(config.vocabularySize * config.embeddingDim) * 2) // bfloat16 = 2 bytes
        rmsnormWeightSize = roundUp16(Int(config.embeddingDim) * 2)

        let attnQkvDim = Int(config.headDim * (config.numHeads + 2 * config.numKvHeads))
        attnQkvWeightSize = roundUp16(attnQkvDim * Int(config.embeddingDim) * 2)
        attnQkvBiasSize = roundUp16(attnQkvDim * 2)
        attnSinkWeightSize = roundUp16(Int(config.numHeads) * 2)
        attnOutWeightSize = roundUp16(Int(config.embeddingDim * config.numHeads * config.headDim) * 2)
        attnOutBiasSize = roundUp16(Int(config.embeddingDim) * 2)
        mlpGateWeightSize = roundUp16(Int(config.numExperts * config.embeddingDim) * 2)
        mlpGateBiasSize = roundUp16(Int(config.numExperts) * 2)
        unembeddingWeightSize = roundUp16(Int(config.vocabularySize * config.embeddingDim) * 2)

        // Calculate per-block shared weights size
        perBlockSharedWeightsSize = rmsnormWeightSize + attnQkvWeightSize + attnQkvBiasSize +
                                   attnSinkWeightSize + attnOutWeightSize + attnOutBiasSize +
                                   rmsnormWeightSize + mlpGateWeightSize + mlpGateBiasSize

        // Calculate shared weights total size
        sharedWeightsSize = roundUpToPageSize(embeddingWeightSize + rmsnormWeightSize +
                                            unembeddingWeightSize + Int(config.numBlocks) * perBlockSharedWeightsSize)

        // Calculate offsets
        attnRmsnormGainOffset = embeddingWeightSize
        attnQkvWeightOffset = attnRmsnormGainOffset + rmsnormWeightSize
        attnQkvBiasOffset = attnQkvWeightOffset + attnQkvWeightSize
        attnSdpaSinkOffset = attnQkvBiasOffset + attnQkvBiasSize
        attnOutWeightOffset = attnSdpaSinkOffset + attnSinkWeightSize
        attnOutBiasOffset = attnOutWeightOffset + attnOutWeightSize
        mlpRmsnormGainOffset = attnOutBiasOffset + attnOutBiasSize
        mlpGateWeightOffset = mlpRmsnormGainOffset + rmsnormWeightSize
        mlpGateBiasOffset = mlpGateWeightOffset + mlpGateWeightSize
        rmsnormWeightOffset = embeddingWeightSize + Int(config.numBlocks) * perBlockSharedWeightsSize
        unembeddingWeightOffset = rmsnormWeightOffset + rmsnormWeightSize

        // MoE per-expert weight calculations (matching C reference model.c:399-411)
        // C: mlp_swiglu_weight_block_size = math_round_up_po2(2 * model->mlp_dim * model->embedding_dim / 2, 16)
        mlpSwigluWeightBlockSize = roundUp16(2 * Int(config.mlpDim * config.embeddingDim) / 2)
        
        // C: mlp_swiglu_weight_scale_size = math_round_up_po2(2 * model->mlp_dim * model->embedding_dim / 32, 16)
        mlpSwigluWeightScaleSize = roundUp16(2 * Int(config.mlpDim * config.embeddingDim) / 32)
        
        // C: mlp_swiglu_bias_size = math_round_up_po2(2 * model->mlp_dim * sizeof(gptoss_bfloat16), 16)
        mlpSwigluBiasSize = roundUp16(2 * Int(config.mlpDim) * 2) // bfloat16 = 2 bytes
        
        // C: mlp_out_weight_block_size = math_round_up_po2(model->embedding_dim * model->mlp_dim / 2, 16)
        mlpOutWeightBlockSize = roundUp16(Int(config.embeddingDim * config.mlpDim) / 2)
        
        // C: mlp_out_weight_scale_size = math_round_up_po2(model->embedding_dim * model->mlp_dim / 32, 16)
        mlpOutWeightScaleSize = roundUp16(Int(config.embeddingDim * config.mlpDim) / 32)
        
        // C: mlp_out_bias_size = math_round_up_po2(model->embedding_dim * sizeof(gptoss_bfloat16), 16)
        mlpOutBiasSize = roundUp16(Int(config.embeddingDim) * 2) // bfloat16 = 2 bytes

        // C: per_expert_block_weight_size = all six components added
        perExpertBlockWeightSize = mlpSwigluWeightBlockSize + mlpSwigluWeightScaleSize + mlpSwigluBiasSize +
                                 mlpOutWeightBlockSize + mlpOutWeightScaleSize + mlpOutBiasSize
    }
}
