import Foundation
import Metal

/// Exports checkpoint files to GPT-OSS Metal format
public class ModelExporter {
    
    public struct ExportConfig {
        public let checkpointDir: URL
        public let outputPath: URL
        public let applyQKScaling: Bool
        
        public init(checkpointDir: URL, outputPath: URL, applyQKScaling: Bool = true) {
            self.checkpointDir = checkpointDir
            self.outputPath = outputPath
            self.applyQKScaling = applyQKScaling
        }
    }
    
    public init() {}
    
    /// Export checkpoint to GPT-OSS Metal format
    public func export(config: ExportConfig) throws {
        print("Exporting model from \(config.checkpointDir.path)...")
        
        // Load config.json
        let configPath = config.checkpointDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        var modelConfig = try JSONDecoder().decode(CheckpointConfig.self, from: configData)
        modelConfig.checkpointDir = config.checkpointDir
        
        print("Model configuration:")
        print("  Blocks: \(modelConfig.num_hidden_layers)")
        print("  Embedding dim: \(modelConfig.hidden_size)")
        print("  Experts: \(modelConfig.num_experts)")
        print("  Heads: \(modelConfig.num_attention_heads)")
        
        // Load safetensors files
        let safetensors = try loadSafetensors(from: config.checkpointDir)
        
        // Create output file
        let output = try FileHandle(forWritingTo: config.outputPath)
        defer { try? output.close() }
        
        // Write file header
        try writeFileHeader(to: output)
        
        // Write model header
        try writeModelHeader(config: modelConfig, to: output)
        
        // Write tokenizer
        try writeTokenizer(to: output)
        
        // Align to page boundary
        try alignToPage(output)
        
        // Write weights
        try writeWeights(
            safetensors: safetensors,
            config: modelConfig,
            to: output,
            applyQKScaling: config.applyQKScaling
        )
        
        print("✅ Model exported to \(config.outputPath.path)")
    }
    
    // MARK: - File Format
    
    private func writeFileHeader(to output: FileHandle) throws {
        // GPT-OSS v1.0 magic
        let magic: [UInt8] = [
            0x47, 0x50, 0x54, 0x2D, // GPT-
            0x4F, 0x53, 0x53, 0x20, // OSS 
            0x76, 0x31, 0x2E, 0x30, // v1.0
            0x00, 0x00, 0x00, 0x00  // padding
        ]
        output.write(Data(magic))
    }
    
    private func writeModelHeader(config: CheckpointConfig, to output: FileHandle) throws {
        // Model UUID
        let modelUUID = UUID(uuidString: "df52dc86-1789-4ed0-a295-66f10508145b")!
        output.write(modelUUID.data)
        
        // Calculate YARN parameters
        let yarnLow = Double(config.head_dim) / 2 * 
            log(Double(config.initial_context_length) / (config.rope_ntk_beta * 2 * .pi)) /
            log(config.rope_theta)
        let yarnHigh = Double(config.head_dim) / 2 *
            log(Double(config.initial_context_length) / (config.rope_ntk_alpha * 2 * .pi)) /
            log(config.rope_theta)
        
        // Write config
        var data = Data()
        data.append(UInt32(config.initial_context_length * config.rope_scaling_factor))  // context_length
        data.append(UInt32(config.num_hidden_layers))  // num_blocks
        data.append(UInt32(config.num_experts))  // num_experts
        data.append(UInt32(4))  // num_active_experts (hardcoded for now)
        data.append(UInt32(config.hidden_size))  // embedding_dim
        data.append(UInt32(config.intermediate_size))  // mlp_dim
        data.append(Float32(config.swiglu_limit ?? 7.0))  // swiglu_limit
        data.append(UInt32(config.head_dim))  // head_dim
        data.append(UInt32(config.num_attention_heads))  // num_heads
        data.append(UInt32(config.num_key_value_heads))  // num_kv_heads
        data.append(UInt32(config.sliding_window))  // attention_window
        data.append(Float32(config.rope_theta))  // rope_theta
        data.append(Float32(1.0 / Double(config.rope_scaling_factor)))  // interpolation_scale
        data.append(Float32(-yarnLow / (yarnHigh - yarnLow)))  // yarn_offset
        data.append(Float32(1.0 / (yarnHigh - yarnLow)))  // yarn_scale
        data.append(Float32(0.1 * log(Double(config.rope_scaling_factor)) + 1.0))  // yarn_multiplier
        data.append(Float32(1e-5))  // rmsnorm_epsilon
        
        output.write(data)
        
        // Layout UUID
        let layoutUUID = UUID(uuidString: "229177a8-5775-4268-bfd8-d588b351c56d")!
        output.write(layoutUUID.data)
    }
    
    private func writeTokenizer(to output: FileHandle) throws {
        // Tokenizer UUID
        let tokenizerUUID = UUID(uuidString: "7401aded-2a95-40cb-b782-9ccebaafe72b")!
        output.write(tokenizerUUID.data)
        
        // Load actual tokenizer data from exported files
        let (regexData, tokenData, specialUUIDs) = try TokenizerDataLoader.loadDefaultTokenizerData()
        
        var data = Data()
        data.append(UInt32(16))  // num_special_tokens
        data.append(UInt32(199998))  // num_text_tokens
        data.append(UInt32(regexData.count + 1))  // regex_size (including null terminator)
        data.append(UInt32(tokenData.count))  // tokens_size
        
        output.write(data)
        
        // Write special token UUIDs
        for uuid in specialUUIDs {
            output.write(uuid)
        }
        
        // Write regex pattern
        output.write(regexData)
        output.write(Data([0]))  // null terminator
        
        // Write token data
        output.write(tokenData)
    }
    
    private func alignToPage(_ output: FileHandle) throws {
        let pageSize = 16384  // 16KB pages on Apple Silicon
        let currentOffset = try output.offset()
        let alignedOffset = ((currentOffset + UInt64(pageSize) - 1) / UInt64(pageSize)) * UInt64(pageSize)
        let padding = Int(alignedOffset - currentOffset)
        if padding > 0 {
            output.write(Data(repeating: 0, count: padding))
        }
    }
    
    private func writeWeights(
        safetensors: [String: Data],
        config: CheckpointConfig,
        to output: FileHandle,
        applyQKScaling: Bool
    ) throws {
        // Load safetensors file
        let safetensorsPath = config.checkpointDir.appendingPathComponent("model.safetensors")
        let loader = SafetensorsLoader(fileURL: safetensorsPath)
        try loader.open()
        defer { loader.close() }
        
        print("Writing model weights...")
        
        // Write embedding weights
        print("  - Embedding weights")
        let embeddingData = try loader.loadTensorData(name: "embedding.weight")
        // Truncate to actual vocabulary size (num_special + num_text tokens)
        let vocabSize = 200014  // Standard o200k_gptoss vocabulary
        let embeddingSize = vocabSize * config.hidden_size * 2  // BFloat16
        output.write(embeddingData.prefix(embeddingSize))
        writePadding(to: output, alignment: 16)
        
        // Process each transformer block
        for blockIdx in 0..<config.num_hidden_layers {
            print("  - Block \(blockIdx + 1)/\(config.num_hidden_layers)")
            
            // Attention RMSNorm
            let attnNormData = try loader.loadTensorData(name: "block.\(blockIdx).attn.norm.scale")
            output.write(attnNormData)
            writePadding(to: output, alignment: 16)
            
            // QKV weights with Q/K scaling
            if applyQKScaling {
                let qkvWeightData = try processQKVWeights(
                    loader: loader,
                    blockIdx: blockIdx,
                    config: config
                )
                output.write(qkvWeightData)
            } else {
                let qkvWeightData = try loader.loadTensorData(name: "block.\(blockIdx).attn.qkv.weight")
                output.write(qkvWeightData)
            }
            writePadding(to: output, alignment: 16)
            
            // QKV bias
            let qkvBiasData = try loader.loadTensorData(name: "block.\(blockIdx).attn.qkv.bias")
            output.write(qkvBiasData)
            writePadding(to: output, alignment: 16)
            
            // Attention sinks
            let sinksData = try loader.loadTensorData(name: "block.\(blockIdx).attn.sinks")
            output.write(sinksData)
            writePadding(to: output, alignment: 16)
            
            // Attention output
            let attnOutWeight = try loader.loadTensorData(name: "block.\(blockIdx).attn.out.weight")
            output.write(attnOutWeight)
            writePadding(to: output, alignment: 16)
            
            let attnOutBias = try loader.loadTensorData(name: "block.\(blockIdx).attn.out.bias")
            output.write(attnOutBias)
            writePadding(to: output, alignment: 16)
            
            // MLP RMSNorm
            let mlpNormData = try loader.loadTensorData(name: "block.\(blockIdx).mlp.norm.scale")
            output.write(mlpNormData)
            writePadding(to: output, alignment: 16)
            
            // MLP gate weights
            let gateWeight = try loader.loadTensorData(name: "block.\(blockIdx).mlp.gate.weight")
            output.write(gateWeight)
            writePadding(to: output, alignment: 16)
            
            let gateBias = try loader.loadTensorData(name: "block.\(blockIdx).mlp.gate.bias")
            output.write(gateBias)
            writePadding(to: output, alignment: 16)
        }
        
        // Final RMSNorm
        print("  - Final RMSNorm")
        let finalNormData = try loader.loadTensorData(name: "norm.scale")
        output.write(finalNormData)
        writePadding(to: output, alignment: 16)
        
        // Unembedding weights
        print("  - Unembedding weights")
        let unembedData = try loader.loadTensorData(name: "unembedding.weight")
        output.write(unembedData.prefix(embeddingSize))
        writePadding(to: output, alignment: 16)
        
        // Write MoE expert weights for each block
        print("  - MoE expert weights")
        for blockIdx in 0..<config.num_hidden_layers {
            try writeMoEWeights(loader: loader, blockIdx: blockIdx, config: config, to: output)
        }
        
        print("✅ Weights written successfully")
    }
    
    private func processQKVWeights(
        loader: SafetensorsLoader,
        blockIdx: Int,
        config: CheckpointConfig
    ) throws -> Data {
        // Load and apply Q/K scaling as per GPT-OSS requirements
        let weights = try loader.loadBFloat16Tensor(name: "block.\(blockIdx).attn.qkv.weight")
        
        let headDim = config.head_dim
        let numQHeads = config.num_attention_heads
        let numKVHeads = config.num_key_value_heads
        // let totalDim = headDim * (numQHeads + 2 * numKVHeads)
        
        var processed = weights
        
        // Apply Q scaling (first numQHeads * headDim elements)
        let qEnd = numQHeads * headDim
        for i in 0..<qEnd {
            processed[i] *= 0.5
        }
        
        // Apply K scaling (next numKVHeads * headDim elements)  
        let kStart = qEnd
        let kEnd = kStart + numKVHeads * headDim
        for i in kStart..<kEnd {
            processed[i] *= 0.25
        }
        
        // V weights remain unchanged
        
        // Convert back to BFloat16
        var data = Data(capacity: processed.count * 2)
        for float in processed {
            let bf16 = float.bfloat16
            data.append(bf16.littleEndianData)
        }
        
        return data
    }
    
    private func writeMoEWeights(
        loader: SafetensorsLoader,
        blockIdx: Int,
        config: CheckpointConfig,
        to output: FileHandle
    ) throws {
        // Write expert weights in MXFP4 format
        // This is simplified - full implementation would handle MXFP4 quantization
        
        for expertIdx in 0..<config.num_experts {
            // SwiGLU weights (gate and up projections)
            let gateWeight = try loader.loadTensorData(name: "block.\(blockIdx).mlp.experts.\(expertIdx).gate.weight")
            output.write(gateWeight)
            writePadding(to: output, alignment: 16)
            
            let upWeight = try loader.loadTensorData(name: "block.\(blockIdx).mlp.experts.\(expertIdx).up.weight")
            output.write(upWeight)
            writePadding(to: output, alignment: 16)
            
            // Down projection
            let downWeight = try loader.loadTensorData(name: "block.\(blockIdx).mlp.experts.\(expertIdx).down.weight")
            output.write(downWeight)
            writePadding(to: output, alignment: 16)
        }
    }
    
    func writePadding(to output: FileHandle, alignment: Int) {
        let currentOffset = (try? output.offset()) ?? 0
        let alignedOffset = ((currentOffset + UInt64(alignment) - 1) / UInt64(alignment)) * UInt64(alignment)
        let padding = Int(alignedOffset - currentOffset)
        if padding > 0 {
            output.write(Data(repeating: 0, count: padding))
        }
    }
    
    private func loadSafetensors(from dir: URL) throws -> [String: Data] {
        // Legacy method - now using SafetensorsLoader directly
        return [:]
    }
    
    // MARK: - Types
    
    struct CheckpointConfig: Codable {
        let num_hidden_layers: Int
        let hidden_size: Int
        let intermediate_size: Int
        let num_attention_heads: Int
        let num_key_value_heads: Int
        let head_dim: Int
        let num_experts: Int
        let sliding_window: Int
        let rope_theta: Double
        let rope_scaling_factor: Int
        let rope_ntk_alpha: Double
        let rope_ntk_beta: Double
        let initial_context_length: Int
        let swiglu_limit: Float?
        
        // Non-codable properties
        var checkpointDir: URL = URL(fileURLWithPath: ".")
    }
}

// MARK: - Data Extensions

extension UUID {
    var data: Data {
        return withUnsafeBytes(of: self.uuid) { Data($0) }
    }
}

extension Data {
    mutating func append<T>(_ value: T) {
        Swift.withUnsafeBytes(of: value) { bytes in
            self.append(contentsOf: bytes)
        }
    }
}