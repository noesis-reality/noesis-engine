import Foundation
import Metal

// MARK: - RMSNorm Dispatcher

@MainActor
public final class RmsnormDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_bf16w_rmsnorm")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossRmsnormArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        weights: MTLBuffer,
        weightsOffset: Int = 0,  // NEW: Support weights offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        tokens: Int,
        threadsPerThreadgroup: Int = 1024
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossRmsnormArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(weights, offset: weightsOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)

        // Legacy explicit size preserved for reference; computed below from pipeline
        // let threadsPerThreadgroupSize = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(width: tokens, height: 1, depth: 1)
        // Derive thread geometry from pipeline where reasonable
        let w: Int = pipeline.threadExecutionWidth
        let maxThreads: Int = pipeline.maxTotalThreadsPerThreadgroup
        let tptWidth: Int = Swift.min(Swift.max(w, threadsPerThreadgroup), maxThreads)
        let tpt = MTLSize(width: tptWidth, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: tpt)
        encoder.endEncoding()
    }
}

// MARK: - Embeddings Dispatcher

@MainActor
public final class EmbeddingsDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_bf16_f32_embeddings")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossEmbeddingsArgs,
        tokens: MTLBuffer,
        tokenOffset: Int,  // NEW: Add token offset parameter
        weights: MTLBuffer,
        output: MTLBuffer,
        numTokens: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossEmbeddingsArgs>.size, index: 0)
        encoder.setBuffer(tokens, offset: tokenOffset, index: 1)  // ← FIX: Use proper token offset
        encoder.setBuffer(weights, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        // CRITICAL FIX: Use 512 threads per threadgroup to match C reference (context.c:137)
        let threadsPerThreadgroup = MTLSize(width: 512, height: 1, depth: 1)
        let threadgroups = MTLSize(width: numTokens, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Matmul Dispatcher

@MainActor
public final class MatmulDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_bf16w_matmul")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossMatmulArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        weight: MTLBuffer,
        weightOffset: Int = 0,  // NEW: Support weight offset
        bias: MTLBuffer,
        biasOffset: Int = 0,  // NEW: Support bias offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        numTokens: Int,
        simdgroupsPerThreadgroup: Int = 32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossMatmulArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(bias, offset: biasOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)

        // CRITICAL FIX: Use 256 threads per threadgroup to match C reference (context.c:173)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: Int(args.numRows) / 8,  // 256 threads = 8 simdgroups of 32
            height: numTokens,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Unembedding Dispatcher

@MainActor
public final class UnembeddingDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_bf16w_unembedding")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossUnembeddingArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        weight: MTLBuffer,
        weightOffset: Int = 0,  // NEW: Support weight offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        argmax: MTLBuffer,
        argmaxOffset: Int = 0,  // NEW: Support argmax offset
        numTokens: Int,
        simdgroupsPerThreadgroup: Int = 32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossUnembeddingArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBuffer(argmax, offset: argmaxOffset, index: 4)

        // CRITICAL FIX: Use 256 threads per threadgroup to match C reference (context.c:425)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (Int(args.numRows) + Int(args.numRowsPerThreadgroup) - 1) / Int(args.numRowsPerThreadgroup)
        let threadgroups = MTLSize(width: numThreadgroups, height: numTokens, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - RoPE Dispatcher

@MainActor
public final class RopeDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_rope")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossRopeArgs,
        activations: MTLBuffer,
        numHeadPairs: Int,
        numTokens: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossRopeArgs>.size, index: 0)
        encoder.setBuffer(activations, offset: 0, index: 1)

        let threadsPerGrid = MTLSize(width: numHeadPairs, height: numTokens, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

// MARK: - SDPA Dispatcher

@MainActor
public final class SdpaDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_sdpa_q8_d64")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossSdpaArgs,
        q: MTLBuffer,
        qOffset: Int = 0,  // NEW: Support q offset
        k: MTLBuffer,
        kOffset: Int = 0,  // NEW: Support k offset
        v: MTLBuffer,
        vOffset: Int = 0,  // NEW: Support v offset
        s: MTLBuffer,
        sOffset: Int = 0,  // NEW: Support s offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        numTokens: Int,
        numKvHeads: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossSdpaArgs>.size, index: 0)
        encoder.setBuffer(q, offset: qOffset, index: 1)
        encoder.setBuffer(k, offset: kOffset, index: 2)
        encoder.setBuffer(v, offset: vOffset, index: 3)
        encoder.setBuffer(s, offset: sOffset, index: 4)
        encoder.setBuffer(output, offset: outputOffset, index: 5)

        let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroups = MTLSize(width: numTokens, height: numKvHeads, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - TopK Dispatcher

@MainActor
public final class TopKDispatcher {
    private let pipelineE32: MTLComputePipelineState
    private let pipelineE128: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipelineE32 = try model.getPipeline(for: "gptoss_f32_topk_softmax_e32_k4")
        self.pipelineE128 = try model.getPipeline(for: "gptoss_f32_topk_softmax_e128_k4")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossTopkArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        numTokens: Int,
        numExperts: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        let pipeline = (numExperts == 32) ? pipelineE32 : pipelineE128
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossTopkArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)

        let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroups = MTLSize(width: numTokens, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - MoE Matmul Dispatcher

@MainActor
public final class MoeMatmulDispatcher {
    private let swigluPipeline: MTLComputePipelineState
    private let matmulPipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.swigluPipeline = try model.getPipeline(for: "gptoss_f32_mf4w_moe_matmul_swiglu")
        self.matmulPipeline = try model.getPipeline(for: "gptoss_f32_mf4w_moe_matmul")
    }

    public func encodeSwiglu(
        commandBuffer: MTLCommandBuffer,
        args: GptossMoeMatmulSwigluArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        expert: MTLBuffer,
        expertOffset: Int = 0,  // NEW: Support expert offset
        weightBlocks: MTLBuffer,
        weightBlocksOffset: Int = 0,  // NEW: Support weight blocks offset
        weightScales: MTLBuffer,
        weightScalesOffset: Int = 0,  // NEW: Support weight scales offset
        bias: MTLBuffer,
        biasOffset: Int = 0,  // NEW: Support bias offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        numTokens: Int,
        simdgroupsPerThreadgroup: Int = 32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(swigluPipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossMoeMatmulSwigluArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(expert, offset: expertOffset, index: 2)
        encoder.setBuffer(weightBlocks, offset: weightBlocksOffset, index: 3)
        encoder.setBuffer(weightScales, offset: weightScalesOffset, index: 4)
        encoder.setBuffer(bias, offset: biasOffset, index: 5)
        encoder.setBuffer(output, offset: outputOffset, index: 6)

        let threadsPerThreadgroup = MTLSize(width: 32 * simdgroupsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: Int(args.numRows) / simdgroupsPerThreadgroup,
            height: numTokens,
            depth: Int(args.numActiveExperts)
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    public func encodeMatmul(
        commandBuffer: MTLCommandBuffer,
        args: GptossMoeMatmulArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        expert: MTLBuffer,
        expertOffset: Int = 0,  // NEW: Support expert offset
        weightBlocks: MTLBuffer,
        weightBlocksOffset: Int = 0,  // NEW: Support weight blocks offset
        weightScales: MTLBuffer,
        weightScalesOffset: Int = 0,  // NEW: Support weight scales offset
        bias: MTLBuffer,
        biasOffset: Int = 0,  // NEW: Support bias offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        numTokens: Int,
        simdgroupsPerThreadgroup: Int = 32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(matmulPipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossMoeMatmulArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(expert, offset: expertOffset, index: 2)
        encoder.setBuffer(weightBlocks, offset: weightBlocksOffset, index: 3)
        encoder.setBuffer(weightScales, offset: weightScalesOffset, index: 4)
        encoder.setBuffer(bias, offset: biasOffset, index: 5)
        encoder.setBuffer(output, offset: outputOffset, index: 6)

        let threadsPerThreadgroup = MTLSize(width: 32 * simdgroupsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: Int(args.numRows) / simdgroupsPerThreadgroup,
            height: numTokens,
            depth: Int(args.numActiveExperts)
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Accumulate Dispatcher

@MainActor
public final class AccumulateDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_accumulate_e4")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossAccumulateArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,  // NEW: Support input offset
        expert: MTLBuffer,
        expertOffset: Int = 0,  // NEW: Support expert offset
        output: MTLBuffer,
        outputOffset: Int = 0,  // NEW: Support output offset
        numTokens: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossAccumulateArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(expert, offset: expertOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (Int(args.numVecs) + Int(args.numVecsPerThreadgroup) - 1) / Int(args.numVecsPerThreadgroup)
        let threadgroups = MTLSize(width: numThreadgroups, height: numTokens, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Random Dispatcher

@MainActor
public final class RandomDispatcher {
    private let u32Pipeline: MTLComputePipelineState
    private let f32Pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.u32Pipeline = try model.getPipeline(for: "gptoss_u32_fill_random")
        self.f32Pipeline = try model.getPipeline(for: "gptoss_f32_fill_random")
    }

    public func encodeU32(
        commandBuffer: MTLCommandBuffer,
        args: GptossU32FillRandomArgs,
        output: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(u32Pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossU32FillRandomArgs>.size, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (Int(args.numVecs) + Int(args.numVecsPerThreadgroup) - 1) / Int(args.numVecsPerThreadgroup)
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    public func encodeF32(
        commandBuffer: MTLCommandBuffer,
        args: GptossF32FillRandomArgs,
        output: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(f32Pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossF32FillRandomArgs>.size, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (Int(args.numVecs) + Int(args.numVecsPerThreadgroup) - 1) / Int(args.numVecsPerThreadgroup)
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Softmax Dispatcher

@MainActor
public final class SoftmaxDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_softmax")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossSoftmaxArgs,
        score: MTLBuffer,
        argmax: MTLBuffer,
        prob: MTLBuffer,
        sum: MTLBuffer,
        numTokens: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossSoftmaxArgs>.size, index: 0)
        encoder.setBuffer(score, offset: 0, index: 1)
        encoder.setBuffer(argmax, offset: 0, index: 2)
        encoder.setBuffer(prob, offset: 0, index: 3)
        encoder.setBuffer(sum, offset: 0, index: 4)

        // Calculate actual number of threadgroups needed
        let numThreadgroups = min(Int(args.maxThreadgroups), (Int(args.numVecs) + Int(args.numVecsPerThreadgroup) - 1) / Int(args.numVecsPerThreadgroup))
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadgroups, height: numTokens, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Convert Dispatcher

@MainActor
public final class ConvertDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_mf4_f32_convert")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossConvertArgs,
        blocks: MTLBuffer,
        scales: MTLBuffer,
        output: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossConvertArgs>.size, index: 0)
        encoder.setBuffer(blocks, offset: 0, index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (Int(args.numVecs) + Int(args.numVecsPerThreadgroup) - 1) / Int(args.numVecsPerThreadgroup)
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

// MARK: - Missing Dispatchers from GPT-OSS Reference

@MainActor
public final class U32FillRandomDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_u32_fill_random")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossU32FillRandomArgs,
        output: MTLBuffer,
        outputOffset: Int = 0,
        threadsPerThreadgroup: Int = 1024
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossU32FillRandomArgs>.size, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)

        let numThreadgroups = Int(args.numVecs + UInt64(threadsPerThreadgroup) - 1) / threadsPerThreadgroup
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

@MainActor
public final class F32FillRandomDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_f32_fill_random")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossF32FillRandomArgs,
        output: MTLBuffer,
        outputOffset: Int = 0,
        threadsPerThreadgroup: Int = 1024
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossF32FillRandomArgs>.size, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)

        let numThreadgroups = Int(args.numVecs + UInt64(threadsPerThreadgroup) - 1) / threadsPerThreadgroup
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

@MainActor
public final class Bf16FillRandomDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_bf16_fill_random")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossF32FillRandomArgs, // Uses same args as f32 version
        output: MTLBuffer,
        outputOffset: Int = 0,
        threadsPerThreadgroup: Int = 1024
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossF32FillRandomArgs>.size, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)

        let numThreadgroups = Int(args.numVecs + UInt64(threadsPerThreadgroup) - 1) / threadsPerThreadgroup
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

@MainActor
public final class Mf4F32ConvertDispatcher {
    private let pipeline: MTLComputePipelineState

    public init(model: GptossModel) throws {
        self.pipeline = try model.getPipeline(for: "gptoss_mf4_f32_convert")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        args: GptossConvertArgs,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        threadsPerThreadgroup: Int = 1024
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)

        var argsValue = args
        encoder.setBytes(&argsValue, length: MemoryLayout<GptossConvertArgs>.size, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)

        let numThreadgroups = Int(args.numVecs + UInt64(threadsPerThreadgroup) - 1) / threadsPerThreadgroup
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

