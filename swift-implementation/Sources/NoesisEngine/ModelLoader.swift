import Foundation
import Metal

// MARK: - GPT-OSS Model Loader

@MainActor public final class ModelLoader {
    public enum LoadError: Error {
        case invalidFile
        case invalidFormat
        case unsupportedVersion
        case corruptedData
        case insufficientMemory
        case ioError(Error)
        case invalidUUID
        case unsupportedLayout
        case unsupportedTokenizer
    }

    private struct FileReader {
        let fd: Int32
        let path: String
        private var offset: UInt64 = 0

        init(path: String) throws {
            self.path = path
            self.fd = open(path, O_RDONLY)
            if fd == -1 {
                throw LoadError.ioError(POSIXError(.init(rawValue: errno)!))
            }
        }

        mutating func read<T>(_ type: T.Type) throws -> T {
            let size = MemoryLayout<T>.size
            var value = T.self as! T
            let result = withUnsafeMutablePointer(to: &value) { ptr in
                Darwin.read(fd, ptr, size)
            }
            if result != size {
                throw LoadError.corruptedData
            }
            offset += UInt64(size)
            return value
        }

        mutating func readBytes(count: Int) throws -> Data {
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
            defer { buffer.deallocate() }

            let result = Darwin.read(fd, buffer, count)
            if result != count {
                throw LoadError.corruptedData
            }
            offset += UInt64(count)
            return Data(bytes: buffer, count: count)
        }

        func createMemoryMapping(offset: UInt64, size: Int) throws -> UnsafeRawPointer {
            let pageSize = 16384 // 16KB page size on Apple Silicon
            let alignedOffset = (offset / UInt64(pageSize)) * UInt64(pageSize)
            let prefixSize = Int(offset - alignedOffset)
            let mappingSize = ((size + prefixSize + pageSize - 1) / pageSize) * pageSize

            let mapping = mmap(nil, mappingSize, PROT_READ, MAP_PRIVATE, fd, off_t(alignedOffset))
            if mapping == MAP_FAILED {
                throw LoadError.ioError(POSIXError(.init(rawValue: errno)!))
            }
            guard let base = mapping else {
                throw LoadError.ioError(POSIXError(.init(rawValue: errno)!))
            }
            let advanced = base.advanced(by: prefixSize)
            return UnsafeRawPointer(advanced)
        }

        var currentOffset: UInt64 { offset }

        func close() {
            Darwin.close(fd)
        }
    }

    @MainActor
    public static func loadModel(from url: URL, device: MTLDevice) throws -> GptossModel {
        var reader = try FileReader(path: url.path)
        defer { reader.close() }

        // Read file header
        let fileHeaderData = try reader.readBytes(count: GptossFileHeader.size)
        let fileHeader = try parseFileHeader(data: fileHeaderData)
        guard fileHeader.isValid else {
            throw LoadError.invalidFormat
        }

        // Read model UUID
        let modelUUIDData = try reader.readBytes(count: 16)
        let modelUUID = GptossUUID(Array(modelUUIDData))
        guard modelUUID.equals(GptossUUID.gptossModelUUID) else {
            throw LoadError.invalidUUID
        }

        // Read model header
        let modelHeaderData = try reader.readBytes(count: GptossModelHeader.size)
        let modelHeader = try parseModelHeader(data: modelHeaderData)

        // Read layout UUID
        let layoutUUIDData = try reader.readBytes(count: 16)
        let layoutUUID = GptossUUID(Array(layoutUUIDData))
        guard layoutUUID.equals(GptossUUID.appleGPULayoutUUID) else {
            throw LoadError.unsupportedLayout
        }

        // Read tokenizer UUID
        let tokenizerUUIDData = try reader.readBytes(count: 16)
        let tokenizerUUID = GptossUUID(Array(tokenizerUUIDData))
        guard tokenizerUUID.equals(GptossUUID.tiktokenTokenizerUUID) else {
            throw LoadError.unsupportedTokenizer
        }

        // Read tokenizer header
        let tokenizerHeaderData = try reader.readBytes(count: GptossTiktokenTokenizerHeader.size)
        let tokenizerHeader = try parseTokenizerHeader(data: tokenizerHeaderData)

        // Create model config
        let config = ModelConfig(
            contextLength: modelHeader.contextLength,
            numBlocks: modelHeader.numBlocks,
            numExperts: modelHeader.numExperts,
            numActiveExperts: modelHeader.numActiveExperts,
            embeddingDim: modelHeader.embeddingDim,
            mlpDim: modelHeader.mlpDim,
            swigluLimit: modelHeader.swigluLimit,
            headDim: modelHeader.headDim,
            numHeads: modelHeader.numHeads,
            numKvHeads: modelHeader.numKvHeads,
            attentionWindow: modelHeader.attentionWindow,
            ropeTheta: modelHeader.ropeTheta,
            interpolationScale: modelHeader.interpolationScale,
            yarnOffset: modelHeader.yarnOffset,
            yarnScale: modelHeader.yarnScale,
            yarnMultiplier: modelHeader.yarnMultiplier,
            rmsnormEpsilon: modelHeader.rmsnormEpsilon,
            vocabularySize: tokenizerHeader.numSpecialTokens + tokenizerHeader.numTextTokens
        )

        // Create engine and model
        guard let engine = NoesisEngine.shared else {
            throw LoadError.insufficientMemory
        }

        let model = try GptossModel(config: config, engine: engine)
        let layout = WeightLayout(config: config)

        // Load tokenizer with memory mapping (matching C reference model.c:217-265)
        let tokenizer = try loadTokenizer(
            reader: &reader,
            tokenizerHeader: tokenizerHeader
        )
        model.tokenizer = tokenizer

        // CRITICAL FIX: Align to page boundary after tokenizer, matching C reference (model.c:275)
        // The weights must start at a page-aligned offset
        let currentOffset = reader.currentOffset
        let pageSize = 16384  // 16KB pages on Apple Silicon
        let alignedWeightOffset = ((currentOffset + UInt64(pageSize) - 1) / UInt64(pageSize)) * UInt64(pageSize)
        
        // Calculate and load weights with memory mapping from aligned offset
        try loadWeights(
            model: model,
            reader: reader,
            weightOffset: alignedWeightOffset,
            config: config,
            device: device
        )

        // Sanity checks: validate mapped weights sizes and basic non-zero content
        if let sharedBuf = model.sharedWeightBuffer {
            let sharedLen = sharedBuf.length
            if sharedLen < layout.sharedWeightsSize {
                print("🚨 ERROR: Shared weights buffer smaller than expected (\(sharedLen) < \(layout.sharedWeightsSize))")
                throw LoadError.corruptedData
            }

            // Check embedding and unembedding regions exist fully within buffer
            let embedEnd = layout.embeddingWeightSize
            let unembedStart = layout.unembeddingWeightOffset
            let unembedEnd = unembedStart + layout.unembeddingWeightSize
            if embedEnd > sharedLen || unembedEnd > sharedLen {
                print("🚨 ERROR: Weight offsets exceed shared buffer length")
                print("    embedEnd=\(embedEnd), unembedEnd=\(unembedEnd), sharedLen=\(sharedLen)")
                throw LoadError.corruptedData
            }

            // Lightweight checksum: sample a window from embedding and unembedding
            func bf16SampleSum(_ buf: MTLBuffer, offset: Int, bytes: Int, stride: Int = 4) -> UInt64 {
                let base = buf.contents().advanced(by: offset)
                let count = max(0, min(bytes, buf.length - offset))
                if count <= 0 { return 0 }
                let u16 = base.bindMemory(to: UInt16.self, capacity: count / 2)
                var acc: UInt64 = 0
                let n = count / 2
                var i = 0
                while i < n {
                    acc &+= UInt64(u16[i])
                    i += stride
                }
                return acc
            }

            let embedSample = bf16SampleSum(sharedBuf, offset: 0, bytes: min(1 << 20, layout.embeddingWeightSize))
            let unembedSample = bf16SampleSum(sharedBuf, offset: layout.unembeddingWeightOffset, bytes: min(1 << 20, layout.unembeddingWeightSize))

            if embedSample == 0 || unembedSample == 0 {
                print("⚠️ WARNING: Weight samples are zero (embed=\(embedSample), unembed=\(unembedSample))")
                print("   This often indicates the model file is empty, corrupted, or mis-aligned.")
            }
        }

        return model
    }

    private static func loadWeights(
        model: GptossModel,
        reader: FileReader,
        weightOffset: UInt64,  // Page-aligned offset where weights start
        config: ModelConfig,
        device: MTLDevice
    ) throws {
        let layout = WeightLayout(config: config)

        // Create memory-mapped shared weights buffer from the page-aligned offset
        let sharedWeightsPtr = try reader.createMemoryMapping(
            offset: weightOffset,
            size: layout.sharedWeightsSize
        )

        guard let sharedWeights = device.makeBuffer(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: sharedWeightsPtr),
            length: layout.sharedWeightsSize,
            options: .storageModeShared
        ) else {
            throw LoadError.insufficientMemory
        }

        model.sharedWeightBuffer = sharedWeights

        // Set up weight offsets matching gpt-oss reference
        model.attnRmsnormGainOffset = layout.attnRmsnormGainOffset
        model.attnQkvWeightOffset = layout.attnQkvWeightOffset
        model.attnQkvBiasOffset = layout.attnQkvBiasOffset
        model.attnSdpaSinkOffset = layout.attnSdpaSinkOffset
        model.attnOutWeightOffset = layout.attnOutWeightOffset
        model.attnOutBiasOffset = layout.attnOutBiasOffset
        model.mlpRmsnormGainOffset = layout.mlpRmsnormGainOffset
        model.mlpGateWeightOffset = layout.mlpGateWeightOffset
        model.mlpGateBiasOffset = layout.mlpGateBiasOffset
        model.rmsnormWeightOffset = layout.rmsnormWeightOffset
        model.unembeddingWeightOffset = layout.unembeddingWeightOffset
        model.perBlockSharedWeightsSize = layout.perBlockSharedWeightsSize

        // MoE expert weight offsets (matching C reference model.c:400-409)
        // C: model->mlp_swiglu_scale_offset = mlp_swiglu_weight_block_size
        model.mlpSwigluScaleOffset = layout.mlpSwigluWeightBlockSize
        
        // C: model->mlp_swiglu_bias_offset = model->mlp_swiglu_scale_offset + mlp_swiglu_weight_scale_size
        model.mlpSwigluBiasOffset = model.mlpSwigluScaleOffset + layout.mlpSwigluWeightScaleSize
        
        // C: model->mlp_out_block_offset = model->mlp_swiglu_bias_offset + mlp_swiglu_bias_size
        model.mlpOutBlockOffset = model.mlpSwigluBiasOffset + layout.mlpSwigluBiasSize
        
        // C: model->mlp_out_scale_offset = model->mlp_out_block_offset + mlp_out_weight_block_size
        model.mlpOutScaleOffset = model.mlpOutBlockOffset + layout.mlpOutWeightBlockSize
        
        // C: model->mlp_out_bias_offset = model->mlp_out_scale_offset + mlp_out_weight_scale_size
        model.mlpOutBiasOffset = model.mlpOutScaleOffset + layout.mlpOutWeightScaleSize
        
        model.perExpertBlockWeightSize = layout.perExpertBlockWeightSize

        // Create per-block MoE weight buffers
        model.blockWeightBuffers = []
        let weightsStartOffset = weightOffset + UInt64(layout.sharedWeightsSize)
        let pageSize = 16384
        let moeBlockWeightSize = ((Int(config.numExperts) * layout.perExpertBlockWeightSize + pageSize - 1) / pageSize) * pageSize

        for blockIdx in 0..<Int(config.numBlocks) {
            let blockOffset = weightsStartOffset + UInt64(blockIdx * moeBlockWeightSize)
            let blockWeightsPtr = try reader.createMemoryMapping(
                offset: blockOffset,
                size: moeBlockWeightSize
            )

            guard let blockWeights = device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: blockWeightsPtr),
                length: moeBlockWeightSize,
                options: .storageModeShared
            ) else {
                throw LoadError.insufficientMemory
            }

            model.blockWeightBuffers.append(blockWeights)
        }
    }

    private static func loadTokenizer(
        reader: inout FileReader,
        tokenizerHeader: GptossTiktokenTokenizerHeader
    ) throws -> GptossTokenizer {
        // Create tokenizer object (matching C reference model.c:217-226)
        let tokenizer = GptossTokenizer(
            mappingPtr: nil,  // Will be set after memory mapping
            mappingSize: 0,
            regexPtr: nil,
            tokensPtr: nil,
            numTextTokens: tokenizerHeader.numTextTokens,
            numSpecialTokens: tokenizerHeader.numSpecialTokens
        )
        
        // Read and process special token UUIDs (matching C reference model.c:231-243)
        for t in 0..<tokenizerHeader.numSpecialTokens {
            let tokenUUIDData = try reader.readBytes(count: 16)
            let tokenUUID = GptossUUID(Array(tokenUUIDData))
            
            // Decode special token type from UUID
            if let specialToken = decodeSpecialTokenUUID(tokenUUID) {
                // Set special token ID: text_tokens + special_token_index
                let tokenId = tokenizerHeader.numTextTokens + t
                tokenizer.setSpecialTokenID(specialToken, id: tokenId)
            }
        }
        
        // Calculate tokenizer data offsets (matching C reference model.c:245-248)
        let tokenizerStartOffset = reader.currentOffset
        let tokenizerEndOffset = tokenizerStartOffset + UInt64(tokenizerHeader.regexSize + tokenizerHeader.tokensSize)
        
        // Round to page boundaries for memory mapping
        let pageSize = 16384  // 16KB pages on Apple Silicon
        let tokenizerMappingStart = (tokenizerStartOffset / UInt64(pageSize)) * UInt64(pageSize)
        let tokenizerMappingEnd = ((tokenizerEndOffset + UInt64(pageSize) - 1) / UInt64(pageSize)) * UInt64(pageSize)
        let tokenizerMappingSize = Int(tokenizerMappingEnd - tokenizerMappingStart)
        
        // Memory map the tokenizer data (matching C reference model.c:249-260)
        let tokenizerMappingPtr = mmap(
            nil,
            tokenizerMappingSize,
            PROT_READ,
            MAP_PRIVATE,
            reader.fd,
            off_t(tokenizerMappingStart)
        )
        
        if tokenizerMappingPtr == MAP_FAILED {
            throw LoadError.ioError(POSIXError(.init(rawValue: errno)!))
        }
        
        guard let mappingBase = tokenizerMappingPtr else {
            throw LoadError.ioError(POSIXError(.init(rawValue: errno)!))
        }
        
        // Calculate actual data pointers within the mapping
        let offsetInMapping = Int(tokenizerStartOffset - tokenizerMappingStart)
        let regexPtr = UnsafeRawPointer(mappingBase).advanced(by: offsetInMapping)
        let tokensPtr = regexPtr.advanced(by: Int(tokenizerHeader.regexSize))
        
        // Create new tokenizer with proper memory mapping
        let finalTokenizer = GptossTokenizer(
            mappingPtr: UnsafeRawPointer(mappingBase),
            mappingSize: tokenizerMappingSize,
            regexPtr: regexPtr,
            tokensPtr: tokensPtr,
            numTextTokens: tokenizerHeader.numTextTokens,
            numSpecialTokens: tokenizerHeader.numSpecialTokens
        )
        
        // Copy over special token IDs
        finalTokenizer.specialTokenIDs = tokenizer.specialTokenIDs
        
        // Advise kernel about memory access pattern (matching C reference model.c:261-263)
        if madvise(UnsafeMutableRawPointer(mutating: mappingBase), tokenizerMappingSize, MADV_RANDOM | MADV_WILLNEED) != 0 {
            print("WARNING: madvise for tokenizer failed with error \(errno)")
        }
        
        // Advance reader past tokenizer data
        _ = try reader.readBytes(count: Int(tokenizerHeader.regexSize))
        _ = try reader.readBytes(count: Int(tokenizerHeader.tokensSize))
        
        return finalTokenizer
    }

    private static func parseFileHeader(data: Data) throws -> GptossFileHeader {
        return data.withUnsafeBytes { bytes in
            let magic = Array(bytes.prefix(12))
            let zero = bytes.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
            return GptossFileHeader(magic: magic, zero: zero)
        }
    }

    private static func parseModelHeader(data: Data) throws -> GptossModelHeader {
        return data.withUnsafeBytes { bytes in
            return GptossModelHeader(
                contextLength: bytes.loadUnaligned(fromByteOffset: 0, as: UInt32.self),
                numBlocks: bytes.loadUnaligned(fromByteOffset: 4, as: UInt32.self),
                numExperts: bytes.loadUnaligned(fromByteOffset: 8, as: UInt32.self),
                numActiveExperts: bytes.loadUnaligned(fromByteOffset: 12, as: UInt32.self),
                embeddingDim: bytes.loadUnaligned(fromByteOffset: 16, as: UInt32.self),
                mlpDim: bytes.loadUnaligned(fromByteOffset: 20, as: UInt32.self),
                swigluLimit: bytes.loadUnaligned(fromByteOffset: 24, as: Float32.self),
                headDim: bytes.loadUnaligned(fromByteOffset: 28, as: UInt32.self),
                numHeads: bytes.loadUnaligned(fromByteOffset: 32, as: UInt32.self),
                numKvHeads: bytes.loadUnaligned(fromByteOffset: 36, as: UInt32.self),
                attentionWindow: bytes.loadUnaligned(fromByteOffset: 40, as: UInt32.self),
                ropeTheta: bytes.loadUnaligned(fromByteOffset: 44, as: Float32.self),
                interpolationScale: bytes.loadUnaligned(fromByteOffset: 48, as: Float32.self),
                yarnOffset: bytes.loadUnaligned(fromByteOffset: 52, as: Float32.self),
                yarnScale: bytes.loadUnaligned(fromByteOffset: 56, as: Float32.self),
                yarnMultiplier: bytes.loadUnaligned(fromByteOffset: 60, as: Float32.self),
                rmsnormEpsilon: bytes.loadUnaligned(fromByteOffset: 64, as: Float32.self)
            )
        }
    }

    private static func parseTokenizerHeader(data: Data) throws -> GptossTiktokenTokenizerHeader {
        return data.withUnsafeBytes { bytes in
            return GptossTiktokenTokenizerHeader(
                numSpecialTokens: bytes.loadUnaligned(fromByteOffset: 0, as: UInt32.self),
                numTextTokens: bytes.loadUnaligned(fromByteOffset: 4, as: UInt32.self),
                regexSize: bytes.loadUnaligned(fromByteOffset: 8, as: UInt32.self),
                tokensSize: bytes.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
            )
        }
    }

    private static func decodeSpecialTokenUUID(_ uuid: GptossUUID) -> GptossSpecialToken? {
        // Decode special token UUIDs from the actual model file
        // These UUIDs match the create-local-model.py script
        
        // Special token UUIDs from GPT-OSS create-local-model.py
        let specialTokenUUIDs: [GptossUUID: GptossSpecialToken] = [
            // <|start|> token: UUID('55a77c2f-8a01-4c54-8ac2-313bfc7e208d')
            GptossUUID([0x55, 0xa7, 0x7c, 0x2f, 0x8a, 0x01, 0x4c, 0x54,
                       0x8a, 0xc2, 0x31, 0x3b, 0xfc, 0x7e, 0x20, 0x8d]): .start,
            
            // <|message|> token: UUID('16e40431-f47f-4b22-b59b-8b278fc30a54')
            GptossUUID([0x16, 0xe4, 0x04, 0x31, 0xf4, 0x7f, 0x4b, 0x22,
                       0xb5, 0x9b, 0x8b, 0x27, 0x8f, 0xc3, 0x0a, 0x54]): .message,
            
            // <|end|> token: UUID('fcac2f6d-4705-4f6b-b228-642accac7238')
            GptossUUID([0xfc, 0xac, 0x2f, 0x6d, 0x47, 0x05, 0x4f, 0x6b,
                       0xb2, 0x28, 0x64, 0x2a, 0xcc, 0xac, 0x72, 0x38]): .end,
            
            // <|return|> token: UUID('f799ff69-1992-43c4-a3d8-d831f475dc75')
            GptossUUID([0xf7, 0x99, 0xff, 0x69, 0x19, 0x92, 0x43, 0xc4,
                       0xa3, 0xd8, 0xd8, 0x31, 0xf4, 0x75, 0xdc, 0x75]): .return,
            
            // <|refusal|> token: UUID('e15ba702-28c4-4292-ab8f-ffa434709128')
            GptossUUID([0xe1, 0x5b, 0xa7, 0x02, 0x28, 0xc4, 0x42, 0x92,
                       0xab, 0x8f, 0xff, 0xa4, 0x34, 0x70, 0x91, 0x28]): .refusal,
            
            // <|constrain|> token: UUID('c0bb14c7-6022-49da-ad08-792d67e8b470')
            GptossUUID([0xc0, 0xbb, 0x14, 0xc7, 0x60, 0x22, 0x49, 0xda,
                       0xad, 0x08, 0x79, 0x2d, 0x67, 0xe8, 0xb4, 0x70]): .constrain,
            
            // <|channel|> token: UUID('fd3dda11-c8ab-4033-876e-d93deb172c93')
            GptossUUID([0xfd, 0x3d, 0xda, 0x11, 0xc8, 0xab, 0x40, 0x33,
                       0x87, 0x6e, 0xd9, 0x3d, 0xeb, 0x17, 0x2c, 0x93]): .channel,
            
            // <|call|> token: UUID('1220f796-e388-4de5-b487-fe2eb5fe03c0')
            GptossUUID([0x12, 0x20, 0xf7, 0x96, 0xe3, 0x88, 0x4d, 0xe5,
                       0xb4, 0x87, 0xfe, 0x2e, 0xb5, 0xfe, 0x03, 0xc0]): .call,
            
            // <|untrusted|> token: UUID('07d7da55-b346-4cff-8b37-7cefacf8a3e8')
            GptossUUID([0x07, 0xd7, 0xda, 0x55, 0xb3, 0x46, 0x4c, 0xff,
                       0x8b, 0x37, 0x7c, 0xef, 0xac, 0xf8, 0xa3, 0xe8]): .untrusted,
            
            // <|end_untrusted|> token: UUID('f265bd9c-c717-469e-a447-920687d65d90')
            GptossUUID([0xf2, 0x65, 0xbd, 0x9c, 0xc7, 0x17, 0x46, 0x9e,
                       0xa4, 0x47, 0x92, 0x06, 0x87, 0xd6, 0x5d, 0x90]): .endUntrusted
        ]
        
        // Check if UUID matches any known special token
        for (knownUUID, token) in specialTokenUUIDs {
            if uuid.equals(knownUUID) {
                return token
            }
        }
        
        return nil
    }
}

// MARK: - Special Token Support

public enum GptossSpecialToken: UInt8 {
    case invalid = 0
    case `return` = 1
    case start = 2
    case message = 3
    case end = 4
    case refusal = 5
    case constrain = 6
    case channel = 7
    case call = 8
    case untrusted = 9
    case endUntrusted = 10
}

// MARK: - Memory Mapping Support

private func roundUpToPageSize(_ size: Int) -> Int {
    let pageSize = 16384  // 16KB pages on Apple Silicon
    return (size + pageSize - 1) & ~(pageSize - 1)
}

private func roundDownToPageSize(_ size: Int) -> Int {
    let pageSize = 16384  // 16KB pages on Apple Silicon
    return size & ~(pageSize - 1)
}
