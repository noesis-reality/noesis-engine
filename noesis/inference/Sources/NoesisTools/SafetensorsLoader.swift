import Foundation

/// Loads and parses safetensors files
public class SafetensorsLoader {
    
    /// Metadata for a tensor in the safetensors file
    public struct TensorInfo {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let dataOffsets: (start: Int, end: Int)
        
        var elementCount: Int {
            shape.reduce(1, *)
        }
        
        var bytesPerElement: Int {
            switch dtype {
            case "BF16": return 2
            case "F16": return 2
            case "F32": return 4
            case "F64": return 8
            case "I8": return 1
            case "I16": return 2
            case "I32": return 4
            case "I64": return 8
            case "U8": return 1
            case "U16": return 2
            case "U32": return 4
            case "U64": return 8
            default: return 0
            }
        }
    }
    
    private let fileURL: URL
    private var fileHandle: FileHandle?
    private var header: [String: Any] = [:]
    private var tensors: [String: TensorInfo] = [:]
    private var dataOffset: Int = 0
    
    public init(fileURL: URL) {
        self.fileURL = fileURL
    }
    
    /// Open and parse the safetensors file
    public func open() throws {
        fileHandle = try FileHandle(forReadingFrom: fileURL)
        
        // Read header size (first 8 bytes, little-endian u64)
        let headerSizeData = try fileHandle!.read(upToCount: 8) ?? Data()
        guard headerSizeData.count == 8 else {
            throw LoaderError.invalidFile("Could not read header size")
        }
        
        let headerSize = headerSizeData.withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self).littleEndian
        }
        
        // Read header JSON
        let headerData = try fileHandle!.read(upToCount: Int(headerSize)) ?? Data()
        guard headerData.count == headerSize else {
            throw LoaderError.invalidFile("Could not read full header")
        }
        
        let headerJSON = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] ?? [:]
        self.header = headerJSON
        
        // Parse tensor metadata
        for (key, value) in headerJSON {
            if key == "__metadata__" { continue }
            
            guard let tensorDict = value as? [String: Any],
                  let dtype = tensorDict["dtype"] as? String,
                  let shape = tensorDict["shape"] as? [Int],
                  let offsets = tensorDict["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                continue
            }
            
            tensors[key] = TensorInfo(
                name: key,
                dtype: dtype,
                shape: shape,
                dataOffsets: (start: offsets[0], end: offsets[1])
            )
        }
        
        // Data starts after header
        dataOffset = 8 + Int(headerSize)
    }
    
    /// Get list of available tensor names
    public func tensorNames() -> [String] {
        Array(tensors.keys).sorted()
    }
    
    /// Get tensor metadata
    public func tensorInfo(name: String) -> TensorInfo? {
        tensors[name]
    }
    
    /// Load a tensor as raw data
    public func loadTensorData(name: String) throws -> Data {
        guard let info = tensors[name] else {
            throw LoaderError.tensorNotFound(name)
        }
        
        guard let handle = fileHandle else {
            throw LoaderError.fileNotOpen
        }
        
        let tensorDataOffset = dataOffset + info.dataOffsets.start
        let tensorDataSize = info.dataOffsets.end - info.dataOffsets.start
        
        try handle.seek(toOffset: UInt64(tensorDataOffset))
        guard let data = try handle.read(upToCount: tensorDataSize),
              data.count == tensorDataSize else {
            throw LoaderError.readError("Failed to read tensor data for \(name)")
        }
        
        return data
    }
    
    /// Load a BFloat16 tensor and convert to Float32
    public func loadBFloat16Tensor(name: String) throws -> [Float] {
        guard let info = tensors[name] else {
            throw LoaderError.tensorNotFound(name)
        }
        
        guard info.dtype == "BF16" else {
            throw LoaderError.wrongDtype("Expected BF16, got \(info.dtype)")
        }
        
        let data = try loadTensorData(name: name)
        var floats: [Float] = []
        floats.reserveCapacity(info.elementCount)
        
        // Convert BFloat16 to Float32
        data.withUnsafeBytes { bytes in
            let uint16Ptr = bytes.bindMemory(to: UInt16.self)
            for i in 0..<info.elementCount {
                // BFloat16 to Float32: shift left by 16 bits
                let bf16Bits = uint16Ptr[i].littleEndian
                let float32Bits = UInt32(bf16Bits) << 16
                let float = Float(bitPattern: float32Bits)
                floats.append(float)
            }
        }
        
        return floats
    }
    
    /// Apply Q/K scaling to attention weights as per GPT-OSS requirements
    public func processAttentionWeights(
        weightName: String,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int
    ) throws -> Data {
        let weights = try loadBFloat16Tensor(name: weightName)
        
        // Reshape and apply scaling
        var processedWeights = weights
        // let totalHeads = numQHeads + 2 * numKVHeads
        
        // Q weights: scale by 0.5
        let qSize = numQHeads * headDim
        for i in 0..<qSize {
            processedWeights[i] *= 0.5
        }
        
        // K weights: scale by 0.25
        let kStart = qSize
        let kSize = numKVHeads * headDim
        for i in kStart..<(kStart + kSize) {
            processedWeights[i] *= 0.25
        }
        
        // V weights remain unchanged
        
        // Convert back to BFloat16
        var bf16Data = Data(capacity: processedWeights.count * 2)
        for float in processedWeights {
            let bits = float.bitPattern
            let bf16 = UInt16((bits >> 16) & 0xFFFF)
            bf16Data.append(bf16.littleEndianData)
        }
        
        return bf16Data
    }
    
    /// Close the file
    public func close() {
        fileHandle?.closeFile()
        fileHandle = nil
    }
    
    deinit {
        close()
    }
    
    // MARK: - Errors
    
    public enum LoaderError: LocalizedError {
        case invalidFile(String)
        case fileNotOpen
        case tensorNotFound(String)
        case wrongDtype(String)
        case readError(String)
        
        public var errorDescription: String? {
            switch self {
            case .invalidFile(let msg): return "Invalid safetensors file: \(msg)"
            case .fileNotOpen: return "File not open"
            case .tensorNotFound(let name): return "Tensor not found: \(name)"
            case .wrongDtype(let msg): return "Wrong dtype: \(msg)"
            case .readError(let msg): return "Read error: \(msg)"
            }
        }
    }
}

// MARK: - Data Extensions

extension UInt16 {
    var littleEndianData: Data {
        var value = self.littleEndian
        return Data(bytes: &value, count: 2)
    }
}

extension UInt32 {
    var littleEndianData: Data {
        var value = self.littleEndian
        return Data(bytes: &value, count: 4)
    }
}

extension Float {
    /// Convert to BFloat16 representation
    var bfloat16: UInt16 {
        let bits = self.bitPattern
        return UInt16((bits >> 16) & 0xFFFF)
    }
    
    /// Create from BFloat16 representation
    init(bfloat16: UInt16) {
        let float32Bits = UInt32(bfloat16) << 16
        self = Float(bitPattern: float32Bits)
    }
}