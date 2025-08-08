import Foundation
import Metal

@MainActor
public final class NoesisEngine {
    public static let shared = NoesisEngine()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    public let libraries: [MTLLibrary]
    public let pipelines: ComputePipelineCache

    private init?() {
        // Enforce Metal 4 only (macOS 15+)
        guard #available(macOS 15.0, *) else {
            fatalError("Metal 4 required: macOS 15.0+ is required for NoesisEngine")
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        guard let cq = device.makeCommandQueue() else { return nil }
        self.commandQueue = cq

        // Load shaders from embedded sources (Metal 4 compile at runtime)
        do {
            let libs = try MetalLibraryLoader.makeLibrariesFromEmbeddedShaders(device: device)
            guard let first = libs.first else { throw NSError(domain: "NoesisEngine", code: -2, userInfo: [NSLocalizedDescriptionKey: "No shader libraries built"]) }
            self.library = first
            self.libraries = libs
            self.pipelines = ComputePipelineCache(device: device, libraries: libs)
            self.pipelines.prewarmAll()
        } catch {
            print("Failed to initialize Metal library: \(error)")
            return nil
        }
    }
}

public struct MetalLibraryLoader {
    public static func makeLibrariesFromEmbeddedShaders(device: MTLDevice) throws -> [MTLLibrary] {
        let bundle = Bundle.module

        // SPM flattens all resources into the bundle root - work with this reality
        let allURLs = bundle.urls(forResourcesWithExtension: nil, subdirectory: nil) ?? []

        // First try to load pre-built metallib (built by MetallibBuilder plugin)
        if let metallibURL = allURLs.first(where: { $0.pathExtension == "metallib" }) {
            do {
                let library = try device.makeLibrary(URL: metallibURL)
                print("🚀 Loaded pre-built Metal library: \(metallibURL.lastPathComponent)")
                return [library]
            } catch {
                print("⚠️ Failed to load metallib, falling back to runtime compilation: \(error)")
            }
        }
        
        // Fallback to runtime compilation from individual shaders
        print("🔧 Compiling Metal shaders at runtime...")

        // Separate Metal shaders and headers
        let shaderURLs = allURLs.filter { $0.pathExtension == "metal" }
        let headerURLs = allURLs.filter { $0.pathExtension == "h" || $0.pathExtension == "hpp" }

        guard !shaderURLs.isEmpty else {
            throw NSError(domain: "MetalLibraryLoader", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "No Metal shaders found in bundle resources"
            ])
        }

        // Load header files for inlining
        var headers: [String: String] = [:]
        for headerURL in headerURLs {
            let headerName = headerURL.lastPathComponent
            headers[headerName] = try String(contentsOf: headerURL, encoding: .utf8)
        }

        // Process shader sources with proper header inlining
        func processShaderSource(_ source: String) -> String {
            var processedSource = source

            // Remove all #include statements since we're adding headers at the top
            for (headerName, _) in headers {
                let includePattern = "#include <internal/\(headerName)>"
                processedSource = processedSource.replacingOccurrences(of: includePattern, with: "")

                let quotePattern = "#include \"internal/\(headerName)\""
                processedSource = processedSource.replacingOccurrences(of: quotePattern, with: "")
            }

            // Remove standalone includes without specific header names
            processedSource = processedSource.replacingOccurrences(of: "#include <internal/kernel-args.h>", with: "")

            return processedSource
        }

        // Build combined Metal library source
        var combinedSource = ""

        // First, add kernel-args.h at the top for all struct definitions
        if let kernelArgsHeader = headers["kernel-args.h"] {
            combinedSource += "// Kernel argument structures - included once at top\n"
            combinedSource += kernelArgsHeader
            combinedSource += "\n\n"
        }

        // Then add all shader sources with includes removed
        for shaderURL in shaderURLs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let source = try String(contentsOf: shaderURL, encoding: .utf8)
            let processedSource = processShaderSource(source)
            combinedSource += "// Source: \(shaderURL.lastPathComponent)\n"
            combinedSource += processedSource
            combinedSource += "\n\n"
        }

        // Compile Metal library with proper options for Metal 4
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1  // Latest Metal Shading Language

        // Use modern mathMode instead of deprecated fastMathEnabled
        if #available(macOS 15.0, *) {
            options.mathMode = .fast
        }

        #if DEBUG
        options.preprocessorMacros = ["DEBUG": NSNumber(value: 1)]
        #endif

        let library = try device.makeLibrary(source: combinedSource, options: options)
        return [library]
    }
}

public enum MetalKernels {
    case embeddings
    case rmsnorm
    case matmul
    case unembedding
    case rope
    case random
    case sdpa
    case sample
    case topk
    case convert
    case accumulate

    public var functionNameHints: [String] {
        switch self {
        case .embeddings: return ["gptoss_bf16_f32_embeddings"]
        case .rmsnorm: return ["gptoss_f32_bf16w_rmsnorm", "_gptoss_f32_bf16w_rmsnorm"]
        case .matmul: return ["gptoss_f32_bf16w_matmul"]
        case .unembedding: return ["gptoss_f32_bf16w_unembedding"]
        case .rope: return ["gptoss_f32_rope"]
        case .random: return ["gptoss_u32_fill_random", "gptoss_f32_fill_random"]
        case .sdpa: return ["gptoss_f32_sdpa_q8_d64"]
        case .sample: return ["gptoss_f32_softmax"]
        case .topk: return ["gptoss_f32_topk_softmax_e128_k4", "gptoss_f32_topk_softmax_e32_k4"]
        case .convert: return ["gptoss_mf4_f32_convert"]
        case .accumulate: return ["gptoss_f32_accumulate_e4"]
        }
    }
}

public final class ComputePipelineCache {
    private let device: MTLDevice
    private let libraries: [MTLLibrary]
    private var cache: [String: MTLComputePipelineState] = [:]

    public init(device: MTLDevice, libraries: [MTLLibrary]) {
        self.device = device
        self.libraries = libraries
    }

    public func pipeline(for kernel: MetalKernels) throws -> MTLComputePipelineState {
        for name in kernel.functionNameHints {
            if let p = cache[name] { return p }
            for lib in libraries {
                if let f = lib.makeFunction(name: name) ?? lib.makeFunction(name: name.trimmingCharacters(in: CharacterSet(charactersIn: ":_"))) {
                    let p = try device.makeComputePipelineState(function: f)
                    cache[name] = p
                    return p
                }
            }
        }
        throw NSError(domain: "NoesisEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Kernel function not found: \(kernel)"])
    }

    public func prewarmAll() {
        // Compile a representative subset ahead of time
        let allHints = [
            MetalKernels.embeddings,
            .rmsnorm,
            .matmul,
            .unembedding,
            .rope,
            .random,
            .sdpa,
            .sample,
            .topk,
            .convert,
            .accumulate
        ].flatMap { $0.functionNameHints }
        for name in allHints {
            if cache[name] != nil { continue }
            for lib in libraries {
                if let f = lib.makeFunction(name: name) {
                    do {
                        let p = try device.makeComputePipelineState(function: f)
                        cache[name] = p
                        break
                    } catch {
                        // Ignore and continue
                    }
                }
            }
        }
    }
}

