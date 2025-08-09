import Foundation
import Metal

public struct MetalDebugger {
    @MainActor
    public static func debugShaderCompilation(device: MTLDevice) {
        print("🔍 Metal Shader Debug Information")
        print("   Device: \(device.name)")
        print("")
        
        let bundle = Bundle.main
        print("📦 Bundle Resources:")
        
        let allURLs: [URL] = []
        let shaderURLs = allURLs.filter { $0.pathExtension == "metal" }
        let metallibURLs = allURLs.filter { $0.pathExtension == "metallib" }
        
        if !metallibURLs.isEmpty {
            print("   Found \(metallibURLs.count) pre-built .metallib files:")
            for url in metallibURLs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                print("     - \(url.lastPathComponent)")
            }
        }
        
        if !shaderURLs.isEmpty {
            print("   Found \(shaderURLs.count) .metal files (fallback):")
            for url in shaderURLs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                print("     - \(url.lastPathComponent)")
            }
        } else {
            print("   ❌ No .metal files found")
        }
        
        let headerURLs = allURLs.filter { $0.pathExtension == "h" || $0.pathExtension == "hpp" }
        if !headerURLs.isEmpty {
            print("   Found \(headerURLs.count) header files:")
            for url in headerURLs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                print("     - \(url.lastPathComponent)")
            }
        } else {
            print("   ❌ No header files found")
        }
        
        // Try to load and show first few lines of kernel-args.h if available
        if let kernelArgsURL = allURLs.first(where: { $0.lastPathComponent == "kernel-args.h" }) {
            if let kernelArgsContent = try? String(contentsOf: kernelArgsURL, encoding: .utf8) {
                let lines = kernelArgsContent.components(separatedBy: CharacterSet.newlines)
                print("   kernel-args.h preview (first 10 lines):")
                for (i, line) in lines.prefix(10).enumerated() {
                    print("     \(i+1): \(line)")
                }
            }
        }
        
        // Try compiling a simple test shader
        print("\n🧪 Test Compilation:")
        let testShader = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct test_args {
            uint32_t value;
        };
        
        kernel void test_kernel(
            constant test_args& args [[buffer(0)]],
            device float* output [[buffer(1)]],
            uint tid [[thread_position_in_grid]]
        ) {
            output[tid] = float(args.value + tid);
        }
        """
        
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        
        if let testLib = try? device.makeLibrary(source: testShader, options: options) {
            print("   ✅ Basic shader compilation successful")
            print("   Test functions: \(testLib.functionNames)")
        } else {
            print("   ❌ Failed to compile test shader")
        }
        
        print("")
    }
    
    /// Debug a specific Metal function library
    @MainActor
    public static func debugLibrary(_ library: MTLLibrary?, kernelName: String? = nil) {
        guard let library = library else {
            print("❌ Library is nil")
            return
        }
        
        print("📚 Metal Library Debug:")
        print("   Functions: \(library.functionNames)")
        
        if let kernelName = kernelName {
            if let function = library.makeFunction(name: kernelName) {
                print("   ✅ Function '\(kernelName)' found")
                print("      Type: \(function.functionType)")
            } else {
                print("   ❌ Function '\(kernelName)' not found")
            }
        }
    }
}