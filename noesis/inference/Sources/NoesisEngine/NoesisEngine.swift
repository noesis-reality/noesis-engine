import Foundation
import Metal

public class NoesisEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "NoesisEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Metal is not available"])
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "NoesisEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = commandQueue
        
        // Load pre-compiled metallib
        self.library = try Self.loadMetalLibrary(device: device)
    }
    
    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        // First try to load from the module bundle (SPM resources)
        if let metallibURL = Bundle.noesisEngine.url(forResource: "default", withExtension: "metallib") {
            do {
                let library = try device.makeLibrary(URL: metallibURL)
                print("✅ Loaded Metal library from bundle resources")
                print("   Available functions: \(library.functionNames.prefix(5).joined(separator: ", "))\(library.functionNames.count > 5 ? ", ..." : "")")
                return library
            } catch {
                print("⚠️ Failed to load metallib from bundle: \(error)")
            }
        }
        
        // Fallback: Try multiple locations for the metallib
        let searchPaths = [
            // In the bundle (for packaged apps)
            Bundle.main.bundleURL.appendingPathComponent("default.metallib"),
            // Next to the executable
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("default.metallib"),
            // In NoesisEngine bundle (SPM resource bundles)
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("NoesisEngine_NoesisEngine.bundle/default.metallib"),
            // Build directory locations
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("../NoesisEngine_NoesisEngine.bundle/default.metallib"),
            // Relative to current directory
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".build/arm64-apple-macosx/release/NoesisEngine_NoesisEngine.bundle/default.metallib"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".build/arm64-apple-macosx/debug/NoesisEngine_NoesisEngine.bundle/default.metallib"),
            // Resources directory
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Sources/NoesisEngine/Resources/default.metallib")
        ]
        
        // Try each path
        for url in searchPaths {
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    let library = try device.makeLibrary(URL: url)
                    print("✅ Loaded Metal library from: \(url.lastPathComponent)")
                    print("   Available functions: \(library.functionNames.prefix(5).joined(separator: ", "))\(library.functionNames.count > 5 ? ", ..." : "")")
                    return library
                } catch {
                    print("⚠️ Failed to load metallib from \(url.path): \(error)")
                }
            }
        }
        
        // If no pre-compiled metallib found, throw error
        throw NSError(domain: "NoesisEngine", code: -1, userInfo: [
            NSLocalizedDescriptionKey: "Could not find or load default.metallib. Searched paths: \(searchPaths.map { $0.path }.joined(separator: ", "))"
        ])
    }
    
    public func createPipeline() throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: "example_kernel") else {
            throw NSError(domain: "NoesisEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Function not found"])
        }
        return try device.makeComputePipelineState(function: function)
    }
}

// Keep this for backward compatibility but it's not used anymore
public struct MetalLibraryLoader {
    public static func makeLibrariesFromEmbeddedShaders(device: MTLDevice) throws -> [MTLLibrary] {
        let engine = try NoesisEngine()
        return [engine.library]
    }
    
    public static func makeLibrariesFromSource(device: MTLDevice) throws -> [MTLLibrary] {
        let engine = try NoesisEngine()
        return [engine.library]
    }
}