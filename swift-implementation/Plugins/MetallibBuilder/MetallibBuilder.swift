import Foundation
import PackagePlugin

@main
struct MetallibBuilder: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        // Only run for the NoesisEngine target
        guard target.name == "NoesisEngine" else {
            return []
        }
        
        // Find Metal shader files from gpt_oss
        let packageRoot = context.package.directoryURL
        let metalSourceDir = packageRoot.appending(path: "gpt_oss/metal/source")
        let includeDir = metalSourceDir.appending(path: "include")
        
        // Check if Metal source directory exists
        guard FileManager.default.fileExists(atPath: metalSourceDir.path) else {
            // If gpt_oss shaders don't exist, skip metallib build
            return []
        }
        
        // Metal shader files to compile
        let metalFiles = [
            "accumulate.metal",
            "convert.metal", 
            "embeddings.metal",
            "matmul.metal",
            "moematmul.metal",
            "random.metal",
            "rmsnorm.metal",
            "rope.metal",
            "sample.metal",
            "sdpa.metal",
            "topk.metal"
        ]
        
        // Output paths
        let outputDir = context.pluginWorkDirectoryURL.appending(path: "MetalLib")
        let metallibPath = outputDir.appending(path: "default.metallib")
        
        // Input and intermediate paths  
        let inputPaths = metalFiles.map { metalSourceDir.appending(path: $0) }
        let airPaths = metalFiles.map { outputDir.appending(path: $0.replacingOccurrences(of: ".metal", with: ".air")) }
        
        var commands: [Command] = []
        
        // Compile each Metal file to AIR
        for i in 0..<metalFiles.count {
            let metalFile = metalFiles[i]
            let metalPath = inputPaths[i]
            let airPath = airPaths[i]
            
            commands.append(.buildCommand(
                displayName: "Compile \(metalFile)",
                executable: URL(filePath: "/usr/bin/xcrun"),
                arguments: [
                    "-sdk", "macosx", 
                    "metal",
                    "-c",
                    "-I", includeDir.path,
                    "-o", airPath.path,
                    metalPath.path
                ],
                inputFiles: [metalPath],
                outputFiles: [airPath]
            ))
        }
        
        // Link AIR files into metallib  
        commands.append(.buildCommand(
            displayName: "Build Metal Library",
            executable: URL(filePath: "/usr/bin/xcrun"),
            arguments: [
                "-sdk", "macosx",
                "metallib"
            ] + airPaths.map(\.path) + [
                "-o", metallibPath.path
            ],
            inputFiles: airPaths,
            outputFiles: [metallibPath]
        ))
        
        return commands
    }
}