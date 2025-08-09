import PackagePlugin
import Foundation

@main
struct FlatBuffersBuilder: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        guard let target = target as? SourceModuleTarget else {
            return []
        }
        
        let schemasPath = context.package.directory.appending(["schemas"])
        let outputPath = context.pluginWorkDirectory.appending("Generated")
        
        // Find all .fbs files in schemas directory
        let fbsFiles = try FileManager.default.contentsOfDirectory(atPath: schemasPath.string)
            .filter { $0.hasSuffix(".fbs") }
            .map { schemasPath.appending($0) }
        
        guard !fbsFiles.isEmpty else {
            return []
        }
        
        // Get flatc executable path
        let flatcPath = try context.tool(named: "flatc").path
        
        var commands: [Command] = []
        
        for fbsFile in fbsFiles {
            let outputFile = outputPath.appending("\(fbsFile.stem)_generated.swift")
            
            let command = Command.buildCommand(
                displayName: "Generating Swift FlatBuffers code for \(fbsFile.lastComponent)",
                executable: flatcPath,
                arguments: [
                    "--swift",
                    "--swift-package-name", "FlatBuffers",
                    "--gen-mutable",
                    "--gen-object-api",
                    "-o", outputPath.string,
                    fbsFile.string
                ],
                inputFiles: [fbsFile],
                outputFiles: [outputFile]
            )
            
            commands.append(command)
        }
        
        return commands
    }
}