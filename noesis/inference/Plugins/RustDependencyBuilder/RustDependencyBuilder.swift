import Foundation
import PackagePlugin

/// Build plugin that automatically clones and builds Rust dependencies
/// Supports developer mode (uses local repos) and user mode (auto-fetches)
@main
struct RustDependencyBuilder: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        var commands: [Command] = []
        
        let workingDirectory = context.pluginWorkDirectory
        let packageDirectory = context.package.directory
        
        // Check for developer mode (NOESIS_DEV_MODE environment variable)
        let isDeveloperMode = ProcessInfo.processInfo.environment["NOESIS_DEV_MODE"] != nil
        
        print("🦀 RustDependencyBuilder: Developer mode = \(isDeveloperMode)")
        
        if isDeveloperMode {
            print("🔧 Using local Rust repositories for development")
            commands.append(contentsOf: try createDeveloperModeCommands(
                context: context,
                packageDirectory: packageDirectory,
                workingDirectory: workingDirectory
            ))
        } else {
            print("📦 Auto-fetching and building Rust dependencies")
            commands.append(contentsOf: try createUserModeCommands(
                context: context,
                packageDirectory: packageDirectory,
                workingDirectory: workingDirectory
            ))
        }
        
        return commands
    }
    
    // MARK: - Developer Mode (Local Repositories)
    
    private func createDeveloperModeCommands(
        context: PluginContext,
        packageDirectory: Path,
        workingDirectory: Path
    ) throws -> [Command] {
        var commands: [Command] = []
        
        // Check if local repos exist
        let tiktokenPath = packageDirectory.appending(subpath: "../../../tiktoken-swift")
        let harmonyPath = packageDirectory.appending(subpath: "../../../harmony-swift")
        
        // Build tiktoken-swift locally
        if FileManager.default.fileExists(atPath: tiktokenPath.string) {
            commands.append(.buildCommand(
                displayName: "Build local tiktoken-swift",
                executable: Path("/bin/sh"),
                arguments: [
                    "-c",
                    "cd '\(tiktokenPath.string)' && cargo build --release --features c_api"
                ],
                environment: [:],
                inputFiles: [
                    tiktokenPath.appending("Cargo.toml"),
                    tiktokenPath.appending("src")
                ],
                outputFiles: [
                    tiktokenPath.appending("target/release/libtiktoken.dylib")
                ]
            ))
        } else {
            print("⚠️ Local tiktoken-swift not found at \(tiktokenPath.string)")
        }
        
        // Build harmony-swift locally
        if FileManager.default.fileExists(atPath: harmonyPath.string) {
            commands.append(.buildCommand(
                displayName: "Build local harmony-swift",
                executable: Path("/bin/sh"),
                arguments: [
                    "-c",
                    "cd '\(harmonyPath.string)' && cargo build --release --features c-api"
                ],
                environment: [:],
                inputFiles: [
                    harmonyPath.appending("Cargo.toml"),
                    harmonyPath.appending("src")
                ],
                outputFiles: [
                    harmonyPath.appending("target/release/libopenai_harmony.dylib")
                ]
            ))
        } else {
            print("⚠️ Local harmony-swift not found at \(harmonyPath.string)")
        }
        
        return commands
    }
    
    // MARK: - User Mode (Auto-fetch Repositories)
    
    private func createUserModeCommands(
        context: PluginContext,
        packageDirectory: Path,
        workingDirectory: Path
    ) throws -> [Command] {
        var commands: [Command] = []
        
        let rustDepsDir = workingDirectory.appending("rust-deps")
        let tiktokenDir = rustDepsDir.appending("tiktoken-swift")
        let harmonyDir = rustDepsDir.appending("harmony-swift")
        
        // Create Rust dependencies directory
        commands.append(.buildCommand(
            displayName: "Create Rust deps directory",
            executable: Path("/bin/mkdir"),
            arguments: ["-p", rustDepsDir.string],
            environment: [:],
            inputFiles: [],
            outputFiles: [rustDepsDir]
        ))
        
        // Clone tiktoken-swift if not exists
        commands.append(.buildCommand(
            displayName: "Clone tiktoken-swift",
            executable: Path("/bin/sh"),
            arguments: [
                "-c",
                "cd '\(rustDepsDir.string)' && git clone https://github.com/noesis-reality/tiktoken-swift.git tiktoken-swift"
            ],
            environment: [:],
            inputFiles: [],
            outputFiles: [tiktokenDir]
        ))
        
        // Clone harmony-swift if not exists
        commands.append(.buildCommand(
            displayName: "Clone harmony-swift", 
            executable: Path("/bin/sh"),
            arguments: [
                "-c",
                "cd '\(rustDepsDir.string)' && git clone https://github.com/noesis-reality/harmony-swift.git harmony-swift"
            ],
            environment: [:],
            inputFiles: [],
            outputFiles: [harmonyDir]
        ))
        
        // Build tiktoken-swift
        commands.append(.buildCommand(
            displayName: "Build tiktoken-swift",
            executable: Path("/bin/sh"),
            arguments: [
                "-c",
                "cd '\(tiktokenDir.string)' && cargo build --release --features c_api"
            ],
            environment: [:],
            inputFiles: [
                tiktokenDir.appending("Cargo.toml"),
                tiktokenDir.appending("src")
            ],
            outputFiles: [
                tiktokenDir.appending("target/release/libtiktoken.dylib")
            ]
        ))
        
        // Build harmony-swift
        commands.append(.buildCommand(
            displayName: "Build harmony-swift",
            executable: Path("/bin/sh"),
            arguments: [
                "-c",
                "cd '\(harmonyDir.string)' && cargo build --release --features c-api"
            ],
            environment: [:],
            inputFiles: [
                harmonyDir.appending("Cargo.toml"),
                harmonyDir.appending("src")
            ],
            outputFiles: [
                harmonyDir.appending("target/release/libopenai_harmony.dylib")
            ]
        ))
        
        // Copy built libraries to expected locations
        let libsOutputDir = workingDirectory.appending("libs")
        
        commands.append(.buildCommand(
            displayName: "Copy built Rust libraries",
            executable: Path("/bin/sh"),
            arguments: [
                "-c",
                """
                mkdir -p "\(libsOutputDir.string)" && \\
                cp "\(tiktokenDir.string)/target/release/libtiktoken.dylib" "\(libsOutputDir.string)/" && \\
                cp "\(harmonyDir.string)/target/release/libopenai_harmony.dylib" "\(libsOutputDir.string)/"
                """
            ],
            environment: [:],
            inputFiles: [
                tiktokenDir.appending("target/release/libtiktoken.dylib"),
                harmonyDir.appending("target/release/libopenai_harmony.dylib")
            ],
            outputFiles: [
                libsOutputDir.appending("libtiktoken.dylib"),
                libsOutputDir.appending("libopenai_harmony.dylib")
            ]
        ))
        
        return commands
    }
}