// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

// Build configuration - mirrors Gradle approach
let flatbuffersRepoPath = ProcessInfo.processInfo.environment["FLATBUFFERS_REPO_PATH"] ?? "../../flatbuffers"
let useLocalFlatBuffers = ProcessInfo.processInfo.environment["USE_LOCAL_FLATBUFFERS"] == "true" || 
                         FileManager.default.fileExists(atPath: flatbuffersRepoPath)

let package = Package(
    name: "NoesisEngine",
    platforms: [
        .macOS("15.5"),
    ],
    products: [
        // Core libraries
        .library(
            name: "NoesisEngine",
            targets: ["NoesisEngine"]),
        .library(
            name: "NoesisTools",
            targets: ["NoesisTools"]),
        // CLI tools
        .executable(
            name: "noesis-chat",
            targets: ["NoesisChat"]
        ),
        .executable(
            name: "noesis-generate", 
            targets: ["NoesisGenerate"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // FlatBuffers dependency - use local fork if available, otherwise GitHub fork
        useLocalFlatBuffers ? 
            .package(path: flatbuffersRepoPath) :
            .package(url: "https://github.com/ariawisp/flatbuffers.git", from: "24.3.25"),
    ],
    targets: [
        .plugin(
            name: "FlatBuffersBuilder",
            capability: .buildTool(),
            dependencies: [
                .product(name: "FlatBuffers", package: useLocalFlatBuffers ? "flatbuffers" : "flatbuffers")
            ]
        ),
        .systemLibrary(
            name: "CHarmony",
            path: "Sources/CHarmony"
        ),
        .target(
            name: "Harmony",
            dependencies: ["CHarmony"],
            linkerSettings: [
                .unsafeFlags([
                    "-L", "Sources/CHarmony",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Sources/CHarmony",
                    "-Xlinker", "-rpath", "-Xlinker", "@loader_path/../Sources/CHarmony"
                ]),
                .linkedLibrary("openai_harmony")
            ]
        ),
        .target(
            name: "HarmonyProtocol",
            dependencies: [
                .product(name: "FlatBuffers", package: useLocalFlatBuffers ? "flatbuffers" : "flatbuffers")
            ],
            plugins: ["FlatBuffersBuilder"]
        ),
        .target(
            name: "NoesisEngine",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .target(
            name: "NoesisTools",
            dependencies: [
                "Harmony"
            ],
            resources: [
                .copy("README.md")
            ]
        ),
        .target(
            name: "NoesisBridge",
            dependencies: [
                "NoesisEngine", 
                "NoesisTools",
                "HarmonyProtocol"
            ],
            linkerSettings: [
                .linkedFramework("JavaVM", .when(platforms: [.macOS]))
            ]
        ),
        .executableTarget(
            name: "NoesisChat",
            dependencies: [
                "NoesisEngine",
                "NoesisTools",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .executableTarget(
            name: "NoesisGenerate",
            dependencies: [
                "NoesisEngine",
                "NoesisTools",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
    ]
)
