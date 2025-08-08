// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NoesisEngine",
    platforms: [
        .macOS("15.5"), // Metal 4 requirement - minimum for GPU inference
        // TODO: Add iPad/iPhone support when Metal 4 becomes available on iOS
    ],
    products: [
        // Core libraries
        .library(
            name: "NoesisEngine",
            targets: ["NoesisEngine"]),
        .library(
            name: "NoesisTools",
            targets: ["NoesisTools"]),
        // CLI tools matching Python gpt-oss
        .executable(
            name: "noesis-chat",
            targets: ["NoesisChat"]
        ),
        .executable(
            name: "noesis-generate", 
            targets: ["NoesisGenerate"]
        ),
        .executable(
            name: "noesis-export",
            targets: ["NoesisExport"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .systemLibrary(
            name: "CHarmony",
            path: "Sources/CHarmony"
        ),
        .systemLibrary(
            name: "CTiktoken",
            path: "Sources/CTiktoken"
        ),
        .target(
            name: "Harmony",
            dependencies: ["CHarmony"],
            linkerSettings: [
                .unsafeFlags(["-L", "Sources/CHarmony"]),
                .linkedLibrary("openai_harmony")
            ]
        ),
        .target(
            name: "NoesisEngine",
            dependencies: [],
            resources: [
                // Fallback: individual shaders for development/debugging
                .copy("../../../gpt_oss/metal/source/accumulate.metal"),
                .copy("../../../gpt_oss/metal/source/convert.metal"),
                .copy("../../../gpt_oss/metal/source/embeddings.metal"),
                .copy("../../../gpt_oss/metal/source/matmul.metal"),
                .copy("../../../gpt_oss/metal/source/moematmul.metal"),
                .copy("../../../gpt_oss/metal/source/random.metal"),
                .copy("../../../gpt_oss/metal/source/rmsnorm.metal"),
                .copy("../../../gpt_oss/metal/source/rope.metal"),
                .copy("../../../gpt_oss/metal/source/sample.metal"),
                .copy("../../../gpt_oss/metal/source/sdpa.metal"),
                .copy("../../../gpt_oss/metal/source/topk.metal"),
                .copy("../../../gpt_oss/metal/source/include")
            ],
            plugins: ["MetallibBuilder"]
        ),
        .target(
            name: "NoesisTools",
            dependencies: [
                "Harmony",
                "CTiktoken"
            ],
            resources: [
                .copy("README.md")
            ],
            linkerSettings: [
                .unsafeFlags(["-L", "Sources/CTiktoken"]),
                .linkedLibrary("tiktoken")
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
        .executableTarget(
            name: "NoesisExport",
            dependencies: [
                "NoesisTools",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        
        // Build plugins
        .plugin(
            name: "MetallibBuilder",
            capability: .buildTool(),
            dependencies: []
        ),
        .plugin(
            name: "RustDependencyBuilder",
            capability: .buildTool(),
            dependencies: []
        ),
    ]
)
