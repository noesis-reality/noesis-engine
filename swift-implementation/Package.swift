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
                "NoesisTools"
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
