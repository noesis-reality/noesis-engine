# NoesisEngine - Swift Implementation

This directory contains the Swift implementation of GPT-OSS, providing Metal-accelerated inference with official OpenAI Harmony tokenization.

## Structure

- **`Package.swift`** - Swift Package Manager configuration
- **`Sources/`** - Swift source code
  - `NoesisEngine/` - Core Metal GPU inference engine
  - `NoesisTools/` - Harmony tokenization integration  
  - `NoesisChat/` - Interactive chat CLI tool
  - `NoesisGenerate/` - Text generation CLI tool
  - `NoesisExport/` - Model export utilities
  - `OfficialHarmony/` - C headers for Harmony FFI
  - `OfficialHarmonyWrapper/` - Swift wrapper for Harmony
- **`Plugins/`** - Swift build plugins
  - `MetallibBuilder/` - Automated Metal shader compilation

## Setup & Building

### Option 1: User Mode (Automatic Dependencies)
For end users who just want to try NoesisEngine:

```bash
cd swift-implementation

# Setup Rust dependencies automatically  
./Scripts/setup-rust-libs.sh

# Build the project
swift build -c release
```

This will automatically:
- Clone tiktoken-swift and harmony-swift from noesis-reality GitHub org
- Build the required Rust libraries with C API features
- Set up all dependencies for Swift linking

### Option 2: Developer Mode (Local Repositories)
For developers working on the forks:

```bash
# Set developer mode
export NOESIS_DEV_MODE=1

cd swift-implementation

# Setup using local repositories (must be co-located)
./Scripts/setup-rust-libs.sh

# Build the project  
swift build -c release
```

Developer mode expects the repository structure:
```
your-workspace/
├── noesis-engine/swift-implementation/  # This project
├── tiktoken-swift/                      # Your local tiktoken fork
└── harmony-swift/                       # Your local harmony fork
```

## Running

```bash
# Interactive chat
swift run noesis-chat

# Generate text  
swift run noesis-generate

# Export models
swift run noesis-export
```

## Requirements

- **macOS 15.5+** - Required for Metal 4 GPU acceleration
- **Metal-capable Mac** - For GPU inference acceleration

## Dependencies

- **Metal 4** - GPU acceleration with pre-compiled metallib support
- **Harmony tokenization** - Official OpenAI tokenizer via C FFI
- **ArgumentParser** - Command-line interface

> **Note:** iPad/iPhone support planned for future release when Metal 4 becomes available on iOS platforms.

This implementation provides the same functionality as the Python reference implementation in `../` but optimized for Apple platforms with Metal GPU acceleration.