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

## Quick Start

```bash
# Test your setup
./Scripts/test-noesis.sh

# Download a model (40GB)
huggingface-cli download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/

# Run chat
./Scripts/run-chat.sh gpt-oss-20b/metal/model.bin
```

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

### Download a Model

First, download the GPT-OSS model weights from Hugging Face:

```bash
# For GPT-OSS 20b (recommended for testing, ~40GB)
huggingface-cli download openai/gpt-oss-20b \
    --include "metal/*" \
    --local-dir gpt-oss-20b/

# For GPT-OSS 120b (requires more memory, ~240GB)
huggingface-cli download openai/gpt-oss-120b \
    --include "metal/*" \
    --local-dir gpt-oss-120b/
```

### Run the CLIs

#### Interactive Chat (noesis-chat)

```bash
# Set library paths for Rust FFI
export DYLD_LIBRARY_PATH=Sources/CHarmony:Sources/CTiktoken

# Run interactive chat
.build/debug/noesis-chat gpt-oss-20b/metal/model.bin

# Or with release build for better performance
.build/release/noesis-chat gpt-oss-20b/metal/model.bin

# With options
.build/release/noesis-chat gpt-oss-20b/metal/model.bin \
    --temperature 0.8 \
    --max-tokens 200 \
    --verbose
```

#### Text Generation (noesis-generate)

```bash
# Simple generation
.build/release/noesis-generate gpt-oss-20b/metal/model.bin \
    "Why did the chicken cross the road?" \
    --max-tokens 50

# With system prompt and JSON output
.build/release/noesis-generate gpt-oss-20b/metal/model.bin \
    "Explain quantum computing" \
    --system "You are a physics professor" \
    --format json \
    --stats

# Using Harmony format
.build/release/noesis-generate gpt-oss-20b/metal/model.bin \
    "Write a haiku about coding" \
    --harmony \
    --temperature 0.9
```

### Available Options

#### noesis-chat
- `--system` - Custom system prompt
- `--temperature` - Sampling temperature (0.0-1.0, default: 0.7)
- `--max-tokens` - Maximum tokens per response (default: 500)
- `--reasoning` - Reasoning effort level (low/medium/high)
- `--channels` - Enable reasoning channels
- `--stats` - Show generation statistics
- `--verbose` - Verbose output

#### noesis-generate
- `--max-tokens` - Maximum tokens to generate (default: 100)
- `--temperature` - Sampling temperature (default: 0.7)
- `--top-p` - Nucleus sampling threshold (default: 0.9)
- `--repetition-penalty` - Repetition penalty (default: 1.1)
- `--system` - System prompt (optional)
- `--format` - Output format: text, tokens, or json
- `--harmony` - Use Harmony conversation format
- `--stats` - Show generation statistics

## Requirements

- **macOS 15.5+** - Required for Metal 4 GPU acceleration
- **Metal-capable Mac** - For GPU inference acceleration

## Dependencies

- **Metal 4** - GPU acceleration with pre-compiled metallib support
- **Harmony tokenization** - Official OpenAI tokenizer via C FFI
- **ArgumentParser** - Command-line interface

> **Note:** iPad/iPhone support planned for future release when Metal 4 becomes available on iOS platforms.

This implementation provides the same functionality as the Python reference implementation in `../` but optimized for Apple platforms with Metal GPU acceleration.