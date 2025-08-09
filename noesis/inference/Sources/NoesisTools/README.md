# NoesisTools - Swift GPT-OSS Toolchain

A comprehensive Swift implementation of the GPT-OSS model toolchain, providing native macOS/iOS support for model export, generation, and chat functionality.

## Overview

NoesisTools provides a full Swift replacement for the Python GPT-OSS tools:

- **Model Export**: Convert Hugging Face checkpoints to GPT-OSS Metal format
- **Model Download**: Download models directly from Hugging Face Hub
- **Tokenization**: Native o200k_gptoss tokenizer implementation
- **Text Generation**: Command-line generation with various sampling options
- **Interactive Chat**: Harmony-format chat with full conversation support

## Components

### 1. Model Exporter (`noesis-export`)

Exports Hugging Face checkpoint files to GPT-OSS Metal binary format:

```bash
# Export a local checkpoint
swift run noesis-export path/to/checkpoint output/model.bin

# Download and export from Hugging Face
swift run noesis-export path/to/checkpoint output/model.bin --download --repo gpt-oss/gpt-oss-20b
```

Features:
- Automatic Q/K scaling (factor of 0.5 and 0.25 baked into weights)
- Safetensors parsing
- MXFP4 quantization support (for MoE weights)
- Proper weight layout for Metal shaders

### 2. Text Generation (`noesis-generate`)

Generate text from prompts:

```bash
# Simple generation
swift run noesis-generate model.bin "The meaning of life is"

# With Harmony format
swift run noesis-generate model.bin "Explain quantum computing" --harmony --system "You are a helpful assistant"

# JSON output with statistics
swift run noesis-generate model.bin "Write a poem" --format json --stats
```

Options:
- `--temperature`: Sampling temperature (0.0 = greedy)
- `--max-tokens`: Maximum tokens to generate
- `--top-p`: Top-p sampling threshold
- `--repetition-penalty`: Penalize repeated tokens
- `--harmony`: Use Harmony conversation format
- `--format`: Output format (text/tokens/json)

### 3. Interactive Chat (`noesis-chat`)

Full interactive chat with Harmony format:

```bash
# Start chat session
swift run noesis-chat model.bin

# With custom system prompt
swift run noesis-chat model.bin --system "You are an expert programmer"

# Enable reasoning channels
swift run noesis-chat model.bin --channels --reasoning high
```

Features:
- Multi-turn conversations
- Harmony format with proper special tokens
- Channel support (analysis, commentary, final)
- Reasoning effort levels
- Session statistics

## Key Classes

### `SafetensorsLoader`
Parses and loads safetensors checkpoint files:
```swift
let loader = SafetensorsLoader(fileURL: checkpointURL)
try loader.open()
let weights = try loader.loadBFloat16Tensor(name: "embedding.weight")
```

### `O200kTokenizer`
Native Swift implementation of the o200k_gptoss tokenizer:
```swift
let tokenizer = O200kTokenizer()
let tokens = tokenizer.encode("Hello, world!")
let text = tokenizer.decode(tokens)
```

### `ModelExporter`
Exports checkpoints to GPT-OSS format:
```swift
let exporter = ModelExporter()
let config = ModelExporter.ExportConfig(
    checkpointDir: checkpointURL,
    outputPath: outputURL,
    applyQKScaling: true
)
try exporter.export(config: config)
```

### `HuggingFaceDownloader`
Downloads models from Hugging Face Hub:
```swift
let downloader = HuggingFaceDownloader()
try await downloader.downloadCheckpoint(
    repoId: "gpt-oss/gpt-oss-20b",
    localDir: localURL
)
```

## Building

```bash
# Build all tools
swift build

# Build specific tool
swift build --target NoesisExport

# Build with optimizations
swift build -c release
```

## Testing

```bash
# Run tests
swift test

# Test model export
swift run noesis-export test-checkpoint test-model.bin --no-scaling

# Test generation
swift run noesis-generate test-model.bin "Test prompt" --max-tokens 10
```

## Implementation Notes

### Weight Layout
The GPT-OSS Metal format requires specific weight ordering:
1. Embedding weights
2. Per-block weights (attention, MLP)
3. Final RMSNorm
4. Unembedding weights
5. MoE expert weights

### Q/K Scaling
During export, Q and K projections are scaled:
- Q weights: multiplied by 0.5
- K weights: multiplied by 0.25
- This bakes the 1/sqrt(d_k) scaling into weights

### Special Tokens
The tokenizer includes 16 special tokens for Harmony format:
- `<|start|>`, `<|end|>`, `<|message|>`
- `<|channel|>`, `<|return|>`, `<|call|>`
- `<|untrusted|>`, `<|endofuntrusted|>`
- And others...

### Memory Mapping
Models are memory-mapped for efficient loading:
- Page-aligned weight buffers
- Direct Metal buffer wrapping
- Minimal memory overhead

## Compatibility

- **Platform**: macOS 15.0+ (Apple Silicon)
- **Swift**: 6.1+
- **Dependencies**: ArgumentParser, HarmonyKit
- **Models**: GPT-OSS 20B checkpoints

## Future Improvements

- [ ] MXFP4 quantization for MoE weights
- [ ] Streaming safetensors parsing
- [ ] Parallel weight processing
- [ ] Model sharding support
- [ ] iOS deployment targets
- [ ] Vision model support

## License

Same as NoesisEngine project.