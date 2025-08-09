#!/bin/bash

# Convenience script to run noesis-chat with proper environment

# Change to script's parent directory (swift-implementation)
cd "$(dirname "$0")/.."

# Set library paths
export DYLD_LIBRARY_PATH=Sources/CHarmony:Sources/CTiktoken

# Check if model path provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model-path> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 gpt-oss-20b/metal/model.bin"
    echo "  $0 gpt-oss-20b/metal/model.bin --temperature 0.8 --verbose"
    echo ""
    echo "To download a model:"
    echo "  huggingface-cli download openai/gpt-oss-20b --include \"metal/*\" --local-dir gpt-oss-20b/"
    exit 1
fi

# Check if model exists
if [ ! -f "$1" ]; then
    echo "Error: Model file not found: $1"
    exit 1
fi

# Build if needed
if [ ! -f ".build/release/noesis-chat" ]; then
    echo "Building release version..."
    swift build -c release --product noesis-chat
fi

# Run the chat
exec .build/release/noesis-chat "$@"