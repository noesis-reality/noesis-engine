#!/bin/bash

# NoesisEngine Rust Library Setup Script
# Supports both developer mode (local repos) and user mode (auto-fetch)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCES_DIR="$PROJECT_ROOT/Sources"

echo "🦀 Setting up Rust dependencies for NoesisEngine"
echo "📁 Project root: $PROJECT_ROOT"

# Check for developer mode
if [[ -n "$NOESIS_DEV_MODE" ]]; then
    echo "🔧 Developer mode enabled"
    
    # Check for local repos (mitramod workspace structure)
    TIKTOKEN_LOCAL="$PROJECT_ROOT/../../tiktoken-swift"
    HARMONY_LOCAL="$PROJECT_ROOT/../../harmony-swift"
    
    if [[ -d "$TIKTOKEN_LOCAL" ]]; then
        echo "📦 Found local tiktoken-swift at $TIKTOKEN_LOCAL"
        cd "$TIKTOKEN_LOCAL"
        cargo build --release --features c_api
        
        # Copy library and header to CTiktoken for linking
        mkdir -p "$SOURCES_DIR/CTiktoken"
        cp "$TIKTOKEN_LOCAL/target/release/libtiktoken.dylib" "$SOURCES_DIR/CTiktoken/"
        cp "$TIKTOKEN_LOCAL/tiktoken_ffi.h" "$SOURCES_DIR/CTiktoken/" 2>/dev/null || echo "⚠️ tiktoken_ffi.h not found, using existing header"
    else
        echo "⚠️ NOESIS_DEV_MODE set but tiktoken-swift not found at $TIKTOKEN_LOCAL"
    fi
    
    if [[ -d "$HARMONY_LOCAL" ]]; then
        echo "📦 Found local harmony-swift at $HARMONY_LOCAL"
        cd "$HARMONY_LOCAL"
        cargo build --release --features c-api
        
        # Copy library and header to CHarmony for linking
        mkdir -p "$SOURCES_DIR/CHarmony"
        cp "$HARMONY_LOCAL/target/release/libopenai_harmony.dylib" "$SOURCES_DIR/CHarmony/"
        cp "$HARMONY_LOCAL/harmony_ffi.h" "$SOURCES_DIR/CHarmony/" 2>/dev/null || echo "⚠️ harmony_ffi.h not found, using existing header"
    else
        echo "⚠️ NOESIS_DEV_MODE set but harmony-swift not found at $HARMONY_LOCAL"
    fi
    
else
    echo "📦 User mode - auto-fetching Rust dependencies"
    
    # Create temp directory for cloning
    TEMP_DIR="$PROJECT_ROOT/.build/rust-deps"
    mkdir -p "$TEMP_DIR"
    
    # Clone tiktoken-swift if needed
    if [[ ! -d "$TEMP_DIR/tiktoken-swift" ]]; then
        echo "🔄 Cloning tiktoken-swift..."
        cd "$TEMP_DIR"
        git clone https://github.com/noesis-reality/tiktoken-swift.git
    fi
    
    # Clone harmony-swift if needed
    if [[ ! -d "$TEMP_DIR/harmony-swift" ]]; then
        echo "🔄 Cloning harmony-swift..."
        cd "$TEMP_DIR"
        git clone https://github.com/noesis-reality/harmony-swift.git
    fi
    
    # Build tiktoken-swift
    echo "🔨 Building tiktoken-swift..."
    cd "$TEMP_DIR/tiktoken-swift"
    cargo build --release --features c_api
    
    # Build harmony-swift
    echo "🔨 Building harmony-swift..."
    cd "$TEMP_DIR/harmony-swift"
    cargo build --release --features c-api
    
    # Copy libraries and headers to source directories
    mkdir -p "$SOURCES_DIR/CTiktoken"
    mkdir -p "$SOURCES_DIR/CHarmony"
    
    cp "$TEMP_DIR/tiktoken-swift/target/release/libtiktoken.dylib" "$SOURCES_DIR/CTiktoken/"
    cp "$TEMP_DIR/tiktoken-swift/tiktoken_ffi.h" "$SOURCES_DIR/CTiktoken/" 2>/dev/null || echo "⚠️ tiktoken_ffi.h not found"
    
    cp "$TEMP_DIR/harmony-swift/target/release/libopenai_harmony.dylib" "$SOURCES_DIR/CHarmony/"
    cp "$TEMP_DIR/harmony-swift/harmony_ffi.h" "$SOURCES_DIR/CHarmony/" 2>/dev/null || echo "⚠️ harmony_ffi.h not found"
fi

echo "✅ Rust libraries setup complete!"
echo "📍 Libraries available at:"
echo "   - $SOURCES_DIR/CTiktoken/libtiktoken.dylib"
echo "   - $SOURCES_DIR/CHarmony/libopenai_harmony.dylib"