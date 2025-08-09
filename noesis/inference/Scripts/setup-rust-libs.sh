#!/bin/bash

# NoesisEngine Rust Library Setup Script
# Sets up harmony-swift (which includes embedded tiktoken functionality)
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
    HARMONY_LOCAL="$PROJECT_ROOT/../../harmony-swift"
    
    # Note: tiktoken-swift is no longer needed - harmony-swift has tiktoken embedded
    
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
    
    # Clone harmony-swift if needed
    if [[ ! -d "$TEMP_DIR/harmony-swift" ]]; then
        echo "🔄 Cloning harmony-swift..."
        cd "$TEMP_DIR"
        git clone https://github.com/noesis-reality/harmony-swift.git
    fi
    
    # Build harmony-swift (includes tiktoken functionality)
    echo "🔨 Building harmony-swift..."
    cd "$TEMP_DIR/harmony-swift"
    cargo build --release --features c-api
    
    # Copy library and header to source directory
    mkdir -p "$SOURCES_DIR/CHarmony"
    
    cp "$TEMP_DIR/harmony-swift/target/release/libopenai_harmony.dylib" "$SOURCES_DIR/CHarmony/"
    cp "$TEMP_DIR/harmony-swift/harmony_ffi.h" "$SOURCES_DIR/CHarmony/" 2>/dev/null || echo "⚠️ harmony_ffi.h not found"
fi

echo "✅ Rust library setup complete!"
echo "📍 Harmony library (with embedded tiktoken) available at:"
echo "   - $SOURCES_DIR/CHarmony/libopenai_harmony.dylib"
echo ""
echo "Note: tiktoken functionality is now provided through harmony-swift,"
echo "      which has tiktoken embedded. No separate tiktoken-swift needed."