#!/bin/bash

# Copy dynamic libraries to build output directory

BUILD_DIR="${1:-.build/release}"

# Ensure build directory exists
mkdir -p "$BUILD_DIR"

# Copy harmony library (includes embedded tiktoken)
if [ -f "Sources/CHarmony/libopenai_harmony.dylib" ]; then
    cp "Sources/CHarmony/libopenai_harmony.dylib" "$BUILD_DIR/"
    echo "✅ Copied libopenai_harmony.dylib to $BUILD_DIR"
fi

# Update rpath for executables
for exe in noesis-generate noesis-chat noesis-export; do
    if [ -f "$BUILD_DIR/$exe" ]; then
        # Add @executable_path to rpath
        install_name_tool -add_rpath @executable_path "$BUILD_DIR/$exe" 2>/dev/null || true
        echo "✅ Updated rpath for $exe"
    fi
done

echo "✅ Libraries copied and rpaths updated"