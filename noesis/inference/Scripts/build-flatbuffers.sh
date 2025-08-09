#!/bin/bash
# Build FlatBuffers Swift components - mirrors Gradle approach

set -e

# Configuration from environment or defaults
FLATBUFFERS_REPO_PATH="${FLATBUFFERS_REPO_PATH:-../../flatbuffers}"
USE_LOCAL_FLATBUFFERS="${USE_LOCAL_FLATBUFFERS:-false}"

# Determine if we should use local FlatBuffers
if [[ "$USE_LOCAL_FLATBUFFERS" == "true" || -d "$FLATBUFFERS_REPO_PATH" ]]; then
    echo "🔨 Using local FlatBuffers fork at: $FLATBUFFERS_REPO_PATH"
    export USE_LOCAL_FLATBUFFERS=true
    export FLATBUFFERS_REPO_PATH="$FLATBUFFERS_REPO_PATH"
    
    # Build local FlatBuffers if needed
    if [[ ! -f "$FLATBUFFERS_REPO_PATH/flatc" ]]; then
        echo "🔨 Building local FlatBuffers fork..."
        cd "$FLATBUFFERS_REPO_PATH"
        cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .
        make -j$(nproc 2>/dev/null || echo 4)
        cd - > /dev/null
    fi
else
    echo "🔨 Using GitHub FlatBuffers fork: ariawisp/flatbuffers"
    export USE_LOCAL_FLATBUFFERS=false
fi

# Generate Swift FlatBuffers code manually (since plugin might not work initially)
echo "🔨 Generating Swift FlatBuffers code for Harmony protocol..."

# Find flatc
if [[ "$USE_LOCAL_FLATBUFFERS" == "true" ]]; then
    FLATC_PATH="$FLATBUFFERS_REPO_PATH/flatc"
else
    # Try to use system flatc or download it
    FLATC_PATH=$(which flatc 2>/dev/null || echo "")
    if [[ -z "$FLATC_PATH" ]]; then
        echo "⚠️  flatc not found. Please install FlatBuffers or set USE_LOCAL_FLATBUFFERS=true"
        echo "   Or run: brew install flatbuffers"
        exit 1
    fi
fi

# Output directory
OUTPUT_DIR="Sources/HarmonyProtocol/Generated"
mkdir -p "$OUTPUT_DIR"

# Generate Swift code for each schema
for schema in schemas/*.fbs; do
    if [[ -f "$schema" ]]; then
        echo "   Generating Swift code for: $(basename "$schema")"
        "$FLATC_PATH" --swift \
                      --swift-package-name FlatBuffers \
                      --gen-mutable \
                      --gen-object-api \
                      -o "$OUTPUT_DIR" \
                      "$schema"
    fi
done

echo "✅ Swift FlatBuffers generation complete"

# Build the Swift package
echo "🔨 Building Swift package with FlatBuffers..."
swift build -c release

echo "✅ Swift build complete"