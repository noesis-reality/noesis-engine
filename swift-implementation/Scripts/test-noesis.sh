#!/bin/bash

# Test script for NoesisEngine Swift implementation

set -e

# Change to script's parent directory (swift-implementation)
cd "$(dirname "$0")/.."

echo "🧪 NoesisEngine Test Suite"
echo "=========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Metal availability
echo "1. Checking Metal support..."
if swift -e 'import Metal; print(MTLCreateSystemDefaultDevice() != nil ? "✅" : "❌")' 2>/dev/null | grep -q "✅"; then
    echo -e "${GREEN}✅ Metal is available${NC}"
else
    echo -e "${RED}❌ Metal is not available${NC}"
    exit 1
fi

# Check libraries
echo ""
echo "2. Checking FFI libraries..."
if [ -f "Sources/CHarmony/libopenai_harmony.dylib" ] && [ -f "Sources/CTiktoken/libtiktoken.dylib" ]; then
    echo -e "${GREEN}✅ FFI libraries found${NC}"
else
    echo -e "${YELLOW}⚠️  FFI libraries missing. Running setup...${NC}"
    ./Scripts/setup-rust-libs.sh
fi

# Build the project
echo ""
echo "3. Building NoesisEngine..."
if swift build --product noesis-chat > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Test help command
echo ""
echo "4. Testing CLI help..."
if .build/debug/noesis-chat --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ CLI help works${NC}"
else
    echo -e "${RED}❌ CLI help failed${NC}"
    exit 1
fi

# Check for models
echo ""
echo "5. Checking for models..."
MODEL_FOUND=false
if [ -f "gpt-oss-20b/metal/model.bin" ]; then
    echo -e "${GREEN}✅ GPT-OSS 20b model found${NC}"
    MODEL_FOUND=true
elif [ -f "gpt-oss-120b/metal/model.bin" ]; then
    echo -e "${GREEN}✅ GPT-OSS 120b model found${NC}"
    MODEL_FOUND=true
else
    echo -e "${YELLOW}⚠️  No models found${NC}"
    echo ""
    echo "To download a model, run:"
    echo "  huggingface-cli download openai/gpt-oss-20b --include \"metal/*\" --local-dir gpt-oss-20b/"
fi

echo ""
echo "=========================="
echo -e "${GREEN}✨ Test suite complete!${NC}"
echo ""

if [ "$MODEL_FOUND" = true ]; then
    echo "Ready to run chat! Use:"
    echo "  export DYLD_LIBRARY_PATH=Sources/CHarmony:Sources/CTiktoken"
    echo "  .build/debug/noesis-chat <model-path>"
else
    echo "Download a model first, then run:"
    echo "  export DYLD_LIBRARY_PATH=Sources/CHarmony:Sources/CTiktoken"
    echo "  .build/debug/noesis-chat <model-path>"
fi