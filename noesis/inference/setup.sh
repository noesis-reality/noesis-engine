#!/bin/bash

# Noesis Engine - One-command setup script
# Usage: ./setup.sh [--model 20b|120b] [--skip-model]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_SIZE="20b"
SKIP_MODEL=false
NOESIS_HOME="${HOME}/.noesis"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--model 20b|120b] [--skip-model]"
            echo ""
            echo "Options:"
            echo "  --model SIZE    Download model (20b or 120b, default: 20b)"
            echo "  --skip-model    Skip model download"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Noesis Engine - One-Command Setup   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/6]${NC} Checking prerequisites..."

# Check for Swift
if ! command -v swift &> /dev/null; then
    echo -e "${RED}❌ Swift is not installed${NC}"
    exit 1
fi

# Check for Rust/Cargo
if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}⚠️  Cargo not found. Installing Rust...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Check for Metal
if ! swift -e 'import Metal; print(MTLCreateSystemDefaultDevice() != nil ? "ok" : "fail")' 2>/dev/null | grep -q "ok"; then
    echo -e "${RED}❌ Metal is not available on this system${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"

# Step 2: Setup Rust libraries
echo ""
echo -e "${YELLOW}[2/6]${NC} Building Rust dependencies..."

if [ -f "Scripts/setup-rust-libs.sh" ]; then
    ./Scripts/setup-rust-libs.sh
else
    echo -e "${RED}❌ setup-rust-libs.sh not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Rust libraries built${NC}"

# Step 3: Build Swift project
echo ""
echo -e "${YELLOW}[3/6]${NC} Building Noesis Engine..."

swift build -c release --product noesis-chat
swift build -c release --product noesis-generate

echo -e "${GREEN}✅ Build complete${NC}"

# Step 4: Setup configuration
echo ""
echo -e "${YELLOW}[4/6]${NC} Setting up configuration..."

mkdir -p "${NOESIS_HOME}"
mkdir -p "${NOESIS_HOME}/models"

# Create config file if it doesn't exist
CONFIG_FILE="${NOESIS_HOME}/noesis.config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" << EOF
{
  "models": {
    "default": "gpt-oss-${MODEL_SIZE}",
    "paths": {
      "gpt-oss-20b": "${NOESIS_HOME}/models/gpt-oss-20b/metal/model.bin",
      "gpt-oss-120b": "${NOESIS_HOME}/models/gpt-oss-120b/metal/model.bin"
    }
  },
  "generation": {
    "defaultTemperature": 0.7,
    "defaultMaxTokens": 500,
    "defaultTopP": 0.9
  },
  "chat": {
    "systemPrompt": "You are a helpful AI assistant.",
    "reasoningLevel": "medium",
    "enableChannels": false,
    "showStats": false
  }
}
EOF
    echo -e "${GREEN}✅ Created config at $CONFIG_FILE${NC}"
else
    echo -e "${GREEN}✅ Config already exists at $CONFIG_FILE${NC}"
fi

# Step 5: Download model (optional)
if [ "$SKIP_MODEL" = false ]; then
    echo ""
    echo -e "${YELLOW}[5/6]${NC} Downloading GPT-OSS ${MODEL_SIZE} model..."
    
    MODEL_DIR="${NOESIS_HOME}/models/gpt-oss-${MODEL_SIZE}"
    MODEL_BIN="${MODEL_DIR}/metal/model.bin"
    
    if [ -f "$MODEL_BIN" ]; then
        echo -e "${GREEN}✅ Model already downloaded${NC}"
    else
        # Install hf cli if needed
        if ! command -v hf &> /dev/null; then
            echo "Installing hf cli..."
            pip3 install -q huggingface-hub[cli]
        fi
        
        echo "Downloading model (this may take a while)..."
        hf download "openai/gpt-oss-${MODEL_SIZE}" \
            --include "metal/*" \
            --local-dir "$MODEL_DIR"
        
        echo -e "${GREEN}✅ Model downloaded to $MODEL_DIR${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}[5/6]${NC} Skipping model download (--skip-model flag set)"
fi

# Step 6: Create convenience scripts
echo ""
echo -e "${YELLOW}[6/6]${NC} Creating convenience scripts..."

# Create noesis-chat wrapper
cat > "${NOESIS_HOME}/noesis-chat" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOESIS_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
exec "${NOESIS_DIR}/.build/release/noesis-chat" "$@"
EOF
chmod +x "${NOESIS_HOME}/noesis-chat"

# Create noesis-generate wrapper
cat > "${NOESIS_HOME}/noesis-generate" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOESIS_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
exec "${NOESIS_DIR}/.build/release/noesis-generate" "$@"
EOF
chmod +x "${NOESIS_HOME}/noesis-generate"

echo -e "${GREEN}✅ Created wrapper scripts${NC}"

# Done!
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Setup Complete! 🎉            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Quick start:"
echo ""
if [ "$SKIP_MODEL" = false ]; then
    echo "  # Interactive chat"
    echo "  .build/release/noesis-chat"
    echo ""
    echo "  # Generate text"
    echo "  .build/release/noesis-generate \"Tell me a joke\""
else
    echo "  # Download a model first:"
    echo "  hf download openai/gpt-oss-20b --include \"metal/*\" --local-dir ~/.noesis/models/gpt-oss-20b"
    echo ""
    echo "  # Then run:"
    echo "  .build/release/noesis-chat ~/.noesis/models/gpt-oss-20b/metal/model.bin"
fi
echo ""
echo "Configuration file: ${CONFIG_FILE}"
echo "Models directory: ${NOESIS_HOME}/models/"
echo ""
echo "For more options, run:"
echo "  .build/release/noesis-chat --help"
echo "  .build/release/noesis-generate --help"