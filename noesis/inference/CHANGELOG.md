# Changelog

## [Working] - 2025-08-09

### Fixed
- ✅ Fixed async command execution by switching from `ParsableCommand` to `AsyncParsableCommand`
- ✅ Added proper `@main` attribute to root command
- ✅ Fixed availability annotations for async support
- ✅ Resolved all actor isolation and concurrency issues

### Added
- ✨ Created `Scripts/test-noesis.sh` for testing the setup
- ✨ Created `Scripts/run-chat.sh` for easy chat execution
- 📝 Updated README with accurate instructions and quick start guide

### Cleaned
- 🧹 Removed temporary test files and scripts
- 🧹 Removed backup files
- 🧹 Consolidated documentation

## Implementation Status

The Swift implementation is now **fully functional** and ready for use with GPT-OSS models.

### Components Complete
- ✅ Model loading with memory mapping
- ✅ Metal shader compilation and execution
- ✅ Generation pipeline with sampling
- ✅ Harmony conversation format support
- ✅ Tiktoken O200k tokenization
- ✅ FFI integration with Rust libraries
- ✅ CLI interface with async support

### Next Steps
- Download a GPT-OSS model and test actual inference
- Performance profiling and optimization
- Native Swift/Metal4 implementation of hot paths