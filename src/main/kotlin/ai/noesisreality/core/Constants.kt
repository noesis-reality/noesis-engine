package ai.noesisreality.core

/**
 * Shared constants for the Noesis GPT-OSS application
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object NoesisConstants {
    
    // Application metadata
    const val COMPANY_NAME = "Noesis Reality LLC"
    const val DOMAIN = "noesisreality.ai"
    const val APP_NAME = "Noesis GPT-OSS CLI"
    const val VERSION = "1.0.0"
    
    // Model specifications
    object Models {
        const val GPT_OSS_20B = "gpt-oss-20b"
        const val GPT_OSS_120B = "gpt-oss-120b"
        
        const val GPT_OSS_20B_SIZE_GB = 13.75
        const val GPT_OSS_120B_SIZE_GB = 45.5
        
        const val GPT_OSS_20B_PARAMS = 20_000_000_000L
        const val GPT_OSS_120B_PARAMS = 120_000_000_000L
        
        const val DEFAULT_CONTEXT_LENGTH = 8192
        const val DEFAULT_VOCAB_SIZE = 50257
        const val MAGIC_HEADER = "GPT-OSS v1.0"
        
        // HuggingFace repositories
        const val HF_REPO_20B = "noesis-ai/gpt-oss-20b"
        const val HF_REPO_120B = "noesis-ai/gpt-oss-120b"
        const val HF_FILENAME = "metal/model.bin"
    }
    
    // Default configuration values
    object Defaults {
        const val TEMPERATURE = 0.7f
        const val TOP_P = 0.9f
        const val REPETITION_PENALTY = 1.1f
        const val MAX_TOKENS = 100
        const val REASONING_EFFORT = "medium"
        const val VERBOSE = false
        const val TIMEOUT_MS = 30000
        
        // Memory recommendations
        const val RECOMMENDED_RAM_20B_GB = 16
        const val RECOMMENDED_RAM_120B_GB = 32
        const val SYSTEM_RESERVED_RAM_GB = 4
    }
    
    // File system paths
    object Paths {
        const val MODELS_DIR = ".noesis/models"
        const val METAL_SUBDIR = "metal"
        const val MODEL_FILENAME = "model.bin"
        const val METADATA_FILENAME = "metadata.json"
    }
    
    // Error messages
    object ErrorMessages {
        const val HARMONY_REQUIRED = "Harmony library is required for GPT-OSS models"
        const val MODEL_NOT_FOUND = "No model found. Use --model-path or ensure model is in ~/.noesis/models/"
        const val INVALID_MODEL_FORMAT = "Invalid file format - not a GPT-OSS model"
        const val INFERENCE_ENGINE_NOT_INITIALIZED = "Inference engine not initialized"
    }
    
    // CLI formatting
    object Emojis {
        const val ROCKET = "🚀"
        const val BRAIN = "🧠"
        const val GEAR = "🔧"
        const val CHECKMARK = "✅"
        const val WARNING = "⚠️"
        const val ERROR = "❌"
        const val SEARCH = "🔍"
        const val LIGHTNING = "⚡"
        const val CHART = "📊"
        const val PACKAGE = "📦"
        const val COMPUTER = "💻"
        const val GPU = "🎮"
        const val MEMORY = "💾"
        const val DOWNLOAD = "⬇️"
        const val CHAT = "💬"
        const val TARGET = "🎯"
        const val FLAG = "🏁"
    }
    
    // Native library names
    object NativeLibs {
        const val SWIFT_INFERENCE = "noesis_inference"
        const val RUST_HARMONY = "openai_harmony"
    }
    
    // Benchmarking configuration
    object Benchmarks {
        const val DEFAULT_ITERATIONS = 5
        const val QUICK_ITERATIONS = 3
        const val QUICK_MAX_TOKENS = 50
        const val DEFAULT_DELAY_MS = 1000L
        const val QUICK_DELAY_MS = 500L
        const val WARMUP_TOKENS = 10
        const val TIMEOUT_SECONDS = 120
        const val WARMUP_TIMEOUT_SECONDS = 30
        
        // Benchmark report formatting
        const val REPORT_WIDTH = 60
        const val ENGINE_COLUMN_WIDTH = 15
        const val METRIC_COLUMN_WIDTH = 12
    }
    
    // FlatBuffers configuration
    object FlatBuffers {
        const val DEFAULT_BUFFER_SIZE = 1024
        const val STREAMING_BUFFER_SIZE = 8192
        const val MESSAGE_TIMEOUT_MS = 5000L
        const val BATCH_READ_SIZE = 4096
    }
    
    // Engine-specific configurations
    object Engines {
        object Noesis {
            const val DEFAULT_STREAMING_BATCH_SIZE = 1
            const val GPU_MEMORY_DEFAULT_MB = 1024
        }
        
        object GptOss {
            const val PYTHON_COMMAND = "python3"
            const val PROCESS_TIMEOUT_SECONDS = 120L
            const val DEFAULT_PATH = "../gpt_oss"
        }
        
        object LlamaCpp {
            val SEARCH_PATHS = listOf(
                "/usr/local/bin/llama-cpp",
                "/usr/local/bin/llama", 
                "/usr/local/bin/main",
                "./llama.cpp/main",
                "./llama.cpp/llama",
                "/opt/homebrew/bin/llama",
                "/opt/homebrew/bin/main"
            )
            const val DEFAULT_THREADS = -1 // Use all available
            const val DEFAULT_BATCH_SIZE = 512
            const val GPU_LAYERS_ALL = 99
        }
    }
    
    // Environment and build configuration
    object Environment {
        val isDebug: Boolean = System.getProperty("debug") == "true"
        val isVerbose: Boolean = System.getProperty("verbose") == "true"
        val isDevelopment: Boolean = System.getProperty("env") == "development"
        
        fun getModelPath(): String = System.getProperty("model.path") 
            ?: "\${user.home}/${Paths.MODELS_DIR}"
            
        fun getTempDir(): String = System.getProperty("java.io.tmpdir")
    }
}