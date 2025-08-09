package ai.noesisreality.engine

import ai.noesisreality.core.NoesisConstants
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Noesis Inference Engine - Pure LLM inference interface
 * 
 * This class provides a clean Kotlin interface to the Swift Metal 4 GPT-OSS implementation.
 * The Swift layer handles ONLY the low-level inference operations - model loading,
 * tokenization, and text generation. All higher-level logic (sessions, diagnostics,
 * model management) is handled in Kotlin.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object NoesisInferenceEngine {
    
    private var instance: NoesisInferenceEngine? = null
    private var isInitialized = false
    private var verbose = false
    private var currentModelPath: String? = null
    
    init {
        try {
            System.loadLibrary(NoesisConstants.NativeLibs.SWIFT_INFERENCE)
            isInitialized = true
        } catch (e: UnsatisfiedLinkError) {
            System.err.println("${NoesisConstants.Emojis.ERROR} Failed to load Noesis inference library: ${e.message}")
            System.err.println("Run './gradlew buildNativeLibrary' to build the Swift inference engine")
        }
    }
    
    // Native method declarations - minimal interface to Swift
    private external fun nativeInitialize(modelPath: String?, verbose: Boolean): String
    private external fun nativeGenerate(requestJson: String): String
    private external fun nativeGetModelInfo(): String
    private external fun nativeShutdown()
    
    /**
     * Configure the inference engine
     */
    fun configure(verbose: Boolean = false, modelPath: String? = null, configPath: String? = null) {
        this.verbose = verbose
        this.currentModelPath = modelPath
        
        if (verbose) {
            println("🔧 Configuring Noesis Inference Engine")
            modelPath?.let { println("   Model: $it") }
            configPath?.let { println("   Config: $it") }
        }
    }
    
    /**
     * Get singleton instance of the inference engine
     */
    fun getInstance(): NoesisInferenceEngine {
        if (!isInitialized) {
            throw RuntimeException("Noesis inference engine not available - native library not loaded")
        }
        
        if (instance == null) {
            synchronized(this) {
                if (instance == null) {
                    // Initialize the Swift backend
                    val initResult = nativeInitialize(currentModelPath, verbose)
                    if (verbose) {
                        println("🚀 Swift inference engine initialized: $initResult")
                    }
                    instance = NoesisInferenceEngine
                }
            }
        }
        
        return instance!!
    }
    
    /**
     * Core text generation - the only thing Swift handles
     */
    suspend fun generate(
        prompt: String,
        systemPrompt: String? = null,
        maxTokens: Int = 100,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        repetitionPenalty: Float = 1.1f,
        reasoningEffort: String = "medium",
        seed: Int? = null
    ): InferenceResult {
        if (!isInitialized) {
            throw RuntimeException("Inference engine not initialized")
        }
        
        val request = InferenceRequest(
            prompt = prompt,
            systemPrompt = systemPrompt,
            maxTokens = maxTokens,
            temperature = temperature,
            topP = topP,
            repetitionPenalty = repetitionPenalty,
            reasoningEffort = reasoningEffort,
            seed = seed
        )
        
        val requestJson = Json.encodeToString(InferenceRequest.serializer(), request)
        val resultJson = nativeGenerate(requestJson)
        
        return try {
            Json.decodeFromString(InferenceResult.serializer(), resultJson)
        } catch (e: Exception) {
            // Handle error responses from Swift
            InferenceResult(
                text = "",
                tokens = emptyList(),
                tokensGenerated = 0,
                timeMs = 0,
                tokensPerSecond = 0.0,
                gpuMemoryMB = 0,
                error = "Generation failed: ${e.message}"
            )
        }
    }
    
    /**
     * Get basic model information from Swift
     */
    suspend fun getModelInfo(): ModelInfo {
        if (!isInitialized) {
            throw RuntimeException("Inference engine not initialized")
        }
        
        val infoJson = nativeGetModelInfo()
        return Json.decodeFromString(ModelInfo.serializer(), infoJson)
    }
    
    /**
     * Check if verbose logging is enabled
     */
    fun isVerbose(): Boolean = verbose
    
    /**
     * Shutdown the inference engine
     */
    fun shutdown() {
        if (isInitialized && instance != null) {
            nativeShutdown()
            instance = null
            
            if (verbose) {
                println("🛑 Swift inference engine shut down")
            }
        }
    }
}

/**
 * Request format for Swift inference engine
 */
@Serializable
data class InferenceRequest(
    val prompt: String,
    val systemPrompt: String? = null,
    val maxTokens: Int = 100,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val repetitionPenalty: Float = 1.1f,
    val reasoningEffort: String = "medium",
    val seed: Int? = null
)

/**
 * Result format from Swift inference engine
 */
@Serializable
data class InferenceResult(
    val text: String,
    val tokens: List<Int>,
    val tokensGenerated: Int,
    val timeMs: Long,
    val tokensPerSecond: Double,
    val gpuMemoryMB: Int,
    val error: String? = null
)

/**
 * Basic model information from Swift
 */
@Serializable
data class ModelInfo(
    val name: String,
    val parameters: Long,
    val contextLength: Int,
    val vocabularySize: Int,
    val embeddingDim: Int,
    val numBlocks: Int,
    val isLoaded: Boolean
)