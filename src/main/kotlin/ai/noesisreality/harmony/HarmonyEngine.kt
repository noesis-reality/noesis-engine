package ai.noesisreality.harmony

import ai.noesisreality.core.NoesisConstants
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Harmony Engine - Direct Kotlin interface to Rust Harmony library
 * 
 * Provides structured reasoning, multi-modal processing, and advanced prompt formatting
 * capabilities from OpenAI's Harmony library. This complements the Swift inference engine
 * by handling the advanced reasoning and encoding layers.
 * 
 * Key Capabilities:
 * - Structured reasoning and thinking
 * - Multi-modal prompt encoding
 * - Advanced tokenization with Harmony formatting
 * - Multi-channel stream management (future)
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object HarmonyEngine {
    
    private var instance: HarmonyEngine? = null
    private var isInitialized = false
    private var verbose = false
    
    init {
        try {
            System.loadLibrary(NoesisConstants.NativeLibs.RUST_HARMONY)
            isInitialized = true
        } catch (e: UnsatisfiedLinkError) {
            System.err.println("${NoesisConstants.Emojis.ERROR} CRITICAL: ${NoesisConstants.ErrorMessages.HARMONY_REQUIRED}")
            System.err.println("Failed to load native library: ${e.message}")
            System.err.println("Build the Rust Harmony library with: ./gradlew buildRustHarmonyBridge")
            throw RuntimeException(NoesisConstants.ErrorMessages.HARMONY_REQUIRED, e)
        }
    }
    
    // Native method declarations - direct interface to Rust Harmony
    internal external fun nativeCreateEncoder(): Long
    internal external fun nativeFreeEncoder(encoderPtr: Long)
    internal external fun nativeEncodePlain(encoderPtr: Long, text: String): IntArray?
    internal external fun nativeRenderPrompt(
        encoderPtr: Long,
        systemMessage: String?,
        userMessage: String,
        assistantPrefix: String?
    ): IntArray?
    internal external fun nativeDecode(encoderPtr: Long, tokens: IntArray): String?
    internal external fun nativeGetStopTokens(encoderPtr: Long): IntArray?
    
    /**
     * Configure the Harmony engine
     */
    fun configure(verbose: Boolean = false) {
        this.verbose = verbose
        
        if (verbose) {
            println("🔧 Configuring Harmony Engine")
        }
    }
    
    /**
     * Get singleton instance - Harmony is required for GPT-OSS
     */
    fun getInstance(): HarmonyEngine {
        // Harmony is initialized in init block and will throw if not available
        if (instance == null) {
            synchronized(this) {
                if (instance == null) {
                    instance = HarmonyEngine
                }
            }
        }
        
        return instance!!
    }
    
    /**
     * Create a new Harmony encoder instance
     */
    fun createEncoder(): HarmonyEncoder {
        val encoderPtr = nativeCreateEncoder()
        if (encoderPtr == 0L) {
            throw RuntimeException("Failed to create Harmony encoder - required for GPT-OSS models")
        }
        
        if (verbose) {
            println("🔤 Created Harmony encoder")
        }
        
        return HarmonyEncoder(encoderPtr, verbose)
    }
    
    /**
     * Check if verbose logging is enabled
     */
    fun isVerbose(): Boolean = verbose
}

/**
 * Harmony Encoder - handles tokenization and prompt formatting
 */
class HarmonyEncoder(
    private val encoderPtr: Long,
    private val verbose: Boolean = false
) : AutoCloseable {
    
    private var isClosed = false
    
    /**
     * Encode plain text without Harmony formatting
     */
    fun encodePlain(text: String): HarmonyTokens {
        checkNotClosed()
        
        if (verbose) {
            println("🔤 Encoding plain text: \"${text.take(50)}...\"")
        }
        
        val tokens = HarmonyEngine.nativeEncodePlain(encoderPtr, text)
            ?: throw HarmonyException("Failed to encode plain text")
        
        return HarmonyTokens(
            tokens = tokens,
            text = text,
            type = HarmonyTokenType.PLAIN
        )
    }
    
    /**
     * Render a structured prompt with Harmony formatting
     */
    fun renderPrompt(
        systemMessage: String? = null,
        userMessage: String,
        assistantPrefix: String? = null
    ): HarmonyTokens {
        checkNotClosed()
        
        if (verbose) {
            println("🔤 Rendering Harmony prompt")
            systemMessage?.let { println("   System: \"${it.take(50)}...\"") }
            println("   User: \"${userMessage.take(50)}...\"")
            assistantPrefix?.let { println("   Assistant prefix: \"$it\"") }
        }
        
        val tokens = HarmonyEngine.nativeRenderPrompt(
            encoderPtr,
            systemMessage,
            userMessage,
            assistantPrefix
        ) ?: throw HarmonyException("Failed to render Harmony prompt")
        
        val fullText = buildString {
            systemMessage?.let { append("System: $it\n\n") }
            append("User: $userMessage\n\n")
            assistantPrefix?.let { append("Assistant: $it") } ?: append("Assistant:")
        }
        
        return HarmonyTokens(
            tokens = tokens,
            text = fullText,
            type = HarmonyTokenType.STRUCTURED_PROMPT,
            systemMessage = systemMessage,
            userMessage = userMessage,
            assistantPrefix = assistantPrefix
        )
    }
    
    /**
     * Decode tokens back to text
     */
    fun decode(tokens: IntArray): String {
        checkNotClosed()
        
        return HarmonyEngine.nativeDecode(encoderPtr, tokens)
            ?: throw HarmonyException("Failed to decode tokens")
    }
    
    /**
     * Get stop tokens for this encoder
     */
    fun getStopTokens(): IntArray {
        checkNotClosed()
        
        return HarmonyEngine.nativeGetStopTokens(encoderPtr)
            ?: throw HarmonyException("Failed to get stop tokens")
    }
    
    /**
     * Create a reasoning context for structured thinking
     */
    fun createReasoningContext(
        task: String,
        context: String? = null,
        reasoningLevel: HarmonyReasoningLevel = HarmonyReasoningLevel.MEDIUM
    ): HarmonyReasoningContext {
        checkNotClosed()
        
        return HarmonyReasoningContext(
            encoder = this,
            task = task,
            context = context,
            reasoningLevel = reasoningLevel,
            verbose = verbose
        )
    }
    
    private fun checkNotClosed() {
        if (isClosed) {
            throw HarmonyException("Encoder has been closed")
        }
    }
    
    override fun close() {
        if (!isClosed) {
            HarmonyEngine.nativeFreeEncoder(encoderPtr)
            isClosed = true
            
            if (verbose) {
                println("🔤 Harmony encoder closed")
            }
        }
    }
}

/**
 * Harmony Reasoning Context - manages structured thinking processes
 */
class HarmonyReasoningContext(
    private val encoder: HarmonyEncoder,
    val task: String,
    val context: String?,
    val reasoningLevel: HarmonyReasoningLevel,
    private val verbose: Boolean = false
) {
    
    private val reasoningSteps = mutableListOf<HarmonyReasoningStep>()
    
    /**
     * Add a reasoning step to the context
     */
    fun addReasoningStep(
        thought: String,
        analysis: String? = null,
        conclusion: String? = null
    ): HarmonyReasoningStep {
        val step = HarmonyReasoningStep(
            stepNumber = reasoningSteps.size + 1,
            thought = thought,
            analysis = analysis,
            conclusion = conclusion,
            timestamp = System.currentTimeMillis()
        )
        
        reasoningSteps.add(step)
        
        if (verbose) {
            println("🧠 Reasoning step ${step.stepNumber}: ${thought.take(60)}...")
        }
        
        return step
    }
    
    /**
     * Generate a structured prompt with reasoning context
     */
    fun generateStructuredPrompt(): HarmonyTokens {
        val systemMessage = buildString {
            append("You are an AI assistant capable of structured reasoning.\n\n")
            append("Task: $task\n")
            context?.let { append("Context: $it\n") }
            append("Reasoning Level: ${reasoningLevel.description}\n\n")
            
            if (reasoningSteps.isNotEmpty()) {
                append("Previous Reasoning Steps:\n")
                reasoningSteps.forEach { step ->
                    append("${step.stepNumber}. ${step.thought}\n")
                    step.analysis?.let { append("   Analysis: $it\n") }
                    step.conclusion?.let { append("   Conclusion: $it\n") }
                }
                append("\n")
            }
        }
        
        return encoder.renderPrompt(
            systemMessage = systemMessage,
            userMessage = "Please continue with structured reasoning for this task.",
            assistantPrefix = "I'll think through this step by step:\n\n<thinking>"
        )
    }
    
    fun getReasoningSteps(): List<HarmonyReasoningStep> = reasoningSteps.toList()
}

// Data classes for Harmony operations
@Serializable
data class HarmonyTokens(
    val tokens: IntArray,
    val text: String,
    val type: HarmonyTokenType,
    val systemMessage: String? = null,
    val userMessage: String? = null,
    val assistantPrefix: String? = null,
    val metadata: Map<String, String> = emptyMap()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as HarmonyTokens

        if (!tokens.contentEquals(other.tokens)) return false
        if (text != other.text) return false
        if (type != other.type) return false

        return true
    }

    override fun hashCode(): Int {
        var result = tokens.contentHashCode()
        result = 31 * result + text.hashCode()
        result = 31 * result + type.hashCode()
        return result
    }
}

@Serializable
enum class HarmonyTokenType {
    PLAIN,
    STRUCTURED_PROMPT,
    REASONING_CONTEXT,
    MULTI_MODAL
}

@Serializable
enum class HarmonyReasoningLevel(val description: String) {
    LOW("Basic reasoning with simple steps"),
    MEDIUM("Structured reasoning with analysis"),
    HIGH("Deep reasoning with detailed analysis and verification")
}

@Serializable
data class HarmonyReasoningStep(
    val stepNumber: Int,
    val thought: String,
    val analysis: String? = null,
    val conclusion: String? = null,
    val timestamp: Long
)

class HarmonyException(message: String, cause: Throwable? = null) : Exception(message, cause)