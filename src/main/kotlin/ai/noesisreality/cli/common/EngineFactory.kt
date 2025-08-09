package ai.noesisreality.cli.common

import ai.noesisreality.engine.NoesisInferenceEngine
import ai.noesisreality.harmony.HarmonyEngine
import ai.noesisreality.harmony.HarmonyReasoningLevel

/**
 * Factory for creating and configuring engines
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object EngineFactory {
    
    /**
     * Initialize both inference and harmony engines with common configuration
     */
    fun initialize(
        verbose: Boolean,
        modelPath: String? = null,
        configPath: String? = null
    ) {
        NoesisInferenceEngine.configure(
            verbose = verbose,
            modelPath = modelPath,
            configPath = configPath
        )
        
        HarmonyEngine.configure(verbose = verbose)
        
        if (verbose) {
            println("🚀 Noesis engines initialized")
            println("   • Swift Metal 4 inference engine")
            println("   • Rust Harmony encoding engine (required for GPT-OSS)")
        }
    }
    
    /**
     * Get both engines as a pair
     */
    fun getEngines(): Pair<NoesisInferenceEngine, HarmonyEngine> =
        NoesisInferenceEngine.getInstance() to HarmonyEngine.getInstance()
}

/**
 * Extension function to convert string reasoning effort to enum
 */
fun String.toHarmonyReasoningLevel(): HarmonyReasoningLevel = when (this) {
    "low" -> HarmonyReasoningLevel.LOW
    "high" -> HarmonyReasoningLevel.HIGH
    else -> HarmonyReasoningLevel.MEDIUM
}