package ai.noesisreality.cli.common

import ai.noesisreality.harmony.HarmonyEngine

/**
 * Utility for building prompts with Harmony encoding
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object HarmonyPromptBuilder {
    
    /**
     * Build a standard prompt using Harmony encoding
     */
    fun buildStandardPrompt(
        userPrompt: String,
        systemPrompt: String? = null
    ): String = HarmonyEngine.getInstance().createEncoder().use { encoder ->
        val harmonyTokens = encoder.renderPrompt(
            systemMessage = systemPrompt,
            userMessage = userPrompt,
            assistantPrefix = null
        )
        encoder.decode(harmonyTokens.tokens)
    }
    
    /**
     * Build a structured reasoning prompt using Harmony
     */
    fun buildReasoningPrompt(
        task: String,
        reasoningLevel: String,
        context: String? = null
    ): String = HarmonyEngine.getInstance().createEncoder().use { encoder ->
        val reasoningContext = encoder.createReasoningContext(
            task = task,
            context = context,
            reasoningLevel = reasoningLevel.toHarmonyReasoningLevel()
        )
        val harmonyTokens = reasoningContext.generateStructuredPrompt()
        encoder.decode(harmonyTokens.tokens)
    }
}