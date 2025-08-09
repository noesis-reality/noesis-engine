package ai.noesisreality.cli.common

import ai.noesisreality.engine.InferenceResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.encodeToJsonElement

/**
 * Output formatting utilities for CLI commands
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object OutputFormatters {
    
    private val json = Json { prettyPrint = true }
    
    /**
     * Format inference result based on requested output format
     */
    fun formatInferenceResult(
        result: InferenceResult,
        format: String,
        showStats: Boolean = false
    ): String = buildString {
        when (format) {
            "json" -> {
                val jsonResult = JsonObject(mapOf(
                    "text" to JsonPrimitive(result.text),
                    "tokens" to json.encodeToJsonElement(result.tokens),
                    "metadata" to JsonObject(mapOf(
                        "tokensGenerated" to JsonPrimitive(result.tokensGenerated),
                        "timeMs" to JsonPrimitive(result.timeMs),
                        "tokensPerSecond" to JsonPrimitive(result.tokensPerSecond),
                        "gpuMemoryMB" to JsonPrimitive(result.gpuMemoryMB)
                    ))
                ))
                append(json.encodeToString(JsonObject.serializer(), jsonResult))
            }
            
            "tokens" -> {
                appendLine("Generated tokens: ${result.tokens}")
                appendLine("Token count: ${result.tokensGenerated}")
                appendLine("Text output:")
                append(result.text)
            }
            
            else -> append(result.text)
        }
        
        if (showStats && format != "json") {
            appendLine("\n\n📊 Generation Statistics:")
            appendLine("  Tokens: ${result.tokensGenerated}")
            appendLine("  Time: ${result.timeMs}ms")
            appendLine("  Speed: ${"%.1f".format(result.tokensPerSecond)} tok/sec")
            appendLine("  GPU memory: ${result.gpuMemoryMB}MB")
        }
    }
    
    /**
     * Format benchmark results
     */
    fun formatBenchmarkResults(results: List<BenchmarkResult>): String = buildString {
        appendLine("🏆 Benchmark Results:")
        val peakThroughput = results.maxOf { it.tokensPerSecond }
        val avgLatency = results.map { it.avgLatencyMs }.average()
        
        results.forEach { result ->
            appendLine("  ${"%4d".format(result.sequenceLength)} tokens: " +
                "${"%6.1f".format(result.avgLatencyMs)}ms, " +
                "${"%5.1f".format(result.tokensPerSecond)} tok/sec")
        }
        
        appendLine()
        appendLine("📈 Summary:")
        appendLine("  Peak throughput: ${"%.1f".format(peakThroughput)} tokens/sec")
        appendLine("  Average latency: ${"%.1f".format(avgLatency)}ms")
    }
}

// Data class for benchmark results (moved here for better organization)
@kotlinx.serialization.Serializable
data class BenchmarkResult(
    val sequenceLength: Int,
    val avgLatencyMs: Double,
    val avgTokensGenerated: Int,
    val tokensPerSecond: Double
)