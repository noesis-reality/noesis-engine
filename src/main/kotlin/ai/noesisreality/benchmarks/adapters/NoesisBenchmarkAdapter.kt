package ai.noesisreality.benchmarks.adapters

import ai.noesisreality.benchmarks.BenchmarkAdapter
import ai.noesisreality.benchmarks.BenchmarkConfig
import ai.noesisreality.benchmarks.BenchmarkRun
import ai.noesisreality.protocol.StreamingInferenceEngine
import kotlin.system.measureTimeMillis

/**
 * Benchmark adapter for Noesis Engine using Kotlin + Swift Metal 4 implementation
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class NoesisBenchmarkAdapter(
    private val modelPath: String? = null
) : BenchmarkAdapter {
    
    private lateinit var inferenceEngine: StreamingInferenceEngine
    private var isInitialized = false
    
    override suspend fun warmup(config: BenchmarkConfig) {
        if (!isInitialized) {
            // Initialize Noesis engine with specified model
            inferenceEngine = StreamingInferenceEngine()
            isInitialized = true
            
            // Warmup run
            val warmupResult = inferenceEngine.generateSingleShot(
                prompt = "Hello",
                maxTokens = 10,
                temperature = 0.7f
            )
            
            // Verify engine is working
            if (warmupResult.tokensGenerated == 0) {
                throw RuntimeException("Noesis engine warmup failed - no tokens generated")
            }
        }
    }
    
    override suspend fun runInference(config: BenchmarkConfig): BenchmarkRun {
        if (!isInitialized) {
            throw IllegalStateException("Noesis adapter not warmed up")
        }
        
        val startTime = System.currentTimeMillis()
        var tokensGenerated = 0
        var gpuMemoryMB = 0
        
        val inferenceTime = measureTimeMillis {
            val result = inferenceEngine.generateSingleShot(
                prompt = config.prompt,
                maxTokens = config.maxTokens,
                temperature = config.temperature
            )
            
            tokensGenerated = result.tokensGenerated
            gpuMemoryMB = result.gpuMemoryMB
        }
        
        val endTime = System.currentTimeMillis()
        val latency = endTime - startTime
        val tokensPerSecond = if (inferenceTime > 0) {
            (tokensGenerated * 1000.0) / inferenceTime
        } else 0.0
        
        return BenchmarkRun(
            tokensGenerated = tokensGenerated,
            inferenceTimeMs = inferenceTime,
            tokensPerSecond = tokensPerSecond,
            latencyMs = latency.toDouble(),
            memoryUsageMB = getCurrentMemoryUsage(),
            gpuUsageMB = gpuMemoryMB
        )
    }
    
    override fun isAvailable(): Boolean {
        // Check if Swift Metal libraries are available
        return try {
            StreamingInferenceEngine()
            true
        } catch (e: Exception) {
            false
        }
    }
    
    override fun getDisplayName(): String = "Noesis Engine (Kotlin + Swift Metal 4)"
    
    override suspend fun cleanup() {
        if (isInitialized) {
            try {
                // Release any resources held by the inference engine
                // For now, just mark as uninitialized to force reinitialization
                // TODO: Add proper resource cleanup when StreamingInferenceEngine supports it
                isInitialized = false
            } catch (e: Exception) {
                // Ignore cleanup errors but log them
                println("Warning: Noesis engine cleanup failed: ${e.message}")
            }
        }
    }
    
    private fun getCurrentMemoryUsage(): Int {
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val usedMemory = totalMemory - freeMemory
        return (usedMemory / 1024 / 1024).toInt()
    }
}