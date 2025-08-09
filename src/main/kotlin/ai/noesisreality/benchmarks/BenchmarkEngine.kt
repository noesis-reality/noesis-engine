package ai.noesisreality.benchmarks

import ai.noesisreality.core.NoesisConstants
import kotlinx.coroutines.*
import kotlinx.serialization.Serializable
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

/**
 * Comprehensive benchmarking engine for comparing inference performance across:
 * - Noesis Engine (Kotlin + Swift Metal 4)
 * - GPT-OSS Reference (Python/C/Metal) 
 * - llama.cpp (C++)
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class BenchmarkEngine {
    
    private val benchmarkAdapters = mutableMapOf<EngineType, BenchmarkAdapter>()
    
    fun registerAdapter(engineType: EngineType, adapter: BenchmarkAdapter) {
        benchmarkAdapters[engineType] = adapter
    }
    
    suspend fun runComparisonBenchmark(
        config: BenchmarkConfig,
        engines: List<EngineType> = EngineType.values().toList()
    ): BenchmarkReport = withContext(Dispatchers.Default) {
        
        val startTime = System.currentTimeMillis()
        val results = mutableMapOf<EngineType, EngineResults>()
        
        println("${NoesisConstants.Emojis.ROCKET} Starting benchmark comparison...")
        println("Prompt: \"${config.prompt.take(50)}${if (config.prompt.length > 50) "..." else ""}\"")
        println("Engines: ${engines.joinToString(", ")}")
        println("Iterations: ${config.iterations}")
        println("Max tokens: ${config.maxTokens}")
        println()
        
        for (engine in engines) {
            val adapter = benchmarkAdapters[engine]
            if (adapter == null) {
                println("${NoesisConstants.Emojis.WARNING} No adapter found for $engine, skipping...")
                continue
            }
            
            println("${NoesisConstants.Emojis.GEAR} Benchmarking $engine...")
            
            try {
                val engineResult = runEnginebenchmark(adapter, config)
                results[engine] = engineResult
                
                println("${NoesisConstants.Emojis.CHECKMARK} $engine completed")
                println("  Average: ${String.format("%.2f", engineResult.averageTokensPerSecond)} tokens/sec")
                println("  Best: ${String.format("%.2f", engineResult.bestTokensPerSecond)} tokens/sec")
                println("  Total tokens: ${engineResult.totalTokens}")
                println()
                
            } catch (e: Exception) {
                println("${NoesisConstants.Emojis.ERROR} $engine failed: ${e.message}")
                println()
            } finally {
                // Always cleanup after each engine to prevent memory leaks and interference
                try {
                    println("  🧹 Cleaning up $engine...")
                    adapter.cleanup()
                    System.gc() // Request garbage collection
                    delay(1000) // Allow time for cleanup
                } catch (cleanupError: Exception) {
                    println("  ⚠️  Cleanup warning for $engine: ${cleanupError.message}")
                }
            }
        }
        
        val totalTime = System.currentTimeMillis() - startTime
        
        BenchmarkReport(
            config = config,
            results = results,
            totalBenchmarkTimeMs = totalTime,
            timestamp = LocalDateTime.now()
        )
    }
    
    private suspend fun runEnginebenchmark(
        adapter: BenchmarkAdapter, 
        config: BenchmarkConfig
    ): EngineResults {
        
        val runs = mutableListOf<BenchmarkRun>()
        var totalTokens = 0
        
        // Warmup run
        println("  ${NoesisConstants.Emojis.LIGHTNING} Warmup...")
        try {
            adapter.warmup(config)
        } catch (e: Exception) {
            println("  ${NoesisConstants.Emojis.WARNING} Warmup failed: ${e.message}")
        }
        
        // Benchmark runs
        for (i in 1..config.iterations) {
            print("  Run $i/${config.iterations}... ")
            
            val runResult = adapter.runInference(config)
            runs.add(runResult)
            totalTokens += runResult.tokensGenerated
            
            println("${String.format("%.2f", runResult.tokensPerSecond)} tokens/sec")
            
            if (i < config.iterations) {
                delay(config.delayBetweenRunsMs)
            }
        }
        
        return EngineResults(
            runs = runs,
            totalTokens = totalTokens,
            averageTokensPerSecond = runs.map { it.tokensPerSecond }.average(),
            bestTokensPerSecond = runs.maxOf { it.tokensPerSecond },
            worstTokensPerSecond = runs.minOf { it.tokensPerSecond },
            standardDeviation = calculateStandardDeviation(runs.map { it.tokensPerSecond }),
            averageLatency = runs.map { it.latencyMs }.average(),
            totalInferenceTimeMs = runs.sumOf { it.inferenceTimeMs }
        )
    }
    
    private fun calculateStandardDeviation(values: List<Double>): Double {
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance)
    }
    
    fun generateReport(report: BenchmarkReport): String {
        val sb = StringBuilder()
        
        sb.appendLine("${NoesisConstants.Emojis.CHART} NOESIS INFERENCE BENCHMARK REPORT")
        sb.appendLine("=" * 60)
        sb.appendLine()
        sb.appendLine("Timestamp: ${report.timestamp.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)}")
        sb.appendLine("Prompt: ${report.config.prompt}")
        sb.appendLine("Max Tokens: ${report.config.maxTokens}")
        sb.appendLine("Temperature: ${report.config.temperature}")
        sb.appendLine("Iterations: ${report.config.iterations}")
        sb.appendLine("Total Benchmark Time: ${String.format("%.2f", report.totalBenchmarkTimeMs / 1000.0)}s")
        sb.appendLine()
        
        // Performance comparison table
        sb.appendLine("PERFORMANCE COMPARISON")
        sb.appendLine("-" * 60)
        sb.appendLine(String.format("%-15s %-12s %-12s %-12s %-8s", 
            "Engine", "Avg tok/s", "Best tok/s", "Worst tok/s", "StdDev"))
        sb.appendLine("-" * 60)
        
        val sortedResults = report.results.toList().sortedByDescending { it.second.averageTokensPerSecond }
        
        for ((engine, results) in sortedResults) {
            sb.appendLine(String.format("%-15s %-12.2f %-12.2f %-12.2f %-8.2f",
                engine.displayName,
                results.averageTokensPerSecond,
                results.bestTokensPerSecond,
                results.worstTokensPerSecond,
                results.standardDeviation
            ))
        }
        
        sb.appendLine()
        
        // Relative performance analysis
        if (sortedResults.size >= 2) {
            sb.appendLine("RELATIVE PERFORMANCE")
            sb.appendLine("-" * 30)
            
            val (bestEngine, bestResults) = sortedResults.first()
            sb.appendLine("${NoesisConstants.Emojis.TARGET} Best: ${bestEngine.displayName} (${String.format("%.2f", bestResults.averageTokensPerSecond)} tok/s)")
            
            for ((engine, results) in sortedResults.drop(1)) {
                val relativeDiff = ((bestResults.averageTokensPerSecond - results.averageTokensPerSecond) / results.averageTokensPerSecond) * 100
                sb.appendLine("${engine.displayName}: ${String.format("%.1f", relativeDiff)}% slower than ${bestEngine.displayName}")
            }
            sb.appendLine()
        }
        
        // Detailed statistics per engine
        for ((engine, results) in report.results) {
            sb.appendLine("DETAILED STATS: ${engine.displayName}")
            sb.appendLine("-" * 30)
            sb.appendLine("Total Tokens Generated: ${results.totalTokens}")
            sb.appendLine("Total Inference Time: ${String.format("%.2f", results.totalInferenceTimeMs / 1000.0)}s")
            sb.appendLine("Average Latency: ${String.format("%.2f", results.averageLatency)}ms")
            sb.appendLine("Tokens/Second Range: ${String.format("%.2f", results.worstTokensPerSecond)} - ${String.format("%.2f", results.bestTokensPerSecond)}")
            sb.appendLine("Consistency (1σ): ±${String.format("%.2f", results.standardDeviation)} tok/s")
            sb.appendLine()
        }
        
        // System info
        sb.appendLine("SYSTEM INFORMATION")
        sb.appendLine("-" * 30)
        sb.appendLine("OS: ${System.getProperty("os.name")} ${System.getProperty("os.version")}")
        sb.appendLine("Architecture: ${System.getProperty("os.arch")}")
        sb.appendLine("JVM: ${System.getProperty("java.version")}")
        sb.appendLine("Available Processors: ${Runtime.getRuntime().availableProcessors()}")
        sb.appendLine("Max Memory: ${Runtime.getRuntime().maxMemory() / 1024 / 1024}MB")
        sb.appendLine()
        
        return sb.toString()
    }
    
    fun saveReport(report: BenchmarkReport, outputDir: File = File("benchmark-results")) {
        outputDir.mkdirs()
        
        val timestamp = report.timestamp.format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
        val reportFile = File(outputDir, "benchmark-$timestamp.txt")
        val jsonFile = File(outputDir, "benchmark-$timestamp.json")
        
        // Save human-readable report
        reportFile.writeText(generateReport(report))
        
        // Save machine-readable JSON
        // jsonFile.writeText(Json.encodeToString(report))
        
        println("${NoesisConstants.Emojis.PACKAGE} Reports saved:")
        println("  Text: ${reportFile.absolutePath}")
        println("  JSON: ${jsonFile.absolutePath}")
    }
}

enum class EngineType(val displayName: String) {
    NOESIS("Noesis Engine"),
    GPT_OSS_REFERENCE("GPT-OSS Reference"),
    LLAMA_CPP("llama.cpp")
}

@Serializable
data class BenchmarkConfig(
    val prompt: String,
    val maxTokens: Int = 100,
    val temperature: Float = 0.7f,
    val iterations: Int = 5,
    val delayBetweenRunsMs: Long = 1000,
    val modelPath: String? = null,
    val useGPU: Boolean = true,
    val contextLength: Int = 2048
)

@Serializable
data class BenchmarkRun(
    val tokensGenerated: Int,
    val inferenceTimeMs: Long,
    val tokensPerSecond: Double,
    val latencyMs: Double,
    val memoryUsageMB: Int = 0,
    val gpuUsageMB: Int = 0
)

@Serializable 
data class EngineResults(
    val runs: List<BenchmarkRun>,
    val totalTokens: Int,
    val averageTokensPerSecond: Double,
    val bestTokensPerSecond: Double,
    val worstTokensPerSecond: Double,
    val standardDeviation: Double,
    val averageLatency: Double,
    val totalInferenceTimeMs: Long
)

@Serializable
data class BenchmarkReport(
    val config: BenchmarkConfig,
    val results: Map<EngineType, EngineResults>,
    val totalBenchmarkTimeMs: Long,
    val timestampString: String
) {
    @kotlinx.serialization.Transient
    val timestamp: LocalDateTime = if (timestampString.isNotEmpty()) {
        LocalDateTime.parse(timestampString)
    } else LocalDateTime.now()
    
    constructor(
        config: BenchmarkConfig,
        results: Map<EngineType, EngineResults>,
        totalBenchmarkTimeMs: Long,
        timestamp: LocalDateTime
    ) : this(
        config = config,
        results = results,
        totalBenchmarkTimeMs = totalBenchmarkTimeMs,
        timestampString = timestamp.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
    )
}

interface BenchmarkAdapter {
    suspend fun warmup(config: BenchmarkConfig)
    suspend fun runInference(config: BenchmarkConfig): BenchmarkRun
    suspend fun cleanup()
    fun isAvailable(): Boolean
    fun getDisplayName(): String
}

private operator fun String.times(n: Int): String = repeat(n)