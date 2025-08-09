package ai.noesisreality.benchmarks.adapters

import ai.noesisreality.benchmarks.BenchmarkAdapter
import ai.noesisreality.benchmarks.BenchmarkConfig
import ai.noesisreality.benchmarks.BenchmarkRun
import ai.noesisreality.core.NoesisConstants
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.system.measureTimeMillis

/**
 * Benchmark adapter for llama.cpp implementation
 * 
 * This adapter runs llama.cpp's main executable and parses its output
 * to extract performance metrics for comparison with Noesis and GPT-OSS.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class LlamaCppBenchmarkAdapter(
    private val llamaCppPath: String = NoesisConstants.Engines.LlamaCpp.SEARCH_PATHS.first(),
    private val modelPath: String? = null,
    private val useGpu: Boolean = true,
    private val threads: Int = if (NoesisConstants.Engines.LlamaCpp.DEFAULT_THREADS == -1) 
        Runtime.getRuntime().availableProcessors() else NoesisConstants.Engines.LlamaCpp.DEFAULT_THREADS
) : BenchmarkAdapter {
    
    private var isVerified = false
    private var actualExecutablePath: String? = null
    
    override suspend fun warmup(config: BenchmarkConfig) = withContext(Dispatchers.IO) {
        if (!isVerified) {
            findLlamaCppExecutable()
            
            if (modelPath == null) {
                throw RuntimeException("Model path is required for llama.cpp")
            }
            
            // Run a quick warmup inference
            val warmupCommand = buildCommand(
                prompt = "Hello",
                maxTokens = 5,
                temperature = 0.7f
            )
            
            val result = executeCommand(warmupCommand, timeoutSeconds = 30)
            if (result.exitCode != 0) {
                throw RuntimeException("llama.cpp warmup failed: ${result.stderr}")
            }
            
            isVerified = true
        }
    }
    
    override suspend fun runInference(config: BenchmarkConfig): BenchmarkRun = withContext(Dispatchers.IO) {
        if (!isVerified) {
            throw IllegalStateException("llama.cpp adapter not warmed up")
        }
        
        val command = buildCommand(
            prompt = config.prompt,
            maxTokens = config.maxTokens,
            temperature = config.temperature
        )
        
        val startTime = System.currentTimeMillis()
        val result = executeCommand(command, timeoutSeconds = 120)
        val endTime = System.currentTimeMillis()
        
        if (result.exitCode != 0) {
            throw RuntimeException("llama.cpp inference failed: ${result.stderr}")
        }
        
        val metrics = parseLlamaCppOutput(result.stdout + result.stderr)
        val latency = endTime - startTime
        
        return@withContext BenchmarkRun(
            tokensGenerated = metrics.tokensGenerated,
            inferenceTimeMs = metrics.inferenceTimeMs,
            tokensPerSecond = metrics.tokensPerSecond,
            latencyMs = latency.toDouble(),
            memoryUsageMB = getCurrentMemoryUsage(),
            gpuUsageMB = 0 // llama.cpp doesn't report GPU memory in standard output
        )
    }
    
    override fun isAvailable(): Boolean {
        return try {
            findLlamaCppExecutable()
            true
        } catch (e: Exception) {
            false
        }
    }
    
    override fun getDisplayName(): String = "llama.cpp${if (useGpu) " (GPU)" else " (CPU)"}"
    
    override suspend fun cleanup() {
        // llama.cpp is process-based, so cleanup involves terminating any lingering processes
        isVerified = false
        actualExecutablePath = null
        
        try {
            // Kill any remaining llama processes
            val killCommand = listOf("pkill", "-f", "llama")
            executeCommand(killCommand, timeoutSeconds = 5)
        } catch (e: Exception) {
            // Ignore cleanup errors, just log
            println("Warning: llama.cpp process cleanup failed: ${e.message}")
        }
    }
    
    private fun findLlamaCppExecutable() {
        if (actualExecutablePath != null) return
        
        val possiblePaths = listOf(llamaCppPath) + NoesisConstants.Engines.LlamaCpp.SEARCH_PATHS
        
        for (path in possiblePaths) {
            val file = File(path)
            if (file.exists() && file.canExecute()) {
                actualExecutablePath = path
                return
            }
        }
        
        // Try to find in PATH
        val pathCheck = executeCommand(listOf("which", "llama"), timeoutSeconds = 5)
        if (pathCheck.exitCode == 0 && pathCheck.stdout.trim().isNotEmpty()) {
            actualExecutablePath = pathCheck.stdout.trim()
            return
        }
        
        val mainCheck = executeCommand(listOf("which", "main"), timeoutSeconds = 5)
        if (mainCheck.exitCode == 0 && mainCheck.stdout.trim().isNotEmpty()) {
            actualExecutablePath = mainCheck.stdout.trim()
            return
        }
        
        throw RuntimeException(
            """llama.cpp executable not found. Tried:
               ${possiblePaths.joinToString("\n  ")}
               
               Install llama.cpp:
               git clone https://github.com/ggerganov/llama.cpp
               cd llama.cpp && make
               
               Or install via Homebrew:
               brew install llama.cpp
            """.trimIndent()
        )
    }
    
    private fun buildCommand(
        prompt: String,
        maxTokens: Int,
        temperature: Float
    ): List<String> {
        val command = mutableListOf(
            actualExecutablePath!!,
            "--model", modelPath!!,
            "--prompt", prompt,
            "--n-predict", maxTokens.toString(),
            "--temp", temperature.toString(),
            "--threads", threads.toString(),
            "--batch-size", NoesisConstants.Engines.LlamaCpp.DEFAULT_BATCH_SIZE.toString(),
            "--no-display-prompt", // Don't echo the prompt back
            "--log-disable" // Reduce verbose logging
        )
        
        if (useGpu) {
            // Try to enable GPU acceleration
            command.addAll(listOf(
                "--n-gpu-layers", NoesisConstants.Engines.LlamaCpp.GPU_LAYERS_ALL.toString()
            ))
        }
        
        return command
    }
    
    private fun executeCommand(
        command: List<String>,
        timeoutSeconds: Long = 60,
        workingDir: File? = null
    ): CommandResult {
        val processBuilder = ProcessBuilder(command)
            .redirectErrorStream(false)
        
        workingDir?.let { processBuilder.directory(it) }
        
        val process = processBuilder.start()
        val completed = process.waitFor(timeoutSeconds, TimeUnit.SECONDS)
        
        return if (completed) {
            val stdout = process.inputStream.bufferedReader().readText()
            val stderr = process.errorStream.bufferedReader().readText()
            CommandResult(process.exitValue(), stdout, stderr)
        } else {
            process.destroyForcibly()
            CommandResult(-1, "", "Command timed out after ${timeoutSeconds}s")
        }
    }
    
    private fun parseLlamaCppOutput(output: String): LlamaCppMetrics {
        var tokensGenerated = 0
        var inferenceTimeMs = 0L
        var tokensPerSecond = 0.0
        
        val lines = output.lines()
        
        for (line in lines) {
            when {
                // Pattern: "llama_print_timings:     load time =   123.45 ms"
                line.contains("llama_print_timings:") && line.contains("eval time") -> {
                    val timeMatch = Regex("""eval time\s*=\s*([\d.]+)\s*ms""").find(line)
                    timeMatch?.let {
                        inferenceTimeMs = it.groupValues[1].toDouble().toLong()
                    }
                }
                
                // Pattern: "llama_print_timings:        eval: 100 tokens, 50.0 ms per token, 20.0 tokens per second"
                line.contains("eval:") && line.contains("tokens per second") -> {
                    val tokenMatch = Regex("""eval:\s*(\d+)\s*tokens""").find(line)
                    val speedMatch = Regex("""([\d.]+)\s*tokens per second""").find(line)
                    
                    tokenMatch?.let {
                        tokensGenerated = it.groupValues[1].toInt()
                    }
                    speedMatch?.let {
                        tokensPerSecond = it.groupValues[1].toDouble()
                    }
                }
                
                // Alternative pattern for token count in generation output
                line.trim().isNotEmpty() && !line.startsWith("llama") && tokensGenerated == 0 -> {
                    // Count words as approximation of tokens (rough estimate)
                    val wordCount = line.trim().split(Regex("\\s+")).size
                    if (wordCount > tokensGenerated) {
                        tokensGenerated = wordCount
                    }
                }
            }
        }
        
        // Calculate tokens/sec if not provided but we have timing data
        if (tokensPerSecond == 0.0 && tokensGenerated > 0 && inferenceTimeMs > 0) {
            tokensPerSecond = (tokensGenerated * 1000.0) / inferenceTimeMs
        }
        
        // If we still don't have inference time, estimate from total timing
        if (inferenceTimeMs == 0L && tokensPerSecond > 0.0 && tokensGenerated > 0) {
            inferenceTimeMs = ((tokensGenerated / tokensPerSecond) * 1000).toLong()
        }
        
        return LlamaCppMetrics(
            tokensGenerated = tokensGenerated,
            inferenceTimeMs = inferenceTimeMs,
            tokensPerSecond = tokensPerSecond
        )
    }
    
    private fun getCurrentMemoryUsage(): Int {
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val usedMemory = totalMemory - freeMemory
        return (usedMemory / 1024 / 1024).toInt()
    }
    
    private data class CommandResult(
        val exitCode: Int,
        val stdout: String,
        val stderr: String
    )
    
    private data class LlamaCppMetrics(
        val tokensGenerated: Int,
        val inferenceTimeMs: Long,
        val tokensPerSecond: Double
    )
}