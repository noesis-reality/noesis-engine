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
 * Benchmark adapter for GPT-OSS reference implementation (Python/C/Metal)
 * 
 * This adapter runs the original GPT-OSS Python implementation and parses its output
 * to extract performance metrics for fair comparison.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class GptOssReferenceBenchmarkAdapter(
    private val gptOssPath: String = NoesisConstants.Engines.GptOss.DEFAULT_PATH,
    private val modelPath: String? = null,
    private val pythonCommand: String = NoesisConstants.Engines.GptOss.PYTHON_COMMAND
) : BenchmarkAdapter {
    
    private var isVerified = false
    
    override suspend fun warmup(config: BenchmarkConfig) = withContext(Dispatchers.IO) {
        if (!isVerified) {
            verifyGptOssInstallation()
            
            // Run a quick warmup inference
            val warmupCommand = buildCommand(
                prompt = "Hello",
                maxTokens = 5,
                temperature = 0.7f
            )
            
            val result = executeCommand(warmupCommand, timeoutSeconds = 30)
            if (result.exitCode != 0) {
                throw RuntimeException("GPT-OSS warmup failed: ${result.stderr}")
            }
            
            isVerified = true
        }
    }
    
    override suspend fun runInference(config: BenchmarkConfig): BenchmarkRun = withContext(Dispatchers.IO) {
        if (!isVerified) {
            throw IllegalStateException("GPT-OSS adapter not warmed up")
        }
        
        val command = buildCommand(
            prompt = config.prompt,
            maxTokens = config.maxTokens,
            temperature = config.temperature
        )
        
        val startTime = System.currentTimeMillis()
        val result = executeCommand(command, timeoutSeconds = NoesisConstants.Engines.GptOss.PROCESS_TIMEOUT_SECONDS)
        val endTime = System.currentTimeMillis()
        
        if (result.exitCode != 0) {
            throw RuntimeException("GPT-OSS inference failed: ${result.stderr}")
        }
        
        val metrics = parseGptOssOutput(result.stdout)
        val latency = endTime - startTime
        
        return@withContext BenchmarkRun(
            tokensGenerated = metrics.tokensGenerated,
            inferenceTimeMs = metrics.inferenceTimeMs,
            tokensPerSecond = metrics.tokensPerSecond,
            latencyMs = latency.toDouble(),
            memoryUsageMB = metrics.memoryUsageMB,
            gpuUsageMB = metrics.gpuMemoryMB
        )
    }
    
    override fun isAvailable(): Boolean {
        return try {
            val gptOssDir = File(gptOssPath)
            val generateScript = File(gptOssDir, "generate.py")
            val chatScript = File(gptOssDir, "chat.py")
            
            gptOssDir.exists() && gptOssDir.isDirectory &&
            (generateScript.exists() || chatScript.exists())
        } catch (e: Exception) {
            false
        }
    }
    
    override fun getDisplayName(): String = "GPT-OSS Reference (Python + Metal)"
    
    override suspend fun cleanup() {
        // GPT-OSS is process-based, so cleanup involves terminating any lingering processes
        isVerified = false
        
        try {
            // Kill any remaining gpt-oss processes
            val killCommand = listOf("pkill", "-f", "gpt_oss")
            executeCommand(killCommand, timeoutSeconds = 5)
        } catch (e: Exception) {
            // Ignore cleanup errors, just log
            println("Warning: GPT-OSS process cleanup failed: ${e.message}")
        }
    }
    
    private fun verifyGptOssInstallation() {
        val gptOssDir = File(gptOssPath)
        if (!gptOssDir.exists() || !gptOssDir.isDirectory) {
            throw RuntimeException("GPT-OSS directory not found at: $gptOssPath")
        }
        
        // Check for Python installation
        val pythonCheck = executeCommand(listOf(pythonCommand, "--version"), timeoutSeconds = 5)
        if (pythonCheck.exitCode != 0) {
            throw RuntimeException("Python not found. Install Python 3.8+ and ensure '$pythonCommand' is in PATH")
        }
        
        // Check for required Python modules
        val moduleCheck = executeCommand(
            listOf(pythonCommand, "-c", "import torch, numpy, transformers"),
            timeoutSeconds = 10,
            workingDir = gptOssDir
        )
        if (moduleCheck.exitCode != 0) {
            throw RuntimeException("Required Python modules not installed. Run: pip install torch numpy transformers")
        }
    }
    
    private fun buildCommand(
        prompt: String,
        maxTokens: Int,
        temperature: Float
    ): List<String> {
        val command = mutableListOf(
            pythonCommand,
            "-m", "gpt_oss.generate",
            "--prompt", prompt,
            "--max-tokens", maxTokens.toString(),
            "--temperature", temperature.toString(),
            "--benchmark", // Enable benchmark mode for detailed output
            "--no-stream" // Disable streaming for consistent timing
        )
        
        modelPath?.let { 
            command.addAll(listOf("--model-path", it))
        }
        
        return command
    }
    
    private fun executeCommand(
        command: List<String>,
        timeoutSeconds: Long = NoesisConstants.Engines.GptOss.PROCESS_TIMEOUT_SECONDS,
        workingDir: File = File(gptOssPath)
    ): CommandResult {
        val processBuilder = ProcessBuilder(command)
            .directory(workingDir)
            .redirectErrorStream(false)
        
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
    
    private fun parseGptOssOutput(output: String): GptOssMetrics {
        var tokensGenerated = 0
        var inferenceTimeMs = 0L
        var tokensPerSecond = 0.0
        var memoryUsageMB = 0
        var gpuMemoryMB = 0
        
        val lines = output.lines()
        
        for (line in lines) {
            when {
                line.contains("Generated tokens:") -> {
                    tokensGenerated = extractNumber(line).toInt()
                }
                line.contains("Inference time:") -> {
                    // Extract time in seconds and convert to milliseconds
                    val timeSeconds = extractNumber(line)
                    inferenceTimeMs = (timeSeconds * 1000).toLong()
                }
                line.contains("Tokens per second:") -> {
                    tokensPerSecond = extractNumber(line)
                }
                line.contains("Memory usage:") -> {
                    memoryUsageMB = extractNumber(line).toInt()
                }
                line.contains("GPU memory:") -> {
                    gpuMemoryMB = extractNumber(line).toInt()
                }
            }
        }
        
        // Calculate tokens/sec if not provided
        if (tokensPerSecond == 0.0 && tokensGenerated > 0 && inferenceTimeMs > 0) {
            tokensPerSecond = (tokensGenerated * 1000.0) / inferenceTimeMs
        }
        
        return GptOssMetrics(
            tokensGenerated = tokensGenerated,
            inferenceTimeMs = inferenceTimeMs,
            tokensPerSecond = tokensPerSecond,
            memoryUsageMB = memoryUsageMB,
            gpuMemoryMB = gpuMemoryMB
        )
    }
    
    private fun extractNumber(line: String): Double {
        val regex = Regex("""\d+(?:\.\d+)?""")
        val match = regex.find(line)
        return match?.value?.toDoubleOrNull() ?: 0.0
    }
    
    private data class CommandResult(
        val exitCode: Int,
        val stdout: String,
        val stderr: String
    )
    
    private data class GptOssMetrics(
        val tokensGenerated: Int,
        val inferenceTimeMs: Long,
        val tokensPerSecond: Double,
        val memoryUsageMB: Int,
        val gpuMemoryMB: Int
    )
}