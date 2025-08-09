package ai.noesisreality.cli.commands

import ai.noesisreality.benchmarks.*
import ai.noesisreality.benchmarks.adapters.*
import ai.noesisreality.core.NoesisConstants
import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.context
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.output.MordantHelpFormatter
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.default
import com.github.ajalt.clikt.parameters.options.*
import com.github.ajalt.clikt.parameters.types.file
import com.github.ajalt.clikt.parameters.types.float
import com.github.ajalt.clikt.parameters.types.int
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Benchmark CLI commands for comparing inference performance across engines
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class BenchmarkCommand : CliktCommand(
    name = "benchmark",
    help = """
    ${NoesisConstants.Emojis.CHART} Benchmark and compare inference performance across engines
    
    Compare token/s performance between:
    • Noesis Engine (Kotlin + Swift Metal 4)  
    • GPT-OSS Reference (Python + Metal)
    • llama.cpp (C++)
    
    Examples:
      noesis benchmark compare "Explain quantum computing" --iterations 10
      noesis benchmark quick --engines noesis,llama-cpp
      noesis benchmark full --model-path ./gpt-oss-20b --output ./results/
    """.trimIndent()
) {
    
    init {
        context {
            helpFormatter = { MordantHelpFormatter(it, showDefaultValues = true) }
        }
        
        subcommands(
            QuickBenchmarkCommand(),
            CompareBenchmarkCommand(),
            FullBenchmarkCommand(),
            ListEnginesCommand()
        )
    }
    
    override fun run() = Unit
}

class QuickBenchmarkCommand : CliktCommand(
    name = "quick",
    help = "Run a quick benchmark comparison (3 iterations, 50 tokens)"
) {
    
    private val prompt by argument(
        help = "Prompt to benchmark with"
    ).default("Explain the concept of machine learning in simple terms.")
    
    private val engines by option(
        "--engines", "-e",
        help = "Comma-separated list of engines to benchmark (noesis,gpt-oss,llama-cpp)"
    ).default("noesis,gpt-oss,llama-cpp")
    
    private val modelPath by option(
        "--model-path", "-m",
        help = "Path to model file (required for some engines)"
    ).file(mustExist = false)
    
    override fun run() = runBlocking {
        val config = BenchmarkConfig(
            prompt = prompt,
            maxTokens = NoesisConstants.Benchmarks.QUICK_MAX_TOKENS,
            temperature = NoesisConstants.Defaults.TEMPERATURE,
            iterations = NoesisConstants.Benchmarks.QUICK_ITERATIONS,
            delayBetweenRunsMs = NoesisConstants.Benchmarks.QUICK_DELAY_MS,
            modelPath = modelPath?.absolutePath
        )
        
        runBenchmark(config, parseEngines(engines))
    }
}

class CompareBenchmarkCommand : CliktCommand(
    name = "compare",
    help = "Compare engines with custom parameters"
) {
    
    private val prompt by argument(
        help = "Prompt to benchmark with"
    )
    
    private val maxTokens by option(
        "--max-tokens", "-n",
        help = "Maximum tokens to generate"
    ).int().default(100)
    
    private val temperature by option(
        "--temperature", "-t", 
        help = "Temperature for generation"
    ).float().default(0.7f)
    
    private val iterations by option(
        "--iterations", "-i",
        help = "Number of benchmark iterations"
    ).int().default(5)
    
    private val engines by option(
        "--engines", "-e",
        help = "Comma-separated list of engines to benchmark"
    ).default("noesis,gpt-oss,llama-cpp")
    
    private val modelPath by option(
        "--model-path", "-m",
        help = "Path to model file"
    ).file(mustExist = false)
    
    private val delay by option(
        "--delay",
        help = "Delay between runs in milliseconds"
    ).int().default(1000)
    
    override fun run() = runBlocking {
        val config = BenchmarkConfig(
            prompt = prompt,
            maxTokens = maxTokens,
            temperature = temperature,
            iterations = iterations,
            delayBetweenRunsMs = delay.toLong(),
            modelPath = modelPath?.absolutePath
        )
        
        runBenchmark(config, parseEngines(engines))
    }
}

class FullBenchmarkCommand : CliktCommand(
    name = "full",
    help = "Comprehensive benchmark suite with multiple prompts and configurations"
) {
    
    private val modelPath by option(
        "--model-path", "-m",
        help = "Path to model file"
    ).file(mustExist = false)
    
    private val outputDir by option(
        "--output", "-o",
        help = "Output directory for detailed reports"
    ).file().default(File("benchmark-results"))
    
    private val engines by option(
        "--engines", "-e",
        help = "Comma-separated list of engines to benchmark"
    ).default("noesis,gpt-oss,llama-cpp")
    
    private val iterations by option(
        "--iterations", "-i",
        help = "Number of iterations per test"
    ).int().default(5)
    
    override fun run() = runBlocking {
        val engineList = parseEngines(engines)
        
        // Test suite with different prompt types and lengths
        val testSuite = listOf(
            BenchmarkConfig(
                prompt = "Hello, world!",
                maxTokens = 20,
                temperature = 0.7f,
                iterations = iterations,
                modelPath = modelPath?.absolutePath
            ),
            BenchmarkConfig(
                prompt = "Explain quantum computing in detail, covering the fundamental principles.",
                maxTokens = 150,
                temperature = 0.7f,
                iterations = iterations,
                modelPath = modelPath?.absolutePath
            ),
            BenchmarkConfig(
                prompt = "Write a creative story about artificial intelligence and the future of humanity.",
                maxTokens = 300,
                temperature = 1.0f,
                iterations = iterations,
                modelPath = modelPath?.absolutePath
            ),
            BenchmarkConfig(
                prompt = "1 + 1 = ",
                maxTokens = 5,
                temperature = 0.0f,
                iterations = iterations * 2, // More iterations for simple prompts
                modelPath = modelPath?.absolutePath
            )
        )
        
        println("${NoesisConstants.Emojis.ROCKET} Running comprehensive benchmark suite...")
        println("${testSuite.size} test configurations, ${engineList.size} engines")
        println()
        
        for ((index, config) in testSuite.withIndex()) {
            println("${NoesisConstants.Emojis.TARGET} Test ${index + 1}/${testSuite.size}")
            println("Prompt: \"${config.prompt.take(50)}${if (config.prompt.length > 50) "..." else ""}\"")
            println("Max tokens: ${config.maxTokens}, Temperature: ${config.temperature}")
            println()
            
            runBenchmark(config, engineList, outputDir)
            
            if (index < testSuite.size - 1) {
                println("${NoesisConstants.Emojis.GEAR} Cooling down before next test...")
                kotlinx.coroutines.delay(2000)
                println()
            }
        }
        
        println("${NoesisConstants.Emojis.FLAG} Comprehensive benchmark completed!")
        println("${NoesisConstants.Emojis.PACKAGE} Reports saved to: ${outputDir.absolutePath}")
    }
}

class ListEnginesCommand : CliktCommand(
    name = "list",
    help = "List available benchmark engines and their status"
) {
    
    override fun run() {
        println("${NoesisConstants.Emojis.GEAR} Available Benchmark Engines")
        println("=" * 50)
        println()
        
        val adapters = listOf(
            "noesis" to NoesisBenchmarkAdapter(),
            "gpt-oss" to GptOssReferenceBenchmarkAdapter(),
            "llama-cpp" to LlamaCppBenchmarkAdapter()
        )
        
        for ((name, adapter) in adapters) {
            val status = if (adapter.isAvailable()) {
                "${NoesisConstants.Emojis.CHECKMARK} Available"
            } else {
                "${NoesisConstants.Emojis.ERROR} Not Available"
            }
            
            println(String.format("%-15s %-15s %s", name, status, adapter.getDisplayName()))
        }
        
        println()
        println("Usage: noesis benchmark compare \"Your prompt\" --engines noesis,gpt-oss")
    }
}

private fun parseEngines(enginesStr: String): List<EngineType> {
    return enginesStr.split(",").mapNotNull { name ->
        when (name.trim().lowercase()) {
            "noesis", "noesis-engine" -> EngineType.NOESIS
            "gpt-oss", "gptoss", "gpt-oss-reference" -> EngineType.GPT_OSS_REFERENCE
            "llama", "llama-cpp", "llamacpp" -> EngineType.LLAMA_CPP
            else -> {
                println("${NoesisConstants.Emojis.WARNING} Unknown engine: $name")
                null
            }
        }
    }
}

private suspend fun runBenchmark(
    config: BenchmarkConfig,
    engines: List<EngineType>,
    outputDir: File? = null
) {
    val benchmarkEngine = BenchmarkEngine()
    
    // Register available adapters
    benchmarkEngine.registerAdapter(EngineType.NOESIS, NoesisBenchmarkAdapter(config.modelPath))
    benchmarkEngine.registerAdapter(EngineType.GPT_OSS_REFERENCE, GptOssReferenceBenchmarkAdapter(modelPath = config.modelPath))
    benchmarkEngine.registerAdapter(EngineType.LLAMA_CPP, LlamaCppBenchmarkAdapter(modelPath = config.modelPath))
    
    val report = benchmarkEngine.runComparisonBenchmark(config, engines)
    
    // Print results to console
    println(benchmarkEngine.generateReport(report))
    
    // Save detailed reports if output directory specified
    outputDir?.let {
        benchmarkEngine.saveReport(report, it)
    }
}

private operator fun String.times(n: Int): String = repeat(n)