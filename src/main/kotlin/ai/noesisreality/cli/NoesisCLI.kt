package ai.noesisreality.cli

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.*
import com.github.ajalt.clikt.parameters.types.*
import com.github.ajalt.clikt.parameters.arguments.*
import ai.noesisreality.cli.commands.BenchmarkCommand
import ai.noesisreality.cli.common.EngineFactory
import ai.noesisreality.cli.common.HarmonyPromptBuilder
import ai.noesisreality.cli.common.OutputFormatters
import ai.noesisreality.cli.common.BenchmarkResult
import ai.noesisreality.engine.NoesisInferenceEngine
import ai.noesisreality.sessions.ChatSessionManager
import ai.noesisreality.diagnostics.SystemDiagnostics
import ai.noesisreality.models.ModelManager
import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.system.exitProcess

/**
 * Noesis CLI - Advanced GPT-OSS inference with Metal 4 acceleration
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 * Domain: noesisreality.ai
 * 
 * Unified command-line interface providing high-performance text generation
 * on Apple Silicon GPUs via Swift Metal 4 backend.
 */
class NoesisCLI : CliktCommand(
    name = "noesis",
    help = """
        🚀 Noesis - Advanced GPT-OSS inference with Metal 4 acceleration
        
        High-performance text generation and chat on Apple Silicon.
        Powered by Swift Metal 4 backend with Kotlin user experience layer.
        
        Examples:
          noesis generate "Explain quantum computing"
          noesis chat --system "You are a helpful assistant"
          noesis benchmark --iterations 50
          noesis diagnose --export diagnostics.json
    """.trimIndent()
) {
    
    // Global configuration
    private val verbose by option("-v", "--verbose", help = "Enable verbose output").flag(default = false)
    private val modelPath by option("--model-path", help = "Path to model.bin file")
    private val configFile by option("--config", help = "Configuration file path").file(canBeDir = false)
    
    override fun run() {
        EngineFactory.initialize(
            verbose = verbose,
            modelPath = modelPath,
            configPath = configFile?.absolutePath
        )
    }
    
    init {
        subcommands(
            GenerateCommand(),
            ChatCommand(), 
            BenchmarkCommand(),
            DiagnoseCommand(),
            ModelCommand()
        )
    }
}

fun main(args: Array<String>) = NoesisCLI().main(args)

/**
 * Text generation command - single-shot inference
 */
class GenerateCommand : CliktCommand(
    name = "generate",
    help = "Generate text from a prompt using GPT-OSS models"
) {
    
    private val prompt by argument(name = "PROMPT", help = "Input prompt text")
    
    // Generation parameters
    private val maxTokens by option("-m", "--max-tokens", help = "Maximum tokens to generate").int().default(100)
    private val temperature by option("-t", "--temperature", help = "Temperature for sampling").float().default(0.7f)
    private val topP by option("--top-p", help = "Top-p sampling threshold").float().default(0.9f)
    private val repetitionPenalty by option("--repetition-penalty", help = "Repetition penalty").float().default(1.1f)
    private val seed by option("--seed", help = "Random seed").int()
    
    // Advanced options
    private val systemPrompt by option("--system", help = "System prompt")
    private val reasoningEffort by option("--reasoning", help = "Reasoning effort").choice("low", "medium", "high").default("medium")
    private val format by option("--format", help = "Output format").choice("text", "json", "tokens").default("text")
    private val useHarmony by option("--harmony", help = "Use Harmony structured reasoning").flag()
    
    // Performance monitoring
    private val stats by option("--stats", help = "Show generation statistics").flag()
    private val benchmark by option("--benchmark", help = "Run micro-benchmark").flag()
    
    override fun run() {
        runBlocking {
            try {
                val engine = NoesisInferenceEngine.getInstance()
                
                if (benchmark) {
                    echo("🏁 Running micro-benchmark...")
                }
                
                echo("🎯 Generating text...")
                
                val finalPrompt = if (useHarmony) {
                    HarmonyPromptBuilder.buildReasoningPrompt(prompt, reasoningEffort)
                } else {
                    HarmonyPromptBuilder.buildStandardPrompt(prompt, systemPrompt)
                }
                
                val result = engine.generate(
                    prompt = finalPrompt,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topP = topP,
                    repetitionPenalty = repetitionPenalty,
                    reasoningEffort = reasoningEffort,
                    seed = seed
                )
                
                echo(OutputFormatters.formatInferenceResult(result, format, stats))
                
            } catch (e: Exception) {
                echo("❌ Generation failed: ${e.message}", err = true)
                if (NoesisInferenceEngine.isVerbose()) {
                    e.printStackTrace()
                }
                exitProcess(1)
            }
        }
    }
}

/**
 * Interactive chat command - managed entirely in Kotlin
 */
class ChatCommand : CliktCommand(
    name = "chat", 
    help = "Start an interactive chat session"
) {
    
    // Session configuration
    private val systemPrompt by option("--system", help = "System prompt")
    private val contextLength by option("-c", "--context", help = "Context length").int().default(8192)
    private val temperature by option("-t", "--temperature", help = "Temperature").float().default(0.7f)
    private val reasoningEffort by option("--reasoning", help = "Reasoning effort").choice("low", "medium", "high").default("medium")
    
    // Session management
    private val sessionFile by option("--session", help = "Save/load session file").file()
    private val multiline by option("--multiline", help = "Enable multiline input").flag()
    private val autoSave by option("--auto-save", help = "Auto-save every N messages").int()
    
    // Tools (handled in Kotlin)
    private val enableTools by option("--tools", help = "Enable tool usage").flag()
    
    override fun run() {
        runBlocking {
            try {
                val sessionManager = ChatSessionManager(
                    systemPrompt = systemPrompt,
                    contextLength = contextLength,
                    temperature = temperature,
                    reasoningEffort = reasoningEffort,
                    multilineMode = multiline,
                    autoSaveInterval = autoSave,
                    toolsEnabled = enableTools,
                    verbose = NoesisInferenceEngine.isVerbose()
                )
                
                // Load existing session if specified
                sessionFile?.let { file ->
                    if (file.exists()) {
                        sessionManager.loadSession(file.absolutePath)
                        echo("📚 Loaded session from ${file.name}")
                    }
                }
                
                echo("💬 Starting Noesis Chat Session")
                echo("   Model: GPT-OSS with Metal 4 acceleration")
                echo("   Context: ${contextLength} tokens")
                echo("   Reasoning: ${reasoningEffort}")
                if (enableTools) echo("   Tools: enabled")
                echo("   Type '/help' for commands or '/quit' to exit\n")
                
                // Start interactive session (all logic in Kotlin)
                sessionManager.start()
                
                // Save session if specified
                sessionFile?.let { file ->
                    sessionManager.saveSession(file.absolutePath)
                    echo("💾 Session saved to ${file.name}")
                }
                
            } catch (e: Exception) {
                echo("❌ Chat failed: ${e.message}", err = true)
                if (NoesisInferenceEngine.isVerbose()) {
                    e.printStackTrace()
                }
                exitProcess(1)
            }
        }
    }
}


/**
 * System diagnostics command - Kotlin orchestrated
 */
class DiagnoseCommand : CliktCommand(
    name = "diagnose",
    help = "Run system diagnostics"
) {
    
    private val checkModel by option("--check-model", help = "Validate model file").file()
    private val checkGpu by option("--check-gpu", help = "GPU diagnostics").flag()
    private val checkMemory by option("--check-memory", help = "Memory diagnostics").flag()
    private val export by option("--export", help = "Export to file").file()
    private val fix by option("--fix", help = "Attempt to fix issues").flag()
    
    override fun run() {
        runBlocking {
            try {
                echo("🔍 Noesis System Diagnostics")
                echo()
                
                val diagnostics = SystemDiagnostics(verbose = NoesisInferenceEngine.isVerbose())
                val report = diagnostics.runComplete(
                    checkModelFile = checkModel?.absolutePath,
                    checkGpu = checkGpu,
                    checkMemory = checkMemory
                )
                
                // Display results
                echo("💻 System Information:")
                echo("  Platform: ${report.platform}")
                echo("  macOS: ${report.macosVersion}")
                echo("  Xcode CLI: ${if (report.xcodeInstalled) "✅" else "❌"}")
                echo()
                
                echo("🎮 GPU Status:")
                echo("  Device: ${report.gpuName}")
                echo("  Metal 4: ${if (report.metal4Supported) "✅" else "❌"}")
                echo("  Memory: ${report.gpuMemoryGB}GB")
                echo()
                
                echo("📦 Models:")
                report.models.forEach { model ->
                    val status = if (model.isValid) "✅" else "❌"
                    echo("  ${model.name}: $status (${String.format("%.1f", model.sizeGB)}GB)")
                }
                echo()
                
                if (report.issues.isNotEmpty()) {
                    echo("⚠️  Issues Found:")
                    report.issues.forEach { issue ->
                        echo("  • $issue")
                    }
                    
                    if (fix) {
                        echo()
                        echo("🔧 Attempting fixes...")
                        val fixed = diagnostics.attemptFixes(report.issues)
                        echo("Fixed $fixed issues")
                    }
                } else {
                    echo("✅ All checks passed!")
                }
                
                export?.let { file ->
                    file.writeText(report.toJson())
                    echo("\n📄 Report saved to: ${file.name}")
                }
                
            } catch (e: Exception) {
                echo("❌ Diagnostics failed: ${e.message}", err = true)
                exitProcess(1)
            }
        }
    }
}

/**
 * Model management - pure Kotlin implementation
 */
class ModelCommand : CliktCommand(name = "model", help = "Manage GPT-OSS models") {
    override fun run() {
        echo("📦 Model Management")
        echo("Commands: list, download, verify, clean, info")
    }
    
    init {
        subcommands(
            ModelListCommand(),
            ModelDownloadCommand(), 
            ModelVerifyCommand(),
            ModelCleanCommand(),
            ModelInfoCommand()
        )
    }
}

class ModelListCommand : CliktCommand(name = "list", help = "List models") {
    override fun run() {
        runBlocking {
            val manager = ModelManager()
            val models = manager.listAvailable()
            
            echo("📋 Available Models:")
            models.forEach { model ->
                val status = when {
                    model.isInstalled && model.isValid -> "✅ Ready"
                    model.isInstalled -> "⚠️  Installed (validation failed)"
                    else -> "⬇️  Available"
                }
                echo("  ${model.name} (${String.format("%.1f", model.sizeGB)}GB): $status")
                if (model.isInstalled) {
                    echo("    ${model.path}")
                }
            }
        }
    }
}

class ModelDownloadCommand : CliktCommand(name = "download", help = "Download model") {
    private val modelName by argument(help = "Model name").choice("gpt-oss-20b", "gpt-oss-120b")
    private val force by option("--force", help = "Force redownload").flag()
    
    override fun run() {
        runBlocking {
            val manager = ModelManager()
            echo("⬇️  Downloading $modelName...")
            
            manager.download(modelName, force) { progress ->
                echo("Progress: ${String.format("%.1f", progress * 100)}%")
            }
            
            echo("✅ Download complete!")
        }
    }
}

class ModelVerifyCommand : CliktCommand(name = "verify", help = "Verify model") {
    private val modelPath by argument(help = "Model path").file(mustExist = true)
    
    override fun run() {
        runBlocking {
            val manager = ModelManager()
            echo("🔍 Verifying ${modelPath.name}...")
            
            val result = manager.verify(modelPath.absolutePath)
            if (result.isValid) {
                echo("✅ Model is valid")
                echo("  Size: ${String.format("%.1f", result.sizeGB)}GB")
                echo("  Checksum: ${result.checksum}")
            } else {
                echo("❌ Verification failed: ${result.error}")
                exitProcess(1)
            }
        }
    }
}

class ModelCleanCommand : CliktCommand(name = "clean", help = "Clean model cache") {
    private val dryRun by option("--dry-run", help = "Show what would be cleaned").flag()
    
    override fun run() {
        runBlocking {
            val manager = ModelManager()
            val result = manager.cleanup(dryRun)
            
            echo("🧹 ${if (dryRun) "Would clean" else "Cleaned"}:")
            echo("  ${result.filesRemoved} files")
            echo("  ${String.format("%.1f", result.spaceMB)}MB")
        }
    }
}

class ModelInfoCommand : CliktCommand(name = "info", help = "Show model information") {
    private val modelPath by argument(help = "Model path").file(mustExist = true)
    
    override fun run() {
        runBlocking {
            val manager = ModelManager()
            val info = manager.getInfo(modelPath.absolutePath)
            
            echo("📊 Model Information:")
            echo("  Name: ${info.name}")
            echo("  Size: ${String.format("%.1f", info.sizeGB)}GB")
            echo("  Format: ${info.format}")
            echo("  Parameters: ${info.parameters}")
            echo("  Context length: ${info.contextLength}")
            echo("  Vocabulary size: ${info.vocabularySize}")
        }
    }
}

// Common data classes moved to ai.noesisreality.cli.common package