import org.gradle.api.DefaultTask
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.TaskAction
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.io.path.exists

/**
 * Task that handles model directory setup and download
 * Solves the SwiftPM sandbox permission issues
 */
abstract class ModelSetupTask : DefaultTask() {
    
    @get:Input
    @get:Optional
    abstract val modelName: Property<String>
    
    @get:Input
    @get:Optional  
    abstract val skipDownload: Property<Boolean>
    
    init {
        // Set defaults
        modelName.convention("gpt-oss-20b")
        skipDownload.convention(false)
    }
    
    @TaskAction
    fun setupModel() {
        val home = System.getProperty("user.home")
        val modelDir = File("$home/.noesis/models/${modelName.get()}")
        val metalDir = File(modelDir, "metal")
        val modelBin = File(metalDir, "model.bin")
        
        logger.lifecycle("🏗️  Setting up Noesis model: ${modelName.get()}")
        
        // Create directory structure
        logger.lifecycle("📁 Creating model directory structure...")
        if (!modelDir.exists()) {
            if (modelDir.mkdirs()) {
                logger.lifecycle("   ✅ Created: ${modelDir.absolutePath}")
            } else {
                throw RuntimeException("❌ Failed to create directory: ${modelDir.absolutePath}")
            }
        } else {
            logger.lifecycle("   ✅ Directory exists: ${modelDir.absolutePath}")
        }
        
        // Check if valid model already exists
        if (isValidModel(modelBin)) {
            logger.lifecycle("✅ Valid pre-converted Metal model found (${modelBin.length()} bytes)")
            logger.lifecycle("   Location: ${modelBin.absolutePath}")
            // Still need to setup Rust dependencies even if model exists
            buildHarmonyRustDependency()
            return
        }
        
        if (skipDownload.get()) {
            logger.lifecycle("⚠️  Model download skipped (--skip-download flag)")
            return
        }
        
        // Download model
        downloadModel(modelDir, modelBin)
        
        // Build Rust/Harmony dependencies
        buildHarmonyRustDependency()
    }
    
    private fun isValidModel(modelBin: File): Boolean {
        if (!modelBin.exists() || modelBin.length() < 10_000_000) {
            return false
        }
        
        // Check GPT-OSS magic header (only read first 12 bytes)
        try {
            modelBin.inputStream().use { input ->
                val header = ByteArray(12)
                val bytesRead = input.read(header)
                if (bytesRead < 12) {
                    return false
                }
                val expected = "GPT-OSS v1.0".toByteArray()
                return header.contentEquals(expected)
            }
        } catch (e: Exception) {
            logger.debug("Failed to verify model header: ${e.message}")
            return false
        }
    }
    
    private fun downloadModel(modelDir: File, modelBin: File) {
        logger.lifecycle("📥 Downloading pre-converted Metal model from HuggingFace...")
        
        try {
            // Try to find hf command
            val hfCommand = findHuggingFaceCommand()
            if (hfCommand == null) {
                logger.lifecycle("⚠️  HuggingFace CLI not found")
                logger.lifecycle("   Install: pip install huggingface-hub[cli]")
                logger.lifecycle("   Manual: hf download openai/${modelName.get()} --include \"metal/*\" --local-dir ${modelDir.absolutePath}")
                return
            }
            
            logger.lifecycle("   Using HF command: $hfCommand")
            
            // Execute download
            val process = ProcessBuilder(
                hfCommand,
                "download",
                "openai/${modelName.get()}",
                "--include", "metal/*",
                "--local-dir", modelDir.absolutePath
            ).apply {
                // Inherit environment to get proper PATH
                environment().putAll(System.getenv())
                // Add common Python paths
                val path = environment()["PATH"] ?: ""
                environment()["PATH"] = "${System.getProperty("user.home")}/.pyenv/shims:" +
                        "${System.getProperty("user.home")}/.local/bin:" +
                        "/opt/homebrew/bin:/usr/local/bin:$path"
            }.start()
            
            val exitCode = process.waitFor()
            
            if (exitCode == 0) {
                if (isValidModel(modelBin)) {
                    logger.lifecycle("✅ Download successful: ${modelBin.length()} bytes")
                    logger.lifecycle("   Location: ${modelBin.absolutePath}")
                } else {
                    logger.lifecycle("⚠️  Download completed but model validation failed")
                }
            } else {
                val errorOutput = process.errorStream.bufferedReader().readText()
                logger.lifecycle("⚠️  Download failed with exit code: $exitCode")
                logger.lifecycle("   Error: $errorOutput")
                logger.lifecycle("   Manual: hf download openai/${modelName.get()} --include \"metal/*\" --local-dir ${modelDir.absolutePath}")
            }
            
        } catch (e: Exception) {
            logger.lifecycle("⚠️  Download error: ${e.message}")
            logger.lifecycle("   Manual: hf download openai/${modelName.get()} --include \"metal/*\" --local-dir ${modelDir.absolutePath}")
        }
    }
    
    private fun findHuggingFaceCommand(): String? {
        val possiblePaths = listOf(
            "hf",
            "/usr/local/bin/hf",
            "/opt/homebrew/bin/hf",
            "${System.getProperty("user.home")}/.pyenv/shims/hf",
            "${System.getProperty("user.home")}/.local/bin/hf"
        )
        
        for (path in possiblePaths) {
            if (commandExists(path)) {
                return path
            }
        }
        
        return null
    }
    
    private fun commandExists(command: String): Boolean {
        return try {
            val process = ProcessBuilder("which", command).start()
            process.waitFor() == 0
        } catch (e: Exception) {
            false
        }
    }
    
    private fun buildHarmonyRustDependency() {
        logger.lifecycle("🦀 Setting up Harmony Rust dependency...")
        
        val projectDir = project.projectDir
        val swiftImplDir = File(projectDir, "swift-implementation")
        val sourceDylib = File(swiftImplDir, "Sources/CHarmony/libopenai_harmony.dylib")
        val rustDepsDir = File(swiftImplDir, ".build/rust-deps/harmony-swift")
        val targetDir = File(rustDepsDir, "target/release/deps")
        val dylibFile = File(targetDir, "libopenai_harmony.dylib")
        
        // Check if dylib already exists and is recent
        if (dylibFile.exists() && dylibFile.lastModified() >= sourceDylib.lastModified()) {
            logger.lifecycle("✅ Harmony dylib already up-to-date: ${dylibFile.absolutePath}")
            return
        }
        
        // Check if source dylib exists
        if (!sourceDylib.exists()) {
            logger.lifecycle("⚠️  Source dylib not found at: ${sourceDylib.absolutePath}")
            logger.lifecycle("   This may indicate missing Harmony dependency")
            return
        }
        
        // Create target directory and copy dylib
        if (!targetDir.exists()) {
            val created = targetDir.mkdirs()
            if (!created) {
                logger.lifecycle("⚠️  Failed to create target directory: ${targetDir.absolutePath}")
                return
            }
        }
        
        try {
            sourceDylib.copyTo(dylibFile, overwrite = true)
            logger.lifecycle("✅ Harmony dylib copied successfully")
            logger.lifecycle("   From: ${sourceDylib.absolutePath}")
            logger.lifecycle("   To: ${dylibFile.absolutePath}")
        } catch (e: Exception) {
            logger.lifecycle("⚠️  Failed to copy Harmony dylib: ${e.message}")
            logger.lifecycle("   Source exists: ${sourceDylib.exists()}")
            logger.lifecycle("   Target dir exists: ${targetDir.exists()}")
        }
    }
}