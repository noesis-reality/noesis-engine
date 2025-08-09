package ai.noesisreality.models

import ai.noesisreality.core.NoesisConstants
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.coroutines.delay
import java.io.File
import java.net.URL
import java.nio.channels.Channels
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.TimeUnit
import java.security.MessageDigest

/**
 * Model Manager - Pure Kotlin implementation
 * 
 * Handles all model management operations including downloading, verification,
 * caching, and metadata management. The Swift layer only handles inference
 * once models are properly installed and verified.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class ModelManager(
    private val modelsDir: File = File("${System.getProperty("user.home")}/${NoesisConstants.Paths.MODELS_DIR}"),
    private val verbose: Boolean = NoesisConstants.Defaults.VERBOSE
) {

    init {
        // Ensure models directory exists
        modelsDir.mkdirs()
    }

    /**
     * List all available models (installed and downloadable)
     */
    suspend fun listAvailable(): List<ModelInfo> {
        val models = mutableListOf<ModelInfo>()
        
        // Check for officially supported models
        SUPPORTED_MODELS.forEach { (name, metadata) ->
            val modelDir = File(modelsDir, name)
            val modelFile = File(modelDir, "${NoesisConstants.Paths.METAL_SUBDIR}/${NoesisConstants.Paths.MODEL_FILENAME}")
            
            if (modelFile.exists()) {
                // Model is installed, check if valid
                val verification = verify(modelFile.absolutePath)
                models.add(ModelInfo(
                    name = name,
                    path = modelFile.absolutePath,
                    isInstalled = true,
                    isValid = verification.isValid,
                    sizeGB = verification.sizeGB,
                    format = "GPT-OSS Metal",
                    parameters = metadata.parameters,
                    contextLength = metadata.contextLength,
                    vocabularySize = metadata.vocabularySize,
                    checksum = verification.checksum,
                    issue = if (!verification.isValid) verification.error else null
                ))
            } else {
                // Model not installed
                models.add(ModelInfo(
                    name = name,
                    path = modelFile.absolutePath,
                    isInstalled = false,
                    isValid = false,
                    sizeGB = metadata.expectedSizeGB,
                    format = "GPT-OSS Metal",
                    parameters = metadata.parameters,
                    contextLength = metadata.contextLength,
                    vocabularySize = metadata.vocabularySize,
                    issue = "Not downloaded"
                ))
            }
        }
        
        return models
    }
    
    /**
     * Download a model from HuggingFace
     */
    suspend fun download(
        modelName: String,
        force: Boolean = false,
        progressCallback: (Double) -> Unit = {}
    ) {
        val metadata = SUPPORTED_MODELS[modelName] 
            ?: throw IllegalArgumentException("Unsupported model: $modelName")
        
        val modelDir = File(modelsDir, modelName)
        val metalDir = File(modelDir, "metal")
        val modelFile = File(metalDir, "model.bin")
        
        // Check if already exists and valid
        if (!force && modelFile.exists()) {
            val verification = verify(modelFile.absolutePath)
            if (verification.isValid) {
                if (verbose) println("✅ Model $modelName already installed and valid")
                return
            } else {
                if (verbose) println("⚠️ Existing model invalid, re-downloading...")
            }
        }
        
        // Create directories
        metalDir.mkdirs()
        
        if (verbose) {
            println("⬇️ Downloading $modelName...")
            println("   Source: ${metadata.downloadUrl}")
            println("   Target: ${modelFile.absolutePath}")
            println("   Expected size: ${String.format("%.1f", metadata.expectedSizeGB)}GB")
        }
        
        try {
            // Use HuggingFace CLI if available, otherwise direct download
            if (isHuggingFaceCliAvailable()) {
                downloadViaHuggingFaceCli(metadata, modelFile, progressCallback)
            } else {
                downloadDirect(metadata.downloadUrl, modelFile, progressCallback)
            }
            
            // Verify download
            val verification = verify(modelFile.absolutePath)
            if (!verification.isValid) {
                modelFile.delete()
                throw RuntimeException("Downloaded model failed verification: ${verification.error}")
            }
            
            // Create metadata file
            val metadataFile = File(modelDir, "metadata.json")
            val json = Json { prettyPrint = true }
            metadataFile.writeText(json.encodeToString(ModelMetadata.serializer(), metadata))
            
            if (verbose) {
                println("✅ Download completed and verified")
                println("   Size: ${String.format("%.1f", verification.sizeGB)}GB")
                println("   Checksum: ${verification.checksum?.take(8)}...")
            }
            
        } catch (e: Exception) {
            // Cleanup failed download
            modelFile.delete()
            throw RuntimeException("Download failed: ${e.message}", e)
        }
    }
    
    /**
     * Verify model file integrity and format
     */
    suspend fun verify(modelPath: String): VerificationResult {
        val file = File(modelPath)
        
        if (!file.exists()) {
            return VerificationResult(
                isValid = false,
                sizeGB = 0.0,
                error = "File does not exist"
            )
        }
        
        val sizeGB = file.length().toDouble() / (1024 * 1024 * 1024)
        
        try {
            // Check file header for GPT-OSS magic bytes
            val header = file.inputStream().use { stream ->
                val headerBytes = ByteArray(32)
                val bytesRead = stream.read(headerBytes)
                if (bytesRead < 12) {
                    return VerificationResult(
                        isValid = false,
                        sizeGB = sizeGB,
                        error = "File too small to contain valid header"
                    )
                }
                headerBytes
            }
            
            // Check for GPT-OSS magic header
            val magicBytes = "GPT-OSS v1.0".toByteArray()
            val hasValidHeader = header.sliceArray(0..11).contentEquals(magicBytes)
            
            if (!hasValidHeader) {
                return VerificationResult(
                    isValid = false,
                    sizeGB = sizeGB,
                    error = "Invalid file format - not a GPT-OSS model"
                )
            }
            
            // Calculate checksum if file is reasonable size (< 100GB)
            val checksum = if (sizeGB < 100) {
                calculateMD5(file)
            } else null
            
            // Basic size validation
            if (sizeGB < 1.0) {
                return VerificationResult(
                    isValid = false,
                    sizeGB = sizeGB,
                    error = "File too small to be a valid model"
                )
            }
            
            return VerificationResult(
                isValid = true,
                sizeGB = sizeGB,
                checksum = checksum
            )
            
        } catch (e: Exception) {
            return VerificationResult(
                isValid = false,
                sizeGB = sizeGB,
                error = "Verification failed: ${e.message}"
            )
        }
    }
    
    /**
     * Get detailed model information
     */
    suspend fun getInfo(modelPath: String): ModelInfo {
        val file = File(modelPath)
        val verification = verify(modelPath)
        
        // Try to find metadata file
        val modelDir = file.parentFile?.parentFile // Go up from metal/model.bin to model dir
        val metadataFile = modelDir?.let { File(it, "metadata.json") }
        
        val metadata = if (metadataFile?.exists() == true) {
            try {
                Json.decodeFromString(ModelMetadata.serializer(), metadataFile.readText())
            } catch (e: Exception) {
                null
            }
        } else null
        
        return ModelInfo(
            name = metadata?.name ?: file.nameWithoutExtension,
            path = modelPath,
            isInstalled = file.exists(),
            isValid = verification.isValid,
            sizeGB = verification.sizeGB,
            format = "GPT-OSS Metal",
            parameters = metadata?.parameters ?: 0,
            contextLength = metadata?.contextLength ?: 0,
            vocabularySize = metadata?.vocabularySize ?: 0,
            checksum = verification.checksum,
            issue = if (!verification.isValid) verification.error else null
        )
    }
    
    /**
     * Clean up model cache and temporary files
     */
    suspend fun cleanup(dryRun: Boolean = false): CleanupResult {
        var filesRemoved = 0
        var totalSizeMB = 0.0
        
        // Look for temporary download files
        val tempFiles = mutableListOf<File>()
        
        modelsDir.walkTopDown().forEach { file ->
            when {
                file.name.endsWith(".tmp") -> tempFiles.add(file)
                file.name.endsWith(".partial") -> tempFiles.add(file)
                file.name == ".DS_Store" -> tempFiles.add(file)
                // Corrupted models (exist but fail verification)
                file.name == "model.bin" -> {
                    val verification = verify(file.absolutePath)
                    if (!verification.isValid && file.length() > 0) {
                        tempFiles.add(file)
                    }
                }
            }
        }
        
        for (file in tempFiles) {
            val sizeMB = file.length().toDouble() / (1024 * 1024)
            totalSizeMB += sizeMB
            
            if (verbose) {
                println("${if (dryRun) "Would remove" else "Removing"}: ${file.absolutePath} (${String.format("%.1f", sizeMB)}MB)")
            }
            
            if (!dryRun) {
                try {
                    if (file.delete()) {
                        filesRemoved++
                    }
                } catch (e: Exception) {
                    if (verbose) {
                        println("Failed to delete ${file.absolutePath}: ${e.message}")
                    }
                }
            } else {
                filesRemoved++
            }
        }
        
        return CleanupResult(
            filesRemoved = filesRemoved,
            spaceMB = totalSizeMB
        )
    }
    
    // Private helper methods
    
    private fun isHuggingFaceCliAvailable(): Boolean {
        return try {
            val process = ProcessBuilder("hf", "--version").start()
            process.waitFor(5, TimeUnit.SECONDS) && process.exitValue() == 0
        } catch (e: Exception) {
            false
        }
    }
    
    private suspend fun downloadViaHuggingFaceCli(
        metadata: ModelMetadata,
        targetFile: File,
        progressCallback: (Double) -> Unit
    ) {
        val tempDir = Files.createTempDirectory("noesis-model-download").toFile()
        
        try {
            val process = ProcessBuilder(
                "hf", "download",
                metadata.huggingfaceRepo,
                metadata.filename,
                "--local-dir", tempDir.absolutePath,
                "--local-dir-use-symlinks", "false"
            ).start()
            
            // Monitor progress by checking file size
            var lastProgress = 0.0
            while (process.isAlive) {
                val downloadedFile = File(tempDir, metadata.filename)
                if (downloadedFile.exists()) {
                    val currentSize = downloadedFile.length().toDouble()
                    val expectedSize = metadata.expectedSizeGB * 1024 * 1024 * 1024
                    val progress = (currentSize / expectedSize).coerceAtMost(1.0)
                    
                    if (progress > lastProgress + 0.05) { // Report every 5%
                        progressCallback(progress)
                        lastProgress = progress
                    }
                }
                delay(1000)
            }
            
            val exitCode = process.waitFor()
            if (exitCode != 0) {
                throw RuntimeException("HuggingFace CLI download failed with exit code $exitCode")
            }
            
            // Move downloaded file to final location
            val downloadedFile = File(tempDir, metadata.filename)
            Files.move(downloadedFile.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
            progressCallback(1.0)
            
        } finally {
            tempDir.deleteRecursively()
        }
    }
    
    private suspend fun downloadDirect(
        url: String,
        targetFile: File,
        progressCallback: (Double) -> Unit
    ) {
        val connection = URL(url).openConnection()
        val contentLength = connection.contentLengthLong
        
        targetFile.parentFile.mkdirs()
        val tempFile = File(targetFile.absolutePath + ".tmp")
        
        try {
            connection.getInputStream().use { input ->
                Channels.newChannel(input).use { inputChannel ->
                    targetFile.outputStream().use { output ->
                        output.channel.use { outputChannel ->
                            var transferred = 0L
                            var lastProgress = 0.0
                            
                            while (transferred < contentLength) {
                                val chunk = outputChannel.transferFrom(inputChannel, transferred, 8192)
                                if (chunk == 0L) break
                                
                                transferred += chunk
                                val progress = transferred.toDouble() / contentLength
                                
                                if (progress > lastProgress + 0.01) { // Report every 1%
                                    progressCallback(progress)
                                    lastProgress = progress
                                }
                            }
                        }
                    }
                }
            }
            
            // Move temp file to final location
            Files.move(tempFile.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
            progressCallback(1.0)
            
        } catch (e: Exception) {
            tempFile.delete()
            throw e
        }
    }
    
    private fun calculateMD5(file: File): String {
        val md = MessageDigest.getInstance("MD5")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                md.update(buffer, 0, bytesRead)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
    
    companion object {
        // Supported model configurations
        private val SUPPORTED_MODELS = mapOf(
            NoesisConstants.Models.GPT_OSS_20B to ModelMetadata(
                name = NoesisConstants.Models.GPT_OSS_20B,
                huggingfaceRepo = NoesisConstants.Models.HF_REPO_20B,
                filename = NoesisConstants.Models.HF_FILENAME,
                downloadUrl = "https://huggingface.co/${NoesisConstants.Models.HF_REPO_20B}/resolve/main/${NoesisConstants.Models.HF_FILENAME}",
                expectedSizeGB = NoesisConstants.Models.GPT_OSS_20B_SIZE_GB,
                parameters = NoesisConstants.Models.GPT_OSS_20B_PARAMS,
                contextLength = NoesisConstants.Models.DEFAULT_CONTEXT_LENGTH,
                vocabularySize = NoesisConstants.Models.DEFAULT_VOCAB_SIZE
            ),
            NoesisConstants.Models.GPT_OSS_120B to ModelMetadata(
                name = NoesisConstants.Models.GPT_OSS_120B, 
                huggingfaceRepo = NoesisConstants.Models.HF_REPO_120B,
                filename = NoesisConstants.Models.HF_FILENAME,
                downloadUrl = "https://huggingface.co/${NoesisConstants.Models.HF_REPO_120B}/resolve/main/${NoesisConstants.Models.HF_FILENAME}",
                expectedSizeGB = NoesisConstants.Models.GPT_OSS_120B_SIZE_GB,
                parameters = NoesisConstants.Models.GPT_OSS_120B_PARAMS,
                contextLength = NoesisConstants.Models.DEFAULT_CONTEXT_LENGTH,
                vocabularySize = NoesisConstants.Models.DEFAULT_VOCAB_SIZE
            )
        )
    }
}

// Data classes for model management
@Serializable
data class ModelInfo(
    val name: String,
    val path: String,
    val isInstalled: Boolean,
    val isValid: Boolean,
    val sizeGB: Double,
    val format: String = "GPT-OSS Metal",
    val parameters: Long = 0,
    val contextLength: Int = 0,
    val vocabularySize: Int = 0,
    val checksum: String? = null,
    val issue: String? = null
)

@Serializable
data class ModelMetadata(
    val name: String,
    val huggingfaceRepo: String,
    val filename: String,
    val downloadUrl: String,
    val expectedSizeGB: Double,
    val parameters: Long,
    val contextLength: Int,
    val vocabularySize: Int
)

@Serializable
data class VerificationResult(
    val isValid: Boolean,
    val sizeGB: Double,
    val checksum: String? = null,
    val error: String? = null
)

@Serializable
data class CleanupResult(
    val filesRemoved: Int,
    val spaceMB: Double
)