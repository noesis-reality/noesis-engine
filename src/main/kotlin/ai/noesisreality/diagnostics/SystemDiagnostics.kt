package ai.noesisreality.diagnostics

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * System Diagnostics - Pure Kotlin implementation
 * 
 * Handles all system validation, environment checking, and issue detection
 * without relying on the Swift inference engine. Only uses native system
 * commands and file system operations.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class SystemDiagnostics(private val verbose: Boolean = false) {
    
    /**
     * Run complete system diagnostics
     */
    suspend fun runComplete(
        checkModelFile: String? = null,
        checkGpu: Boolean = false,
        checkMemory: Boolean = false
    ): DiagnosticReport {
        if (verbose) {
            println("🔍 Running comprehensive system diagnostics...")
        }
        
        val systemInfo = checkSystemInfo()
        val gpuInfo = if (checkGpu) checkGpuInfo() else GpuInfo.placeholder()
        val memoryInfo = if (checkMemory) checkMemoryInfo() else MemoryInfo.placeholder()
        val modelInfo = checkModels(checkModelFile)
        val environmentInfo = checkEnvironment()
        
        val issues = mutableListOf<String>()
        
        // Analyze findings and identify issues
        if (!systemInfo.xcodeInstalled) {
            issues.add("Xcode Command Line Tools not installed")
        }
        
        if (systemInfo.macosVersion.startsWith("14.") || systemInfo.macosVersion.startsWith("13.")) {
            issues.add("macOS version ${systemInfo.macosVersion} may not support Metal 4 - update to macOS 15.5+")
        }
        
        if (memoryInfo.totalMemoryGB < 16) {
            issues.add("System has less than 16GB RAM - may struggle with gpt-oss-20b")
        }
        
        if (!environmentInfo.huggingfaceCliAvailable) {
            issues.add("HuggingFace CLI not available - model downloads may fail")
        }
        
        modelInfo.forEach { model ->
            if (model.isInstalled && !model.isValid) {
                issues.add("Model ${model.name} is corrupted or invalid")
            }
        }
        
        val recommendations = generateRecommendations(systemInfo, memoryInfo, modelInfo)
        
        return DiagnosticReport(
            platform = systemInfo.platform,
            macosVersion = systemInfo.macosVersion,
            xcodeInstalled = systemInfo.xcodeInstalled,
            gpuName = gpuInfo.name,
            metal4Supported = gpuInfo.metal4Supported,
            gpuMemoryGB = gpuInfo.memoryGB,
            systemMemoryGB = memoryInfo.totalMemoryGB,
            availableMemoryGB = memoryInfo.availableMemoryGB,
            models = modelInfo,
            environment = environmentInfo,
            issues = issues,
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Check basic system information
     */
    private fun checkSystemInfo(): SystemInfo {
        val osName = System.getProperty("os.name")
        val osVersion = System.getProperty("os.version")
        val arch = System.getProperty("os.arch")
        
        // Check for Xcode CLI tools
        val xcodeInstalled = try {
            val process = ProcessBuilder("xcode-select", "--print-path").start()
            process.waitFor(5, TimeUnit.SECONDS) && process.exitValue() == 0
        } catch (e: Exception) {
            false
        }
        
        // Get more detailed macOS version
        val macosVersion = try {
            val process = ProcessBuilder("sw_vers", "-productVersion").start()
            if (process.waitFor(5, TimeUnit.SECONDS) && process.exitValue() == 0) {
                process.inputStream.bufferedReader().readText().trim()
            } else {
                osVersion
            }
        } catch (e: Exception) {
            osVersion
        }
        
        val platform = "$osName $arch"
        val isAppleSilicon = arch.contains("aarch64") || arch.contains("arm64")
        
        return SystemInfo(
            platform = platform,
            macosVersion = macosVersion,
            xcodeInstalled = xcodeInstalled,
            isAppleSilicon = isAppleSilicon
        )
    }
    
    /**
     * Check GPU information (basic detection)
     */
    private fun checkGpuInfo(): GpuInfo {
        // Basic GPU detection - would need more sophisticated detection for real Metal info
        val gpuName = try {
            val process = ProcessBuilder("system_profiler", "SPDisplaysDataType").start()
            if (process.waitFor(10, TimeUnit.SECONDS) && process.exitValue() == 0) {
                val output = process.inputStream.bufferedReader().readText()
                // Parse for GPU name - this is simplified
                when {
                    output.contains("Apple M3") -> "Apple M3"
                    output.contains("Apple M2") -> "Apple M2" 
                    output.contains("Apple M1") -> "Apple M1"
                    else -> "Apple Silicon GPU"
                }
            } else {
                "Unknown GPU"
            }
        } catch (e: Exception) {
            "Unknown GPU"
        }
        
        // Estimate based on GPU type
        val memoryGB = when {
            gpuName.contains("M3 Max") -> 48
            gpuName.contains("M3 Pro") -> 24
            gpuName.contains("M3") -> 16
            gpuName.contains("M2 Max") -> 32
            gpuName.contains("M2 Pro") -> 16
            gpuName.contains("M2") -> 8
            gpuName.contains("M1 Max") -> 32
            gpuName.contains("M1 Pro") -> 16
            gpuName.contains("M1") -> 8
            else -> 8
        }
        
        // Metal 4 support estimation
        val metal4Supported = when {
            gpuName.contains("M3") -> true
            gpuName.contains("M2") -> true
            gpuName.contains("M1") -> false // Estimate - M1 may not have full Metal 4
            else -> false
        }
        
        return GpuInfo(
            name = gpuName,
            metal4Supported = metal4Supported,
            memoryGB = memoryGB
        )
    }
    
    /**
     * Check memory information
     */
    private fun checkMemoryInfo(): MemoryInfo {
        val totalMemoryBytes = try {
            val process = ProcessBuilder("sysctl", "-n", "hw.memsize").start()
            if (process.waitFor(5, TimeUnit.SECONDS) && process.exitValue() == 0) {
                process.inputStream.bufferedReader().readText().trim().toLongOrNull() ?: 0L
            } else {
                0L
            }
        } catch (e: Exception) {
            0L
        }
        
        val totalMemoryGB = (totalMemoryBytes / (1024 * 1024 * 1024)).toInt()
        val availableMemoryGB = maxOf(0, totalMemoryGB - 4) // Reserve 4GB for system
        
        return MemoryInfo(
            totalMemoryGB = totalMemoryGB,
            availableMemoryGB = availableMemoryGB
        )
    }
    
    /**
     * Check available models
     */
    private fun checkModels(specificModelPath: String? = null): List<ModelInfo> {
        val models = mutableListOf<ModelInfo>()
        
        // Check specific model if provided
        specificModelPath?.let { path ->
            models.add(checkSingleModel(path))
        }
        
        // Check standard model locations
        val homeDir = System.getProperty("user.home")
        val noesisModelsDir = File("$homeDir/.noesis/models")
        
        if (noesisModelsDir.exists()) {
            listOf("gpt-oss-20b", "gpt-oss-120b").forEach { modelName ->
                val modelFile = File(noesisModelsDir, "$modelName/metal/model.bin")
                if (modelFile.exists()) {
                    models.add(checkSingleModel(modelFile.absolutePath, modelName))
                } else {
                    // Model not installed
                    val expectedSize = if (modelName.contains("20b")) 13.75 else 45.5
                    models.add(ModelInfo(
                        name = modelName,
                        path = modelFile.absolutePath,
                        isInstalled = false,
                        isValid = false,
                        sizeGB = expectedSize,
                        issue = "Not downloaded"
                    ))
                }
            }
        }
        
        return models
    }
    
    /**
     * Check a single model file
     */
    private fun checkSingleModel(path: String, name: String? = null): ModelInfo {
        val file = File(path)
        val modelName = name ?: file.nameWithoutExtension
        
        if (!file.exists()) {
            return ModelInfo(
                name = modelName,
                path = path,
                isInstalled = false,
                isValid = false,
                sizeGB = 0.0,
                issue = "File not found"
            )
        }
        
        val sizeGB = file.length().toDouble() / (1024 * 1024 * 1024)
        
        // Basic validation - check for GPT-OSS magic header
        val isValid = try {
            val header = file.inputStream().use { it.readNBytes(12) }
            val magicHeader = "GPT-OSS v1.0".toByteArray()
            header.contentEquals(magicHeader) && sizeGB > 1.0 // Must be at least 1GB
        } catch (e: Exception) {
            false
        }
        
        return ModelInfo(
            name = modelName,
            path = path,
            isInstalled = true,
            isValid = isValid,
            sizeGB = sizeGB,
            issue = if (!isValid) "Invalid or corrupted model file" else null
        )
    }
    
    /**
     * Check environment dependencies
     */
    private fun checkEnvironment(): EnvironmentInfo {
        val pythonAvailable = checkCommand("python3", "--version")
        val pipAvailable = checkCommand("pip3", "--version")
        val hfAvailable = checkCommand("hf", "--version")
        val gitAvailable = checkCommand("git", "--version")
        val curlAvailable = checkCommand("curl", "--version")
        
        return EnvironmentInfo(
            pythonAvailable = pythonAvailable,
            pipAvailable = pipAvailable,
            huggingfaceCliAvailable = hfAvailable,
            gitAvailable = gitAvailable,
            curlAvailable = curlAvailable
        )
    }
    
    /**
     * Check if a command is available
     */
    private fun checkCommand(command: String, vararg args: String): Boolean {
        return try {
            val process = ProcessBuilder(command, *args).start()
            process.waitFor(5, TimeUnit.SECONDS) && process.exitValue() == 0
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Generate recommendations based on findings
     */
    private fun generateRecommendations(
        systemInfo: SystemInfo,
        memoryInfo: MemoryInfo,
        models: List<ModelInfo>
    ): List<String> {
        val recommendations = mutableListOf<String>()
        
        if (!systemInfo.xcodeInstalled) {
            recommendations.add("Install Xcode Command Line Tools: xcode-select --install")
        }
        
        when {
            memoryInfo.totalMemoryGB >= 32 -> {
                recommendations.add("Your system can run gpt-oss-120b (32GB+ RAM)")
                recommendations.add("Consider gpt-oss-20b for faster inference")
            }
            memoryInfo.totalMemoryGB >= 16 -> {
                recommendations.add("Your system is optimized for gpt-oss-20b (16GB+ RAM)")
                recommendations.add("gpt-oss-120b may work but could be slow")
            }
            else -> {
                recommendations.add("Consider upgrading to 16GB+ RAM for better performance")
                recommendations.add("gpt-oss-20b may run but with limited context")
            }
        }
        
        val hasValidModels = models.any { it.isInstalled && it.isValid }
        if (!hasValidModels) {
            recommendations.add("Download a model: noesis model download gpt-oss-20b")
        }
        
        if (systemInfo.macosVersion.startsWith("14.")) {
            recommendations.add("Update to macOS 15.5+ for full Metal 4 support")
        }
        
        return recommendations
    }
    
    /**
     * Attempt to fix common issues
     */
    fun attemptFixes(issues: List<String>): Int {
        var fixedCount = 0
        
        for (issue in issues) {
            try {
                when {
                    issue.contains("HuggingFace CLI") -> {
                        if (verbose) println("🔧 Attempting to install HuggingFace CLI...")
                        val process = ProcessBuilder("pip3", "install", "huggingface-hub[cli]").start()
                        if (process.waitFor(30, TimeUnit.SECONDS) && process.exitValue() == 0) {
                            fixedCount++
                            if (verbose) println("✅ HuggingFace CLI installed")
                        }
                    }
                    issue.contains("corrupted") -> {
                        // Could attempt to re-download corrupted models
                        if (verbose) println("⚠️ Cannot auto-fix corrupted models - manual re-download required")
                    }
                }
            } catch (e: Exception) {
                if (verbose) {
                    println("❌ Failed to fix: $issue - ${e.message}")
                }
            }
        }
        
        return fixedCount
    }
}

// Data classes for diagnostic results
@Serializable
data class DiagnosticReport(
    val platform: String,
    val macosVersion: String,
    val xcodeInstalled: Boolean,
    val gpuName: String,
    val metal4Supported: Boolean,
    val gpuMemoryGB: Int,
    val systemMemoryGB: Int,
    val availableMemoryGB: Int,
    val models: List<ModelInfo>,
    val environment: EnvironmentInfo,
    val issues: List<String>,
    val recommendations: List<String>,
    val timestamp: Long
) {
    fun toJson(): String = DiagnosticReport.json.encodeToString(serializer(), this)
    
    companion object {
        private val json = Json { prettyPrint = true }
    }
}

@Serializable
data class SystemInfo(
    val platform: String,
    val macosVersion: String,
    val xcodeInstalled: Boolean,
    val isAppleSilicon: Boolean
)

@Serializable
data class GpuInfo(
    val name: String,
    val metal4Supported: Boolean,
    val memoryGB: Int
) {
    companion object {
        fun placeholder() = GpuInfo("Unknown", false, 8)
    }
}

@Serializable
data class MemoryInfo(
    val totalMemoryGB: Int,
    val availableMemoryGB: Int
) {
    companion object {
        fun placeholder() = MemoryInfo(16, 12)
    }
}

@Serializable
data class ModelInfo(
    val name: String,
    val path: String,
    val isInstalled: Boolean,
    val isValid: Boolean,
    val sizeGB: Double,
    val issue: String? = null
)

@Serializable
data class EnvironmentInfo(
    val pythonAvailable: Boolean,
    val pipAvailable: Boolean,
    val huggingfaceCliAvailable: Boolean,
    val gitAvailable: Boolean,
    val curlAvailable: Boolean
)