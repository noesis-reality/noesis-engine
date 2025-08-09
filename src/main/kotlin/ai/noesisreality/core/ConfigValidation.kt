package ai.noesisreality.core

import ai.noesisreality.benchmarks.BenchmarkConfig

/**
 * Configuration validation utilities
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
object ConfigValidation {
    
    /**
     * Validate benchmark configuration parameters
     */
    fun validateBenchmarkConfig(config: BenchmarkConfig): Result<Unit> {
        val errors = mutableListOf<String>()
        
        // Validate basic parameters
        if (config.maxTokens <= 0) {
            errors.add("maxTokens must be positive, got: ${config.maxTokens}")
        }
        if (config.maxTokens > 2048) {
            errors.add("maxTokens too large (max 2048), got: ${config.maxTokens}")
        }
        
        if (config.temperature < 0.0f || config.temperature > 2.0f) {
            errors.add("temperature must be in range [0.0, 2.0], got: ${config.temperature}")
        }
        
        if (config.iterations <= 0) {
            errors.add("iterations must be positive, got: ${config.iterations}")
        }
        if (config.iterations > 100) {
            errors.add("iterations too large (max 100), got: ${config.iterations}")
        }
        
        if (config.prompt.isBlank()) {
            errors.add("prompt cannot be blank")
        }
        if (config.prompt.length > 4096) {
            errors.add("prompt too long (max 4096 chars), got: ${config.prompt.length}")
        }
        
        return if (errors.isEmpty()) {
            Result.success(Unit)
        } else {
            Result.failure(IllegalArgumentException("Configuration errors: ${errors.joinToString("; ")}"))
        }
    }
    
    /**
     * Validate system requirements for benchmarking
     */
    fun validateSystemRequirements(): Result<SystemCapabilities> {
        val capabilities = SystemCapabilities()
        val errors = mutableListOf<String>()
        
        // Check memory
        val runtime = Runtime.getRuntime()
        val totalMemoryGB = runtime.maxMemory() / (1024 * 1024 * 1024)
        
        if (totalMemoryGB < NoesisConstants.Defaults.RECOMMENDED_RAM_20B_GB) {
            errors.add("Insufficient memory: ${totalMemoryGB}GB available, ${NoesisConstants.Defaults.RECOMMENDED_RAM_20B_GB}GB recommended")
        }
        
        capabilities.availableMemoryGB = totalMemoryGB.toInt()
        capabilities.canRunLarge = totalMemoryGB >= NoesisConstants.Defaults.RECOMMENDED_RAM_120B_GB
        
        // Check OS compatibility
        val osName = System.getProperty("os.name").lowercase()
        capabilities.isMacOS = osName.contains("mac")
        capabilities.isLinux = osName.contains("linux")
        
        if (!capabilities.isMacOS && !capabilities.isLinux) {
            errors.add("Unsupported OS: $osName. Only macOS and Linux are supported.")
        }
        
        return if (errors.isEmpty()) {
            Result.success(capabilities)
        } else {
            Result.failure(IllegalStateException("System requirement errors: ${errors.joinToString("; ")}"))
        }
    }
}

/**
 * System capability information
 */
data class SystemCapabilities(
    var availableMemoryGB: Int = 0,
    var canRunLarge: Boolean = false,
    var isMacOS: Boolean = false,
    var isLinux: Boolean = false
)