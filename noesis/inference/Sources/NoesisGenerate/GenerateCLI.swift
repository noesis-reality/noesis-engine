import Foundation
import ArgumentParser
import NoesisEngine
import NoesisTools
@preconcurrency import Metal

/// Command-line tool for text generation with GPT-OSS
@main
@available(macOS 10.15, *)
struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "noesis-generate",
        abstract: "Generate text using GPT-OSS models"
    )
    
    @Argument(help: "Input prompt text")
    var prompt: String
    
    @Option(help: "Path to model.bin file (uses config default if not specified)")
    var modelPath: String?
    
    @Option(name: .shortAndLong, help: "Maximum tokens to generate")
    var maxTokens: Int = 100
    
    @Option(name: .shortAndLong, help: "Temperature for sampling (0.0 = greedy)")
    var temperature: Float = 0.7
    
    @Option(name: .long, help: "Top-p sampling threshold")
    var topP: Float = 0.9
    
    @Option(name: .long, help: "Repetition penalty")
    var repetitionPenalty: Float = 1.1
    
    @Option(name: .long, help: "System prompt")
    var system: String?
    
    @Option(name: .long, help: "Output format (text, tokens, json)")
    var format: OutputFormat = .text
    
    @Flag(name: .long, help: "Show generation statistics")
    var stats = false
    
    @Flag(name: .long, help: "Verbose output")
    var verbose = false
    
    enum OutputFormat: String, ExpressibleByArgument {
        case text
        case tokens
        case json
    }
    
    mutating func run() async throws {
        // Load configuration
        let configFile = ConfigLoader.load()
        
        // Resolve model path
        guard let resolvedModelPath = ConfigLoader.resolveModelPath(modelPath) else {
            throw ValidationError.fileNotFound("No model path specified and no default model in config. Use --model-path or set up config.")
        }
        
        // Use config defaults if values weren't explicitly set
        let finalTemperature = temperature != 0.7 ? temperature : (configFile?.generation.defaultTemperature ?? temperature)
        let finalMaxTokens = maxTokens != 100 ? maxTokens : (configFile?.generation.defaultMaxTokens ?? maxTokens)
        let finalTopP = topP != 0.9 ? topP : (configFile?.generation.defaultTopP ?? topP)
        
        let config = GenerateConfig(
            modelPath: resolvedModelPath,
            prompt: prompt,
            maxTokens: finalMaxTokens,
            temperature: finalTemperature,
            topP: finalTopP,
            repetitionPenalty: repetitionPenalty,
            system: system,
            format: format,
            stats: stats,
            verbose: verbose
        )
        
        try await Self.runGeneration(config: config)
    }
    
    @MainActor
    static func runGeneration(config: GenerateConfig) async throws {
        let modelPath = config.modelPath
        let prompt = config.prompt
        let maxTokens = config.maxTokens
        let temperature = config.temperature
        let topP = config.topP
        // let repetitionPenalty = config.repetitionPenalty // Not used in GptossSampler
        let system = config.system
        let format = config.format
        let stats = config.stats
        let verbose = config.verbose
        let modelURL = URL(fileURLWithPath: modelPath)
        
        // Verify model exists
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ValidationError.fileNotFound("Model not found at \(modelURL.path)")
        }
        
        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RuntimeError.metalNotAvailable
        }
        
        if verbose {
            print("Loading model from \(modelURL.path)...")
        }
        
        // Load model
        let model = try ModelLoader.loadModel(from: modelURL, device: device)
        
        print("🎯 Model loaded successfully")
        
        if verbose {
            print("Model loaded:")
            print("  Vocabulary: \(model.config.vocabularySize)")
            print("  Blocks: \(model.config.numBlocks)")
            print("  Embedding dim: \(model.config.embeddingDim)")
        }
        
        print("🔧 Creating context...")
        // Create context
        let context = GptossContext(model: model, contextLength: 4096)
        
        print("📝 Initializing tokenizer...")
        // Prepare prompt with Harmony format (required for GPT-OSS)
        let tokenizer = try O200kTokenizer()
        
        // Use Python Metal example's system prompt if not specified
        let defaultSystemPrompt = """
        You are ChatGPT, a large language model trained by OpenAI.
        Knowledge cutoff: 2024-06
        Current date: \(Date().formatted(date: .abbreviated, time: .omitted))
        
        reasoning effort high
        
        # Valid channels: analysis, final. Channel must be included for every message.
        """
        
        // Always use Harmony format for GPT-OSS
        let promptTokens = tokenizer.createHarmonyPrompt(
            systemMessage: system ?? defaultSystemPrompt,
            userMessage: prompt
        )
        
        if verbose {
            print("Created Harmony prompt with \(promptTokens.count) tokens")
            print("First 10 tokens: \(Array(promptTokens.prefix(10)))")
            // Decode to see what the prompt looks like
            let promptText = tokenizer.decode(Array(promptTokens.prefix(100)))
            print("Prompt preview: \(promptText.prefix(200))...")
        }
        
        // Create sampler
        let sampler = GptossSampler(
            temperature: temperature,
            topP: topP
        )
        
        // Generate
        let pipeline = try GenerationPipeline(model: model, context: context)
        
        var generatedTokens: [UInt32] = []
        let startTime = Date()
        
        if verbose {
            print("\nGenerating...")
        }
        
        // Get stop tokens from tokenizer (uses Harmony's proper stop tokens)
        let stopTokenSet = tokenizer.stopTokens()
        
        if verbose {
            print("Stop tokens from Harmony: \(stopTokenSet)")
        }
        
        // Capture values before closure
        let outputFormat = format
        let showStats = stats
        
        let tokens = try pipeline.generateTokens(
            prompt: promptTokens,
            maxTokens: maxTokens,
            sampler: sampler
        ) { token in
            generatedTokens.append(token)
            
            // Stop on Harmony stop tokens (properly retrieved from Harmony)
            if stopTokenSet.contains(token) {
                return false
            }
            
            // Stream output if in text mode
            if outputFormat == .text && !showStats {
                let decoded = tokenizer.decode([token])
                print(decoded, terminator: "")
                fflush(stdout)
            }
            
            // Debug: always print token ID
            if ProcessInfo.processInfo.environment["NOESIS_DEBUG"] == "1" {
                print(" [token:\(token)]", terminator: "")
                fflush(stdout)
            }
            
            return true
        }
        
        let endTime = Date()
        let generationTime = endTime.timeIntervalSince(startTime)
        
        // Output results based on format
        switch format {
        case .text:
            if stats {
                // Print full text if we haven't been streaming
                let fullText = tokenizer.decode(tokens)
                print(fullText)
            } else {
                // Already streamed, just add newline
                print()
            }
            
        case .tokens:
            print("Generated tokens: \(tokens)")
            
        case .json:
            let output = GenerationOutput(
                prompt: prompt,
                promptTokens: promptTokens,
                generatedTokens: tokens,
                text: tokenizer.decode(tokens),
                stats: stats ? GenerationStats(
                    tokensGenerated: tokens.count,
                    timeSeconds: generationTime,
                    tokensPerSecond: Double(tokens.count) / generationTime
                ) : nil
            )
            
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let jsonData = try encoder.encode(output)
            print(String(data: jsonData, encoding: .utf8)!)
        }
        
        // Print statistics if requested
        if stats && format != .json {
            print("\n--- Statistics ---")
            print("Tokens generated: \(tokens.count)")
            print("Time: \(String(format: "%.2f", generationTime)) seconds")
            print("Speed: \(String(format: "%.2f", Double(tokens.count) / generationTime)) tokens/sec")
        }
    }
}

// MARK: - Output Types

struct GenerateConfig {
    let modelPath: String
    let prompt: String
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let repetitionPenalty: Float
    let system: String?
    let format: Generate.OutputFormat
    let stats: Bool
    let verbose: Bool
}

struct GenerationOutput: Codable {
    let prompt: String
    let promptTokens: [UInt32]
    let generatedTokens: [UInt32]
    let text: String
    let stats: GenerationStats?
}

struct GenerationStats: Codable {
    let tokensGenerated: Int
    let timeSeconds: Double
    let tokensPerSecond: Double
}

// MARK: - Errors

enum ValidationError: LocalizedError {
    case fileNotFound(String)
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound(let msg):
            return msg
        }
    }
}

enum RuntimeError: LocalizedError {
    case metalNotAvailable
    
    var errorDescription: String? {
        switch self {
        case .metalNotAvailable:
            return "Metal is not available on this system"
        }
    }
}

