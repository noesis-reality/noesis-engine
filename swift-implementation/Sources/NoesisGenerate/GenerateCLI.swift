import Foundation
import ArgumentParser
import NoesisEngine
import NoesisTools
@preconcurrency import Metal

/// Command-line tool for text generation with GPT-OSS
@available(macOS 10.15, *)
struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "noesis-generate",
        abstract: "Generate text using GPT-OSS models"
    )
    
    @Argument(help: "Path to model.bin file")
    var modelPath: String
    
    @Argument(help: "Input prompt text")
    var prompt: String
    
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
    
    @Flag(name: .long, help: "Use Harmony format for conversation")
    var harmony = false
    
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
        let config = GenerateConfig(
            modelPath: modelPath,
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            system: system,
            format: format,
            harmony: harmony,
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
        let harmony = config.harmony
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
        
        if verbose {
            print("Model loaded:")
            print("  Vocabulary: \(model.config.vocabularySize)")
            print("  Blocks: \(model.config.numBlocks)")
            print("  Embedding dim: \(model.config.embeddingDim)")
        }
        
        // Create context
        let context = GptossContext(model: model, contextLength: 4096)
        
        // Prepare prompt
        let tokenizer = try O200kTokenizer()
        var promptTokens: [UInt32]
        
        if harmony {
            // Use Harmony format
            promptTokens = tokenizer.createHarmonyPrompt(
                systemMessage: system,
                userMessage: prompt
            )
            
            if verbose {
                print("Created Harmony prompt with \(promptTokens.count) tokens")
            }
        } else {
            // Simple text prompt
            var fullPrompt = prompt
            if let systemPrompt = system {
                fullPrompt = systemPrompt + "\n\n" + prompt
            }
            promptTokens = tokenizer.encode(fullPrompt)
            
            if verbose {
                print("Encoded prompt to \(promptTokens.count) tokens")
            }
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
        
        // Capture values before closure
        let useHarmony = harmony
        let outputFormat = format
        let showStats = stats
        
        let tokens = try pipeline.generateTokens(
            prompt: promptTokens,
            maxTokens: maxTokens,
            sampler: sampler
        ) { token in
            generatedTokens.append(token)
            
            // Stop on special tokens
            if useHarmony {
                if token == 200002 || // <|return|>
                   token == 200012 || // <|call|>
                   token == 200007 || // <|end|>
                   token == 199999 {  // <|endoftext|>
                    return false
                }
            } else if token == 199999 { // Always stop on EOS
                return false
            }
            
            // Stream output if in text mode
            if outputFormat == .text && !showStats {
                let decoded = tokenizer.decode([token])
                print(decoded, terminator: "")
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
    let harmony: Bool
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

// MARK: - Main

@main
@available(macOS 10.15, *)
struct GenerateCLI {
    static func main() async {
        await Generate.main()
    }
}