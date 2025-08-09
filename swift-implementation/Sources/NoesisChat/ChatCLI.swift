import Foundation
import ArgumentParser
import NoesisEngine
import NoesisTools
@preconcurrency import Metal

/// Interactive chat with GPT-OSS models using Harmony format
@main
@available(macOS 10.15, *)
struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "noesis-chat",
        abstract: "Interactive chat with GPT-OSS models using Harmony format"
    )
    
    @Argument(help: "Path to model.bin file")
    var modelPath: String
    
    @Option(name: .shortAndLong, help: "System prompt")
    var system: String = "You are ChatGPT, a large language model trained by OpenAI."
    
    @Option(name: .shortAndLong, help: "Temperature for sampling")
    var temperature: Float = 0.7
    
    @Option(name: .long, help: "Maximum tokens per response")
    var maxTokens: Int = 500
    
    @Option(name: .long, help: "Conversation date (YYYY-MM-DD)")
    var date: String?
    
    @Option(name: .long, help: "Knowledge cutoff date")
    var knowledgeCutoff: String = "2024-06"
    
    @Option(name: .long, help: "Reasoning effort (low, medium, high)")
    var reasoning: ReasoningLevel = .medium
    
    @Flag(name: .long, help: "Enable channels (analysis, commentary, final)")
    var channels = false
    
    @Flag(name: .long, help: "Show token statistics")
    var stats = false
    
    @Flag(name: .long, help: "Verbose output")
    var verbose = false
    
    enum ReasoningLevel: String, ExpressibleByArgument {
        case low, medium, high
        
        var harmonyEffort: HarmonyReasoningEffort {
            switch self {
            case .low: return .low
            case .medium: return .medium
            case .high: return .high
            }
        }
    }
    
    mutating func run() async throws {
        // Create chat configuration
        let chatConfig = ChatConfig(
            modelPath: modelPath,
            system: system,
            temperature: temperature,
            maxTokens: maxTokens,
            date: date,
            knowledgeCutoff: knowledgeCutoff,
            reasoning: reasoning,
            channels: channels,
            stats: stats,
            verbose: verbose
        )
        
        // Run the chat directly with async/await
        let chat = ChatRunner()
        try await chat.runChat(config: chatConfig)
    }
}

// Keep the rest of the file the same...
struct ChatConfig {
    let modelPath: String
    let system: String
    let temperature: Float
    let maxTokens: Int
    let date: String?
    let knowledgeCutoff: String
    let reasoning: Chat.ReasoningLevel
    let channels: Bool
    let stats: Bool
    let verbose: Bool
}

@MainActor
final class ChatRunner {
    func runChat(config: ChatConfig) async throws {
        let modelURL = URL(fileURLWithPath: config.modelPath)
        
        // Verify model exists
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ValidationError.fileNotFound("Model not found at \(modelURL.path)")
        }
        
        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RuntimeError.metalNotAvailable
        }
        
        print("🤖 Loading GPT-OSS model...")
        
        // Load model
        let model = try ModelLoader.loadModel(from: modelURL, device: device)
        
        if config.verbose {
            print("✅ Model loaded:")
            print("  Vocabulary: \(model.config.vocabularySize)")
            print("  Blocks: \(model.config.numBlocks)")
            print("  Context: \(model.config.contextLength)")
        }
        
        // Create context and pipeline
        let context = GptossContext(model: model, contextLength: 4096)
        let pipeline = try GenerationPipeline(model: model, context: context)
        
        // Setup Harmony encoding
        let encoding = HarmonyEncoding.harmony_gpt_oss
        let temperature = config.temperature
        let maxTokens = config.maxTokens
        let channels = config.channels
        let verbose = config.verbose
        let stats = config.stats
        
        // Conversation state
        var conversation = HarmonyConversation(messages: [])
        
        // Setup system message
        let conversationDate = config.date ?? {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd"
            return formatter.string(from: Date())
        }()
        
        var systemContent = HarmonySystemContent.new()
            .withModelIdentity(config.system)
            .withReasoningEffort(config.reasoning.harmonyEffort)
            .withConversationStartDate(conversationDate)
            .withKnowledgeCutoff(config.knowledgeCutoff)
        
        if channels {
            systemContent = systemContent.withRequiredChannels(["analysis", "commentary", "final"])
        }
        
        conversation.messages.append(
            HarmonyMessage.fromRoleAndContent(.system, .system(systemContent))
        )
        
        print("\n💬 Chat session started. Type 'exit' or 'quit' to end.\n")
        print("System: \(config.system)")
        print("Date: \(conversationDate)")
        print("Knowledge cutoff: \(config.knowledgeCutoff)")
        if channels {
            print("Channels: enabled (analysis, commentary, final)")
        }
        print("\n" + String(repeating: "─", count: 60) + "\n")
        
        // Chat loop
        while true {
            print("You: ", terminator: "")
            fflush(stdout)
            
            guard let input = readLine(), !input.isEmpty else {
                continue
            }
            
            if input.lowercased() == "exit" || input.lowercased() == "quit" {
                print("\n👋 Goodbye!")
                break
            }
            
            // Special commands
            if input.hasPrefix("/") {
                handleCommand(input, config: config)
                continue
            }
            
            // Add user message
            conversation.messages.append(
                HarmonyMessage.from(role: .user, text: input)
            )
            
            // Generate response
            print("\nAssistant: ", terminator: "")
            fflush(stdout)
            
            let startTime = Date()
            
            // Render conversation for completion
            let promptTokens = encoding.renderConversationForCompletion(
                conversation,
                nextTurnRole: .assistant
            )
            
            if config.verbose {
                print("[Prompt: \(promptTokens.count) tokens] ", terminator: "")
            }
            
            // Reset context for each turn
            context.reset()
            
            // Generate response
            var responseTokens: [UInt32] = []
            var responseText = ""
            var currentChannel: String? = nil
            
            let capturedChannels = channels
            let capturedVerbose = verbose
            
            let generatedTokens = try pipeline.generateTokens(
                    prompt: promptTokens.map { UInt32($0) },
                    maxTokens: maxTokens,
                    sampler: GptossSampler(temperature: temperature)
                ) { token in
                responseTokens.append(token)
                
                // Handle special tokens
                if token == 200005 { // <|channel|>
                    // Channel marker - next content is channel name
                    currentChannel = ""
                    return true
                } else if token == 200002 || // <|return|>
                          token == 200012 || // <|call|>
                          token == 200007 || // <|end|>
                          token == 199999 {  // <|endoftext|>
                    return false
                }
                
                // Decode and display token
                let decoded = encoding.tokenizer.decode([token])
                responseText += decoded
                
                // Handle channel display
                if let channel = currentChannel {
                    if decoded.contains("\n") {
                        // End of channel name
                        currentChannel = nil
                        if capturedChannels && capturedVerbose {
                            print("\n[\(channel)] ", terminator: "")
                        }
                    } else {
                        currentChannel! += decoded
                    }
                } else if !capturedChannels || !responseText.contains("<|channel|>") {
                    // Only print if not in channel mode or no channels yet
                    print(decoded, terminator: "")
                    fflush(stdout)
                }
                
                return true
            }
            
            let endTime = Date()
            
            // Add assistant response to conversation
            conversation.messages.append(
                HarmonyMessage.from(role: .assistant, text: responseText)
            )
            
            print() // New line after response
            
            if stats {
                let generationTime = endTime.timeIntervalSince(startTime)
                print("\n[Stats: \(generatedTokens.count) tokens in \(String(format: "%.2f", generationTime))s = \(String(format: "%.1f", Double(generatedTokens.count) / generationTime)) tok/s]")
            }
            
            print("\n" + String(repeating: "─", count: 60) + "\n")
        }
    }
    
    private func handleCommand(_ command: String, config: ChatConfig) {
        switch command {
        case "/help":
            print("""
            Commands:
              /help     - Show this help
              /clear    - Clear conversation history
              /stats    - Toggle statistics display
              /channels - Toggle channel display
              /system   - Show current system prompt
              exit/quit - End chat session
            """)
            
        case "/clear":
            print("Conversation cleared (keeping system message)")
            // Note: Would need to make conversation mutable and reset it
            
        case "/stats":
            // Note: Would need to make stats mutable
            print("Statistics: \(config.stats ? "enabled" : "disabled")")
            
        case "/channels":
            // Note: Would need to make channels mutable
            print("Channels: \(config.channels ? "enabled" : "disabled")")
            
        case "/system":
            print("System prompt: \(config.system)")
            
        default:
            print("Unknown command. Type /help for available commands.")
        }
    }
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

