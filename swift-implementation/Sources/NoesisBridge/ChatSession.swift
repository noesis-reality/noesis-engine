import Foundation
import NoesisEngine
import NoesisTools

/**
 * Interactive Chat Session Management
 * 
 * Handles the interactive chat loop, history management, and tool integration
 * for the unified Kotlin CLI system.
 */
public class ChatSession {
    private let engine: NoesisEngine
    private let model: GptossModel
    private let context: InferenceContext
    private let systemPrompt: String?
    private let temperature: Float
    private let reasoningEffort: ReasoningEffort
    private let toolsEnabled: Bool
    private let browserEnabled: Bool
    private let pythonEnabled: Bool
    private let multilineMode: Bool
    private let historyFile: String?
    private let verbose: Bool
    
    private var conversationHistory: [ChatMessage] = []
    private var totalTokens: Int = 0
    
    public init(
        engine: NoesisEngine,
        model: GptossModel,
        context: InferenceContext,
        systemPrompt: String? = nil,
        temperature: Float = 0.7,
        reasoningEffort: ReasoningEffort = .medium,
        toolsEnabled: Bool = false,
        browserEnabled: Bool = false,
        pythonEnabled: Bool = false,
        multilineMode: Bool = false,
        historyFile: String? = nil,
        verbose: Bool = false
    ) {
        self.engine = engine
        self.model = model
        self.context = context
        self.systemPrompt = systemPrompt
        self.temperature = temperature
        self.reasoningEffort = reasoningEffort
        self.toolsEnabled = toolsEnabled
        self.browserEnabled = browserEnabled
        self.pythonEnabled = pythonEnabled
        self.multilineMode = multilineMode
        self.historyFile = historyFile
        self.verbose = verbose
    }
    
    public func start() throws {
        print("💬 Noesis Chat Session Started")
        print("   Type 'exit', 'quit', or press Ctrl+C to end")
        print("   Type 'help' for available commands")
        if multilineMode {
            print("   Multiline mode: Type '###' on a new line to send message")
        }
        print()
        
        // Add system message to history
        if let systemPrompt = systemPrompt {
            conversationHistory.append(ChatMessage(role: .system, content: systemPrompt))
        }
        
        // Main chat loop
        while true {
            do {
                let userInput = try readUserInput()
                
                // Handle special commands
                if let command = parseSpecialCommand(userInput) {
                    try handleSpecialCommand(command)
                    continue
                }
                
                // Add user message to history
                conversationHistory.append(ChatMessage(role: .user, content: userInput))
                
                // Generate response
                let response = try generateResponse(for: userInput)
                
                // Add assistant message to history
                conversationHistory.append(ChatMessage(role: .assistant, content: response.text))
                
                // Display response
                print("\n🤖 Assistant:")
                print(response.text)
                
                if verbose {
                    print("\n📊 Stats: \(response.tokens.count) tokens, \(response.timeMs)ms, \(String(format: "%.1f", response.tokensPerSecond)) tok/sec")
                }
                
                print()
                
                // Save history if configured
                if let historyFile = historyFile {
                    try saveHistory(to: historyFile)
                }
                
            } catch ChatError.userExit {
                break
            } catch {
                print("❌ Error: \(error)")
                print()
            }
        }
        
        print("👋 Chat session ended")
    }
    
    public func loadHistory(from filePath: String) throws {
        let url = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: url)
        let history = try JSONDecoder().decode([ChatMessage].self, from: data)
        conversationHistory = history
        
        if verbose {
            print("📚 Loaded \(history.count) messages from history")
        }
    }
    
    private func saveHistory(to filePath: String) throws {
        let url = URL(fileURLWithPath: filePath)
        let data = try JSONEncoder().encode(conversationHistory)
        try data.write(to: url)
    }
    
    private func readUserInput() throws -> String {
        if multilineMode {
            print("👤 You (multiline - type '###' to send):")
            var lines: [String] = []
            while true {
                guard let line = readLine() else {
                    throw ChatError.userExit
                }
                if line.trimmingCharacters(in: .whitespaces) == "###" {
                    break
                }
                lines.append(line)
            }
            return lines.joined(separator: "\n")
        } else {
            print("👤 You: ", terminator: "")
            guard let input = readLine() else {
                throw ChatError.userExit
            }
            return input
        }
    }
    
    private func parseSpecialCommand(_ input: String) -> ChatCommand? {
        let trimmed = input.trimmingCharacters(in: .whitespaces).lowercased()
        
        switch trimmed {
        case "exit", "quit", "q":
            return .exit
        case "help", "h":
            return .help
        case "history":
            return .history
        case "clear":
            return .clear
        case "stats":
            return .stats
        case "save":
            return .save
        default:
            if trimmed.hasPrefix("save ") {
                let filename = String(trimmed.dropFirst(5))
                return .saveAs(filename)
            }
            if trimmed.hasPrefix("load ") {
                let filename = String(trimmed.dropFirst(5))
                return .loadHistory(filename)
            }
            return nil
        }
    }
    
    private func handleSpecialCommand(_ command: ChatCommand) throws {
        switch command {
        case .exit:
            throw ChatError.userExit
            
        case .help:
            print("""
            📚 Available Commands:
              exit, quit, q     - End chat session
              help, h          - Show this help
              history          - Show conversation history
              clear            - Clear conversation history
              stats            - Show session statistics
              save             - Save history to default file
              save <filename>  - Save history to specific file
              load <filename>  - Load history from file
            
            """)
            
        case .history:
            print("📚 Conversation History:")
            for (index, message) in conversationHistory.enumerated() {
                let roleEmoji = message.role == .user ? "👤" : "🤖"
                let preview = String(message.content.prefix(80))
                print("  \(index + 1). \(roleEmoji) \(message.role.rawValue): \(preview)...")
            }
            print()
            
        case .clear:
            conversationHistory.removeAll()
            totalTokens = 0
            print("🗑️  Conversation history cleared")
            print()
            
        case .stats:
            print("📊 Session Statistics:")
            print("  Messages: \(conversationHistory.count)")
            print("  Total tokens: \(totalTokens)")
            print("  Tools enabled: \(toolsEnabled)")
            print("  Browser: \(browserEnabled)")
            print("  Python: \(pythonEnabled)")
            print()
            
        case .save:
            if let historyFile = historyFile {
                try saveHistory(to: historyFile)
                print("💾 History saved to \(historyFile)")
            } else {
                print("⚠️  No default history file configured")
            }
            print()
            
        case .saveAs(let filename):
            try saveHistory(to: filename)
            print("💾 History saved to \(filename)")
            print()
            
        case .loadHistory(let filename):
            try loadHistory(from: filename)
            print("📚 History loaded from \(filename)")
            print()
        }
    }
    
    private func generateResponse(for input: String) throws -> GenerationResult {
        // Build prompt from conversation history
        var fullPrompt = ""
        
        // Add system prompt if present
        if let systemPrompt = systemPrompt {
            fullPrompt += "System: \(systemPrompt)\n\n"
        }
        
        // Add conversation context (last N messages to fit in context)
        let contextMessages = getContextMessages()
        for message in contextMessages {
            fullPrompt += "\(message.role.rawValue.capitalized): \(message.content)\n\n"
        }
        
        fullPrompt += "User: \(input)\n\nAssistant:"
        
        // Generate response
        let parameters = GenerationParameters(
            temperature: temperature,
            topP: 0.9,
            repetitionPenalty: 1.1,
            maxTokens: 500,
            seed: nil
        )
        
        let startTime = Date()
        let result = try engine.generate(
            model: model,
            context: context,
            prompt: fullPrompt,
            systemPrompt: nil, // Already included in full prompt
            parameters: parameters,
            reasoningEffort: reasoningEffort,
            streaming: false
        )
        let endTime = Date()
        
        let timeMs = Int64((endTime.timeIntervalSince(startTime)) * 1000)
        let tokensPerSecond = Double(result.tokens.count) / endTime.timeIntervalSince(startTime)
        
        totalTokens += result.tokens.count
        
        return GenerationResult(
            text: result.text,
            tokens: result.tokens,
            harmonyFormat: result.harmonyFormat ?? result.text,
            tokensGenerated: result.tokens.count,
            timeMs: timeMs,
            tokensPerSecond: tokensPerSecond,
            gpuMemoryMB: 0 // TODO: Implement GPU memory monitoring
        )
    }
    
    private func getContextMessages() -> [ChatMessage] {
        // Simple context management - return last N messages that fit in context
        // In a more sophisticated implementation, this would use proper token counting
        let maxMessages = 20
        let startIndex = max(0, conversationHistory.count - maxMessages)
        return Array(conversationHistory[startIndex...])
    }
}

// MARK: - Supporting Types

public struct ChatMessage: Codable {
    public let role: ChatRole
    public let content: String
    
    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

public enum ChatRole: String, Codable {
    case system = "system"
    case user = "user"
    case assistant = "assistant"
}

private enum ChatCommand {
    case exit
    case help
    case history
    case clear
    case stats
    case save
    case saveAs(String)
    case loadHistory(String)
}

private enum ChatError: Error {
    case userExit
}