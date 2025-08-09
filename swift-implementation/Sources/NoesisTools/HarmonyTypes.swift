import Foundation

// MARK: - Harmony Role

public enum HarmonyRole: String, CaseIterable {
    case system = "system"
    case developer = "developer"
    case user = "user"
    case assistant = "assistant"
    case tool = "tool"
}

// MARK: - Harmony Channel

public enum HarmonyChannel: String, CaseIterable, Sendable {
    case analysis = "analysis"      // Internal reasoning (never shown to users)
    case commentary = "commentary"  // Tool interactions and planning
    case final = "final"            // User-facing responses
    
    public static let `default` = HarmonyChannel.final
}

// MARK: - Reasoning Effort

public enum HarmonyReasoningEffort: String {
    case low = "low"
    case medium = "medium"
    case high = "high"
}

// MARK: - System Content

public struct HarmonySystemContent {
    public var modelIdentity: String
    public var reasoningEffort: HarmonyReasoningEffort
    public var conversationStartDate: String
    public var knowledgeCutoff: String
    public var requiredChannels: [String]
    
    public init(
        modelIdentity: String = "You are ChatGPT, a large language model trained by OpenAI.",
        reasoningEffort: HarmonyReasoningEffort = .medium,
        conversationStartDate: String? = nil,
        knowledgeCutoff: String = "2024-06",
        requiredChannels: [String] = []
    ) {
        self.modelIdentity = modelIdentity
        self.reasoningEffort = reasoningEffort
        
        if let date = conversationStartDate {
            self.conversationStartDate = date
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd"
            self.conversationStartDate = formatter.string(from: Date())
        }
        
        self.knowledgeCutoff = knowledgeCutoff
        self.requiredChannels = requiredChannels
    }
    
    public static func new() -> HarmonySystemContent {
        return HarmonySystemContent()
    }
    
    public func withModelIdentity(_ identity: String) -> HarmonySystemContent {
        var copy = self
        copy.modelIdentity = identity
        return copy
    }
    
    public func withReasoningEffort(_ effort: HarmonyReasoningEffort) -> HarmonySystemContent {
        var copy = self
        copy.reasoningEffort = effort
        return copy
    }
    
    public func withConversationStartDate(_ date: String) -> HarmonySystemContent {
        var copy = self
        copy.conversationStartDate = date
        return copy
    }
    
    public func withKnowledgeCutoff(_ cutoff: String) -> HarmonySystemContent {
        var copy = self
        copy.knowledgeCutoff = cutoff
        return copy
    }
    
    public func withRequiredChannels(_ channels: [String]) -> HarmonySystemContent {
        var copy = self
        copy.requiredChannels = channels
        return copy
    }
}

// MARK: - Message Content

public enum HarmonyContent {
    case text(String)
    case system(HarmonySystemContent)
    case toolCall(id: String, function: String, arguments: String)
    case toolResult(id: String, content: String)
}

// MARK: - Harmony Message

public struct HarmonyMessage {
    public let role: HarmonyRole
    public let content: HarmonyContent
    public let channel: HarmonyChannel
    public let recipient: String?
    
    public init(
        role: HarmonyRole,
        content: HarmonyContent,
        channel: HarmonyChannel = .final,
        recipient: String? = nil
    ) {
        self.role = role
        self.content = content
        self.channel = channel
        self.recipient = recipient
    }
    
    // Convenience constructors
    public static func from(role: HarmonyRole, text: String, channel: HarmonyChannel = .final) -> HarmonyMessage {
        return HarmonyMessage(role: role, content: .text(text), channel: channel)
    }
    
    public static func fromRoleAndContent(_ role: HarmonyRole, _ content: HarmonyContent) -> HarmonyMessage {
        return HarmonyMessage(role: role, content: content)
    }
    
    public static func system(_ content: HarmonySystemContent) -> HarmonyMessage {
        return HarmonyMessage(role: .system, content: .system(content))
    }
}

// MARK: - Harmony Conversation

public struct HarmonyConversation {
    public var messages: [HarmonyMessage]
    
    public init(messages: [HarmonyMessage] = []) {
        self.messages = messages
    }
    
    public mutating func addMessage(_ message: HarmonyMessage) {
        messages.append(message)
    }
    
    public mutating func addUserMessage(_ text: String) {
        messages.append(.from(role: .user, text: text))
    }
    
    public mutating func addAssistantMessage(_ text: String, channel: HarmonyChannel = .final) {
        messages.append(.from(role: .assistant, text: text, channel: channel))
    }
}

// MARK: - Harmony Encoding

public final class HarmonyEncoding: @unchecked Sendable {
    private let harmonyWrapper: HarmonyWrapper
    public let tokenizer: O200kTokenizer
    
    // Special token IDs for o200k_harmony
    public let startToken: UInt32 = 200006
    public let endToken: UInt32 = 200007
    public let messageToken: UInt32 = 200008
    public let channelToken: UInt32 = 200005
    public let constrainToken: UInt32 = 200003
    public let returnToken: UInt32 = 200002
    public let callToken: UInt32 = 200012
    public let endOfTextToken: UInt32 = 199999
    
    public init() throws {
        self.harmonyWrapper = try HarmonyWrapper()
        self.tokenizer = try O200kTokenizer()
    }
    
    // Harmony encoding singleton for gpt_oss models
    public static let harmony_gpt_oss: HarmonyEncoding = {
        do {
            return try HarmonyEncoding()
        } catch {
            fatalError("Failed to initialize Harmony encoding: \(error)")
        }
    }()
    
    /// Render a conversation for completion with next turn role
    public func renderConversationForCompletion(
        _ conversation: HarmonyConversation,
        nextTurnRole: HarmonyRole
    ) -> [Int] {
        // Build prompt from conversation
        var systemMessage: String? = nil
        var userMessage: String = ""
        var assistantPrefix: String? = nil
        
        for message in conversation.messages {
            switch (message.role, message.content) {
            case (.system, .system(let systemContent)):
                // Format system message with metadata
                var system = systemContent.modelIdentity
                system += "\n\nConversation date: \(systemContent.conversationStartDate)"
                system += "\nKnowledge cutoff: \(systemContent.knowledgeCutoff)"
                system += "\nReasoning: \(systemContent.reasoningEffort.rawValue)"
                if !systemContent.requiredChannels.isEmpty {
                    system += "\nChannels: \(systemContent.requiredChannels.joined(separator: ", "))"
                }
                systemMessage = system
                
            case (.user, .text(let text)):
                userMessage = text
                
            case (.assistant, .text(let text)):
                // For continuing a conversation, use the last assistant message as prefix
                assistantPrefix = text
                
            default:
                break
            }
        }
        
        // Determine what to render based on next turn
        if nextTurnRole == .assistant {
            // Render prompt for assistant completion
            return harmonyWrapper.renderPrompt(
                systemMessage: systemMessage,
                userMessage: userMessage,
                assistantPrefix: assistantPrefix
            )
        } else {
            // For other roles, just encode the conversation
            var allText = ""
            for message in conversation.messages {
                if case .text(let text) = message.content {
                    allText += text + "\n"
                }
            }
            return tokenizer.encode(allText).map { Int($0) }
        }
    }
    
    /// Get stop tokens for generation
    public func stopTokens() throws -> [UInt32] {
        return try harmonyWrapper.stopTokens()
    }
    
    /// Render a prompt with system, user, and optional assistant prefix
    public func renderPrompt(
        systemMessage: String?,
        userMessage: String,
        assistantPrefix: String? = nil
    ) -> [Int] {
        return harmonyWrapper.renderPrompt(
            systemMessage: systemMessage,
            userMessage: userMessage,
            assistantPrefix: assistantPrefix
        )
    }
}