package ai.noesisreality.sessions

import ai.noesisreality.engine.NoesisInferenceEngine
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.coroutines.delay
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Chat Session Manager - Pure Kotlin implementation
 * 
 * Handles all chat session logic, history management, command processing,
 * and user interaction. The Swift layer is only used for inference calls.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class ChatSessionManager(
    private val systemPrompt: String? = null,
    private val contextLength: Int = 8192,
    private val temperature: Float = 0.7f,
    private val reasoningEffort: String = "medium",
    private val multilineMode: Boolean = false,
    private val autoSaveInterval: Int? = null,
    private val toolsEnabled: Boolean = false,
    private val verbose: Boolean = false
) {
    
    private val conversation = mutableListOf<ChatMessage>()
    private val engine = NoesisInferenceEngine.getInstance()
    private var totalTokens = 0
    private var messageCount = 0
    private var sessionStartTime = LocalDateTime.now()
    
    init {
        // Add system message if provided
        systemPrompt?.let { system ->
            conversation.add(ChatMessage(
                role = MessageRole.SYSTEM,
                content = system,
                timestamp = LocalDateTime.now()
            ))
        }
    }
    
    /**
     * Start the interactive chat session
     */
    suspend fun start() {
        println("💬 Chat session active - type '/help' for commands")
        
        while (true) {
            try {
                val userInput = readUserInput()
                
                // Handle special commands
                if (userInput.startsWith("/")) {
                    if (!handleCommand(userInput)) {
                        break // User requested exit
                    }
                    continue
                }
                
                // Add user message
                val userMessage = ChatMessage(
                    role = MessageRole.USER,
                    content = userInput,
                    timestamp = LocalDateTime.now()
                )
                conversation.add(userMessage)
                messageCount++
                
                // Generate response
                println("\n🤖 Assistant:")
                val response = generateResponse(userInput)
                println(response.text)
                
                // Add assistant message
                val assistantMessage = ChatMessage(
                    role = MessageRole.ASSISTANT,
                    content = response.text,
                    timestamp = LocalDateTime.now(),
                    metadata = ChatMessageMetadata(
                        tokensGenerated = response.tokensGenerated,
                        timeMs = response.timeMs,
                        tokensPerSecond = response.tokensPerSecond
                    )
                )
                conversation.add(assistantMessage)
                
                totalTokens += response.tokensGenerated
                
                // Show stats if verbose
                if (verbose) {
                    println("\n📊 ${response.tokensGenerated} tokens, ${response.timeMs}ms, ${String.format("%.1f", response.tokensPerSecond)} tok/sec")
                }
                
                println() // Add spacing
                
                // Auto-save if configured
                autoSaveInterval?.let { interval ->
                    if (messageCount % interval == 0) {
                        autoSave()
                    }
                }
                
            } catch (e: ChatExitException) {
                break
            } catch (e: Exception) {
                println("❌ Error: ${e.message}")
                if (verbose) {
                    e.printStackTrace()
                }
            }
        }
        
        showSessionSummary()
    }
    
    /**
     * Read user input with support for multiline mode
     */
    private fun readUserInput(): String {
        if (multilineMode) {
            print("👤 You (multiline - '###' to send):\n")
            val lines = mutableListOf<String>()
            while (true) {
                val line = readLine() ?: throw ChatExitException()
                if (line.trim() == "###") break
                lines.add(line)
            }
            return lines.joinToString("\n")
        } else {
            print("👤 You: ")
            return readLine() ?: throw ChatExitException()
        }
    }
    
    /**
     * Handle chat commands (pure Kotlin logic)
     */
    private suspend fun handleCommand(command: String): Boolean {
        val parts = command.substring(1).split(" ", limit = 2)
        val cmd = parts[0].lowercase()
        val arg = parts.getOrNull(1)
        
        when (cmd) {
            "help", "h" -> {
                showHelp()
            }
            "quit", "exit", "q" -> {
                println("👋 Goodbye!")
                return false
            }
            "history", "hist" -> {
                showHistory()
            }
            "clear", "reset" -> {
                clearConversation()
            }
            "save" -> {
                val filename = arg ?: "chat_${System.currentTimeMillis()}.json"
                saveSession(filename)
                println("💾 Session saved to $filename")
            }
            "load" -> {
                if (arg == null) {
                    println("❌ Usage: /load <filename>")
                } else {
                    loadSession(arg)
                    println("📚 Session loaded from $arg")
                }
            }
            "stats", "info" -> {
                showSessionStats()
            }
            "context" -> {
                showContextInfo()
            }
            "system" -> {
                if (arg == null) {
                    showSystemPrompt()
                } else {
                    setSystemPrompt(arg)
                }
            }
            "temp", "temperature" -> {
                if (arg == null) {
                    println("Current temperature: $temperature")
                } else {
                    // Note: This would require creating a new session manager
                    println("💡 Temperature changes require restarting the session")
                }
            }
            "tokens" -> {
                val tokenCount = estimateTokenCount()
                println("📊 Estimated context tokens: $tokenCount / $contextLength")
            }
            "export" -> {
                val filename = arg ?: "chat_export_${System.currentTimeMillis()}.md"
                exportToMarkdown(filename)
                println("📄 Conversation exported to $filename")
            }
            "multiline", "ml" -> {
                println("💡 Multiline mode is set when starting the session")
                println("Current mode: ${if (multilineMode) "enabled" else "disabled"}")
            }
            else -> {
                println("❓ Unknown command: /$cmd")
                println("Type '/help' for available commands")
            }
        }
        
        return true
    }
    
    /**
     * Generate response using the inference engine
     */
    private suspend fun generateResponse(userInput: String): ai.noesisreality.engine.InferenceResult {
        // Build context from conversation history
        val context = buildContextPrompt()
        
        // Call Swift inference engine with just the core parameters
        return engine.generate(
            prompt = context,
            maxTokens = estimateMaxResponseTokens(),
            temperature = temperature,
            reasoningEffort = reasoningEffort
        )
    }
    
    /**
     * Build prompt context from conversation history
     */
    private fun buildContextPrompt(): String {
        val contextMessages = getContextMessages()
        val prompt = StringBuilder()
        
        contextMessages.forEach { message ->
            when (message.role) {
                MessageRole.SYSTEM -> prompt.append("System: ${message.content}\n\n")
                MessageRole.USER -> prompt.append("User: ${message.content}\n\n")
                MessageRole.ASSISTANT -> prompt.append("Assistant: ${message.content}\n\n")
            }
        }
        
        prompt.append("Assistant:")
        return prompt.toString()
    }
    
    /**
     * Get messages that fit within context window
     */
    private fun getContextMessages(): List<ChatMessage> {
        // Simple token estimation: ~4 chars per token
        val maxContextChars = contextLength * 4
        var currentChars = 0
        val contextMessages = mutableListOf<ChatMessage>()
        
        // Always include system message first
        conversation.filter { it.role == MessageRole.SYSTEM }.firstOrNull()?.let { systemMsg ->
            contextMessages.add(systemMsg)
            currentChars += systemMsg.content.length
        }
        
        // Add messages from most recent backwards
        val nonSystemMessages = conversation.filter { it.role != MessageRole.SYSTEM }
        for (message in nonSystemMessages.asReversed()) {
            val messageChars = message.content.length + 20 // Add overhead for formatting
            if (currentChars + messageChars > maxContextChars) {
                break
            }
            contextMessages.add(0, message) // Insert at beginning
            currentChars += messageChars
        }
        
        return contextMessages
    }
    
    /**
     * Estimate appropriate max tokens for response
     */
    private fun estimateMaxResponseTokens(): Int {
        val contextTokens = estimateTokenCount()
        val availableTokens = contextLength - contextTokens
        return minOf(availableTokens / 2, 500) // Leave room, cap at reasonable response size
    }
    
    /**
     * Estimate current token count
     */
    private fun estimateTokenCount(): Int {
        return getContextMessages().sumOf { it.content.length / 4 }
    }
    
    /**
     * Save session to file
     */
    fun saveSession(filename: String) {
        val session = ChatSession(
            conversation = conversation,
            sessionInfo = SessionInfo(
                startTime = sessionStartTime,
                totalTokens = totalTokens,
                messageCount = messageCount,
                systemPrompt = systemPrompt,
                contextLength = contextLength,
                temperature = temperature,
                reasoningEffort = reasoningEffort
            )
        )
        
        val json = Json { prettyPrint = true }
        val jsonString = json.encodeToString(ChatSession.serializer(), session)
        File(filename).writeText(jsonString)
    }
    
    /**
     * Load session from file
     */
    fun loadSession(filename: String) {
        val jsonString = File(filename).readText()
        val session = Json.decodeFromString(ChatSession.serializer(), jsonString)
        
        conversation.clear()
        conversation.addAll(session.conversation)
        
        totalTokens = session.sessionInfo.totalTokens
        messageCount = session.sessionInfo.messageCount
        sessionStartTime = session.sessionInfo.startTime
    }
    
    // Helper methods for commands
    private fun showHelp() {
        println("""
        📚 Available Commands:
          /help, /h           - Show this help
          /quit, /exit, /q    - End session
          /history, /hist     - Show conversation history  
          /clear, /reset      - Clear conversation
          /save [filename]    - Save session
          /load <filename>    - Load session
          /stats, /info       - Show session statistics
          /context            - Show context information
          /tokens             - Show token usage
          /export [filename]  - Export to markdown
          /system [prompt]    - Show/set system prompt
          /temp               - Show current temperature
        """.trimIndent())
    }
    
    private fun showHistory() {
        println("📚 Conversation History (${conversation.size} messages):")
        conversation.forEachIndexed { index, message ->
            val roleEmoji = when (message.role) {
                MessageRole.SYSTEM -> "⚙️"
                MessageRole.USER -> "👤"
                MessageRole.ASSISTANT -> "🤖"
            }
            val time = message.timestamp.format(DateTimeFormatter.ofPattern("HH:mm"))
            val preview = message.content.take(60).replace("\n", " ")
            println("  ${index + 1}. $roleEmoji $time ${message.role}: $preview${if (message.content.length > 60) "..." else ""}")
        }
    }
    
    private fun clearConversation() {
        val systemMessages = conversation.filter { it.role == MessageRole.SYSTEM }
        conversation.clear()
        conversation.addAll(systemMessages)
        totalTokens = 0
        messageCount = 0
        println("🗑️ Conversation cleared (system prompt preserved)")
    }
    
    private fun showSessionStats() {
        val duration = java.time.Duration.between(sessionStartTime, LocalDateTime.now())
        println("""
        📊 Session Statistics:
          Messages: $messageCount
          Total tokens: $totalTokens
          Duration: ${duration.toMinutes()}m ${duration.seconds % 60}s
          System prompt: ${if (systemPrompt != null) "Set" else "None"}
          Context: $contextLength tokens
          Temperature: $temperature
          Reasoning: $reasoningEffort
          Tools: ${if (toolsEnabled) "Enabled" else "Disabled"}
        """.trimIndent())
    }
    
    private fun showContextInfo() {
        val contextMessages = getContextMessages()
        val estimatedTokens = estimateTokenCount()
        println("""
        🧠 Context Information:
          Context window: $contextLength tokens
          Estimated usage: $estimatedTokens tokens (${(estimatedTokens.toFloat() / contextLength * 100).toInt()}%)
          Messages in context: ${contextMessages.size}
          Available for response: ${contextLength - estimatedTokens} tokens
        """.trimIndent())
    }
    
    private fun showSystemPrompt() {
        if (systemPrompt != null) {
            println("⚙️ System Prompt:")
            println(systemPrompt)
        } else {
            println("⚙️ No system prompt set")
        }
    }
    
    private fun setSystemPrompt(prompt: String) {
        // Remove old system messages
        conversation.removeIf { it.role == MessageRole.SYSTEM }
        
        // Add new system message
        conversation.add(0, ChatMessage(
            role = MessageRole.SYSTEM,
            content = prompt,
            timestamp = LocalDateTime.now()
        ))
        
        println("⚙️ System prompt updated")
    }
    
    private fun exportToMarkdown(filename: String) {
        val markdown = StringBuilder()
        markdown.append("# Noesis Chat Session\n\n")
        markdown.append("**Started:** ${sessionStartTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))}\n")
        markdown.append("**Messages:** $messageCount\n")
        markdown.append("**Total Tokens:** $totalTokens\n\n")
        
        if (systemPrompt != null) {
            markdown.append("## System Prompt\n\n")
            markdown.append("$systemPrompt\n\n")
        }
        
        markdown.append("## Conversation\n\n")
        
        conversation.filter { it.role != MessageRole.SYSTEM }.forEach { message ->
            val roleHeader = when (message.role) {
                MessageRole.USER -> "### 👤 User"
                MessageRole.ASSISTANT -> "### 🤖 Assistant"
                else -> "### System"
            }
            val time = message.timestamp.format(DateTimeFormatter.ofPattern("HH:mm"))
            
            markdown.append("$roleHeader ($time)\n\n")
            markdown.append("${message.content}\n\n")
        }
        
        File(filename).writeText(markdown.toString())
    }
    
    private fun autoSave() {
        val filename = "autosave_${System.currentTimeMillis()}.json"
        try {
            saveSession(filename)
            if (verbose) {
                println("💾 Auto-saved to $filename")
            }
        } catch (e: Exception) {
            if (verbose) {
                println("⚠️ Auto-save failed: ${e.message}")
            }
        }
    }
    
    private fun showSessionSummary() {
        val duration = java.time.Duration.between(sessionStartTime, LocalDateTime.now())
        println("""
        
        📊 Session Summary:
          Duration: ${duration.toMinutes()}m ${duration.seconds % 60}s
          Messages: $messageCount
          Total tokens: $totalTokens
          Average response time: ${if (messageCount > 0) "~${totalTokens / messageCount}" else "N/A"} tokens per exchange
        
        Thanks for using Noesis! 👋
        """.trimIndent())
    }
}

// Data classes for session management
@Serializable
data class ChatMessage(
    val role: MessageRole,
    val content: String,
    @Serializable(with = LocalDateTimeSerializer::class)
    val timestamp: LocalDateTime,
    val metadata: ChatMessageMetadata? = null
)

@Serializable
data class ChatMessageMetadata(
    val tokensGenerated: Int = 0,
    val timeMs: Long = 0,
    val tokensPerSecond: Double = 0.0
)

@Serializable
enum class MessageRole {
    SYSTEM, USER, ASSISTANT
}

@Serializable
data class ChatSession(
    val conversation: List<ChatMessage>,
    val sessionInfo: SessionInfo
)

@Serializable
data class SessionInfo(
    @Serializable(with = LocalDateTimeSerializer::class)
    val startTime: LocalDateTime,
    val totalTokens: Int,
    val messageCount: Int,
    val systemPrompt: String?,
    val contextLength: Int,
    val temperature: Float,
    val reasoningEffort: String
)

class ChatExitException : Exception()

// Custom serializer for LocalDateTime
object LocalDateTimeSerializer : KSerializer<LocalDateTime> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("LocalDateTime", PrimitiveKind.STRING)
    
    override fun serialize(encoder: Encoder, value: LocalDateTime) {
        encoder.encodeString(value.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME))
    }
    
    override fun deserialize(decoder: Decoder): LocalDateTime {
        return LocalDateTime.parse(decoder.decodeString(), DateTimeFormatter.ISO_LOCAL_DATE_TIME)
    }
}