import Foundation
import CHarmony

// MARK: - Harmony C FFI Wrapper

/// Wrapper around harmony-swift C FFI functions
public final class HarmonyWrapper {
    private var harmonyEncoding: OpaquePointer?
    
    public init() throws {
        harmonyEncoding = harmony_encoding_new()
        guard harmonyEncoding != nil else {
            throw HarmonyError.initializationFailed("Failed to create HarmonyEncoding")
        }
    }
    
    deinit {
        if let encoding = harmonyEncoding {
            harmony_encoding_free(encoding)
        }
    }
    
    /// Render a prompt with system, user, and optional assistant prefix
    public func renderPrompt(
        systemMessage: String?,
        userMessage: String,
        assistantPrefix: String? = nil
    ) -> [Int] {
        guard let encoding = harmonyEncoding else { return [] }
        
        var tokensOut: UnsafeMutablePointer<UInt32>?
        var tokensLen: size_t = 0
        
        let result: HarmonyResult
        if let systemMessage = systemMessage {
            result = systemMessage.withCString { systemCStr in
                userMessage.withCString { userCStr in
                    if let assistantPrefix = assistantPrefix {
                        return assistantPrefix.withCString { prefixCStr in
                            harmony_encoding_render_prompt(
                                encoding,
                                systemCStr,
                                userCStr,
                                prefixCStr,
                                &tokensOut,
                                &tokensLen
                            )
                        }
                    } else {
                        return harmony_encoding_render_prompt(
                            encoding,
                            systemCStr,
                            userCStr,
                            nil,
                            &tokensOut,
                            &tokensLen
                        )
                    }
                }
            }
        } else {
            result = userMessage.withCString { userCStr in
                if let assistantPrefix = assistantPrefix {
                    return assistantPrefix.withCString { prefixCStr in
                        harmony_encoding_render_prompt(
                            encoding,
                            nil,
                            userCStr,
                            prefixCStr,
                            &tokensOut,
                            &tokensLen
                        )
                    }
                } else {
                    return harmony_encoding_render_prompt(
                        encoding,
                        nil,
                        userCStr,
                        nil,
                        &tokensOut,
                        &tokensLen
                    )
                }
            }
        }
        
        guard result.success else {
            if let errorMsg = result.error_message {
                let error = String(cString: errorMsg)
                harmony_free_string(errorMsg)
                print("Harmony render prompt error: \(error)")
            }
            return []
        }
        
        guard let tokens = tokensOut, tokensLen > 0 else {
            return []
        }
        
        defer {
            harmony_free_tokens(tokens, tokensLen)
        }
        
        // Convert UInt32 tokens to Int for compatibility
        var intTokens: [Int] = []
        for i in 0..<tokensLen {
            intTokens.append(Int(tokens[i]))
        }
        
        return intTokens
    }
    
    /// Encode plain text without conversation formatting
    public func encodePlain(_ text: String) -> [Int] {
        guard let encoding = harmonyEncoding else { return [] }
        
        var tokensOut: UnsafeMutablePointer<UInt32>?
        var tokensLen: size_t = 0
        
        let result = text.withCString { textCStr in
            harmony_encoding_encode_plain(
                encoding,
                textCStr,
                &tokensOut,
                &tokensLen
            )
        }
        
        guard result.success else {
            if let errorMsg = result.error_message {
                let error = String(cString: errorMsg)
                harmony_free_string(errorMsg)
                print("Harmony encode plain error: \(error)")
            }
            return []
        }
        
        guard let tokens = tokensOut, tokensLen > 0 else {
            return []
        }
        
        defer {
            harmony_free_tokens(tokens, tokensLen)
        }
        
        // Convert UInt32 tokens to Int for compatibility
        var intTokens: [Int] = []
        for i in 0..<tokensLen {
            intTokens.append(Int(tokens[i]))
        }
        
        return intTokens
    }
    
    /// Decode tokens to text
    public func decode(_ tokens: [UInt32]) -> String {
        guard let encoding = harmonyEncoding else { return "" }
        
        let cString = tokens.withUnsafeBufferPointer { buffer in
            harmony_encoding_decode(encoding, buffer.baseAddress, tokens.count)
        }
        
        guard let result = cString else { return "" }
        
        let decoded = String(cString: result)
        harmony_free_string(result)
        
        return decoded
    }
    
    /// Get stop tokens for generation
    public func stopTokens() throws -> [UInt32] {
        guard let encoding = harmonyEncoding else {
            throw HarmonyError.initializationFailed("Harmony encoding not initialized")
        }
        
        var tokensOut: UnsafeMutablePointer<UInt32>?
        var tokensLen: size_t = 0
        
        let result = harmony_encoding_stop_tokens(encoding, &tokensOut, &tokensLen)
        
        guard result.success else {
            if let errorMsg = result.error_message {
                let error = String(cString: errorMsg)
                harmony_free_string(errorMsg)
                throw HarmonyError.encodingFailed("Failed to get stop tokens: \(error)")
            }
            throw HarmonyError.encodingFailed("Failed to get stop tokens")
        }
        
        guard let tokens = tokensOut, tokensLen > 0 else {
            return []
        }
        
        defer {
            harmony_free_tokens(tokens, tokensLen)
        }
        
        // Convert to array
        var stopTokens: [UInt32] = []
        for i in 0..<tokensLen {
            stopTokens.append(tokens[i])
        }
        
        return stopTokens
    }
}

// MARK: - Error Types

public enum HarmonyError: Error {
    case initializationFailed(String)
    case encodingFailed(String)
    case decodingFailed(String)
}

