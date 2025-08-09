import Foundation
import CHarmony

/// Result type for FFI operations
public struct HarmonyFFIError: LocalizedError {
    let message: String
    
    public var errorDescription: String? {
        return message
    }
}

/// Harmony encoding wrapper using official OpenAI harmony library
public class HarmonyEncoding {
    private let wrapper: OpaquePointer
    
    public init() throws {
        guard let ptr = harmony_encoding_new() else {
            throw HarmonyFFIError(message: "Failed to create harmony encoding")
        }
        self.wrapper = ptr
    }
    
    deinit {
        harmony_encoding_free(wrapper)
    }
    
    /// Encode plain text without Harmony formatting
    public func encodePlain(_ text: String) throws -> [UInt32] {
        var tokensPtr: UnsafeMutablePointer<UInt32>?
        var tokensLen: Int = 0
        
        let result = harmony_encoding_encode_plain(
            wrapper,
            text.cString(using: .utf8),
            &tokensPtr,
            &tokensLen
        )
        
        if !result.success {
            let errorMsg = result.error_message != nil
                ? String(cString: result.error_message!)
                : "Unknown error"
            if result.error_message != nil {
                harmony_free_string(result.error_message)
            }
            throw HarmonyFFIError(message: errorMsg)
        }
        
        guard let tokens = tokensPtr else {
            throw HarmonyFFIError(message: "No tokens returned")
        }
        
        // Copy tokens to Swift array
        let tokensArray = Array(UnsafeBufferPointer(start: tokens, count: tokensLen))
        
        // Free the allocated memory
        harmony_free_tokens(tokens, tokensLen)
        
        return tokensArray
    }
    
    /// Render a prompt with optional system message and assistant prefix
    public func renderPrompt(
        systemMessage: String? = nil,
        userMessage: String,
        assistantPrefix: String? = nil
    ) throws -> [UInt32] {
        var tokensPtr: UnsafeMutablePointer<UInt32>?
        var tokensLen: Int = 0
        
        let result = harmony_encoding_render_prompt(
            wrapper,
            systemMessage?.cString(using: .utf8),
            userMessage.cString(using: .utf8),
            assistantPrefix?.cString(using: .utf8),
            &tokensPtr,
            &tokensLen
        )
        
        if !result.success {
            let errorMsg = result.error_message != nil
                ? String(cString: result.error_message!)
                : "Unknown error"
            if result.error_message != nil {
                harmony_free_string(result.error_message)
            }
            throw HarmonyFFIError(message: errorMsg)
        }
        
        guard let tokens = tokensPtr else {
            throw HarmonyFFIError(message: "No tokens returned")
        }
        
        // Copy tokens to Swift array
        let tokensArray = Array(UnsafeBufferPointer(start: tokens, count: tokensLen))
        
        // Free the allocated memory
        harmony_free_tokens(tokens, tokensLen)
        
        return tokensArray
    }
    
    /// Decode tokens to text
    public func decode(tokens: [UInt32]) -> String? {
        let ptr = harmony_encoding_decode(
            wrapper,
            tokens,
            tokens.count
        )
        
        guard let cStr = ptr else {
            return nil
        }
        
        let result = String(cString: cStr)
        harmony_free_string(cStr)
        return result
    }
    
    /// Get stop tokens
    public func stopTokens() throws -> [UInt32] {
        var tokensPtr: UnsafeMutablePointer<UInt32>?
        var tokensLen: Int = 0
        
        let result = harmony_encoding_stop_tokens(
            wrapper,
            &tokensPtr,
            &tokensLen
        )
        
        if !result.success {
            let errorMsg = result.error_message != nil
                ? String(cString: result.error_message!)
                : "Unknown error"
            if result.error_message != nil {
                harmony_free_string(result.error_message)
            }
            throw HarmonyFFIError(message: errorMsg)
        }
        
        guard let tokens = tokensPtr else {
            throw HarmonyFFIError(message: "No tokens returned")
        }
        
        // Copy tokens to Swift array
        let tokensArray = Array(UnsafeBufferPointer(start: tokens, count: tokensLen))
        
        // Free the allocated memory
        harmony_free_tokens(tokens, tokensLen)
        
        return tokensArray
    }
}