import Foundation

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
        var tokensLen: Int = 0
        
        let result = harmony_encoding_render_prompt(
            encoding,
            systemMessage,
            userMessage,
            assistantPrefix,
            &tokensOut,
            &tokensLen
        )
        
        guard result == HARMONY_SUCCESS, let tokens = tokensOut else {
            return []
        }
        
        defer {
            harmony_free_tokens(tokens)
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
        var tokensLen: Int = 0
        
        let result = harmony_encoding_encode_plain(
            encoding,
            text,
            &tokensOut,
            &tokensLen
        )
        
        guard result == HARMONY_SUCCESS, let tokens = tokensOut else {
            return []
        }
        
        defer {
            harmony_free_tokens(tokens)
        }
        
        // Convert UInt32 tokens to Int for compatibility
        var intTokens: [Int] = []
        for i in 0..<tokensLen {
            intTokens.append(Int(tokens[i]))
        }
        
        return intTokens
    }
    
    /// Get stop tokens for generation
    public func stopTokens() throws -> [UInt32] {
        // Common stop tokens for o200k_harmony encoding
        return [
            200002,  // <|return|>
            200007,  // <|end|>
            200012,  // <|call|>
            199999   // <|endoftext|>
        ]
    }
}

// MARK: - Error Types

public enum HarmonyError: Error {
    case initializationFailed(String)
    case encodingFailed(String)
    case decodingFailed(String)
}

// MARK: - C FFI Declarations

// These would normally be imported from the harmony-swift C header
// For now, declaring them inline to match the expected interface

private let HARMONY_SUCCESS: Int32 = 0
private let HARMONY_ERROR: Int32 = 1

@_silgen_name("harmony_encoding_new")
private func harmony_encoding_new() -> OpaquePointer?

@_silgen_name("harmony_encoding_free")
private func harmony_encoding_free(_ encoding: OpaquePointer)

@_silgen_name("harmony_encoding_render_prompt")
private func harmony_encoding_render_prompt(
    _ encoding: OpaquePointer,
    _ systemMsg: UnsafePointer<CChar>?,
    _ userMsg: UnsafePointer<CChar>?,
    _ assistantPrefix: UnsafePointer<CChar>?,
    _ tokensOut: UnsafeMutablePointer<UnsafeMutablePointer<UInt32>?>,
    _ tokensLen: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("harmony_encoding_encode_plain")
private func harmony_encoding_encode_plain(
    _ encoding: OpaquePointer,
    _ text: UnsafePointer<CChar>?,
    _ tokensOut: UnsafeMutablePointer<UnsafeMutablePointer<UInt32>?>,
    _ tokensLen: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("harmony_free_tokens")
private func harmony_free_tokens(_ tokens: UnsafeMutablePointer<UInt32>)