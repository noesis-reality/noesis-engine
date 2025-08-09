import Foundation
import Harmony
import CTiktoken

/// O200k_gptoss tokenizer using HarmonyFFI
public class O200kTokenizer {
    
    /// Special tokens for GPT-OSS
    public enum SpecialToken: String, CaseIterable {
        case reversed199998 = "<|reversed199998|>"
        case endOfText = "<|endoftext|>"
        case untrusted = "<|untrusted|>"
        case endOfUntrusted = "<|endofuntrusted|>"
        case `return` = "<|return|>"
        case constrain = "<|constrain|>"
        case reversed200004 = "<|reversed200004|>"
        case channel = "<|channel|>"
        case start = "<|start|>"
        case end = "<|end|>"
        case message = "<|message|>"
        case reversed200008 = "<|reversed200008|>"
        case reversed200009 = "<|reversed200009|>"
        case reversed200010 = "<|reversed200010|>"
        case reversed200011 = "<|reversed200011|>"
        case call = "<|call|>"
        case refusal = "<|refusal|>"
        
        var tokenId: UInt32 {
            switch self {
            case .reversed199998: return 199998
            case .endOfText: return 199999
            case .untrusted: return 200000
            case .endOfUntrusted: return 200001
            case .`return`: return 200002
            case .constrain: return 200003
            case .reversed200004: return 200004
            case .channel: return 200005
            case .start: return 200006
            case .end: return 200007
            case .message: return 200008
            case .reversed200008: return 200008
            case .reversed200009: return 200009
            case .reversed200010: return 200010
            case .reversed200011: return 200011
            case .call: return 200012
            case .refusal: return 200013
            }
        }
    }
    
    private let harmonyEncoding: HarmonyEncoding
    private let tiktokenBPE: OpaquePointer?
    private let specialTokenMap: [String: UInt32]
    private let reverseSpecialTokenMap: [UInt32: String]
    
    public init() throws {
        // Initialize HarmonyFFI encoding for conversation formatting
        self.harmonyEncoding = try HarmonyEncoding()
        
        // Initialize tiktoken CoreBPE for base tokenization
        self.tiktokenBPE = tiktoken_get_o200k_base()
        if self.tiktokenBPE == nil {
            throw TokenizerError.initializationFailed("Failed to initialize tiktoken o200k_base")
        }
        
        // Build special token maps
        var specialMap: [String: UInt32] = [:]
        var reverseMap: [UInt32: String] = [:]
        
        for token in SpecialToken.allCases {
            specialMap[token.rawValue] = token.tokenId
            reverseMap[token.tokenId] = token.rawValue
        }
        
        self.specialTokenMap = specialMap
        self.reverseSpecialTokenMap = reverseMap
    }
    
    deinit {
        if let bpe = tiktokenBPE {
            tiktoken_free_core_bpe(bpe)
        }
    }
    
    enum TokenizerError: Error {
        case initializationFailed(String)
        case encodingFailed(String)
        case decodingFailed(String)
    }
    
    /// Encode text to token IDs using tiktoken for base tokenization
    public func encode(_ text: String, allowedSpecial: String = "all") -> [UInt32] {
        guard let bpe = tiktokenBPE else { return [] }
        
        var tokensPtr: UnsafeMutablePointer<UInt32>?
        var tokensLen: size_t = 0
        
        let result = tiktoken_encode_ordinary(bpe, text, &tokensPtr, &tokensLen)
        
        guard result.success else {
            if let errorMsg = result.error_message {
                print("Tiktoken encoding error: \(String(cString: errorMsg))")
                tiktoken_free_string(errorMsg)
            }
            return []
        }
        
        guard let tokens = tokensPtr, tokensLen > 0 else { return [] }
        
        let tokenArray = Array(UnsafeBufferPointer(start: tokens, count: tokensLen))
        tiktoken_free_tokens(tokensPtr, tokensLen)
        
        return tokenArray
    }
    
    /// Decode token IDs to text using tiktoken
    public func decode(_ tokens: [UInt32]) -> String {
        guard let bpe = tiktokenBPE else { return "" }
        
        let result = tiktoken_decode(bpe, tokens, tokens.count)
        
        guard let cString = result else { return "" }
        
        let decoded = String(cString: cString)
        tiktoken_free_string(cString)
        
        return decoded
    }
    
    /// Check if a token ID is a special token
    public func isSpecialToken(_ tokenId: UInt32) -> Bool {
        return reverseSpecialTokenMap[tokenId] != nil
    }
    
    /// Get the special token string for a token ID
    public func getSpecialTokenString(_ tokenId: UInt32) -> String? {
        return reverseSpecialTokenMap[tokenId]
    }
    
    /// Create a proper Harmony conversation for generation
    public func createHarmonyPrompt(
        systemMessage: String? = nil,
        userMessage: String,
        assistantPrefix: String? = nil
    ) -> [UInt32] {
        do {
            let intTokens = try harmonyEncoding.renderPrompt(
                systemMessage: systemMessage,
                userMessage: userMessage,
                assistantPrefix: assistantPrefix
            )
            return intTokens.map { UInt32($0) }
        } catch {
            print("Failed to create harmony prompt: \(error)")
            return []
        }
    }
    
    /// Vocabulary size
    public var vocabularySize: Int {
        return 200014  // 199998 text tokens + 16 special tokens
    }
    
    /// Number of text tokens
    public var numTextTokens: Int {
        return 199998
    }
    
    /// Number of special tokens
    public var numSpecialTokens: Int {
        return 16
    }
    
    /// Get stop tokens for assistant actions
    public func stopTokensForAssistantActions() -> [UInt32] {
        return [
            SpecialToken.return.tokenId,    // <|return|>
            SpecialToken.call.tokenId,       // <|call|>
            SpecialToken.end.tokenId,        // <|end|>
            SpecialToken.endOfText.tokenId   // <|endoftext|>
        ]
    }
    
    /// Get stop tokens for generation
    public func stopTokens() -> Set<UInt32> {
        // Get stop tokens from HarmonyFFI
        do {
            let tokens = try harmonyEncoding.stopTokens()
            return Set(tokens)
        } catch {
            // Fallback to default stop tokens
            return Set([
                SpecialToken.endOfText.tokenId,  // Always stop on EOS
                SpecialToken.end.tokenId,        // Stop on message end
            ])
        }
    }
}

// MARK: - Tokenizer Builder

/// Builds the tokenizer data for model export
public class TokenizerBuilder {
    
    /// Build tokenizer data for GPT-OSS model file
    public static func buildTokenizerData() -> (regex: Data, tokens: Data, specialUUIDs: [Data]) {
        // Use the standard O200k tiktoken regex pattern (matches Python tiktoken implementation)
        let regexPattern = #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        let regexData = regexPattern.data(using: String.Encoding.utf8) ?? Data()
        
        // For tokens, we'll use the pre-exported data from Python
        // since we can't directly access tiktoken's internal data
        let tokenData = loadPreExportedTokenData() ?? Data()
        
        // Build special token UUIDs
        var specialUUIDs: [Data] = []
        let specialTokenUUIDs: [String: UUID] = [
            "<|start|>": UUID(uuidString: "55a77c2f-8a01-4c54-8ac2-313bfc7e208d")!,
            "<|message|>": UUID(uuidString: "16e40431-f47f-4b22-b59b-8b278fc30a54")!,
            "<|end|>": UUID(uuidString: "fcac2f6d-4705-4f6b-b228-642accac7238")!,
            "<|return|>": UUID(uuidString: "f799ff69-1992-43c4-a3d8-d831f475dc75")!,
            "<|refusal|>": UUID(uuidString: "e15ba702-28c4-4292-ab8f-ffa434709128")!,
            "<|constrain|>": UUID(uuidString: "c0bb14c7-6022-49da-ad08-792d67e8b470")!,
            "<|channel|>": UUID(uuidString: "fd3dda11-c8ab-4033-876e-d93deb172c93")!,
            "<|call|>": UUID(uuidString: "1220f796-e388-4de5-b487-fe2eb5fe03c0")!,
            "<|untrusted|>": UUID(uuidString: "07d7da55-b346-4cff-8b37-7cefacf8a3e8")!,
            "<|endofuntrusted|>": UUID(uuidString: "f265bd9c-c717-469e-a447-920687d65d90")!,
        ]
        
        // Add UUIDs for special tokens in order
        for token in O200kTokenizer.SpecialToken.allCases {
            if let uuid = specialTokenUUIDs[token.rawValue] {
                specialUUIDs.append(uuid.data)
            } else {
                // Zero UUID for unspecified/reversed tokens
                specialUUIDs.append(Data(repeating: 0, count: 16))
            }
        }
        
        return (regex: regexData, tokens: tokenData, specialUUIDs: specialUUIDs)
    }
    
    /// Load pre-exported token data from Python export
    private static func loadPreExportedTokenData() -> Data? {
        let workingDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let tokensPath = workingDir.appendingPathComponent("tokenizer_tokens.bin")
        
        if FileManager.default.fileExists(atPath: tokensPath.path) {
            return try? Data(contentsOf: tokensPath)
        }
        
        // Fallback: create placeholder data
        print("Warning: tokenizer_tokens.bin not found, using placeholder data")
        var tokenData = Data()
        let numTextTokens = 199998
        
        for _ in 0..<numTextTokens {
            // Placeholder: single byte token
            var length = UInt16(1).littleEndian
            tokenData.append(Data(bytes: &length, count: 2))
            tokenData.append(Data([0]))
        }
        
        return tokenData
    }
}

