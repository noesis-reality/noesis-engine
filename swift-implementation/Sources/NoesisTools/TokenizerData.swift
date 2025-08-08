import Foundation

/// Loads pre-exported tokenizer data from Python tiktoken
public class TokenizerDataLoader {
    
    /// Load tokenizer data files exported from Python
    public static func loadTokenizerData(
        patternPath: URL,
        tokensPath: URL
    ) throws -> (regex: Data, tokens: Data) {
        // Load regex pattern
        let regexString = try String(contentsOf: patternPath, encoding: .utf8)
        guard let regexData = regexString.data(using: .ascii) else {
            throw TokenizerError.invalidPattern
        }
        
        // Load binary token data
        let tokenData = try Data(contentsOf: tokensPath)
        
        return (regex: regexData, tokens: tokenData)
    }
    
    /// Load tokenizer data from embedded resources or default paths
    public static func loadDefaultTokenizerData() throws -> (regex: Data, tokens: Data, specialUUIDs: [Data]) {
        // Try to load from working directory first
        let workingDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let patternPath = workingDir.appendingPathComponent("tokenizer_pattern.txt")
        let tokensPath = workingDir.appendingPathComponent("tokenizer_tokens.bin")
        
        if FileManager.default.fileExists(atPath: patternPath.path) &&
           FileManager.default.fileExists(atPath: tokensPath.path) {
            let (regex, tokens) = try loadTokenizerData(
                patternPath: patternPath,
                tokensPath: tokensPath
            )
            
            // Build special token UUIDs
            let specialUUIDs = buildSpecialTokenUUIDs()
            
            return (regex: regex, tokens: tokens, specialUUIDs: specialUUIDs)
        } else {
            // Fall back to placeholder data
            print("Warning: Tokenizer data files not found, using placeholders")
            return TokenizerBuilder.buildTokenizerData()
        }
    }
    
    /// Build special token UUIDs for GPT-OSS
    private static func buildSpecialTokenUUIDs() -> [Data] {
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
        
        // Map special tokens in order (199998-200013)
        let orderedTokens = [
            "<|reversed199998|>",  // 199998
            "<|endoftext|>",       // 199999
            "<|untrusted|>",       // 200000
            "<|endofuntrusted|>",  // 200001
            "<|return|>",          // 200002
            "<|constrain|>",       // 200003
            "<|reversed200004|>",  // 200004
            "<|channel|>",         // 200005
            "<|start|>",           // 200006
            "<|end|>",             // 200007
            "<|message|>",         // 200008
            "<|reversed200009|>",  // 200009
            "<|reversed200010|>",  // 200010
            "<|reversed200011|>",  // 200011
            "<|call|>",            // 200012
            "<|refusal|>",         // 200013
        ]
        
        for token in orderedTokens {
            if let uuid = specialTokenUUIDs[token] {
                specialUUIDs.append(uuid.data)
            } else {
                // Zero UUID for unspecified/reversed tokens
                specialUUIDs.append(Data(repeating: 0, count: 16))
            }
        }
        
        return specialUUIDs
    }
    
    enum TokenizerError: LocalizedError {
        case invalidPattern
        case dataNotFound
        
        var errorDescription: String? {
            switch self {
            case .invalidPattern:
                return "Invalid tokenizer pattern"
            case .dataNotFound:
                return "Tokenizer data files not found"
            }
        }
    }
}

// UUID extension moved to ModelExporter.swift to avoid duplication