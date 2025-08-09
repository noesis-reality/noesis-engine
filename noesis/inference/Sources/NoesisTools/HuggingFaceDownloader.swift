import Foundation

/// Downloads models from Hugging Face Hub
public class HuggingFaceDownloader {
    public struct DownloadConfig {
        public let repoId: String
        public let localDir: URL
        public let patterns: [String]?
        public let token: String?
        
        public init(repoId: String, localDir: URL, patterns: [String]? = nil, token: String? = nil) {
            self.repoId = repoId
            self.localDir = localDir
            self.patterns = patterns
            self.token = token
        }
    }
    
    private let baseURL = "https://huggingface.co"
    private let session: URLSession
    
    public init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 600
        self.session = URLSession(configuration: config)
    }
    
    /// List files in a Hugging Face repository
    public func listFiles(repoId: String, revision: String = "main") async throws -> [String] {
        let apiURL = "\(baseURL)/api/models/\(repoId)/tree/\(revision)"
        
        var request = URLRequest(url: URL(string: apiURL)!)
        request.httpMethod = "GET"
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw DownloadError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        
        let files = try JSONDecoder().decode([FileInfo].self, from: data)
        return files.map { $0.path }
    }
    
    /// Download specific files from a repository
    public func downloadFiles(config: DownloadConfig, files: [String]) async throws {
        // Create local directory if needed
        try FileManager.default.createDirectory(at: config.localDir, withIntermediateDirectories: true)
        
        for file in files {
            // Check if file matches patterns
            if let patterns = config.patterns {
                let matches = patterns.contains { pattern in
                    file.hasSuffix(pattern.replacingOccurrences(of: "*", with: ""))
                }
                if !matches { continue }
            }
            
            try await downloadFile(
                repoId: config.repoId,
                filename: file,
                localPath: config.localDir.appendingPathComponent(file)
            )
        }
    }
    
    /// Download a single file
    public func downloadFile(repoId: String, filename: String, localPath: URL) async throws {
        let downloadURL = "\(baseURL)/\(repoId)/resolve/main/\(filename)"
        
        print("Downloading \(filename)...")
        
        var request = URLRequest(url: URL(string: downloadURL)!)
        request.httpMethod = "GET"
        
        // Create parent directory if needed
        let parentDir = localPath.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
        
        // Download with progress tracking
        let (tempURL, response) = try await session.download(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw DownloadError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        
        // Move to final location
        if FileManager.default.fileExists(atPath: localPath.path) {
            try FileManager.default.removeItem(at: localPath)
        }
        try FileManager.default.moveItem(at: tempURL, to: localPath)
        
        print("✓ Downloaded \(filename)")
    }
    
    /// Download checkpoint files needed for model export
    public func downloadCheckpoint(repoId: String, localDir: URL) async throws {
        print("Downloading checkpoint from \(repoId)...")
        
        let config = DownloadConfig(
            repoId: repoId,
            localDir: localDir,
            patterns: ["*.safetensors", "*.json"]
        )
        
        // List available files
        let files = try await listFiles(repoId: repoId)
        
        // Filter for checkpoint files
        let checkpointFiles = files.filter { file in
            file.hasSuffix(".safetensors") || file.hasSuffix(".json")
        }
        
        print("Found \(checkpointFiles.count) checkpoint files")
        
        // Download them
        try await downloadFiles(config: config, files: checkpointFiles)
        
        print("✅ Checkpoint downloaded to \(localDir.path)")
    }
    
    // MARK: - Types
    
    private struct FileInfo: Codable {
        let path: String
        let size: Int?
        let type: String
    }
    
    public enum DownloadError: Error {
        case httpError(Int)
        case invalidURL
        case noData
    }
}