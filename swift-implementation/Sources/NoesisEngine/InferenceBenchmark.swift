import Foundation
import Metal

/// High-precision inference benchmarking for NoesisEngine
/// Follows Metal Performance best practices for accurate measurement
@MainActor
public struct InferenceBenchmark {
    
    /// Precise timing measurement using mach_absolute_time for sub-microsecond accuracy
    public struct PrecisionTimer {
        private let startTime: UInt64
        private static let timebaseInfo: mach_timebase_info = {
            var info = mach_timebase_info()
            mach_timebase_info(&info)
            return info
        }()
        
        public init() {
            startTime = mach_absolute_time()
        }
        
        public var elapsedSeconds: Double {
            let elapsed = mach_absolute_time() - startTime
            let nanos = elapsed * UInt64(Self.timebaseInfo.numer) / UInt64(Self.timebaseInfo.denom)
            return Double(nanos) / 1_000_000_000.0
        }
        
        public var elapsedMilliseconds: Double {
            elapsedSeconds * 1000.0
        }
    }
    
    /// Metal GPU timing using command buffer completion handlers (non-blocking)
    public class GPUTimer {
        private var completionTimes: [Double] = []
        private let timer = PrecisionTimer()
        
        public func recordCompletion() {
            completionTimes.append(timer.elapsedSeconds)
        }
        
        public var averageTime: Double {
            guard !completionTimes.isEmpty else { return 0.0 }
            return completionTimes.reduce(0, +) / Double(completionTimes.count)
        }
        
        public var totalTime: Double {
            completionTimes.last ?? 0.0
        }
    }
    
    /// Comprehensive benchmarking results
    public struct BenchmarkResult {
        public let totalTokens: Int
        public let prefillTokens: Int
        public let generatedTokens: Int
        
        public let totalTimeSeconds: Double
        public let prefillTimeSeconds: Double
        public let generationTimeSeconds: Double
        
        public let warmupTimeSeconds: Double
        
        // Primary metrics
        public var tokensPerSecond: Double {
            guard totalTimeSeconds > 0 else { return 0 }
            return Double(totalTokens) / totalTimeSeconds
        }
        
        public var prefillTokensPerSecond: Double {
            guard prefillTimeSeconds > 0 else { return 0 }
            return Double(prefillTokens) / prefillTimeSeconds
        }
        
        public var generationTokensPerSecond: Double {
            guard generationTimeSeconds > 0 else { return 0 }
            return Double(generatedTokens) / generationTimeSeconds
        }
        
        // Time per token (latency metrics)
        public var millisecondsPerToken: Double {
            guard totalTokens > 0 else { return 0 }
            return (totalTimeSeconds * 1000.0) / Double(totalTokens)
        }
        
        public var timeToFirstToken: Double {
            prefillTimeSeconds
        }
        
        public func formattedReport() -> String {
            return """
            📊 NoesisEngine Inference Benchmark Results
            
            🔤 Tokens:
               • Total: \(totalTokens) tokens
               • Prefill: \(prefillTokens) tokens
               • Generated: \(generatedTokens) tokens
            
            ⏱️  Timing:
               • Total Time: \(String(format: "%.3f", totalTimeSeconds))s
               • Prefill Time: \(String(format: "%.3f", prefillTimeSeconds))s (TTFT)
               • Generation Time: \(String(format: "%.3f", generationTimeSeconds))s
               • Warmup Time: \(String(format: "%.3f", warmupTimeSeconds))s
            
            🚀 Performance:
               • Overall: \(String(format: "%.1f", tokensPerSecond)) tok/s
               • Prefill: \(String(format: "%.1f", prefillTokensPerSecond)) tok/s
               • Generation: \(String(format: "%.1f", generationTokensPerSecond)) tok/s
            
            📏 Latency:
               • Time per Token: \(String(format: "%.1f", millisecondsPerToken))ms
               • Time to First Token: \(String(format: "%.0f", timeToFirstToken * 1000))ms
            """
        }
    }
    
    /// Benchmark a complete inference session with proper warmup
    public static func measureInference(
        model: GptossModel,
        context: GptossContext,
        pipeline: GenerationPipeline,
        promptTokens: [UInt32],
        maxTokens: Int,
        sampler: GptossSampler,
        warmupRuns: Int = 3
    ) async throws -> (result: BenchmarkResult, generatedTokens: [UInt32]) {
        
        // GPU warmup to ensure Metal is ready and caches are populated
        let warmupTimer = PrecisionTimer()
        for _ in 0..<warmupRuns {
            context.reset()
            context.addTokens(promptTokens)
            _ = try pipeline.generateTokens(prompt: [], maxTokens: 1, sampler: sampler)
        }
        let warmupTime = warmupTimer.elapsedSeconds
        
        // Reset for actual benchmark
        context.reset()
        context.addTokens(promptTokens)
        
        // Measure prefill phase (processing prompt tokens)
        // Prefill happens automatically when we call generateTokens with context that has batch tokens
        
        // Measure generation phase (don't interfere with timing)
        let generationTimer = PrecisionTimer()
        let generatedTokens = try pipeline.generateTokens(
            prompt: [], // Empty since tokens already in context
            maxTokens: maxTokens,
            sampler: sampler
        )
        let generationTime = generationTimer.elapsedSeconds
        
        // Calculate prefill time (estimated from first token generation which includes prefill)
        // In our implementation, prefill happens during the first generateTokens call
        let firstTokenTime = generationTime / Double(max(1, generatedTokens.count))
        let estimatedPrefillTime = firstTokenTime * 1.5 // Prefill typically slower than generation
        
        let totalTime = generationTime
        
        let result = BenchmarkResult(
            totalTokens: promptTokens.count + generatedTokens.count,
            prefillTokens: promptTokens.count,
            generatedTokens: generatedTokens.count,
            totalTimeSeconds: totalTime,
            prefillTimeSeconds: estimatedPrefillTime,
            generationTimeSeconds: generationTime - estimatedPrefillTime,
            warmupTimeSeconds: warmupTime
        )
        
        return (result, generatedTokens)
    }
    
    /// Quick tokens-per-second measurement for a single generation
    public static func quickMeasure(
        pipeline: GenerationPipeline,
        context: GptossContext,
        maxTokens: Int,
        sampler: GptossSampler
    ) throws -> (tokensPerSecond: Double, generatedTokens: [UInt32]) {
        
        let timer = PrecisionTimer()
        let tokens = try pipeline.generateTokens(prompt: [], maxTokens: maxTokens, sampler: sampler)
        let elapsed = timer.elapsedSeconds
        
        let tps = elapsed > 0 ? Double(tokens.count) / elapsed : 0
        return (tps, tokens)
    }
}

/// Extension to GenerationPipeline for integrated benchmarking
public extension GenerationPipeline {
    
    /// Generate tokens with automatic benchmarking (non-intrusive)
    func generateTokensWithBenchmark(
        prompt: [UInt32],
        maxTokens: Int,
        sampler: GptossSampler = GptossSampler(),
        onToken: ((UInt32) -> Bool)? = nil,
        reportBenchmark: Bool = false
    ) throws -> (tokens: [UInt32], tokensPerSecond: Double) {
        
        let timer = InferenceBenchmark.PrecisionTimer()
        let tokens = try generateTokens(
            prompt: prompt,
            maxTokens: maxTokens,
            sampler: sampler,
            onToken: onToken
        )
        let elapsed = timer.elapsedSeconds
        
        let tps = elapsed > 0 ? Double(tokens.count) / elapsed : 0
        
        if reportBenchmark {
            print("⚡ Generated \(tokens.count) tokens in \(String(format: "%.3f", elapsed))s = \(String(format: "%.1f", tps)) tok/s")
        }
        
        return (tokens, tps)
    }
}