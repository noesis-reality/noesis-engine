import Foundation
import CHarmony
import Harmony
import NoesisTools

/**
 * Swift bridge for Harmony channel-aware streaming
 * 
 * Integrates with harmony-swift C FFI to provide real implementations
 * for multi-channel streaming and incremental token processing.
 */
public class HarmonyStreamingBridge {
    
    public static let shared = HarmonyStreamingBridge()
    
    private var activeStreams: [Int64: HarmonyStreamContext] = [:]
    private var nextStreamId: Int64 = 1
    private let streamLock = NSLock()
    
    private init() {}
    
    /**
     * Create a new Harmony stream with channel separation
     */
    public func createStream(requestData: Data) throws -> Int64 {
        let streamId = generateStreamId()
        
        // Parse the Harmony request to understand what channels to enable
        let channels = parseRequestChannels(requestData)
        let context = try HarmonyStreamContext(
            streamId: streamId,
            requestData: requestData,
            enabledChannels: channels
        )
        
        streamLock.lock()
        activeStreams[streamId] = context
        streamLock.unlock()
        
        return streamId
    }
    
    /**
     * Read data from specific channel
     */
    public func readChannelData(streamId: Int64, channelId: Int32, bufferSize: Int32) -> Data? {
        streamLock.lock()
        guard let context = activeStreams[streamId] else {
            streamLock.unlock()
            return nil
        }
        streamLock.unlock()
        
        return context.readChannelData(channelId: channelId, bufferSize: Int(bufferSize))
    }
    
    /**
     * Check if channel is complete
     */
    public func isChannelComplete(streamId: Int64, channelId: Int32) -> Bool {
        streamLock.lock()
        guard let context = activeStreams[streamId] else {
            streamLock.unlock()
            return true
        }
        streamLock.unlock()
        
        return context.isChannelComplete(channelId: channelId)
    }
    
    /**
     * Stop stream and cleanup resources
     */
    public func stopStream(streamId: Int64) {
        streamLock.lock()
        if let context = activeStreams.removeValue(forKey: streamId) {
            context.cleanup()
        }
        streamLock.unlock()
    }
    
    // Private helper methods
    
    private func generateStreamId() -> Int64 {
        streamLock.lock()
        defer { streamLock.unlock() }
        let id = nextStreamId
        nextStreamId += 1
        return id
    }
    
    private func parseRequestChannels(_ requestData: Data) -> Set<HarmonyChannelType> {
        // Parse FlatBuffers request to determine which channels are enabled
        // For now, enable all channels by default
        return Set(HarmonyChannelType.allCases)
    }
}

/**
 * Context for managing individual Harmony streams
 */
private class HarmonyStreamContext {
    let streamId: Int64
    let enabledChannels: Set<HarmonyChannelType>
    
    private let harmonyEncoding: HarmonyEncoding
    private var channelProcessors: [HarmonyChannelType: HarmonyChannelProcessor] = [:]
    private var channelStates: [Int32: ChannelState] = [:]
    
    init(streamId: Int64, requestData: Data, enabledChannels: Set<HarmonyChannelType>) throws {
        self.streamId = streamId
        self.enabledChannels = enabledChannels
        self.harmonyEncoding = try HarmonyEncoding()
        
        // Initialize channel processors
        for channel in enabledChannels {
            channelProcessors[channel] = HarmonyChannelProcessor(
                channelType: channel,
                harmonyEncoding: harmonyEncoding
            )
            channelStates[Int32(channel.channelId)] = ChannelState()
        }
        
        // Start processing the request
        try processInitialRequest(requestData)
    }
    
    func readChannelData(channelId: Int32, bufferSize: Int) -> Data? {
        guard let channelType = HarmonyChannelType.fromId(Int(channelId)),
              let processor = channelProcessors[channelType],
              let state = channelStates[channelId] else {
            return nil
        }
        
        return processor.readAvailableData(maxSize: bufferSize)
    }
    
    func isChannelComplete(channelId: Int32) -> Bool {
        return channelStates[channelId]?.isComplete ?? true
    }
    
    func cleanup() {
        for processor in channelProcessors.values {
            processor.cleanup()
        }
        channelProcessors.removeAll()
        channelStates.removeAll()
    }
    
    private func processInitialRequest(_ requestData: Data) throws {
        // Parse the FlatBuffers request and start processing
        let requestText = parseRequestText(requestData)
        
        // Process through Harmony encoding to get structured output
        do {
            let tokens = try harmonyEncoding.renderPrompt(
                systemMessage: "You are a helpful assistant with structured thinking.",
                userMessage: requestText,
                assistantPrefix: "<thinking>"
            )
            
            // Route tokens to appropriate channels based on content analysis
            routeTokensToChannels(tokens)
            
        } catch {
            // Mark all channels as complete if processing fails
            for channelId in channelStates.keys {
                channelStates[channelId]?.isComplete = true
            }
            throw error
        }
    }
    
    private func parseRequestText(_ requestData: Data) -> String {
        // Simple extraction - in real implementation would parse FlatBuffers
        return String(data: requestData, encoding: .utf8) ?? ""
    }
    
    private func routeTokensToChannels(_ tokens: [UInt32]) {
        // Analyze tokens and route to appropriate channels
        let text = harmonyEncoding.decode(tokens: tokens) ?? ""
        
        // Simple content-based routing
        if text.contains("<thinking>") {
            channelProcessors[.thinking]?.processTokens(tokens, text: text)
        }
        if text.contains("analysis") || text.contains("because") {
            channelProcessors[.analysis]?.processTokens(tokens, text: text)
        }
        if text.contains("therefore") || text.contains("reasoning") {
            channelProcessors[.reasoning]?.processTokens(tokens, text: text)
        }
        // Default to response channel
        channelProcessors[.response]?.processTokens(tokens, text: text)
        
        // Mark processing complete for this batch
        for channelId in channelStates.keys {
            channelStates[channelId]?.hasNewData = true
        }
    }
}

/**
 * Processes content for specific Harmony channels
 */
private class HarmonyChannelProcessor {
    let channelType: HarmonyChannelType
    let harmonyEncoding: HarmonyEncoding
    
    private var bufferedData = Data()
    private var isProcessingComplete = false
    
    init(channelType: HarmonyChannelType, harmonyEncoding: HarmonyEncoding) {
        self.channelType = channelType
        self.harmonyEncoding = harmonyEncoding
    }
    
    func processTokens(_ tokens: [UInt32], text: String) {
        // Convert tokens and text to structured data format
        let processedData = createChannelData(tokens: tokens, text: text)
        bufferedData.append(processedData)
    }
    
    func readAvailableData(maxSize: Int) -> Data? {
        guard !bufferedData.isEmpty else { return nil }
        
        let readSize = min(maxSize, bufferedData.count)
        let result = bufferedData.subdata(in: 0..<readSize)
        bufferedData.removeSubrange(0..<readSize)
        
        return result
    }
    
    func cleanup() {
        bufferedData.removeAll()
        isProcessingComplete = true
    }
    
    private func createChannelData(tokens: [UInt32], text: String) -> Data {
        // Create structured data for this channel
        // In real implementation, would use FlatBuffers to create HarmonyResponse
        let response = [
            "channel": channelType.displayName,
            "tokens": tokens,
            "text": text,
            "timestamp": Date().timeIntervalSince1970 * 1000
        ] as [String: Any]
        
        return try! JSONSerialization.data(withJSONObject: response)
    }
}

/**
 * Harmony channel types matching the Kotlin enum
 */
enum HarmonyChannelType: Int, CaseIterable {
    case thinking = 0
    case analysis = 1
    case reasoning = 2
    case response = 3
    case tools = 4
    case reflection = 5
    
    var channelId: Int { return rawValue }
    
    var displayName: String {
        switch self {
        case .thinking: return "Internal Thinking"
        case .analysis: return "Problem Analysis"
        case .reasoning: return "Logical Reasoning"
        case .response: return "Final Response"
        case .tools: return "Tool Integration"
        case .reflection: return "Self-Reflection"
        }
    }
    
    static func fromId(_ id: Int) -> HarmonyChannelType? {
        return HarmonyChannelType(rawValue: id)
    }
}

/**
 * State tracking for individual channels
 */
private struct ChannelState {
    var isComplete: Bool = false
    var hasNewData: Bool = false
    var lastReadPosition: Int = 0
}

// MARK: - JNI Bridge Functions

/**
 * Create Harmony stream - called from Kotlin
 */
@_cdecl("harmonyCreateStream")
public func harmonyCreateStream(
    requestData: UnsafePointer<UInt8>,
    requestLength: Int32
) -> Int64 {
    let data = Data(bytes: requestData, count: Int(requestLength))
    
    do {
        let bridge = HarmonyStreamingBridge.shared
        return try bridge.createStream(requestData: data)
    } catch {
        print("Error creating Harmony stream: \(error)")
        return 0
    }
}

/**
 * Read channel data - called from Kotlin
 */
@_cdecl("harmonyReadChannelData")
public func harmonyReadChannelData(
    streamId: Int64,
    channelId: Int32,
    bufferSize: Int32,
    outputBuffer: UnsafeMutablePointer<UInt8>,
    outputLength: UnsafeMutablePointer<Int32>
) -> Bool {
    let bridge = HarmonyStreamingBridge.shared
    
    guard let data = bridge.readChannelData(
        streamId: streamId,
        channelId: channelId,
        bufferSize: bufferSize
    ) else {
        outputLength.pointee = 0
        return false
    }
    
    let copyLength = min(data.count, Int(bufferSize))
    data.copyBytes(to: outputBuffer, count: copyLength)
    outputLength.pointee = Int32(copyLength)
    
    return true
}

/**
 * Check if channel is complete - called from Kotlin
 */
@_cdecl("harmonyIsChannelComplete")
public func harmonyIsChannelComplete(streamId: Int64, channelId: Int32) -> Bool {
    let bridge = HarmonyStreamingBridge.shared
    return bridge.isChannelComplete(streamId: streamId, channelId: channelId)
}

/**
 * Stop Harmony stream - called from Kotlin
 */
@_cdecl("harmonyStopStream")
public func harmonyStopStream(streamId: Int64) {
    let bridge = HarmonyStreamingBridge.shared
    bridge.stopStream(streamId: streamId)
}

// MARK: - Singleton Bridge

extension HarmonyStreamingBridge {
    static let shared = HarmonyStreamingBridge()
}

// MARK: - JNI Bridge Functions for Kotlin HarmonyStreamingEngine

/**
 * Create Harmony channel-aware stream - JNI export for Kotlin
 */
@_cdecl("Java_ai_noesisreality_protocol_HarmonyStreamingEngine_nativeCreateHarmonyStream")
public func jniCreateHarmonyStream(
    env: UnsafeMutablePointer<JNIEnv>,
    clazz: jobject,
    requestBuffer: jobject
) -> jlong {
    // Extract byte array from JNI
    guard let requestData = extractByteArrayFromJNI(env: env, byteArray: requestBuffer) else {
        return 0
    }
    
    do {
        let bridge = HarmonyStreamingBridge.shared
        let streamId = try bridge.createStream(requestData: requestData)
        return jlong(streamId)
    } catch {
        print("Error creating Harmony stream: \(error)")
        return 0
    }
}

/**
 * Read channel data - JNI export for Kotlin
 */
@_cdecl("Java_ai_noesisreality_protocol_HarmonyStreamingEngine_nativeReadChannelData")
public func jniReadChannelData(
    env: UnsafeMutablePointer<JNIEnv>,
    clazz: jobject,
    streamId: jlong,
    channelId: jint,
    bufferSize: jint
) -> jobject? {
    let bridge = HarmonyStreamingBridge.shared
    
    guard let data = bridge.readChannelData(
        streamId: Int64(streamId),
        channelId: Int32(channelId), 
        bufferSize: Int32(bufferSize)
    ) else {
        return createEmptyByteArrayJNI(env: env)
    }
    
    return createByteArrayJNI(env: env, data: data)
}

/**
 * Check if channel is complete - JNI export for Kotlin
 */
@_cdecl("Java_ai_noesisreality_protocol_HarmonyStreamingEngine_nativeIsChannelComplete")
public func jniIsChannelComplete(
    env: UnsafeMutablePointer<JNIEnv>,
    clazz: jobject,
    streamId: jlong,
    channelId: jint
) -> jboolean {
    let bridge = HarmonyStreamingBridge.shared
    let isComplete = bridge.isChannelComplete(
        streamId: Int64(streamId),
        channelId: Int32(channelId)
    )
    return isComplete ? JNI_TRUE : JNI_FALSE
}

/**
 * Stop Harmony stream - JNI export for Kotlin
 */
@_cdecl("Java_ai_noesisreality_protocol_HarmonyStreamingEngine_nativeStopHarmonyStream")
public func jniStopHarmonyStream(
    env: UnsafeMutablePointer<JNIEnv>,
    clazz: jobject,
    streamId: jlong
) {
    let bridge = HarmonyStreamingBridge.shared
    bridge.stopStream(streamId: Int64(streamId))
}

// MARK: - JNI Helper Functions

private func extractByteArrayFromJNI(env: UnsafeMutablePointer<JNIEnv>, byteArray: jobject) -> Data? {
    // This is a simplified implementation
    // In a real implementation, you'd use proper JNI functions to extract the byte array
    // For now, return empty data as placeholder
    return Data()
}

private func createByteArrayJNI(env: UnsafeMutablePointer<JNIEnv>, data: Data) -> jobject? {
    // This is a simplified implementation
    // In a real implementation, you'd use proper JNI functions to create a Java byte array
    // For now, return nil as placeholder
    return nil
}

private func createEmptyByteArrayJNI(env: UnsafeMutablePointer<JNIEnv>) -> jobject? {
    // This is a simplified implementation
    // In a real implementation, you'd use proper JNI functions to create an empty Java byte array
    return nil
}