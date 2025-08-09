package ai.noesisreality.protocol

import ai.noesisreality.core.NoesisConstants
import com.google.flatbuffers.kotlin.FlatBufferBuilder
import com.google.flatbuffers.kotlin.ReadWriteBuffer
import com.google.flatbuffers.kotlin.ArrayReadWriteBuffer
import noesis.protocol.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import java.util.concurrent.atomic.AtomicLong

/**
 * High-Performance Streaming Inference Engine using FlatBuffers
 * 
 * Provides real-time token streaming with zero-copy deserialization
 * for optimal performance during live inference.
 * 
 * Copyright (c) 2025 Noesis Reality LLC
 */
class StreamingInferenceEngine {
    
    private val requestIdGenerator = AtomicLong(0)
    
    // Native JNI methods using FlatBuffers
    private external fun nativeStreamingInference(requestBuffer: ByteArray): ByteArray
    private external fun nativeStartStream(requestBuffer: ByteArray): Long
    private external fun nativeReadStream(streamId: Long, bufferSize: Int): ByteArray?
    private external fun nativeStopStream(streamId: Long)
    
    companion object {
        init {
            System.loadLibrary(NoesisConstants.NativeLibs.SWIFT_INFERENCE)
        }
    }
    
    /**
     * Create a streaming inference flow with backpressure support
     */
    fun createInferenceStream(
        prompt: String,
        systemPrompt: String? = null,
        maxTokens: Int = NoesisConstants.Defaults.MAX_TOKENS,
        temperature: Float = NoesisConstants.Defaults.TEMPERATURE,
        streamingBatchSize: Int = 1
    ): Flow<StreamingInferenceResult> = flow {
        
        val requestId = generateRequestId()
        
        // Build FlatBuffers request using ariawisp fork
        val requestBuffer = buildInferenceRequest(
            requestId = requestId,
            prompt = prompt,
            systemPrompt = systemPrompt,
            maxTokens = maxTokens,
            temperature = temperature,
            topP = 0.9f,
            repetitionPenalty = 1.1f,
            reasoningEffort = ReasoningLevel.Medium,
            streamingMode = StreamingMode.Tokens
        )
        
        // Start native streaming
        val streamId = nativeStartStream(requestBuffer)
        if (streamId == 0L) {
            throw RuntimeException("Failed to start inference stream")
        }
        
        try {
            var sequenceId = 0u
            var isComplete = false
            
            while (!isComplete) {
                // Read streaming batch from native layer
                val responseBuffer = nativeReadStream(streamId, 8192)
                
                if (responseBuffer != null) {
                    val responseReadBuffer = ArrayReadWriteBuffer(responseBuffer)
        val message = NoesisMessage.asRoot(responseReadBuffer)
                    
                    when (message.messageType) {
                        MessageType.TokenStream -> {
                            val tokenStream = message.tokenStream
                            if (tokenStream != null) {
                                val tokensArray = IntArray(tokenStream.tokensLength)
                                for (i in tokensArray.indices) {
                                    tokensArray[i] = tokenStream.tokens(i).toInt()
                                }
                                
                                emit(StreamingInferenceResult.TokenBatch(
                                    requestId = tokenStream.id ?: "",
                                    sequenceId = tokenStream.sequenceId,
                                    tokens = tokensArray.toList(),
                                    textDelta = tokenStream.textDelta ?: "",
                                    isComplete = tokenStream.isFinal,
                                    batchTimeMs = tokenStream.batchTimeMs.toLong(),
                                    cumulativeTokens = tokenStream.cumulativeTokens.toInt(),
                                    instantaneousSpeed = tokenStream.instantaneousSpeed
                                ))
                                
                                isComplete = tokenStream.isFinal
                                sequenceId++
                            }
                        }
                        
                        MessageType.ReasoningStream -> {
                            val reasoningStream = message.reasoningStream
                            if (reasoningStream != null) {
                                emit(StreamingInferenceResult.ReasoningStep(
                                    requestId = reasoningStream.id ?: "",
                                    sequenceId = reasoningStream.sequenceId,
                                    step = reasoningStream.reasoningStep?.let { step ->
                                        ReasoningStepData(
                                            stepNumber = step.stepNumber.toInt(),
                                            thought = step.thought ?: "",
                                            analysis = step.analysis ?: "",
                                            conclusion = step.conclusion,
                                            confidence = step.confidence,
                                            timestampMs = step.timestampMs.toLong()
                                        )
                                    },
                                    isComplete = reasoningStream.isFinal
                                ))
                            }
                        }
                        
                        MessageType.ErrorResponse -> {
                            val error = message.errorResponse
                            if (error != null) {
                                throw StreamingInferenceException(
                                    error.errorMessage ?: "Unknown error",
                                    error.errorCode.toInt(),
                                    error.nativeLayer ?: "unknown"
                                )
                            }
                        }
                        
                        else -> {
                            // Unexpected message type
                            println("${NoesisConstants.Emojis.WARNING} Unexpected message type: ${message.messageType}")
                        }
                    }
                } else {
                    // No more data available, check if stream is still active
                    delay(10) // Small delay to prevent busy waiting
                }
            }
            
        } finally {
            nativeStopStream(streamId)
        }
    }
    
    /**
     * Non-streaming inference for single-shot generation
     */
    suspend fun generateSingleShot(
        prompt: String,
        systemPrompt: String? = null,
        maxTokens: Int = NoesisConstants.Defaults.MAX_TOKENS,
        temperature: Float = NoesisConstants.Defaults.TEMPERATURE,
        topP: Float = NoesisConstants.Defaults.TOP_P,
        repetitionPenalty: Float = NoesisConstants.Defaults.REPETITION_PENALTY,
        reasoningEffort: String = NoesisConstants.Defaults.REASONING_EFFORT
    ): CompleteInferenceResult = withContext(Dispatchers.Default) {
        
        val requestId = generateRequestId()
        
        // Build FlatBuffers request
        val reasoningLevel = when(reasoningEffort) {
            "low" -> ReasoningLevel.Low
            "high" -> ReasoningLevel.High
            else -> ReasoningLevel.Medium
        }
        
        val requestBuffer = buildInferenceRequest(
            requestId = requestId,
            prompt = prompt,
            systemPrompt = systemPrompt,
            maxTokens = maxTokens,
            temperature = temperature,
            topP = topP,
            repetitionPenalty = repetitionPenalty,
            reasoningEffort = reasoningLevel,
            streamingMode = StreamingMode.Disabled
        )
        
        // Execute single-shot inference
        val responseBuffer = nativeStreamingInference(requestBuffer)
        val responseReadBuffer = ArrayReadWriteBuffer(responseBuffer)
        val message = NoesisMessage.asRoot(responseReadBuffer)
        
        when (message.messageType) {
            MessageType.InferenceResponse -> {
                val response = message.inferenceResponse!!
                
                // Parse tokens array
                val tokensArray = IntArray(response.tokensLength)
                for (i in tokensArray.indices) {
                    tokensArray[i] = response.tokens(i).toInt()
                }
                
                // Parse reasoning steps
                val reasoningSteps = (0 until response.reasoningStepsLength).map { i ->
                    val step = response.reasoningSteps(i)!!
                    ReasoningStepData(
                        stepNumber = step.stepNumber.toInt(),
                        thought = step.thought ?: "",
                        analysis = step.analysis,
                        conclusion = step.conclusion,
                        confidence = step.confidence,
                        timestampMs = step.timestampMs.toLong()
                    )
                }
                
                CompleteInferenceResult(
                    requestId = response.id ?: "",
                    text = response.text ?: "",
                    tokens = tokensArray.toList(),
                    tokensGenerated = response.tokensGenerated.toInt(),
                    timeMs = response.timeMs.toLong(),
                    tokensPerSecond = response.tokensPerSecond,
                    gpuMemoryMB = response.gpuMemoryMb.toInt(),
                    harmonyFormat = response.harmonyFormat,
                    reasoningSteps = reasoningSteps
                )
            }
            
            MessageType.ErrorResponse -> {
                val error = message.errorResponse!!
                throw StreamingInferenceException(
                    error.errorMessage ?: "Unknown error",
                    error.errorCode.toInt(),
                    error.nativeLayer ?: "unknown"
                )
            }
            
            else -> throw IllegalStateException("Unexpected response type: ${message.messageType}")
        }
    }
    
    /**
     * Create a high-throughput inference channel for batch processing
     */
    fun createBatchChannel(
        bufferSize: Int = 100
    ): Pair<Channel<InferenceBatchRequest>, Flow<InferenceBatchResult>> {
        val requestChannel = Channel<InferenceBatchRequest>(bufferSize)
        
        val resultFlow = requestChannel.receiveAsFlow()
            .buffer(bufferSize)
            .map { request ->
                val startTime = System.currentTimeMillis()
                
                val results = request.prompts.map { prompt ->
                    generateSingleShot(
                        prompt = prompt.text,
                        systemPrompt = prompt.systemPrompt,
                        maxTokens = prompt.maxTokens,
                        temperature = prompt.temperature
                    )
                }
                
                InferenceBatchResult(
                    batchId = request.batchId,
                    results = results,
                    totalTimeMs = System.currentTimeMillis() - startTime
                )
            }
        
        return requestChannel to resultFlow
    }
    
    private fun generateRequestId(): String = 
        "req_${requestIdGenerator.incrementAndGet()}_${System.currentTimeMillis()}"
        
    private fun buildInferenceRequest(
        requestId: String,
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        reasoningEffort: ReasoningLevel,
        streamingMode: StreamingMode
    ): ByteArray {
        val builder = FlatBufferBuilder(NoesisConstants.FlatBuffers.DEFAULT_BUFFER_SIZE)
        
        // Create strings
        val idOffset = builder.createString(requestId)
        val promptOffset = builder.createString(prompt)
        val systemPromptOffset = systemPrompt?.let { builder.createString(it) }
        
        // Create inference request
        val requestOffset = InferenceRequest.createInferenceRequest(
            builder = builder,
            idOffset = idOffset,
            promptOffset = promptOffset,
            systemPromptOffset = systemPromptOffset ?: builder.createString(""),
            maxTokens = maxTokens.toUInt(),
            temperature = temperature,
            topP = topP,
            repetitionPenalty = repetitionPenalty,
            reasoningEffort = reasoningEffort,
            seed = 0u,
            streamingMode = streamingMode,
            streamingBatchSize = NoesisConstants.Engines.Noesis.DEFAULT_STREAMING_BATCH_SIZE.toUInt(),
            priority = 0,
            timeoutMs = NoesisConstants.Defaults.TIMEOUT_MS.toUInt()
        )
        
        // Start NoesisMessage table
        NoesisMessage.startNoesisMessage(builder)
        NoesisMessage.addMessageType(builder, MessageType.InferenceRequest)
        NoesisMessage.addTimestampMs(builder, System.currentTimeMillis().toULong())
        NoesisMessage.addInferenceRequest(builder, requestOffset)
        val messageOffset = NoesisMessage.endNoesisMessage(builder)
        
        builder.finish(messageOffset)
        return builder.sizedByteArray()
    }
}

// Result data classes
sealed class StreamingInferenceResult {
    data class TokenBatch(
        val requestId: String,
        val sequenceId: UInt,
        val tokens: List<Int>,
        val textDelta: String,
        val isComplete: Boolean,
        val batchTimeMs: Long,
        val cumulativeTokens: Int,
        val instantaneousSpeed: Double
    ) : StreamingInferenceResult()
    
    data class ReasoningStep(
        val requestId: String,
        val sequenceId: UInt,
        val step: ReasoningStepData?,
        val isComplete: Boolean
    ) : StreamingInferenceResult()
}

data class CompleteInferenceResult(
    val requestId: String,
    val text: String,
    val tokens: List<Int>,
    val tokensGenerated: Int,
    val timeMs: Long,
    val tokensPerSecond: Double,
    val gpuMemoryMB: Int,
    val harmonyFormat: String?,
    val reasoningSteps: List<ReasoningStepData>
)

data class ReasoningStepData(
    val stepNumber: Int,
    val thought: String,
    val analysis: String?,
    val conclusion: String?,
    val confidence: Float,
    val timestampMs: Long
)

data class InferenceBatchRequest(
    val batchId: String,
    val prompts: List<BatchPrompt>
)

data class BatchPrompt(
    val text: String,
    val systemPrompt: String? = null,
    val maxTokens: Int = NoesisConstants.Defaults.MAX_TOKENS,
    val temperature: Float = NoesisConstants.Defaults.TEMPERATURE
)

data class InferenceBatchResult(
    val batchId: String,
    val results: List<CompleteInferenceResult>,
    val totalTimeMs: Long
)

class StreamingInferenceException(
    message: String,
    val errorCode: Int,
    val nativeLayer: String
) : Exception("[$nativeLayer:$errorCode] $message")