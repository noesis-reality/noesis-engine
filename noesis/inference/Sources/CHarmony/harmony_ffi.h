#ifndef HARMONY_FFI_H
#define HARMONY_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types
typedef struct HarmonyEncodingWrapper HarmonyEncodingWrapper;
typedef struct StreamableParserWrapper StreamableParserWrapper;

// Result type for operations that can fail
typedef struct {
    bool success;
    char* error_message;
} HarmonyResult;

// Memory management
void harmony_free_string(char* s);
void harmony_free_tokens(uint32_t* tokens, size_t len);

// Harmony Encoding functions
HarmonyEncodingWrapper* harmony_encoding_new(void);
void harmony_encoding_free(HarmonyEncodingWrapper* wrapper);

// Plain text encoding - encode text without Harmony formatting
HarmonyResult harmony_encoding_encode_plain(
    const HarmonyEncodingWrapper* wrapper,
    const char* text,
    uint32_t** tokens_out,
    size_t* tokens_len
);

// Harmony prompt rendering
HarmonyResult harmony_encoding_render_prompt(
    const HarmonyEncodingWrapper* wrapper,
    const char* system_msg,
    const char* user_msg,
    const char* assistant_prefix,
    uint32_t** tokens_out,
    size_t* tokens_len
);

// Decode tokens to text
char* harmony_encoding_decode(
    const HarmonyEncodingWrapper* wrapper,
    const uint32_t* tokens,
    size_t tokens_len
);

// Get stop tokens
HarmonyResult harmony_encoding_stop_tokens(
    const HarmonyEncodingWrapper* wrapper,
    uint32_t** tokens_out,
    size_t* tokens_len
);

// StreamableParser functions for incremental token processing
StreamableParserWrapper* harmony_streamable_parser_new(const HarmonyEncodingWrapper* encoding);
void harmony_streamable_parser_free(StreamableParserWrapper* parser);

// Incremental parsing - feed data chunks and get tokens as they become available
HarmonyResult harmony_streamable_parser_feed(
    const StreamableParserWrapper* parser,
    const uint8_t* data,
    size_t data_len,
    uint32_t** tokens_out,
    size_t* tokens_len
);

// Stream state management
bool harmony_streamable_parser_has_pending(const StreamableParserWrapper* parser);
HarmonyResult harmony_streamable_parser_flush(
    const StreamableParserWrapper* parser,
    uint32_t** tokens_out,
    size_t* tokens_len
);

// Reset parser state for new stream
void harmony_streamable_parser_reset(StreamableParserWrapper* parser);

#ifdef __cplusplus
}
#endif

#endif // HARMONY_FFI_H