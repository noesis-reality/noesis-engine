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

#ifdef __cplusplus
}
#endif

#endif // HARMONY_FFI_H