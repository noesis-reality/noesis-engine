#ifndef TIKTOKEN_FFI_H
#define TIKTOKEN_FFI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to tiktoken CoreBPE
typedef struct CoreBPE CoreBPE;

// Result type for operations that can fail
typedef struct {
    bool success;
    char* error_message;
} TiktokenResult;

// Memory management
void tiktoken_free_string(char* s);
void tiktoken_free_tokens(uint32_t* tokens, size_t len);

// CoreBPE functions
CoreBPE* tiktoken_get_o200k_base(void);
void tiktoken_free_core_bpe(CoreBPE* bpe);

// Encoding/decoding functions
TiktokenResult tiktoken_encode_ordinary(
    const CoreBPE* bpe,
    const char* text,
    uint32_t** tokens_out,
    size_t* tokens_len
);

char* tiktoken_decode(
    const CoreBPE* bpe,
    const uint32_t* tokens,
    size_t tokens_len
);

#ifdef __cplusplus
}
#endif

#endif // TIKTOKEN_FFI_H