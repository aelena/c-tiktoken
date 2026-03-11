// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 7: High-Level Encoding API
//
// The top-level API that combines regex splitting, BPE encoding, special
// token handling, and decoding into a unified interface.

#ifndef TIKTOKEN_ENCODING_H
#define TIKTOKEN_ENCODING_H

#include "tiktoken/bpe.h"
#include "tiktoken/bytes.h"
#include "tiktoken/hash.h"
#include "tiktoken/regex.h"
#include "tiktoken/vocab.h"

#include <stddef.h>
#include <stdint.h>

// ── C23 feature: enum with fixed underlying type ──────────────────────
//
// Controls how special tokens (like <|endoftext|>) are handled during
// encoding. The `: int` suffix guarantees the enum's size and signedness.

enum TiktokenSpecialMode : int {
    TIKTOKEN_SPECIAL_DISALLOW = 0,   // error if special tokens appear
    TIKTOKEN_SPECIAL_ALLOW,          // encode special tokens as special
    TIKTOKEN_SPECIAL_IGNORE,         // treat special token text as ordinary
};

// ── Special token entry ───────────────────────────────────────────────

typedef struct {
    const char *text;       // e.g., "<|endoftext|>"
    size_t      text_len;
    uint32_t    token_id;
} SpecialToken;

// ── TiktokenEncoding: the top-level encoder ───────────────────────────

typedef struct {
    const char      *name;              // e.g., "cl100k_base"
    VocabResult      vocab;             // loaded vocabulary + arena
    Regex           *pattern;           // compiled regex for splitting
    SpecialToken    *special_tokens;    // array of special tokens
    size_t           n_special;         // number of special tokens
} TiktokenEncoding;

// ── Lifecycle ──────────────────────────────────────────────────────────

// Create an encoding from components. Takes ownership of all arguments.
[[nodiscard]]
TiktokenEncoding *tiktoken_new(const char       *name,
                               VocabResult       vocab,
                               Regex            *pattern,
                               SpecialToken     *special_tokens,
                               size_t            n_special);

// Free an encoding and all its resources.
void tiktoken_free(TiktokenEncoding *enc);

// ── Encoding (text → tokens) ──────────────────────────────────────────

// Encode text into token IDs. Special tokens are handled according to
// the mode parameter.
// Returns a TokenVec that the caller must free with tokvec_free().
[[nodiscard]]
TokenVec tiktoken_encode(const TiktokenEncoding *enc,
                         const char             *text,
                         size_t                  text_len,
                         enum TiktokenSpecialMode mode);

// Encode text without special token handling (treats everything as
// ordinary text). Equivalent to tiktoken_encode with IGNORE mode.
[[nodiscard]]
TokenVec tiktoken_encode_ordinary(const TiktokenEncoding *enc,
                                  const char             *text,
                                  size_t                  text_len);

// ── Decoding (tokens → text) ──────────────────────────────────────────

// Decode token IDs back into raw bytes.
// Returns a Bytes that the caller must free with bytes_free().
[[nodiscard]]
Bytes tiktoken_decode(const TiktokenEncoding *enc,
                      const uint32_t         *tokens,
                      size_t                  n_tokens);

// ── Utility ───────────────────────────────────────────────────────────

// Count the number of tokens in the text without returning them.
[[nodiscard]]
size_t tiktoken_count(const TiktokenEncoding *enc,
                      const char             *text,
                      size_t                  text_len);

#endif // TIKTOKEN_ENCODING_H
