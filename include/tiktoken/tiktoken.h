// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 8: Top-Level Convenience Header
//
// Include this single header to get the complete tiktoken API.
//
// Usage:
//     #include <tiktoken/tiktoken.h>
//
//     TiktokenEncoding *enc = tiktoken_from_file(
//         "cl100k_base.tiktoken",
//         tiktoken_pattern_cl100k(),
//         cl100k_special_tokens,
//         n_special
//     );
//
//     TokenVec tokens = tiktoken_encode_ordinary(enc, text, strlen(text));
//     // ... use tokens ...
//     tokvec_free(&tokens);
//     tiktoken_free(enc);

#ifndef TIKTOKEN_TIKTOKEN_H
#define TIKTOKEN_TIKTOKEN_H

// ── C23 feature: #elifdef / #elifndef ──────────────────────────────────
//
// C23 adds #elifdef and #elifndef as shorthand:
//
//   #ifdef FOO
//   // ...
//   #elifdef BAR       // C23: same as #elif defined(BAR)
//   // ...
//   #elifndef BAZ      // C23: same as #elif !defined(BAZ)
//   // ...
//   #endif
//
// These are minor convenience additions, but they reduce visual noise
// in platform-detection headers. We note them here but don't use them
// in this small project (we don't have complex platform conditionals).

#include "tiktoken/base64.h"
#include "tiktoken/bytes.h"
#include "tiktoken/arena.h"
#include "tiktoken/hash.h"
#include "tiktoken/bpe.h"
#include "tiktoken/regex.h"
#include "tiktoken/vocab.h"
#include "tiktoken/encoding.h"

// ── Convenience: create an encoding from a .tiktoken file ─────────────
//
// This is the simplest way to create an encoding. It loads the vocabulary,
// compiles the regex, and sets up special tokens in one call.

[[nodiscard]]
static inline TiktokenEncoding *tiktoken_from_file(
    const char          *vocab_path,
    const char          *regex_pattern,
    const SpecialToken  *special_tokens,
    size_t               n_special)
{
    VocabResult vocab = vocab_load_file(vocab_path);
    if (!vocab.ok) return nullptr;

    Regex *pattern = regex_compile(regex_pattern);
    if (pattern == nullptr) {
        vocab_free(&vocab);
        return nullptr;
    }

    // Copy special tokens into a heap-allocated array.
    SpecialToken *special_copy = nullptr;
    if (n_special > 0 && special_tokens != nullptr) {
        special_copy = malloc(n_special * sizeof(SpecialToken));
        if (special_copy == nullptr) {
            regex_free(pattern);
            vocab_free(&vocab);
            return nullptr;
        }
        memcpy(special_copy, special_tokens, n_special * sizeof(SpecialToken));
    }

    return tiktoken_new("custom", vocab, pattern, special_copy, n_special);
}

// ── Standard special tokens for cl100k_base ───────────────────────────

static inline size_t tiktoken_cl100k_special(const SpecialToken **out) {
    static const SpecialToken tokens[] = {
        { "<|endoftext|>",   13, 100257 },
        { "<|fim_prefix|>",  14, 100258 },
        { "<|fim_middle|>",  14, 100259 },
        { "<|fim_suffix|>",  14, 100260 },
        { "<|endofprompt|>", 15, 100276 },
    };
    *out = tokens;
    return sizeof(tokens) / sizeof(tokens[0]);
}

#endif // TIKTOKEN_TIKTOKEN_H
