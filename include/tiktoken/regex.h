// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 5: Regex Pre-tokenization
//
// Wraps PCRE2 to split input text into chunks before BPE encoding.
// tiktoken uses a complex regex pattern with Unicode properties (\p{L},
// \p{N}) to split text into words, numbers, punctuation, and whitespace.

#ifndef TIKTOKEN_REGEX_H
#define TIKTOKEN_REGEX_H

#include "tiktoken/bytes.h"

#include <stddef.h>
#include <stdint.h>

// Opaque regex handle. Hides the PCRE2 dependency from callers.
typedef struct Regex Regex;

// ── Regex lifecycle ────────────────────────────────────────────────────

// Compile a PCRE2 pattern. Returns nullptr on failure.
// The pattern is expected to be a UTF-8 string with Unicode property
// support (compiled with PCRE2_UTF | PCRE2_UCP).
[[nodiscard]]
Regex *regex_compile(const char *pattern);

// Free a compiled regex.
void regex_free(Regex *re);

// ── Matching ───────────────────────────────────────────────────────────

// A non-owning slice into the original input string.
typedef struct {
    size_t start;
    size_t len;
} RegexMatch;

typedef struct {
    RegexMatch *items;
    size_t      len;
    size_t      cap;
} RegexMatchVec;

// Find all non-overlapping matches of the regex in the input text.
// Returns a vector of (start, len) pairs into the original input.
// The caller must free the returned vector with regexmatchvec_free().
[[nodiscard]]
RegexMatchVec regex_find_all(const Regex *re,
                             const char  *text,
                             size_t       text_len);

// Free a match vector.
void regexmatchvec_free(RegexMatchVec *v);

// ── Predefined patterns ───────────────────────────────────────────────
//
// tiktoken uses different regex patterns for different encodings.

// The cl100k_base pattern (used by GPT-4, GPT-3.5-turbo).
[[nodiscard]]
const char *tiktoken_pattern_cl100k(void);

// The o200k_base pattern (used by GPT-4o).
[[nodiscard]]
const char *tiktoken_pattern_o200k(void);

// The p50k_base pattern (used by GPT-3 codex models).
[[nodiscard]]
const char *tiktoken_pattern_p50k(void);

#endif // TIKTOKEN_REGEX_H
