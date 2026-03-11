// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 6: Vocabulary Loading
//
// Loads .tiktoken vocabulary files into a BpeRanks structure. Each line
// of a .tiktoken file contains a base64-encoded token and its rank:
//
//     IQ==  0
//     Ig==  1
//     ...
//
// This module combines base64 (Ch.1), bytes (Ch.2), arena (Ch.3), and
// hash map (Ch.3) to build the vocabulary.

#ifndef TIKTOKEN_VOCAB_H
#define TIKTOKEN_VOCAB_H

#include "tiktoken/arena.h"
#include "tiktoken/bpe.h"

#include <stddef.h>

// ── VocabResult: loaded vocabulary with its memory ────────────────────

typedef struct {
    BpeRanks  ranks;
    Arena     arena;    // owns all byte data for the token keys
    bool      ok;       // true if loading succeeded
} VocabResult;

// Load a .tiktoken vocabulary from a file path.
// Returns a VocabResult with ok=true on success.
// The arena owns all token byte data; free with vocab_free().
[[nodiscard]]
VocabResult vocab_load_file(const char *path);

// Load a .tiktoken vocabulary from an in-memory buffer.
// `data` points to the file contents; `data_len` is the length.
[[nodiscard]]
VocabResult vocab_load_mem(const char *data, size_t data_len);

// Free all resources associated with a loaded vocabulary.
void vocab_free(VocabResult *v);

#endif // TIKTOKEN_VOCAB_H
