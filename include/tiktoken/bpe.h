// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 4: BPE (Byte Pair Encoding) Algorithm
//
// The core merge algorithm that converts a byte sequence into a sequence
// of token IDs. Given a vocabulary of ranked byte pairs, BPE repeatedly
// merges the highest-priority (lowest-rank) adjacent pair until no more
// merges are possible.

#ifndef TIKTOKEN_BPE_H
#define TIKTOKEN_BPE_H

#include "tiktoken/bytes.h"
#include "tiktoken/hash.h"

#include <stddef.h>
#include <stdint.h>

// ── BpeRanks: the vocabulary ──────────────────────────────────────────
//
// Wraps the two hash maps needed for encode/decode plus the arena that
// owns all the byte data.

typedef struct {
    B2iMap  encoder;       // Bytes → rank (uint32_t)
    I2bMap  decoder;       // rank (uint32_t) → Bytes
    size_t  vocab_size;
} BpeRanks;

// ── Core BPE encode ────────────────────────────────────────────────────
//
// Takes a byte sequence (typically one word/chunk from regex splitting)
// and returns its token IDs by applying the BPE merge algorithm.
//
// The `ranks` map must contain all valid token byte sequences and their
// ranks. Single-byte tokens (0x00–0xFF) should be present with their
// ranks for the algorithm to work correctly.

[[nodiscard]]
TokenVec bpe_encode(const BpeRanks *ranks, const uint8_t *data, size_t len);

// ── BPE decode ─────────────────────────────────────────────────────────
//
// Converts a sequence of token IDs back into raw bytes by concatenating
// the byte sequence for each token.

[[nodiscard]]
Bytes bpe_decode(const BpeRanks *ranks, const uint32_t *tokens, size_t n);

#endif // TIKTOKEN_BPE_H
