// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 4: BPE Algorithm (implementation)
//
// The merge loop that is the heart of all BPE tokenizers.

#include "tiktoken/bpe.h"

#include <stdlib.h>
#include <string.h>

// ── C23 feature note: auto type inference ──────────────────────────────
//
// C23 allows `auto` for local variable declarations when the type can
// be inferred from the initializer:
//
//     auto x = 42;        // x is int
//     auto p = malloc(8); // p is void*
//
// We use it sparingly in this file for simple cases. Unlike C++ auto,
// C23 auto requires a concrete initializer and doesn't work with
// function return types.

// Sentinel rank value meaning "this pair has no merge rank."
static constexpr uint32_t NO_RANK = UINT32_MAX;

// ── Merge list ─────────────────────────────────────────────────────────
//
// During BPE, we maintain the input as a list of "parts" — byte ranges
// that may be merged with their neighbors. We use an indexed linked list:
// each part knows the index of the next part (-1 = end). This lets us
// remove parts in O(1) without shifting an array.
//
// Each part is a span [start, end) into the original input bytes.

typedef struct {
    size_t   start;
    size_t   end;
    int32_t  next;    // index of next part, or -1
    uint32_t rank;    // merge rank of (this part ++ next part), or NO_RANK
} Part;

// Look up the merge rank of the byte sequence input[start..end).
static uint32_t get_rank(const BpeRanks *ranks,
                         const uint8_t *data,
                         size_t start, size_t end)
{
    Bytes key = {
        .data = (uint8_t *)(data + start),
        .len  = end - start,
        .cap  = 0,  // non-owning view
    };
    uint32_t rank = 0;
    if (b2i_get(&ranks->encoder, key, &rank)) {
        return rank;
    }
    return NO_RANK;
}

// Compute the merge rank for part[i] merged with part[next(i)].
// This looks up the concatenation of the two parts' byte ranges.
static uint32_t compute_pair_rank(const BpeRanks *ranks,
                                  const uint8_t *data,
                                  const Part *parts,
                                  int32_t i)
{
    if (i < 0) return NO_RANK;
    int32_t j = parts[i].next;
    if (j < 0) return NO_RANK;
    // The merged span is [parts[i].start, parts[j].end)
    return get_rank(ranks, data, parts[i].start, parts[j].end);
}

TokenVec bpe_encode(const BpeRanks *ranks, const uint8_t *data, size_t len) {
    TokenVec result = tokvec_new();

    if (len == 0) {
        return result;
    }

    // Special case: single byte — just look it up.
    if (len == 1) {
        uint32_t rank = get_rank(ranks, data, 0, 1);
        if (rank != NO_RANK) {
            tokvec_push(&result, rank);
        }
        return result;
    }

    // ── Step 1: Initialize the part list ───────────────────────────
    //
    // Start with one part per byte. Each part is a single-byte range.
    // The `next` pointer chains them together.

    size_t n_parts = len;
    Part *parts = malloc(n_parts * sizeof(Part));
    if (parts == nullptr) {
        return result;
    }

    for (size_t i = 0; i < n_parts; i++) {
        parts[i].start = i;
        parts[i].end   = i + 1;
        parts[i].next  = (i + 1 < n_parts) ? (int32_t)(i + 1) : -1;
        parts[i].rank  = NO_RANK;
    }

    // Compute initial pair ranks for every adjacent pair.
    for (size_t i = 0; i < n_parts; i++) {
        parts[i].rank = compute_pair_rank(ranks, data, parts, (int32_t)i);
    }

    // ── Step 2: The merge loop ─────────────────────────────────────
    //
    // Repeatedly find the pair with the lowest (best) rank and merge it.
    //
    // This is the O(n²) naive approach: scan all parts each iteration
    // to find the minimum. For typical tiktoken inputs (words from regex
    // splitting, usually < 20 bytes), this is fast enough. The O(n log n)
    // priority-queue approach is discussed in the tutorial.

    for (;;) {
        // Find the part with the minimum merge rank.
        uint32_t min_rank = NO_RANK;
        int32_t  min_idx  = -1;

        int32_t idx = 0;  // start at the head of the linked list
        // Walk to find head (it's always index 0 initially, but after
        // merges the first valid part might still be 0).
        // Actually, after merges some parts are "deleted" (skipped via
        // the linked list), but index 0 is always the head unless it
        // was merged into a later part. Since we merge part[i] with
        // part[next(i)] by extending part[i], the head stays at 0.

        // Walk the linked list.
        for (int32_t i = 0; i >= 0; i = parts[i].next) {
            if (parts[i].rank < min_rank) {
                min_rank = parts[i].rank;
                min_idx  = i;
            }
        }

        // If no mergeable pair found, we're done.
        if (min_idx < 0) {
            break;
        }

        // ── Merge the winning pair ─────────────────────────────────
        //
        // Part[min_idx] absorbs part[next]:
        //   - Extend our byte range to cover both parts
        //   - Remove the next part from the linked list
        //   - Recompute ranks for the new neighbors

        int32_t next_idx = parts[min_idx].next;
        // Extend byte range.
        parts[min_idx].end = parts[next_idx].end;
        // Remove next_idx from the list.
        parts[min_idx].next = parts[next_idx].next;

        // Recompute this part's merge rank (with its new next neighbor).
        parts[min_idx].rank = compute_pair_rank(ranks, data, parts, min_idx);

        // Find the previous part (if any) and recompute its rank too,
        // because its next neighbor has changed (it's now merged).
        // We need to walk from the head to find the predecessor.
        // This is O(n) per merge — acceptable for small inputs.
        for (int32_t i = 0; i >= 0; i = parts[i].next) {
            if (parts[i].next == min_idx) {
                parts[i].rank = compute_pair_rank(ranks, data, parts, i);
                break;
            }
        }
    }

    // ── Step 3: Collect results ────────────────────────────────────
    //
    // Walk the remaining parts and look up each one's rank (token ID).

    for (int32_t i = 0; i >= 0; i = parts[i].next) {
        uint32_t rank = get_rank(ranks, data, parts[i].start, parts[i].end);
        if (rank != NO_RANK) {
            tokvec_push(&result, rank);
        }
    }

    free(parts);
    return result;
}

Bytes bpe_decode(const BpeRanks *ranks, const uint32_t *tokens, size_t n) {
    Bytes result = {};

    for (size_t i = 0; i < n; i++) {
        Bytes token_bytes = {};
        if (i2b_get(&ranks->decoder, tokens[i], &token_bytes)) {
            bytes_append(&result, token_bytes.data, token_bytes.len);
        }
    }

    return result;
}
