// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 3: Hash Map
//
// A Robin Hood open-addressing hash map, implemented with macros for
// type genericity. We need two concrete maps:
//   - Bytes → uint32_t  (token bytes to rank, for encoding)
//   - uint32_t → Bytes  (rank to token bytes, for decoding)
//
// Rather than duplicate the implementation, we use C23 `typeof` in
// macros to create type-safe maps with a single code path.

#ifndef TIKTOKEN_HASH_H
#define TIKTOKEN_HASH_H

#include "tiktoken/bytes.h"

#include <stddef.h>
#include <stdint.h>

// ── C23 feature: <stdbit.h> ────────────────────────────────────────────
//
// C23 adds <stdbit.h> with bit manipulation functions. We'd use
// stdc_bit_ceil(n) to round up to the next power of two. However,
// compiler support is still patchy, so we provide our own portable
// version and note the C23 equivalent in comments.

// ── Concrete map types ─────────────────────────────────────────────────
//
// We define concrete structs rather than using void* to maintain type
// safety. Each entry stores a key, a value, a cached hash, and a
// probe distance (PSL = Probe Sequence Length) for Robin Hood insertion.

// ── Map: Bytes → uint32_t (for encoding: token bytes → rank) ──────────

typedef struct {
    Bytes    key;
    uint32_t value;
    uint64_t hash;
    int32_t  psl;      // -1 = empty slot
} B2iEntry;

typedef struct {
    B2iEntry *slots;
    size_t    cap;     // always a power of two
    size_t    len;     // number of occupied entries
} B2iMap;

[[nodiscard]] B2iMap   b2i_new(size_t initial_cap);
void                   b2i_free(B2iMap *m);
[[nodiscard]] bool     b2i_insert(B2iMap *m, Bytes key, uint32_t value);
[[nodiscard]] bool     b2i_get(const B2iMap *m, Bytes key, uint32_t *out);
[[nodiscard]] size_t   b2i_len(const B2iMap *m);

// ── Map: uint32_t → Bytes (for decoding: rank → token bytes) ──────────

typedef struct {
    uint32_t key;
    Bytes    value;
    uint64_t hash;
    int32_t  psl;
} I2bEntry;

typedef struct {
    I2bEntry *slots;
    size_t    cap;
    size_t    len;
} I2bMap;

[[nodiscard]] I2bMap   i2b_new(size_t initial_cap);
void                   i2b_free(I2bMap *m);
[[nodiscard]] bool     i2b_insert(I2bMap *m, uint32_t key, Bytes value);
[[nodiscard]] bool     i2b_get(const I2bMap *m, uint32_t key, Bytes *out);
[[nodiscard]] size_t   i2b_len(const I2bMap *m);

#endif // TIKTOKEN_HASH_H
