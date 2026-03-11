// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 2: Byte Strings
//
// A length-prefixed dynamic byte buffer. Tokens in tiktoken are arbitrary
// byte sequences — not null-terminated C strings. This type is the
// fundamental data unit throughout the project.

#ifndef TIKTOKEN_BYTES_H
#define TIKTOKEN_BYTES_H

#include <stddef.h>
#include <stdint.h>

// ── Bytes: a fat pointer to a byte sequence ────────────────────────────
//
// Unlike C strings, Bytes:
//   - knows its own length (no strlen needed)
//   - can contain null bytes (0x00)
//   - is not null-terminated
//   - tracks capacity separately from length for dynamic growth

typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} Bytes;

// ── Lifecycle ──────────────────────────────────────────────────────────

// Create an empty Bytes with the given initial capacity.
// Returns a zero-initialized Bytes on allocation failure.
[[nodiscard]]
Bytes bytes_with_cap(size_t cap);

// Create a Bytes by copying `len` bytes from `src`.
[[nodiscard]]
Bytes bytes_from_raw(const uint8_t *src, size_t len);

// Create a Bytes from a null-terminated C string (excluding the '\0').
[[nodiscard]]
Bytes bytes_from_str(const char *s);

// Deep-copy a Bytes.
[[nodiscard]]
Bytes bytes_clone(Bytes b);

// Release memory. Safe to call on a zero-initialized Bytes.
void bytes_free(Bytes *b);

// ── Mutation ───────────────────────────────────────────────────────────

// Append `len` bytes from `src` to the end. Grows capacity as needed.
// Returns false on allocation failure.
[[nodiscard]]
bool bytes_append(Bytes *b, const uint8_t *src, size_t len);

// Append another Bytes to the end.
[[nodiscard]]
bool bytes_append_bytes(Bytes *dst, Bytes src);

// Append a single byte.
[[nodiscard]]
bool bytes_push(Bytes *b, uint8_t byte);

// Reset length to 0 without freeing memory (reuse the buffer).
void bytes_clear(Bytes *b);

// ── Observation ────────────────────────────────────────────────────────

// Non-owning view into a subrange of a Bytes. The returned Bytes has
// cap == 0 to signal it does not own its memory.
[[nodiscard]]
Bytes bytes_slice(Bytes b, size_t start, size_t end);

// Compare two byte sequences for equality.
[[nodiscard]]
bool bytes_equal(Bytes a, Bytes b);

// FNV-1a hash of the byte content. Deterministic and fast.
[[nodiscard]]
uint64_t bytes_hash(Bytes b);

// ── ByteVec: a dynamic array of Bytes ──────────────────────────────────
//
// We'll need this for collecting regex match results, BPE parts, etc.

typedef struct {
    Bytes  *items;
    size_t  len;
    size_t  cap;
} ByteVec;

// Create an empty ByteVec.
[[nodiscard]]
ByteVec bytevec_new(void);

// Append a Bytes (transfers ownership — the vec now owns the data).
[[nodiscard]]
bool bytevec_push(ByteVec *v, Bytes b);

// Free the vec and all Bytes it contains.
void bytevec_free(ByteVec *v);

// ── TokenVec: a dynamic array of uint32_t token IDs ────────────────────

typedef struct {
    uint32_t *items;
    size_t    len;
    size_t    cap;
} TokenVec;

[[nodiscard]]
TokenVec tokvec_new(void);

[[nodiscard]]
bool tokvec_push(TokenVec *v, uint32_t token);

// Append `n` token IDs from `src`.
[[nodiscard]]
bool tokvec_extend(TokenVec *v, const uint32_t *src, size_t n);

void tokvec_free(TokenVec *v);

#endif // TIKTOKEN_BYTES_H
