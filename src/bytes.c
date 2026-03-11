// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 2: Byte Strings (implementation)

#include "tiktoken/bytes.h"

#include <stdlib.h>
#include <string.h>

// ── C23 feature: typeof ────────────────────────────────────────────────
//
// C23 promotes `typeof` from a compiler extension to a standard keyword.
// It evaluates to the type of its operand — without evaluating the
// operand itself. This is invaluable for writing type-safe macros.
//
// We use it here in a growth helper macro so we don't repeat the
// growth logic for every dynamic array type.

// ── C23 feature: empty initializer = {} ────────────────────────────────
//
// C23 allows `= {}` to zero-initialize any aggregate or scalar.
// In C17 you needed `= {0}`, and for structs with a pointer first
// member, `{0}` was technically initializing the pointer with integer 0
// rather than a null pointer. `= {}` is unambiguous: it value-initializes
// every member.

// ── Growth policy ──────────────────────────────────────────────────────
//
// When a dynamic buffer needs more space, we double its capacity (with
// a minimum of 16). This is the standard amortized-O(1) approach:
// n appends cost O(n) total, even though individual appends occasionally
// trigger an O(n) realloc+copy.

// Compute the next capacity >= `needed`, starting from `current`.
static inline size_t grow_cap(size_t current, size_t needed) {
    size_t cap = current ? current : 16;
    while (cap < needed) {
        cap *= 2;
    }
    return cap;
}

// ── Bytes lifecycle ────────────────────────────────────────────────────

Bytes bytes_with_cap(size_t cap) {
    if (cap == 0) {
        return (Bytes){};     // C23 empty initializer
    }
    uint8_t *data = malloc(cap);
    if (data == nullptr) {
        return (Bytes){};
    }
    return (Bytes){ .data = data, .len = 0, .cap = cap };
}

Bytes bytes_from_raw(const uint8_t *src, size_t len) {
    if (len == 0) {
        return (Bytes){};
    }
    Bytes b = bytes_with_cap(len);
    if (b.data == nullptr) {
        return b;
    }
    memcpy(b.data, src, len);
    b.len = len;
    return b;
}

Bytes bytes_from_str(const char *s) {
    if (s == nullptr) {
        return (Bytes){};
    }
    return bytes_from_raw((const uint8_t *)s, strlen(s));
}

Bytes bytes_clone(Bytes b) {
    return bytes_from_raw(b.data, b.len);
}

void bytes_free(Bytes *b) {
    // Only free if we own the memory (cap > 0).
    // Slices have cap == 0 and must not be freed.
    if (b != nullptr && b->cap > 0) {
        free(b->data);
    }
    if (b != nullptr) {
        *b = (Bytes){};
    }
}

// ── Bytes mutation ─────────────────────────────────────────────────────

// Ensure capacity for at least `b->len + extra` bytes.
static bool bytes_ensure(Bytes *b, size_t extra) {
    size_t needed = b->len + extra;
    if (needed <= b->cap) {
        return true;
    }
    size_t new_cap = grow_cap(b->cap, needed);
    uint8_t *new_data = realloc(b->data, new_cap);
    if (new_data == nullptr) {
        return false;
    }
    b->data = new_data;
    b->cap  = new_cap;
    return true;
}

bool bytes_append(Bytes *b, const uint8_t *src, size_t len) {
    if (len == 0) {
        return true;
    }
    if (!bytes_ensure(b, len)) {
        return false;
    }
    memcpy(b->data + b->len, src, len);
    b->len += len;
    return true;
}

bool bytes_append_bytes(Bytes *dst, Bytes src) {
    return bytes_append(dst, src.data, src.len);
}

bool bytes_push(Bytes *b, uint8_t byte) {
    return bytes_append(b, &byte, 1);
}

void bytes_clear(Bytes *b) {
    if (b != nullptr) {
        b->len = 0;
    }
}

// ── Bytes observation ──────────────────────────────────────────────────

Bytes bytes_slice(Bytes b, size_t start, size_t end) {
    // Clamp to valid range.
    if (start > b.len) start = b.len;
    if (end   > b.len) end   = b.len;
    if (start >= end) {
        return (Bytes){};
    }
    // Return a non-owning view: cap == 0 signals "don't free me".
    return (Bytes){
        .data = b.data + start,
        .len  = end - start,
        .cap  = 0,
    };
}

bool bytes_equal(Bytes a, Bytes b) {
    if (a.len != b.len) {
        return false;
    }
    if (a.len == 0) {
        return true;
    }
    return memcmp(a.data, b.data, a.len) == 0;
}

// ── C23 feature: [[maybe_unused]] ──────────────────────────────────────
//
// The [[maybe_unused]] attribute suppresses warnings for deliberately
// unused variables or parameters. Here we document the FNV constants
// with names but only use their values.

// FNV-1a hash — simple, fast, and well-distributed for short keys.
//
// FNV-1a works by:
//   1. Starting with an offset basis (a magic constant)
//   2. For each byte: XOR the byte into the hash, then multiply by a prime
//
// The XOR-then-multiply order (FNV-1a) gives better distribution than
// multiply-then-XOR (FNV-1). The constants are chosen to minimize
// collisions over typical inputs.

[[maybe_unused]]
static constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;

[[maybe_unused]]
static constexpr uint64_t FNV_PRIME        = 0x00000100000001B3ULL;

uint64_t bytes_hash(Bytes b) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < b.len; i++) {
        hash ^= (uint64_t)b.data[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

// ── ByteVec ────────────────────────────────────────────────────────────

ByteVec bytevec_new(void) {
    return (ByteVec){};
}

bool bytevec_push(ByteVec *v, Bytes b) {
    if (v->len >= v->cap) {
        size_t new_cap = grow_cap(v->cap, v->len + 1);
        Bytes *new_items = realloc(v->items, new_cap * sizeof(Bytes));
        if (new_items == nullptr) {
            return false;
        }
        v->items = new_items;
        v->cap   = new_cap;
    }
    v->items[v->len++] = b;
    return true;
}

void bytevec_free(ByteVec *v) {
    if (v == nullptr) return;
    for (size_t i = 0; i < v->len; i++) {
        bytes_free(&v->items[i]);
    }
    free(v->items);
    *v = (ByteVec){};
}

// ── TokenVec ───────────────────────────────────────────────────────────

TokenVec tokvec_new(void) {
    return (TokenVec){};
}

bool tokvec_push(TokenVec *v, uint32_t token) {
    if (v->len >= v->cap) {
        size_t new_cap = grow_cap(v->cap, v->len + 1);
        uint32_t *new_items = realloc(v->items, new_cap * sizeof(uint32_t));
        if (new_items == nullptr) {
            return false;
        }
        v->items = new_items;
        v->cap   = new_cap;
    }
    v->items[v->len++] = token;
    return true;
}

bool tokvec_extend(TokenVec *v, const uint32_t *src, size_t n) {
    if (n == 0) return true;
    size_t needed = v->len + n;
    if (needed > v->cap) {
        size_t new_cap = grow_cap(v->cap, needed);
        uint32_t *new_items = realloc(v->items, new_cap * sizeof(uint32_t));
        if (new_items == nullptr) {
            return false;
        }
        v->items = new_items;
        v->cap   = new_cap;
    }
    memcpy(v->items + v->len, src, n * sizeof(uint32_t));
    v->len += n;
    return true;
}

void tokvec_free(TokenVec *v) {
    if (v == nullptr) return;
    free(v->items);
    *v = (TokenVec){};
}
