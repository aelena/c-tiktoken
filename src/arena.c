// SPDX-License-Identifier: MIT
//
// c-tiktoken, Chapter 3: Arena Allocator (implementation)

#include "tiktoken/arena.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

// One block. The payload follows the header in the same allocation, so a block
// is one malloc and the data pointer needs no separate free.
struct ArenaBlock {
    ArenaBlock *next;
    size_t      cap;
    size_t      used;
    uint8_t     data[];   // C99 flexible array member, still the right tool
};

static constexpr size_t ARENA_DEFAULT_CAP = 4096;

// Align `offset` up to the next multiple of `align`, which must be a power of two.
static inline size_t align_up(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

static ArenaBlock *block_new(size_t cap) {
    if (cap == 0) cap = ARENA_DEFAULT_CAP;
    if (cap > SIZE_MAX - sizeof(ArenaBlock)) return nullptr;

    ArenaBlock *b = malloc(sizeof(ArenaBlock) + cap);
    if (b == nullptr) return nullptr;

    b->next = nullptr;
    b->cap  = cap;
    b->used = 0;
    return b;
}

Arena arena_new(size_t cap) {
    ArenaBlock *b = block_new(cap);
    if (b == nullptr) return (Arena){};
    return (Arena){ .blocks = b, .used = 0, .cap = b->cap, .total = 0 };
}

uint8_t *arena_alloc(Arena *a, size_t size, size_t align) {
    if (a == nullptr || a->blocks == nullptr) return nullptr;

    ArenaBlock *b = a->blocks;
    size_t aligned = align_up(b->used, align);

    // Overflow-safe check of `aligned + size <= b->cap`.
    if (aligned > b->cap || size > b->cap - aligned) {
        // The newest block cannot take it, so chain a new one in front. The old
        // block keeps its contents and every pointer into it stays valid, which
        // is the entire point of doing it this way rather than reallocating.
        size_t next_cap = b->cap;
        if (next_cap <= SIZE_MAX / 2) next_cap *= 2;
        if (next_cap < size) next_cap = size;

        ArenaBlock *nb = block_new(next_cap);
        if (nb == nullptr) return nullptr;

        nb->next  = a->blocks;
        a->blocks = nb;
        b = nb;
        aligned = 0;   // a fresh block is already aligned
    }

    uint8_t *ptr = b->data + aligned;
    b->used  = aligned + size;
    a->used  = b->used;
    a->cap   = b->cap;
    a->total += size;
    return ptr;
}

uint8_t *arena_push(Arena *a, size_t size) {
    return arena_alloc(a, size, 8);
}

uint8_t *arena_push_bytes(Arena *a, const uint8_t *src, size_t len) {
    uint8_t *dst = arena_alloc(a, len, 1);  // bytes need no alignment
    if (dst != nullptr && len > 0) {
        memcpy(dst, src, len);
    }
    return dst;
}

void arena_reset(Arena *a) {
    if (a == nullptr || a->blocks == nullptr) return;

    // Keep the newest block, which is also the largest, and drop the rest.
    ArenaBlock *keep = a->blocks;
    for (ArenaBlock *b = keep->next; b != nullptr; ) {
        ArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    keep->next = nullptr;
    keep->used = 0;

    a->blocks = keep;
    a->used   = 0;
    a->cap    = keep->cap;
    a->total  = 0;
}

void arena_free(Arena *a) {
    if (a == nullptr) return;
    for (ArenaBlock *b = a->blocks; b != nullptr; ) {
        ArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    *a = (Arena){};
}
