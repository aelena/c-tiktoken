// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 3: Arena Allocator (implementation)

#include "tiktoken/arena.h"

#include <stdlib.h>
#include <string.h>

Arena arena_new(size_t cap) {
    if (cap == 0) cap = 4096;
    uint8_t *base = malloc(cap);
    if (base == nullptr) {
        return (Arena){};
    }
    return (Arena){ .base = base, .used = 0, .cap = cap };
}

// Align `offset` up to the next multiple of `align`.
// `align` must be a power of two.
static inline size_t align_up(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

uint8_t *arena_alloc(Arena *a, size_t size, size_t align) {
    size_t aligned = align_up(a->used, align);
    size_t needed  = aligned + size;

    if (needed > a->cap) {
        // Grow: at least double, or enough for the request.
        size_t new_cap = a->cap * 2;
        if (new_cap < needed) new_cap = needed;
        uint8_t *new_base = realloc(a->base, new_cap);
        if (new_base == nullptr) {
            return nullptr;
        }
        a->base = new_base;
        a->cap  = new_cap;
    }

    uint8_t *ptr = a->base + aligned;
    a->used = aligned + size;
    return ptr;
}

uint8_t *arena_push(Arena *a, size_t size) {
    return arena_alloc(a, size, 8);
}

uint8_t *arena_push_bytes(Arena *a, const uint8_t *src, size_t len) {
    uint8_t *dst = arena_alloc(a, len, 1);  // bytes need no alignment
    if (dst != nullptr) {
        memcpy(dst, src, len);
    }
    return dst;
}

void arena_reset(Arena *a) {
    if (a != nullptr) {
        a->used = 0;
    }
}

void arena_free(Arena *a) {
    if (a != nullptr) {
        free(a->base);
        *a = (Arena){};
    }
}
