// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 3: Arena Allocator
//
// A bump allocator for bulk allocation of data that shares a lifetime.
// Used for vocabulary byte data: all 100K+ token byte sequences are
// allocated from a single arena and freed in one shot.

#ifndef TIKTOKEN_ARENA_H
#define TIKTOKEN_ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *base;
    size_t   used;
    size_t   cap;
} Arena;

// Create an arena with the given initial capacity.
[[nodiscard]]
Arena arena_new(size_t cap);

// Allocate `size` bytes from the arena, aligned to `align`.
// Returns nullptr on failure (out of space and realloc fails).
[[nodiscard]]
uint8_t *arena_alloc(Arena *a, size_t size, size_t align);

// Convenience: allocate `size` bytes with default alignment (8).
[[nodiscard]]
uint8_t *arena_push(Arena *a, size_t size);

// Copy `len` bytes from `src` into the arena and return a pointer.
[[nodiscard]]
uint8_t *arena_push_bytes(Arena *a, const uint8_t *src, size_t len);

// Reset the arena to empty without freeing the underlying buffer.
void arena_reset(Arena *a);

// Free the arena's buffer entirely.
void arena_free(Arena *a);

#endif // TIKTOKEN_ARENA_H
