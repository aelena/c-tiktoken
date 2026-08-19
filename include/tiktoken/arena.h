// SPDX-License-Identifier: MIT
//
// c-tiktoken, Chapter 3: Arena Allocator
//
// A bump allocator for data that shares a lifetime. Used for vocabulary byte
// data: every token's bytes are allocated here and released in one shot when the
// vocabulary is freed.
//
// ── The property that matters ─────────────────────────────────────────
//
// A pointer returned by this arena stays valid until arena_reset or arena_free.
// Nothing else invalidates it, and in particular a later allocation does not.
//
// That is the whole reason the hash maps can borrow their key bytes from here
// instead of copying them, and it is not a detail. An earlier version of this
// arena grew a single block with realloc, which moved every allocation already
// handed out and left both maps holding dangling pointers. It survived because
// vocab_load_mem pre-sizes the arena at roughly twenty bytes per token and the
// real cl100k_base vocabulary fits, so the growth path almost never ran. A
// fuzzer found it in under a minute on a file with longer tokens.
//
// So the arena is a list of blocks. When the newest block cannot satisfy a
// request, a new block is chained in front and the old ones stay exactly where
// they are. Growth never moves anything.

#ifndef TIKTOKEN_ARENA_H
#define TIKTOKEN_ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct ArenaBlock ArenaBlock;

typedef struct {
    ArenaBlock *blocks;   // newest first; nullptr means creation failed
    size_t      used;     // bytes used in the newest block
    size_t      cap;      // capacity of the newest block
    size_t      total;    // bytes handed out across every block
} Arena;

// Create an arena whose first block has the given capacity.
// On failure, .blocks is nullptr.
[[nodiscard]]
Arena arena_new(size_t cap);

// Allocate `size` bytes, aligned to `align`, which must be a power of two.
// Returns nullptr only if a new block could not be allocated.
[[nodiscard]]
uint8_t *arena_alloc(Arena *a, size_t size, size_t align);

// Convenience: `size` bytes with default alignment (8).
[[nodiscard]]
uint8_t *arena_push(Arena *a, size_t size);

// Copy `len` bytes from `src` into the arena and return a pointer to the copy.
[[nodiscard]]
uint8_t *arena_push_bytes(Arena *a, const uint8_t *src, size_t len);

// Drop every allocation, keeping one block to reuse. Invalidates every pointer
// the arena has returned.
void arena_reset(Arena *a);

// Release every block. Invalidates every pointer the arena has returned.
void arena_free(Arena *a);

#endif // TIKTOKEN_ARENA_H
