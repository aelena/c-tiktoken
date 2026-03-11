# Chapter 3 — Hash Map and Arena Allocator

## The Storage Problem

We've built base64 decoding (Chapter 1) and byte strings (Chapter 2). Now
we need a way to store and look up 100,000+ token entries efficiently.

The tiktoken vocabulary defines a bidirectional mapping:

```
token bytes  ↔  rank (integer ID)
```

**Encoding** (text → tokens) needs: `Bytes → uint32_t`
**Decoding** (tokens → text) needs: `uint32_t → Bytes`

A linear scan through 100K entries for every lookup would be catastrophically
slow. We need a hash map.

## Part 1: The Arena Allocator

Before building the hash map, let's solve a simpler problem: memory
management for vocabulary data.

When we load a vocabulary file, we create 100K+ `Bytes` values — one for
each token. If each one calls `malloc` individually, we end up with:
- 100K+ tiny allocations (fragmentation)
- 100K+ individual `free` calls on cleanup
- Allocator overhead per block (typically 16–32 bytes of metadata)

An **arena allocator** solves all three problems:

```c
typedef struct {
    uint8_t *base;    // one big contiguous block
    size_t   used;    // high-water mark (bump pointer)
    size_t   cap;     // total capacity
} Arena;
```

Allocation is a **bump**: advance the `used` pointer by the requested size.
Deallocation is **all-or-nothing**: free the entire block at once.

### How It Works

```c
uint8_t *arena_alloc(Arena *a, size_t size, size_t align) {
    size_t aligned = align_up(a->used, align);
    size_t needed  = aligned + size;

    if (needed > a->cap) {
        // Grow by doubling
        size_t new_cap = a->cap * 2;
        if (new_cap < needed) new_cap = needed;
        a->base = realloc(a->base, new_cap);
        a->cap  = new_cap;
    }

    uint8_t *ptr = a->base + aligned;
    a->used = aligned + size;
    return ptr;
}
```

**Alignment** matters because some architectures fault on unaligned
accesses, and even x86 is slower when data crosses cache line boundaries.
The `align_up` function rounds the offset to the next multiple of the
alignment:

```c
static inline size_t align_up(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}
```

This works because `align` is always a power of two, so `~(align - 1)`
creates a bitmask that zeros out the low bits.

### When to Use Arenas

Arenas are ideal when you have a group of allocations that share a
lifetime — they're all created together and destroyed together. Our
vocabulary is a perfect example: all token byte data is loaded at
startup and freed when the encoding is destroyed.

Arenas are **not** suitable when individual items need to be freed
independently. For that, you need a general-purpose allocator.

## Part 2: The Hash Map

### Why Robin Hood Hashing?

There are many hash map strategies. We use **Robin Hood open addressing**:

- **Open addressing** means entries are stored directly in the table
  array (no separate linked lists). This is cache-friendly because
  lookups access contiguous memory.

- **Robin Hood** is a refinement where entries can be displaced during
  insertion to balance probe lengths. Named after the principle of
  "taking from the rich to give to the poor."

The key metric is **PSL** (Probe Sequence Length) — how far an entry
is from its ideal position. Robin Hood hashing minimizes the *variance*
of PSL across all entries, making worst-case lookups much better than
standard linear probing.

### The Insertion Algorithm

```
1. Compute hash and ideal slot index
2. Walk forward from ideal slot:
   a. If slot is empty → place entry, done
   b. If slot's PSL < our PSL → swap (Robin Hood!), continue with displaced entry
   c. Otherwise → increment PSL, advance to next slot
```

In pseudocode:

```c
void insert(Entry *slots, size_t cap, Entry entry) {
    size_t idx = entry.hash & (cap - 1);    // power-of-two modulo
    for (;;) {
        if (slots[idx] is empty) {
            slots[idx] = entry;
            return;
        }
        if (slots[idx].psl < entry.psl) {
            swap(slots[idx], entry);        // Robin Hood swap
        }
        entry.psl++;
        idx = (idx + 1) & (cap - 1);       // wrap around
    }
}
```

### The Lookup Algorithm

Robin Hood hashing gives us an early termination condition for lookups:

```c
bool get(Entry *slots, size_t cap, Key key, Value *out) {
    uint64_t h   = hash(key);
    size_t   idx = h & (cap - 1);

    for (int32_t psl = 0; slots[idx].psl >= psl; psl++) {
        if (slots[idx].hash == h && keys_equal(slots[idx].key, key)) {
            *out = slots[idx].value;
            return true;
        }
        idx = (idx + 1) & (cap - 1);
    }
    return false;   // key not found
}
```

The crucial line is `slots[idx].psl >= psl`. If the entry at position
`idx` has a shorter PSL than what ours would be, then our key can't be
further ahead — we would have displaced this entry during insertion.
This means failed lookups terminate quickly instead of scanning the
entire cluster.

### Load Factor and Growth

We grow the table when occupancy exceeds 70%:

```c
static constexpr double MAX_LOAD = 0.70;
```

At 70% load, Robin Hood hashing gives:
- Average successful probe: ~1.37
- Average failed probe: ~2.37

These are excellent numbers — almost as good as a perfect hash function.
At 90% load, probes would increase to ~3.0 and ~10.0, which is why we
keep the threshold moderate.

Growth means allocating a new array of double the size and re-inserting
all entries. This is O(n) but happens infrequently — amortized O(1)
per insertion, the same analysis as dynamic arrays.

### Power-of-Two Sizing

We always keep the table capacity as a power of two. This lets us
replace the expensive modulo operator (`%`) with a cheap bitmask (`&`):

```c
size_t idx = hash & (cap - 1);    // equivalent to hash % cap, but faster
```

This works because `cap - 1` for a power of two has all lower bits set:

```
cap     = 0b10000000  (128)
cap - 1 = 0b01111111  (127)
hash & (cap - 1)  →  extracts lower 7 bits  →  range [0, 127]
```

## C23 Feature: `<stdbit.h>` — Bit Manipulation Functions

C23 adds `<stdbit.h>` with standardized bit manipulation functions.
The one we'd use here is `stdc_bit_ceil`:

```c
// C23 way:
#include <stdbit.h>
size_t cap = stdc_bit_ceil(requested);  // round up to next power of 2

// Our portable way (until compiler support improves):
static size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}
```

The `next_pow2` technique works by "smearing" the highest set bit
downward through all lower positions, then adding 1. After smearing,
the value has all bits set from the highest original bit down to bit 0,
so adding 1 produces the next power of two.

`<stdbit.h>` also provides:
- `stdc_leading_zeros(x)` — count leading zeros
- `stdc_trailing_zeros(x)` — count trailing zeros
- `stdc_popcount(x)` — count set bits
- `stdc_has_single_bit(x)` — check if x is a power of two
- `stdc_bit_floor(x)` — round down to previous power of two

These replace a patchwork of compiler builtins (`__builtin_clz`,
`__builtin_popcount`) and platform-specific intrinsics with a single
standard header.

## C23 Feature: `constexpr` for Hash Constants

We use `constexpr` for the load factor and hash constants:

```c
static constexpr double MAX_LOAD = 0.70;
```

Note that C23 `constexpr` supports floating-point constants too (C23
allows `constexpr` for all scalar types and arrays thereof). This is
a compile-time constant — the compiler can inline it everywhere and
there's no runtime initialization.

## Two Concrete Maps Instead of One Generic

We implement `B2iMap` (Bytes→uint32_t) and `I2bMap` (uint32_t→Bytes) as
separate concrete types rather than one generic map. Why?

1. **Different hash functions.** Bytes keys use FNV-1a (byte-by-byte).
   Integer keys use a fast integer mixer. A generic map would need a
   function pointer for hashing, adding indirection in the hot path.

2. **Different equality checks.** Bytes equality requires `memcmp`.
   Integer equality is just `==`. Again, a function pointer would
   add overhead.

3. **Type safety.** With concrete types, the compiler catches type
   errors at compile time. With `void *` generics, errors become
   runtime bugs.

4. **Debuggability.** When you inspect a `B2iMap` in a debugger, you
   see `Bytes` keys and `uint32_t` values. A `void *` map shows raw
   memory.

The tradeoff is code duplication — the two maps are structurally
identical. In Chapter 4's tutorial commentary we'll discuss how C23's
`typeof` could be used to macro-generate both from a single template,
but for teaching purposes, the explicit duplication is clearer.

## Entry Structure

Each map entry stores four fields:

```c
typedef struct {
    Bytes    key;       // the lookup key
    uint32_t value;     // the stored value
    uint64_t hash;      // cached hash (avoids recomputing during growth)
    int32_t  psl;       // probe sequence length (-1 = empty slot)
} B2iEntry;
```

**Cached hash** is critical for performance during table growth. When we
double the table and re-insert all entries, we need each entry's hash to
compute its new position. Without caching, we'd re-hash every key — for
`Bytes` keys, that means re-scanning every byte of every token.

**PSL as empty marker:** we use `psl == -1` to mark empty slots rather
than a separate boolean. This saves 7 bytes of padding per entry (the
`int32_t` psl needs to exist anyway).

## Building and Running

Updated `CMakeLists.txt` additions:

```cmake
add_library(tiktoken
    src/base64.c
    src/bytes.c
    src/arena.c        # NEW
    src/hash.c         # NEW
)

add_executable(test_hash tests/test_hash.c)
target_link_libraries(test_hash PRIVATE tiktoken)
add_test(NAME hash COMMAND test_hash)
```

## What's Next

We now have the infrastructure to store a vocabulary: an arena for
bulk byte allocation and hash maps for O(1) lookup in both directions.

In [Chapter 4](chapter04_bpe.md), we'll implement the core BPE
(Byte Pair Encoding) merge algorithm — the heart of the tokenizer.
Given a byte sequence and a vocabulary of ranked merges, BPE
repeatedly merges the highest-priority adjacent pair until no more
merges are possible.

## Summary of C23 Features Introduced

| Feature | What It Replaces | Why It's Better |
|---------|-----------------|-----------------|
| `<stdbit.h>` | `__builtin_clz`, `_BitScanReverse` | Standard, portable bit ops |
| `stdc_bit_ceil` | Hand-rolled `next_pow2` | Single function call, well-tested |
| `constexpr` (float) | `static const double` | Compile-time guarantee extends to floats |
