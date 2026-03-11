// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 3: Hash Map (implementation)
//
// Robin Hood open-addressing hash map with two concrete instantiations:
//   B2iMap: Bytes → uint32_t  (encode direction)
//   I2bMap: uint32_t → Bytes  (decode direction)

#include "tiktoken/hash.h"

#include <stdlib.h>
#include <string.h>

// ── C23 feature note: <stdbit.h> ───────────────────────────────────────
//
// C23 provides stdc_bit_ceil(n) in <stdbit.h> which rounds up to the
// next power of two. We implement our own for portability since not all
// compilers support it yet.
//
//   C23 way:    #include <stdbit.h>
//               size_t cap = stdc_bit_ceil(requested);
//
//   Our way:    size_t cap = next_pow2(requested);

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

// ── Load factor threshold ──────────────────────────────────────────────
//
// We grow the table when it exceeds 70% occupancy. This is a good
// balance between memory usage and probe length for Robin Hood hashing.
// At 70%, the average successful probe length is ~1.37.

static constexpr double MAX_LOAD = 0.70;

// ── Hash functions ─────────────────────────────────────────────────────

// Hash for Bytes keys — reuses the FNV-1a from Chapter 2.
// (We call bytes_hash directly.)

// Hash for uint32_t keys — spread bits with a mix function.
static inline uint64_t hash_u32(uint32_t key) {
    uint64_t h = (uint64_t)key;
    h = (h ^ (h >> 16)) * 0x45d9f3b;
    h = (h ^ (h >> 16)) * 0x45d9f3b;
    h = h ^ (h >> 16);
    // Ensure non-zero (we use 0 as a sentinel in some paths).
    return h | 1;
}

// ── Slot index from hash ──────────────────────────────────────────────
//
// Since capacity is always a power of two, modulo is just a bitmask.
// This is faster than the % operator and is why we enforce power-of-two
// sizing.

static inline size_t slot_index(uint64_t hash, size_t cap) {
    return (size_t)(hash & (uint64_t)(cap - 1));
}

// ════════════════════════════════════════════════════════════════════════
// B2iMap: Bytes → uint32_t
// ════════════════════════════════════════════════════════════════════════

B2iMap b2i_new(size_t initial_cap) {
    size_t cap = next_pow2(initial_cap < 16 ? 16 : initial_cap);
    B2iEntry *slots = calloc(cap, sizeof(B2iEntry));
    if (slots == nullptr) {
        return (B2iMap){};
    }
    // Mark all slots as empty.
    for (size_t i = 0; i < cap; i++) {
        slots[i].psl = -1;
    }
    return (B2iMap){ .slots = slots, .cap = cap, .len = 0 };
}

void b2i_free(B2iMap *m) {
    if (m == nullptr) return;
    // Note: we don't free the Bytes keys here because in our design
    // the arena owns the byte data. If keys were independently allocated
    // you'd free them here.
    free(m->slots);
    *m = (B2iMap){};
}

// Internal: insert into a slot array (used by both insert and grow).
static void b2i_insert_entry(B2iEntry *slots, size_t cap, B2iEntry entry) {
    size_t idx = slot_index(entry.hash, cap);

    // ── Robin Hood insertion ───────────────────────────────────────
    //
    // Walk forward from the ideal slot. At each occupied slot, compare
    // PSL (Probe Sequence Length). If the existing entry has a *shorter*
    // PSL than ours, it's "richer" — swap it out and continue inserting
    // the displaced entry. This keeps probe lengths balanced.
    //
    // The key insight: Robin Hood hashing reduces the variance of probe
    // lengths. While the average is the same as linear probing, the
    // worst case is dramatically better. This makes lookups more
    // predictable and cache-friendly.

    for (;;) {
        if (slots[idx].psl < 0) {
            // Empty slot — place the entry here.
            slots[idx] = entry;
            return;
        }
        if (slots[idx].psl < entry.psl) {
            // Robin Hood: steal from the rich. The existing entry has
            // a shorter probe distance, so it's closer to its ideal
            // position than we are to ours. Swap and continue with
            // the displaced entry.
            B2iEntry tmp = slots[idx];
            slots[idx] = entry;
            entry = tmp;
        }
        entry.psl++;
        idx = (idx + 1) & (cap - 1);
    }
}

// Grow the table by doubling capacity and re-inserting all entries.
static bool b2i_grow(B2iMap *m) {
    size_t new_cap;
    if (m->cap > SIZE_MAX / 2) {
        // Can't double without overflow
        return false;
    }
    new_cap = m->cap * 2;
    B2iEntry *new_slots = calloc(new_cap, sizeof(B2iEntry));
    if (new_slots == nullptr) {
        return false;
    }
    for (size_t i = 0; i < new_cap; i++) {
        new_slots[i].psl = -1;
    }

    // Re-insert all existing entries with reset PSL values.
    for (size_t i = 0; i < m->cap; i++) {
        if (m->slots[i].psl >= 0) {
            B2iEntry e = m->slots[i];
            e.psl = 0;
            b2i_insert_entry(new_slots, new_cap, e);
        }
    }

    free(m->slots);
    m->slots = new_slots;
    m->cap   = new_cap;
    return true;
}

bool b2i_insert(B2iMap *m, Bytes key, uint32_t value) {
    // Check load factor.
    if ((double)(m->len + 1) > (double)m->cap * MAX_LOAD) {
        if (!b2i_grow(m)) {
            return false;
        }
    }

    uint64_t h = bytes_hash(key);
    // Check for duplicate key — update in place if found.
    size_t idx = slot_index(h, m->cap);
    for (int32_t psl = 0; m->slots[idx].psl >= 0; psl++) {
        if (m->slots[idx].hash == h && bytes_equal(m->slots[idx].key, key)) {
            m->slots[idx].value = value;
            return true;
        }
        idx = (idx + 1) & (m->cap - 1);
    }

    B2iEntry entry = {
        .key   = key,
        .value = value,
        .hash  = h,
        .psl   = 0,
    };
    b2i_insert_entry(m->slots, m->cap, entry);
    m->len++;
    return true;
}

bool b2i_get(const B2iMap *m, Bytes key, uint32_t *out) {
    if (m->len == 0) return false;

    uint64_t h   = bytes_hash(key);
    size_t   idx = slot_index(h, m->cap);

    // Walk forward. Thanks to Robin Hood, we can stop early: if the
    // current slot's PSL is less than what ours would be at this
    // position, the key cannot be further ahead.
    for (int32_t psl = 0; m->slots[idx].psl >= psl; psl++) {
        if (m->slots[idx].hash == h && bytes_equal(m->slots[idx].key, key)) {
            if (out != nullptr) *out = m->slots[idx].value;
            return true;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    return false;
}

size_t b2i_len(const B2iMap *m) {
    return m->len;
}

// ════════════════════════════════════════════════════════════════════════
// I2bMap: uint32_t → Bytes
// ════════════════════════════════════════════════════════════════════════
//
// This is structurally identical to B2iMap but with swapped key/value
// types and a different hash function. In a more macro-heavy codebase
// we'd generate both from a template. For clarity, we spell it out.

I2bMap i2b_new(size_t initial_cap) {
    size_t cap = next_pow2(initial_cap < 16 ? 16 : initial_cap);
    I2bEntry *slots = calloc(cap, sizeof(I2bEntry));
    if (slots == nullptr) {
        return (I2bMap){};
    }
    for (size_t i = 0; i < cap; i++) {
        slots[i].psl = -1;
    }
    return (I2bMap){ .slots = slots, .cap = cap, .len = 0 };
}

void i2b_free(I2bMap *m) {
    if (m == nullptr) return;
    free(m->slots);
    *m = (I2bMap){};
}

static void i2b_insert_entry(I2bEntry *slots, size_t cap, I2bEntry entry) {
    size_t idx = slot_index(entry.hash, cap);
    for (;;) {
        if (slots[idx].psl < 0) {
            slots[idx] = entry;
            return;
        }
        if (slots[idx].psl < entry.psl) {
            I2bEntry tmp = slots[idx];
            slots[idx] = entry;
            entry = tmp;
        }
        entry.psl++;
        idx = (idx + 1) & (cap - 1);
    }
}

static bool i2b_grow(I2bMap *m) {
    size_t new_cap;
    if (m->cap > SIZE_MAX / 2) {
        // Can't double without overflow
        return false;
    }
    new_cap = m->cap * 2;
    I2bEntry *new_slots = calloc(new_cap, sizeof(I2bEntry));
    if (new_slots == nullptr) {
        return false;
    }
    for (size_t i = 0; i < new_cap; i++) {
        new_slots[i].psl = -1;
    }
    for (size_t i = 0; i < m->cap; i++) {
        if (m->slots[i].psl >= 0) {
            I2bEntry e = m->slots[i];
            e.psl = 0;
            i2b_insert_entry(new_slots, new_cap, e);
        }
    }
    free(m->slots);
    m->slots = new_slots;
    m->cap   = new_cap;
    return true;
}

bool i2b_insert(I2bMap *m, uint32_t key, Bytes value) {
    if ((double)(m->len + 1) > (double)m->cap * MAX_LOAD) {
        if (!i2b_grow(m)) {
            return false;
        }
    }

    uint64_t h = hash_u32(key);
    size_t idx = slot_index(h, m->cap);
    for (int32_t psl = 0; m->slots[idx].psl >= 0; psl++) {
        if (m->slots[idx].hash == h && m->slots[idx].key == key) {
            m->slots[idx].value = value;
            return true;
        }
        idx = (idx + 1) & (m->cap - 1);
        (void)psl;
    }

    I2bEntry entry = {
        .key   = key,
        .value = value,
        .hash  = h,
        .psl   = 0,
    };
    i2b_insert_entry(m->slots, m->cap, entry);
    m->len++;
    return true;
}

bool i2b_get(const I2bMap *m, uint32_t key, Bytes *out) {
    if (m->len == 0) return false;

    uint64_t h   = hash_u32(key);
    size_t   idx = slot_index(h, m->cap);

    for (int32_t psl = 0; m->slots[idx].psl >= psl; psl++) {
        if (m->slots[idx].hash == h && m->slots[idx].key == key) {
            if (out != nullptr) *out = m->slots[idx].value;
            return true;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    return false;
}

size_t i2b_len(const I2bMap *m) {
    return m->len;
}
