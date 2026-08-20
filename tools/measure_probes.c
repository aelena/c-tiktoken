// SPDX-License-Identifier: MIT
//
// Measure the average probe count of the hash map, so Chapter 3 can cite a
// measurement instead of a number somebody remembered.
//
// Build and run:
//
//     cmake -S . -B pm -DTIKTOKEN_PROBE_STATS=ON
//     cmake --build pm --target measure_probes
//     ./pm/measure_probes [path/to/cl100k_base.tiktoken]
//
// A probe here is one slot examined. A lookup that finds its key in the slot it
// hashed to counts as one probe, not zero.
//
// Two things are measured, because they answer different questions:
//
//   1. Controlled load factors. The table is built at a fixed capacity and
//      filled to exactly the target occupancy, which is what a claim like
//      "at 70% load" means.
//
//   2. The real vocabulary. What the load factor actually is after loading
//      cl100k_base, which is the number that matters in practice and is not
//      70%: the table doubles when it crosses the threshold, so it spends its
//      life somewhere between half and seven tenths full.

#include "tiktoken/hash.h"
#include "tiktoken/bytes.h"
#include "tiktoken/vocab.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9E3779B97F4A7C15ULL;

static uint32_t rnd(void) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return (uint32_t)((rng * 0x2545F4914F6CDD1DULL) >> 33);
}

// A table at exactly `cap` slots with `n` entries, so the load factor is known.
// b2i_new rounds to a power of two and b2i_insert grows past 70%, so the
// capacity is requested large enough that no growth happens during filling.
static void measure_at_load(size_t cap_pow2, double target_load) {
    size_t n = (size_t)((double)cap_pow2 * target_load);

    B2iMap m = b2i_new(cap_pow2);
    if (m.slots == nullptr) {
        printf("  out of memory\n");
        return;
    }

    Bytes *keys = malloc(n * sizeof(Bytes));
    if (keys == nullptr) {
        b2i_free(&m);
        return;
    }

    char buf[32];
    for (size_t i = 0; i < n; i++) {
        int len = snprintf(buf, sizeof(buf), "k%08x%08x", rnd(), rnd());
        keys[i] = bytes_from_raw((const uint8_t *)buf, (size_t)len);
        if (!b2i_insert(&m, keys[i], (uint32_t)i)) {
            printf("  insert failed at %zu\n", i);
            break;
        }
    }

    // If the map grew, the load factor is no longer what was asked for and the
    // measurement would be mislabelled. Say so rather than print a wrong number.
    if (m.cap != cap_pow2) {
        printf("  %5.0f%%  (table grew from %zu to %zu, measurement skipped)\n",
               target_load * 100, cap_pow2, m.cap);
        goto done;
    }

    tiktoken_probe_reset();

    // Hits: every key that is in there.
    for (size_t i = 0; i < n; i++) {
        uint32_t v = 0;
        (void)b2i_get(&m, keys[i], &v);
    }
    double hit_avg = (double)tiktoken_probe_slots_hit
                   / (double)(tiktoken_probe_hits ? tiktoken_probe_hits : 1);

    // Misses: the same number of keys that are not.
    unsigned long long before_miss = tiktoken_probe_misses;
    unsigned long long before_slots = tiktoken_probe_slots_miss;
    for (size_t i = 0; i < n; i++) {
        int len = snprintf(buf, sizeof(buf), "absent%08x%08x", rnd(), rnd());
        Bytes probe = bytes_from_raw((const uint8_t *)buf, (size_t)len);
        uint32_t v = 0;
        (void)b2i_get(&m, probe, &v);
        bytes_free(&probe);
    }
    unsigned long long miss_n = tiktoken_probe_misses - before_miss;
    double miss_avg = (double)(tiktoken_probe_slots_miss - before_slots)
                    / (double)(miss_n ? miss_n : 1);

    // The mean is not where Robin Hood earns its keep. The worst case is: the
    // longest probe sequence any key ended up with, read straight off the table.
    int32_t worst = 0;
    for (size_t i = 0; i < m.cap; i++) {
        if (m.slots[i].psl > worst) worst = m.slots[i].psl;
    }

    printf("  %5.0f%%   %8zu / %8zu   %6.2f       %6.2f      %5d\n",
           target_load * 100, n, m.cap, hit_avg, miss_avg, worst + 1);

done:
    for (size_t i = 0; i < n; i++) bytes_free(&keys[i]);
    free(keys);
    b2i_free(&m);
}

static void measure_real_vocabulary(const char *path) {
    VocabResult v = vocab_load_file(path);
    if (!v.ok) {
        printf("  could not load %s, skipping\n", path);
        return;
    }

    size_t n = b2i_len(&v.ranks.encoder);
    size_t cap = v.ranks.encoder.cap;
    printf("  entries %zu, capacity %zu, load %.1f%%\n",
           n, cap, 100.0 * (double)n / (double)cap);

    // Re-look-up every token that is in the vocabulary. The keys live in the
    // arena, so they are read back out of the map itself.
    tiktoken_probe_reset();
    for (size_t i = 0; i < cap; i++) {
        if (v.ranks.encoder.slots[i].psl < 0) continue;
        uint32_t out = 0;
        (void)b2i_get(&v.ranks.encoder, v.ranks.encoder.slots[i].key, &out);
    }
    double hit_avg = (double)tiktoken_probe_slots_hit
                   / (double)(tiktoken_probe_hits ? tiktoken_probe_hits : 1);

    unsigned long long before_miss = tiktoken_probe_misses;
    unsigned long long before_slots = tiktoken_probe_slots_miss;
    char buf[32];
    for (size_t i = 0; i < n; i++) {
        int len = snprintf(buf, sizeof(buf), "absent%08x%08x", rnd(), rnd());
        Bytes probe = bytes_from_raw((const uint8_t *)buf, (size_t)len);
        uint32_t out = 0;
        (void)b2i_get(&v.ranks.encoder, probe, &out);
        bytes_free(&probe);
    }
    unsigned long long miss_n = tiktoken_probe_misses - before_miss;
    double miss_avg = (double)(tiktoken_probe_slots_miss - before_slots)
                    / (double)(miss_n ? miss_n : 1);

    int32_t worst = 0;
    for (size_t i = 0; i < cap; i++) {
        if (v.ranks.encoder.slots[i].psl > worst) worst = v.ranks.encoder.slots[i].psl;
    }
    printf("  average probes: %.2f on a hit, %.2f on a miss, worst case %d\n",
           hit_avg, miss_avg, worst + 1);
    vocab_free(&v);
}

int main(int argc, char **argv) {
    printf("Hash map probe counts, measured\n");
    printf("A probe is one slot examined; finding a key where it hashed is 1.\n\n");

    printf("Synthetic, table held at a fixed capacity of 2^20:\n");
    printf("  load    entries / capacity    hit          miss       worst\n");
    for (double load = 0.10; load < 0.71; load += 0.10) {
        measure_at_load(1u << 20, load);
    }

    printf("\nReal vocabulary:\n");
    measure_real_vocabulary(argc > 1 ? argv[1] : "data/cl100k_base.tiktoken");

    return 0;
}
