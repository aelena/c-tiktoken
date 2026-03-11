// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 3 tests: Hash Map and Arena Allocator

#include "tiktoken/arena.h"
#include "tiktoken/hash.h"
#include "tiktoken/bytes.h"

#include <stdio.h>
#include <stdlib.h>

// ── Test framework ─────────────────────────────────────────────────────

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                          \
    do {                                                    \
        tests_run++;                                        \
        printf("  %-50s", name);                            \
    } while (0)

#define PASS()                                              \
    do {                                                    \
        tests_passed++;                                     \
        printf("[PASS]\n");                                 \
    } while (0)

#define FAIL(msg)                                           \
    do {                                                    \
        tests_failed++;                                     \
        printf("[FAIL] %s (line %d)\n", msg, __LINE__);     \
        return;                                             \
    } while (0)

#define ASSERT_TRUE(expr)                                   \
    do { if (!(expr)) FAIL("expected true: " #expr); } while (0)

#define ASSERT_FALSE(expr)                                  \
    do { if (expr) FAIL("expected false: " #expr); } while (0)

#define ASSERT_EQ(a, b)                                     \
    do { if ((a) != (b)) FAIL(#a " != " #b); } while (0)

// ── Arena tests ────────────────────────────────────────────────────────

static void test_arena_basic(void) {
    TEST("arena: allocate and use");
    Arena a = arena_new(256);
    ASSERT_TRUE(a.base != nullptr);

    uint8_t *p1 = arena_push(&a, 32);
    ASSERT_TRUE(p1 != nullptr);

    uint8_t *p2 = arena_push(&a, 64);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p2 != p1);  // distinct allocations

    arena_free(&a);
    PASS();
}

static void test_arena_push_bytes(void) {
    TEST("arena: push_bytes copies data");
    Arena a = arena_new(256);
    const uint8_t src[] = {0xDE, 0xAD, 0xBE, 0xEF};
    uint8_t *p = arena_push_bytes(&a, src, 4);
    ASSERT_TRUE(p != nullptr);
    ASSERT_TRUE(p[0] == 0xDE && p[3] == 0xEF);
    arena_free(&a);
    PASS();
}

static void test_arena_grow(void) {
    TEST("arena: grows beyond initial capacity");
    Arena a = arena_new(64);
    // Allocate more than 64 bytes total.
    for (int i = 0; i < 100; i++) {
        uint8_t *p = arena_push(&a, 16);
        ASSERT_TRUE(p != nullptr);
    }
    ASSERT_TRUE(a.cap >= 1600);
    arena_free(&a);
    PASS();
}

static void test_arena_reset(void) {
    TEST("arena: reset reuses memory");
    Arena a = arena_new(256);
    arena_push(&a, 128);
    size_t cap_before = a.cap;

    arena_reset(&a);
    ASSERT_EQ(a.used, 0u);
    ASSERT_EQ(a.cap, cap_before);  // capacity unchanged

    // Can allocate again.
    uint8_t *p = arena_push(&a, 64);
    ASSERT_TRUE(p != nullptr);

    arena_free(&a);
    PASS();
}

// ── B2iMap tests (Bytes → uint32_t) ────────────────────────────────────

static void test_b2i_basic(void) {
    TEST("B2iMap: insert and get");
    B2iMap m = b2i_new(16);

    Bytes k1 = bytes_from_str("hello");
    Bytes k2 = bytes_from_str("world");

    ASSERT_TRUE(b2i_insert(&m, k1, 42));
    ASSERT_TRUE(b2i_insert(&m, k2, 99));
    ASSERT_EQ(b2i_len(&m), 2u);

    uint32_t val = 0;
    ASSERT_TRUE(b2i_get(&m, k1, &val));
    ASSERT_EQ(val, 42u);

    ASSERT_TRUE(b2i_get(&m, k2, &val));
    ASSERT_EQ(val, 99u);

    bytes_free(&k1);
    bytes_free(&k2);
    b2i_free(&m);
    PASS();
}

static void test_b2i_missing(void) {
    TEST("B2iMap: get missing key returns false");
    B2iMap m = b2i_new(16);

    Bytes k1 = bytes_from_str("exists");
    Bytes k2 = bytes_from_str("missing");

    b2i_insert(&m, k1, 1);

    uint32_t val = 0;
    ASSERT_FALSE(b2i_get(&m, k2, &val));

    bytes_free(&k1);
    bytes_free(&k2);
    b2i_free(&m);
    PASS();
}

static void test_b2i_overwrite(void) {
    TEST("B2iMap: insert same key overwrites value");
    B2iMap m = b2i_new(16);

    Bytes k = bytes_from_str("key");
    b2i_insert(&m, k, 100);
    b2i_insert(&m, k, 200);

    ASSERT_EQ(b2i_len(&m), 1u);

    uint32_t val = 0;
    b2i_get(&m, k, &val);
    ASSERT_EQ(val, 200u);

    bytes_free(&k);
    b2i_free(&m);
    PASS();
}

static void test_b2i_many(void) {
    TEST("B2iMap: 1000 entries (triggers growth)");
    B2iMap m = b2i_new(16);

    // Insert 1000 entries using numeric strings as keys.
    char buf[32];
    for (uint32_t i = 0; i < 1000; i++) {
        int n = snprintf(buf, sizeof(buf), "key_%u", i);
        Bytes k = bytes_from_raw((const uint8_t *)buf, (size_t)n);
        ASSERT_TRUE(b2i_insert(&m, k, i));
        bytes_free(&k);
    }

    ASSERT_EQ(b2i_len(&m), 1000u);

    // Verify all entries are retrievable.
    for (uint32_t i = 0; i < 1000; i++) {
        int n = snprintf(buf, sizeof(buf), "key_%u", i);
        Bytes k = bytes_from_raw((const uint8_t *)buf, (size_t)n);
        uint32_t val = 0;
        ASSERT_TRUE(b2i_get(&m, k, &val));
        if (val != i) {
            bytes_free(&k);
            b2i_free(&m);
            FAIL("value mismatch in many-entries test");
        }
        bytes_free(&k);
    }

    b2i_free(&m);
    PASS();
}

// ── I2bMap tests (uint32_t → Bytes) ────────────────────────────────────

static void test_i2b_basic(void) {
    TEST("I2bMap: insert and get");
    I2bMap m = i2b_new(16);

    Bytes v1 = bytes_from_str("hello");
    Bytes v2 = bytes_from_str("world");

    ASSERT_TRUE(i2b_insert(&m, 42, v1));
    ASSERT_TRUE(i2b_insert(&m, 99, v2));
    ASSERT_EQ(i2b_len(&m), 2u);

    Bytes out = {};
    ASSERT_TRUE(i2b_get(&m, 42, &out));
    ASSERT_TRUE(bytes_equal(out, v1));

    ASSERT_TRUE(i2b_get(&m, 99, &out));
    ASSERT_TRUE(bytes_equal(out, v2));

    bytes_free(&v1);
    bytes_free(&v2);
    i2b_free(&m);
    PASS();
}

static void test_i2b_missing(void) {
    TEST("I2bMap: get missing key returns false");
    I2bMap m = i2b_new(16);

    Bytes v = bytes_from_str("val");
    i2b_insert(&m, 1, v);

    Bytes out = {};
    ASSERT_FALSE(i2b_get(&m, 999, &out));

    bytes_free(&v);
    i2b_free(&m);
    PASS();
}

static void test_i2b_many(void) {
    TEST("I2bMap: 1000 entries");
    I2bMap m = i2b_new(16);

    char buf[32];
    for (uint32_t i = 0; i < 1000; i++) {
        int n = snprintf(buf, sizeof(buf), "val_%u", i);
        Bytes v = bytes_from_raw((const uint8_t *)buf, (size_t)n);
        ASSERT_TRUE(i2b_insert(&m, i, v));
        bytes_free(&v);
    }

    ASSERT_EQ(i2b_len(&m), 1000u);

    for (uint32_t i = 0; i < 1000; i++) {
        int n = snprintf(buf, sizeof(buf), "val_%u", i);
        Bytes expected = bytes_from_raw((const uint8_t *)buf, (size_t)n);
        Bytes out = {};
        ASSERT_TRUE(i2b_get(&m, i, &out));
        if (!bytes_equal(out, expected)) {
            bytes_free(&expected);
            i2b_free(&m);
            FAIL("value mismatch in many-entries test");
        }
        bytes_free(&expected);
    }

    i2b_free(&m);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 3: Hash Map and Arena Allocator\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_arena_basic();
    test_arena_push_bytes();
    test_arena_grow();
    test_arena_reset();

    test_b2i_basic();
    test_b2i_missing();
    test_b2i_overwrite();
    test_b2i_many();

    test_i2b_basic();
    test_i2b_missing();
    test_i2b_many();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
