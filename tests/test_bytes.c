// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 2 tests: Byte Strings

#include "tiktoken/bytes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Test framework (same as Chapter 1) ─────────────────────────────────

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

// ── Tests ──────────────────────────────────────────────────────────────

static void test_from_str(void) {
    TEST("bytes_from_str: basic ASCII");
    Bytes b = bytes_from_str("hello");
    ASSERT_EQ(b.len, 5u);
    ASSERT_TRUE(memcmp(b.data, "hello", 5) == 0);
    bytes_free(&b);
    PASS();

    TEST("bytes_from_str: empty string");
    Bytes e = bytes_from_str("");
    ASSERT_EQ(e.len, 0u);
    bytes_free(&e);
    PASS();

    TEST("bytes_from_str: nullptr");
    Bytes n = bytes_from_str(nullptr);
    ASSERT_EQ(n.len, 0u);
    ASSERT_TRUE(n.data == nullptr);
    PASS();
}

static void test_from_raw(void) {
    TEST("bytes_from_raw: binary data with null bytes");
    const uint8_t raw[] = {0x00, 0x01, 0x00, 0xFF};
    Bytes b = bytes_from_raw(raw, 4);
    ASSERT_EQ(b.len, 4u);
    ASSERT_TRUE(b.data[0] == 0x00);
    ASSERT_TRUE(b.data[1] == 0x01);
    ASSERT_TRUE(b.data[2] == 0x00);
    ASSERT_TRUE(b.data[3] == 0xFF);
    bytes_free(&b);
    PASS();
}

static void test_clone(void) {
    TEST("bytes_clone: independent copy");
    Bytes orig = bytes_from_str("original");
    Bytes copy = bytes_clone(orig);
    ASSERT_TRUE(bytes_equal(orig, copy));
    ASSERT_TRUE(orig.data != copy.data);  // different allocations
    bytes_free(&orig);
    bytes_free(&copy);
    PASS();
}

static void test_append(void) {
    TEST("bytes_append: grow from empty");
    Bytes b = {};
    ASSERT_TRUE(bytes_append(&b, (const uint8_t *)"hello", 5));
    ASSERT_TRUE(bytes_append(&b, (const uint8_t *)" world", 6));
    ASSERT_EQ(b.len, 11u);
    ASSERT_TRUE(memcmp(b.data, "hello world", 11) == 0);
    bytes_free(&b);
    PASS();

    TEST("bytes_push: individual bytes");
    Bytes b2 = {};
    ASSERT_TRUE(bytes_push(&b2, 'A'));
    ASSERT_TRUE(bytes_push(&b2, 'B'));
    ASSERT_TRUE(bytes_push(&b2, 'C'));
    ASSERT_EQ(b2.len, 3u);
    ASSERT_TRUE(b2.data[0] == 'A');
    ASSERT_TRUE(b2.data[2] == 'C');
    bytes_free(&b2);
    PASS();
}

static void test_append_bytes(void) {
    TEST("bytes_append_bytes: concatenation");
    Bytes a = bytes_from_str("foo");
    Bytes b = bytes_from_str("bar");
    ASSERT_TRUE(bytes_append_bytes(&a, b));
    Bytes expected = bytes_from_str("foobar");
    ASSERT_TRUE(bytes_equal(a, expected));
    bytes_free(&a);
    bytes_free(&b);
    bytes_free(&expected);
    PASS();
}

static void test_slice(void) {
    TEST("bytes_slice: middle subrange");
    Bytes b = bytes_from_str("hello world");
    Bytes s = bytes_slice(b, 6, 11);
    Bytes expected = bytes_from_str("world");
    ASSERT_TRUE(bytes_equal(s, expected));
    ASSERT_EQ(s.cap, 0u);  // non-owning
    bytes_free(&expected);
    // Don't free the slice — it doesn't own memory.
    bytes_free(&b);
    PASS();

    TEST("bytes_slice: clamped out of range");
    Bytes b2 = bytes_from_str("abc");
    Bytes s2 = bytes_slice(b2, 0, 100);
    ASSERT_EQ(s2.len, 3u);
    bytes_free(&b2);
    PASS();

    TEST("bytes_slice: start >= end returns empty");
    Bytes b3 = bytes_from_str("abc");
    Bytes s3 = bytes_slice(b3, 2, 1);
    ASSERT_EQ(s3.len, 0u);
    bytes_free(&b3);
    PASS();
}

static void test_equal(void) {
    TEST("bytes_equal: same content");
    Bytes a = bytes_from_str("test");
    Bytes b = bytes_from_str("test");
    ASSERT_TRUE(bytes_equal(a, b));
    bytes_free(&a);
    bytes_free(&b);
    PASS();

    TEST("bytes_equal: different content");
    Bytes c = bytes_from_str("abc");
    Bytes d = bytes_from_str("xyz");
    ASSERT_FALSE(bytes_equal(c, d));
    bytes_free(&c);
    bytes_free(&d);
    PASS();

    TEST("bytes_equal: different length");
    Bytes e = bytes_from_str("ab");
    Bytes f = bytes_from_str("abc");
    ASSERT_FALSE(bytes_equal(e, f));
    bytes_free(&e);
    bytes_free(&f);
    PASS();

    TEST("bytes_equal: both empty");
    Bytes g = {};
    Bytes h = {};
    ASSERT_TRUE(bytes_equal(g, h));
    PASS();
}

static void test_hash(void) {
    TEST("bytes_hash: equal bytes → equal hash");
    Bytes a = bytes_from_str("hello");
    Bytes b = bytes_from_str("hello");
    ASSERT_EQ(bytes_hash(a), bytes_hash(b));
    bytes_free(&a);
    bytes_free(&b);
    PASS();

    TEST("bytes_hash: different bytes → different hash");
    Bytes c = bytes_from_str("hello");
    Bytes d = bytes_from_str("world");
    // Not a guarantee in general, but for FNV-1a these specific
    // inputs must differ.
    ASSERT_TRUE(bytes_hash(c) != bytes_hash(d));
    bytes_free(&c);
    bytes_free(&d);
    PASS();
}

static void test_clear(void) {
    TEST("bytes_clear: resets length, preserves capacity");
    Bytes b = bytes_from_str("hello");
    size_t old_cap = b.cap;
    bytes_clear(&b);
    ASSERT_EQ(b.len, 0u);
    ASSERT_EQ(b.cap, old_cap);
    ASSERT_TRUE(b.data != nullptr);
    bytes_free(&b);
    PASS();
}

static void test_bytevec(void) {
    TEST("ByteVec: push and access");
    ByteVec v = bytevec_new();
    ASSERT_TRUE(bytevec_push(&v, bytes_from_str("one")));
    ASSERT_TRUE(bytevec_push(&v, bytes_from_str("two")));
    ASSERT_TRUE(bytevec_push(&v, bytes_from_str("three")));
    ASSERT_EQ(v.len, 3u);
    Bytes expected = bytes_from_str("two");
    ASSERT_TRUE(bytes_equal(v.items[1], expected));
    bytes_free(&expected);
    bytevec_free(&v);
    PASS();
}

static void test_tokvec(void) {
    TEST("TokenVec: push and extend");
    TokenVec v = tokvec_new();
    ASSERT_TRUE(tokvec_push(&v, 100));
    ASSERT_TRUE(tokvec_push(&v, 200));
    uint32_t more[] = {300, 400, 500};
    ASSERT_TRUE(tokvec_extend(&v, more, 3));
    ASSERT_EQ(v.len, 5u);
    ASSERT_EQ(v.items[0], 100u);
    ASSERT_EQ(v.items[4], 500u);
    tokvec_free(&v);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 2: Byte Strings\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_from_str();
    test_from_raw();
    test_clone();
    test_append();
    test_append_bytes();
    test_slice();
    test_equal();
    test_hash();
    test_clear();
    test_bytevec();
    test_tokvec();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
