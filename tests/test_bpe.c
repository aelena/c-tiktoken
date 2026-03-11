// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 4 tests: BPE Algorithm
//
// Tests use a small hand-crafted vocabulary to verify the merge algorithm
// step by step, without needing a real tiktoken vocabulary file.

#include "tiktoken/bpe.h"
#include "tiktoken/bytes.h"
#include "tiktoken/hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define ASSERT_EQ(a, b)                                     \
    do { if ((a) != (b)) FAIL(#a " != " #b); } while (0)

// ── Helper: insert a token into the vocabulary ─────────────────────────

static void add_token(BpeRanks *r, const char *token_str, uint32_t rank) {
    Bytes key = bytes_from_str(token_str);
    b2i_insert(&r->encoder, key, rank);
    i2b_insert(&r->decoder, rank, key);
    r->vocab_size++;
    // Note: in tests we leak the Bytes keys. In production, the arena
    // owns this data. Fine for short-lived tests.
}

static void add_token_raw(BpeRanks *r, const uint8_t *data, size_t len,
                           uint32_t rank) {
    Bytes key = bytes_from_raw(data, len);
    b2i_insert(&r->encoder, key, rank);
    i2b_insert(&r->decoder, rank, key);
    r->vocab_size++;
}

static BpeRanks make_test_vocab(void) {
    BpeRanks r = {
        .encoder    = b2i_new(64),
        .decoder    = i2b_new(64),
        .vocab_size = 0,
    };

    // Single-byte tokens (every byte gets a rank).
    // In real tiktoken, bytes 0x00–0xFF have ranks 0–255.
    // We'll just register the ASCII letters we need.
    add_token_raw(&r, (const uint8_t[]){ 'h' }, 1, 104);  // 'h' = 0x68
    add_token_raw(&r, (const uint8_t[]){ 'e' }, 1, 101);  // 'e' = 0x65
    add_token_raw(&r, (const uint8_t[]){ 'l' }, 1, 108);  // 'l' = 0x6C
    add_token_raw(&r, (const uint8_t[]){ 'o' }, 1, 111);  // 'o' = 0x6F

    // Merged tokens with ascending ranks (lower = merge first).
    // "he" merges first (rank 300), then "ll" (301), then "hel" (302),
    // then "lo" (303), then "hell" (304), then "hello" (305).
    add_token(&r, "he",    300);
    add_token(&r, "ll",    301);
    add_token(&r, "lo",    302);
    add_token(&r, "hel",   303);
    add_token(&r, "hell",  304);
    add_token(&r, "hello", 305);

    return r;
}

static void free_test_vocab(BpeRanks *r) {
    b2i_free(&r->encoder);
    i2b_free(&r->decoder);
}

// ── Tests ──────────────────────────────────────────────────────────────

static void test_single_byte(void) {
    TEST("BPE: single byte → single token");
    BpeRanks v = make_test_vocab();

    TokenVec tokens = bpe_encode(&v, (const uint8_t *)"h", 1);
    ASSERT_EQ(tokens.len, 1u);
    ASSERT_EQ(tokens.items[0], 104u);

    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_empty(void) {
    TEST("BPE: empty input → empty output");
    BpeRanks v = make_test_vocab();

    TokenVec tokens = bpe_encode(&v, (const uint8_t *)"", 0);
    ASSERT_EQ(tokens.len, 0u);

    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_two_bytes_merge(void) {
    TEST("BPE: \"he\" → merges to single token");
    BpeRanks v = make_test_vocab();

    TokenVec tokens = bpe_encode(&v, (const uint8_t *)"he", 2);
    ASSERT_EQ(tokens.len, 1u);
    ASSERT_EQ(tokens.items[0], 300u);  // "he" = rank 300

    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_hello(void) {
    TEST("BPE: \"hello\" → merges to single token");
    BpeRanks v = make_test_vocab();

    // Merge sequence:
    //   h e l l o    (start: 5 single-byte parts)
    //   [he] l l o   (merge "he" rank=300)
    //   [he] [ll] o  (merge "ll" rank=301)
    //   [he] [lo]    ... wait, "ll" was merged, so next is "llo"?
    //
    // Actually let's trace carefully:
    //   Parts: h(104) e(101) l(108) l(108) o(111)
    //   Pair ranks: he=300, el=none, ll=301, lo=302
    //   Min = he(300) → merge → [he] l l o
    //   Pair ranks: hel=303, ll=301, lo=302
    //   Min = ll(301) → merge → [he] [ll] o
    //   Pair ranks: hell=304, llo=none
    //   Min = hell(304) → merge → [hell] o
    //   Pair ranks: hello=305
    //   Min = hello(305) → merge → [hello]
    //   Done!

    TokenVec tokens = bpe_encode(&v, (const uint8_t *)"hello", 5);
    ASSERT_EQ(tokens.len, 1u);
    ASSERT_EQ(tokens.items[0], 305u);  // "hello" = rank 305

    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_no_merges(void) {
    TEST("BPE: no mergeable pairs → all single-byte tokens");
    BpeRanks v = make_test_vocab();

    // "ol" has no merge rank, so "ol" stays as two separate tokens.
    TokenVec tokens = bpe_encode(&v, (const uint8_t *)"ol", 2);
    ASSERT_EQ(tokens.len, 2u);
    ASSERT_EQ(tokens.items[0], 111u);  // 'o'
    ASSERT_EQ(tokens.items[1], 108u);  // 'l'

    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_partial_merges(void) {
    TEST("BPE: \"hell\" → merges to single token");
    BpeRanks v = make_test_vocab();

    // h e l l → he(300) → [he] l l → ll(301) → [he] [ll]
    // → hell(304) → [hell]
    TokenVec tokens = bpe_encode(&v, (const uint8_t *)"hell", 4);
    ASSERT_EQ(tokens.len, 1u);
    ASSERT_EQ(tokens.items[0], 304u);

    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_decode_roundtrip(void) {
    TEST("BPE: encode then decode = original");
    BpeRanks v = make_test_vocab();

    const uint8_t *input = (const uint8_t *)"hello";
    size_t input_len = 5;

    TokenVec tokens = bpe_encode(&v, input, input_len);
    Bytes decoded = bpe_decode(&v, tokens.items, tokens.len);

    ASSERT_EQ(decoded.len, input_len);
    ASSERT_TRUE(memcmp(decoded.data, input, input_len) == 0);

    bytes_free(&decoded);
    tokvec_free(&tokens);
    free_test_vocab(&v);
    PASS();
}

static void test_decode_multiple_tokens(void) {
    TEST("BPE: decode concatenates token bytes");
    BpeRanks v = make_test_vocab();

    // Decode ['h', 'e'] = "he"
    uint32_t tokens[] = {104, 101};
    Bytes decoded = bpe_decode(&v, tokens, 2);
    ASSERT_EQ(decoded.len, 2u);
    ASSERT_TRUE(decoded.data[0] == 'h');
    ASSERT_TRUE(decoded.data[1] == 'e');

    bytes_free(&decoded);
    free_test_vocab(&v);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 4: BPE Algorithm\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_single_byte();
    test_empty();
    test_two_bytes_merge();
    test_hello();
    test_no_merges();
    test_partial_merges();
    test_decode_roundtrip();
    test_decode_multiple_tokens();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
