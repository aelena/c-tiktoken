// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 6 tests: Vocabulary Loading

#include "tiktoken/vocab.h"
#include "tiktoken/bpe.h"
#include "tiktoken/bytes.h"

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

// ── Synthetic vocabulary for testing ──────────────────────────────────
//
// We create a small in-memory .tiktoken file. Each line is:
//     base64(token_bytes) rank
//
// Base64 values:
//     "a"     → "YQ=="   rank 0
//     "b"     → "Yg=="   rank 1
//     "c"     → "Yw=="   rank 2
//     "ab"    → "YWI="   rank 3
//     "bc"    → "YmM="   rank 4
//     "abc"   → "YWJj"   rank 5

static const char TINY_VOCAB[] =
    "YQ== 0\n"
    "Yg== 1\n"
    "Yw== 2\n"
    "YWI= 3\n"
    "YmM= 4\n"
    "YWJj 5\n";

// ── Tests ──────────────────────────────────────────────────────────────

static void test_load_mem_basic(void) {
    TEST("vocab: load from memory buffer");
    VocabResult v = vocab_load_mem(TINY_VOCAB, strlen(TINY_VOCAB));
    ASSERT_TRUE(v.ok);
    ASSERT_EQ(v.ranks.vocab_size, 6u);
    vocab_free(&v);
    PASS();
}

static void test_load_mem_lookup(void) {
    TEST("vocab: encode direction lookup");
    VocabResult v = vocab_load_mem(TINY_VOCAB, strlen(TINY_VOCAB));
    ASSERT_TRUE(v.ok);

    // Look up "a" → rank 0
    Bytes key_a = bytes_from_str("a");
    uint32_t rank = 0;
    ASSERT_TRUE(b2i_get(&v.ranks.encoder, key_a, &rank));
    ASSERT_EQ(rank, 0u);
    bytes_free(&key_a);

    // Look up "abc" → rank 5
    Bytes key_abc = bytes_from_str("abc");
    ASSERT_TRUE(b2i_get(&v.ranks.encoder, key_abc, &rank));
    ASSERT_EQ(rank, 5u);
    bytes_free(&key_abc);

    vocab_free(&v);
    PASS();
}

static void test_load_mem_decode_lookup(void) {
    TEST("vocab: decode direction lookup");
    VocabResult v = vocab_load_mem(TINY_VOCAB, strlen(TINY_VOCAB));
    ASSERT_TRUE(v.ok);

    // Look up rank 3 → "ab"
    Bytes out = {};
    ASSERT_TRUE(i2b_get(&v.ranks.decoder, 3, &out));
    Bytes expected = bytes_from_str("ab");
    ASSERT_TRUE(bytes_equal(out, expected));
    bytes_free(&expected);

    vocab_free(&v);
    PASS();
}

static void test_bpe_with_loaded_vocab(void) {
    TEST("vocab: BPE encode with loaded vocab");
    VocabResult v = vocab_load_mem(TINY_VOCAB, strlen(TINY_VOCAB));
    ASSERT_TRUE(v.ok);

    // Encode "abc" — should merge to single token (rank 5).
    // Merge sequence:
    //   a(0) b(1) c(2)
    //   Pair ranks: ab=3, bc=4
    //   Min = ab(3) → [ab] c
    //   Pair ranks: abc=5
    //   Min = abc(5) → [abc]
    TokenVec tokens = bpe_encode(&v.ranks, (const uint8_t *)"abc", 3);
    ASSERT_EQ(tokens.len, 1u);
    ASSERT_EQ(tokens.items[0], 5u);

    tokvec_free(&tokens);
    vocab_free(&v);
    PASS();
}

static void test_bpe_roundtrip_with_vocab(void) {
    TEST("vocab: BPE encode+decode roundtrip");
    VocabResult v = vocab_load_mem(TINY_VOCAB, strlen(TINY_VOCAB));
    ASSERT_TRUE(v.ok);

    const uint8_t *input = (const uint8_t *)"abc";
    TokenVec tokens = bpe_encode(&v.ranks, input, 3);
    Bytes decoded = bpe_decode(&v.ranks, tokens.items, tokens.len);

    ASSERT_EQ(decoded.len, 3u);
    ASSERT_TRUE(memcmp(decoded.data, "abc", 3) == 0);

    bytes_free(&decoded);
    tokvec_free(&tokens);
    vocab_free(&v);
    PASS();
}

static void test_load_empty(void) {
    TEST("vocab: empty input returns ok=false");
    VocabResult v = vocab_load_mem("", 0);
    ASSERT_TRUE(!v.ok);
    PASS();
}

static void test_load_null(void) {
    TEST("vocab: null input returns ok=false");
    VocabResult v = vocab_load_mem(nullptr, 0);
    ASSERT_TRUE(!v.ok);
    PASS();
}

static void test_load_malformed(void) {
    TEST("vocab: malformed lines are skipped");
    // Line 2 has no space separator; line 3 has invalid base64.
    const char *data =
        "YQ== 0\n"
        "BADLINE\n"
        "!!!= 2\n"
        "Yg== 3\n";
    VocabResult v = vocab_load_mem(data, strlen(data));
    // Should load 2 valid entries (rank 0 and rank 3).
    ASSERT_TRUE(v.ok);
    ASSERT_EQ(v.ranks.vocab_size, 2u);
    vocab_free(&v);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 6: Vocabulary Loading\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_load_mem_basic();
    test_load_mem_lookup();
    test_load_mem_decode_lookup();
    test_bpe_with_loaded_vocab();
    test_bpe_roundtrip_with_vocab();
    test_load_empty();
    test_load_null();
    test_load_malformed();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
