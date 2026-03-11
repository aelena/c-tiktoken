// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 7 tests: High-Level Encoding API
//
// Tests the complete encode/decode pipeline using a synthetic vocabulary.
// These tests don't need PCRE2 — we use a simple \S+ regex pattern.

#include "tiktoken/encoding.h"
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

// ── Build a test encoding ──────────────────────────────────────────────
//
// Uses the same tiny vocab as Chapter 6 tests plus a simple regex.

static const char TINY_VOCAB[] =
    "YQ== 0\n"      // "a" → 0
    "Yg== 1\n"      // "b" → 1
    "Yw== 2\n"      // "c" → 2
    "YWI= 3\n"      // "ab" → 3
    "YmM= 4\n"      // "bc" → 4
    "YWJj 5\n"      // "abc" → 5
    "IA== 6\n";     // " " (space) → 6

static TiktokenEncoding *make_test_encoding(void) {
    VocabResult vocab = vocab_load_mem(TINY_VOCAB, strlen(TINY_VOCAB));
    if (!vocab.ok) return nullptr;

    // Simple pattern: match non-whitespace runs OR individual spaces.
    // This is much simpler than the real cl100k_base pattern but enough
    // for testing the pipeline.
    Regex *pattern = regex_compile("\\S+| ");
    if (pattern == nullptr) {
        vocab_free(&vocab);
        return nullptr;
    }

    // One special token.
    SpecialToken *special = malloc(sizeof(SpecialToken));
    special[0] = (SpecialToken){
        .text     = "<|end|>",
        .text_len = 7,
        .token_id = 100,
    };

    return tiktoken_new("test", vocab, pattern, special, 1);
}

// ── Tests ──────────────────────────────────────────────────────────────

static void test_encode_simple(void) {
    TEST("encoding: simple word");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    TokenVec tokens = tiktoken_encode_ordinary(enc, "abc", 3);
    // "abc" should go through regex as one chunk, then BPE merges to rank 5.
    ASSERT_EQ(tokens.len, 1u);
    ASSERT_EQ(tokens.items[0], 5u);

    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

static void test_encode_with_space(void) {
    TEST("encoding: text with space");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    // "a b" → regex splits into ["a", " ", "b"]
    // "a" → rank 0, " " → rank 6, "b" → rank 1
    TokenVec tokens = tiktoken_encode_ordinary(enc, "a b", 3);
    ASSERT_EQ(tokens.len, 3u);
    ASSERT_EQ(tokens.items[0], 0u);   // "a"
    ASSERT_EQ(tokens.items[1], 6u);   // " "
    ASSERT_EQ(tokens.items[2], 1u);   // "b"

    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

static void test_decode_simple(void) {
    TEST("encoding: decode token IDs");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    uint32_t tokens[] = {5};  // "abc"
    Bytes decoded = tiktoken_decode(enc, tokens, 1);
    ASSERT_EQ(decoded.len, 3u);
    ASSERT_TRUE(memcmp(decoded.data, "abc", 3) == 0);

    bytes_free(&decoded);
    tiktoken_free(enc);
    PASS();
}

static void test_roundtrip(void) {
    TEST("encoding: encode→decode roundtrip");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    const char *text = "abc";
    size_t text_len = 3;

    TokenVec tokens = tiktoken_encode_ordinary(enc, text, text_len);
    Bytes decoded = tiktoken_decode(enc, tokens.items, tokens.len);

    ASSERT_EQ(decoded.len, text_len);
    ASSERT_TRUE(memcmp(decoded.data, text, text_len) == 0);

    bytes_free(&decoded);
    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

static void test_special_token_allow(void) {
    TEST("encoding: special token in ALLOW mode");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    const char *text = "a<|end|>b";
    size_t text_len = strlen(text);

    TokenVec tokens = tiktoken_encode(enc, text, text_len,
                                      TIKTOKEN_SPECIAL_ALLOW);

    // "a" → 0, <|end|> → 100, "b" → 1
    ASSERT_EQ(tokens.len, 3u);
    ASSERT_EQ(tokens.items[0], 0u);
    ASSERT_EQ(tokens.items[1], 100u);
    ASSERT_EQ(tokens.items[2], 1u);

    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

static void test_special_token_decode(void) {
    TEST("encoding: decode special token");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    uint32_t tokens[] = {0, 100, 1};  // "a", <|end|>, "b"
    Bytes decoded = tiktoken_decode(enc, tokens, 3);

    const char *expected = "a<|end|>b";
    ASSERT_EQ(decoded.len, strlen(expected));
    ASSERT_TRUE(memcmp(decoded.data, expected, decoded.len) == 0);

    bytes_free(&decoded);
    tiktoken_free(enc);
    PASS();
}

static void test_count(void) {
    TEST("encoding: count tokens");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    size_t count = tiktoken_count(enc, "abc", 3);
    ASSERT_EQ(count, 1u);  // "abc" → single token

    tiktoken_free(enc);
    PASS();
}

static void test_empty(void) {
    TEST("encoding: empty input");
    TiktokenEncoding *enc = make_test_encoding();
    ASSERT_TRUE(enc != nullptr);

    TokenVec tokens = tiktoken_encode_ordinary(enc, "", 0);
    ASSERT_EQ(tokens.len, 0u);

    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 7: Encoding API\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_encode_simple();
    test_encode_with_space();
    test_decode_simple();
    test_roundtrip();
    test_special_token_allow();
    test_special_token_decode();
    test_count();
    test_empty();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
