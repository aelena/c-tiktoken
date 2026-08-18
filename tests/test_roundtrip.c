// SPDX-License-Identifier: MIT
//
// c-tiktoken — Round-trip and property tests
//
// The other suites test units in isolation. This one tests the two invariants
// that decide whether a tokenizer is correct at all:
//
//     decode(encode(x)) == x
//     count(x) == encode(x).len
//
// Both are checked against a synthetic vocabulary built here at runtime: all
// 256 single-byte tokens, plus a handful of multi-byte merges to exercise the
// BPE loop. Byte completeness is the point — it is what makes the round-trip
// total rather than best-effort, and it is the property the real cl100k_base
// has that the six-token fixtures used elsewhere do not.
//
// Nothing here needs the network, Python, or a downloaded vocabulary file.

#include "tiktoken/tiktoken.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Test framework (same shape as the other suites) ────────────────────

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

// A failed allocation while building a fixture is not a test result — it means
// the test never ran. Reporting that as a pass is worse than crashing.
#define MUST(expr)                                                      \
    do {                                                                \
        if (!(expr)) {                                                  \
            fprintf(stderr, "fixture failed: %s (%s:%d)\n",             \
                    #expr, __FILE__, __LINE__);                         \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// ── Base64, encode direction ──────────────────────────────────────────
//
// The library only ships a decoder, because a .tiktoken file is only ever
// read. Building one needs the other direction, so here it is — deliberately
// local to the test rather than added to the public API for one caller.

static const char B64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t b64_encode_local(const uint8_t *in, size_t n, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i + 1 < n) v |= (uint32_t)in[i + 1] << 8;
        if (i + 2 < n) v |= (uint32_t)in[i + 2];

        out[o++] = B64_ALPHABET[(v >> 18) & 0x3Fu];
        out[o++] = B64_ALPHABET[(v >> 12) & 0x3Fu];
        out[o++] = (i + 1 < n) ? B64_ALPHABET[(v >> 6) & 0x3Fu] : '=';
        out[o++] = (i + 2 < n) ? B64_ALPHABET[v & 0x3Fu] : '=';
    }
    return o;
}

// ── The synthetic vocabulary ──────────────────────────────────────────

// Merges get ranks above every single byte, so BPE always prefers a longer
// token when one exists. Order within this list is the merge priority.
static const char *const MERGES[] = {
    "ab", "bc", "abc",
    "th", "he", "the",
    " t", " th", " the",
    "ca", "af", "caf",
};
#define N_MERGES (sizeof(MERGES) / sizeof(MERGES[0]))

// Chosen to sit above every single byte (0-255) and every merge, so it cannot
// be confused with a real token id.
#define SPECIAL_ID 1000u

static char vocab_buf[16384];

static size_t build_vocab_text(void) {
    size_t used = 0;
    char b64[8];

    // All 256 single bytes, rank == byte value. This is what makes the
    // round-trip total: any byte sequence is representable.
    for (unsigned b = 0; b < 256u; b++) {
        uint8_t byte = (uint8_t)b;
        size_t enc_len = b64_encode_local(&byte, 1, b64);
        b64[enc_len] = '\0';

        int n = snprintf(vocab_buf + used, sizeof(vocab_buf) - used,
                         "%s %u\n", b64, b);
        MUST(n > 0 && (size_t)n < sizeof(vocab_buf) - used);
        used += (size_t)n;
    }

    for (size_t i = 0; i < N_MERGES; i++) {
        const char *m = MERGES[i];
        size_t m_len = strlen(m);
        char mb64[16];
        size_t enc_len = b64_encode_local((const uint8_t *)m, m_len, mb64);
        mb64[enc_len] = '\0';

        int n = snprintf(vocab_buf + used, sizeof(vocab_buf) - used,
                         "%s %zu\n", mb64, 256u + i);
        MUST(n > 0 && (size_t)n < sizeof(vocab_buf) - used);
        used += (size_t)n;
    }

    return used;
}

// Returns the rank of a merge string, or the byte value for a single byte.
static uint32_t merge_rank(const char *s) {
    for (size_t i = 0; i < N_MERGES; i++) {
        if (strcmp(MERGES[i], s) == 0) return (uint32_t)(256u + i);
    }
    MUST(strlen(s) == 1);
    return (uint32_t)(uint8_t)s[0];
}

static VocabResult load_test_vocab(void) {
    size_t len = build_vocab_text();
    VocabResult v = vocab_load_mem(vocab_buf, len);
    MUST(v.ok);
    MUST(v.ranks.vocab_size == 256u + N_MERGES);
    return v;
}

static TiktokenEncoding *make_encoding(void) {
    VocabResult vocab = load_test_vocab();

    // Partitions any input with no gaps: every character is either a space or
    // a non-space, so nothing falls between the matches. A pattern that leaves
    // gaps would make a round-trip test silently vacuous.
    Regex *pattern = regex_compile("[^ ]+| ");
    MUST(pattern != nullptr);

    SpecialToken *special = malloc(sizeof(SpecialToken));
    MUST(special != nullptr);
    special[0] = (SpecialToken){
        .text     = "<|end|>",
        .text_len = 7,
        .token_id = SPECIAL_ID,
    };

    TiktokenEncoding *enc = tiktoken_new("roundtrip", vocab, pattern,
                                         special, 1);
    MUST(enc != nullptr);
    return enc;
}

// ── Round-trip: the byte layer ────────────────────────────────────────

static void test_every_byte_survives_bpe(void) {
    TEST("roundtrip: all 256 byte values through BPE");
    VocabResult v = load_test_vocab();

    uint8_t all[256];
    for (unsigned i = 0; i < 256u; i++) all[i] = (uint8_t)i;

    // The regex layer is UTF-8 aware and would reject this buffer, which is
    // not valid UTF-8. BPE works on bytes, so it is exercised directly.
    TokenVec tokens = bpe_encode(&v.ranks, all, sizeof(all));
    ASSERT_TRUE(tokens.len > 0);

    Bytes out = bpe_decode(&v.ranks, tokens.items, tokens.len);
    ASSERT_EQ(out.len, sizeof(all));
    ASSERT_EQ(memcmp(out.data, all, sizeof(all)), 0);

    bytes_free(&out);
    tokvec_free(&tokens);
    vocab_free(&v);
    PASS();
}

static void test_bpe_prefers_longer_merges(void) {
    TEST("bpe: merges apply in rank order");
    VocabResult v = load_test_vocab();

    // "abc" is a token, so it must come out as one, not as "ab" + "c" and
    // certainly not as three bytes.
    TokenVec t = bpe_encode(&v.ranks, (const uint8_t *)"abc", 3);
    ASSERT_EQ(t.len, 1u);
    ASSERT_EQ(t.items[0], merge_rank("abc"));
    tokvec_free(&t);

    // "abcb" has no token, so the greedy path is "abc" then "b".
    TokenVec t2 = bpe_encode(&v.ranks, (const uint8_t *)"abcb", 4);
    ASSERT_EQ(t2.len, 2u);
    ASSERT_EQ(t2.items[0], merge_rank("abc"));
    ASSERT_EQ(t2.items[1], merge_rank("b"));
    tokvec_free(&t2);

    vocab_free(&v);
    PASS();
}

static void test_bpe_empty_input(void) {
    TEST("bpe: empty input yields no tokens");
    VocabResult v = load_test_vocab();

    TokenVec t = bpe_encode(&v.ranks, (const uint8_t *)"", 0);
    ASSERT_EQ(t.len, 0u);
    tokvec_free(&t);

    Bytes b = bpe_decode(&v.ranks, nullptr, 0);
    ASSERT_EQ(b.len, 0u);
    bytes_free(&b);

    vocab_free(&v);
    PASS();
}

// ── Round-trip: the full pipeline ─────────────────────────────────────

static void check_roundtrip(TiktokenEncoding *enc, const char *text) {
    size_t len = strlen(text);
    TokenVec tokens = tiktoken_encode_ordinary(enc, text, len);
    Bytes out = tiktoken_decode(enc, tokens.items, tokens.len);

    bool ok = (out.len == len) && (len == 0 || memcmp(out.data, text, len) == 0);

    bytes_free(&out);
    tokvec_free(&tokens);

    if (!ok) {
        tests_failed++;
        printf("[FAIL] round-trip differs for \"%s\"\n", text);
        return;
    }
    tests_passed++;
    printf("[PASS]\n");
}

static void test_roundtrip_ascii(void) {
    TEST("roundtrip: ascii through the full pipeline");
    TiktokenEncoding *enc = make_encoding();
    check_roundtrip(enc, "the cat sat on the mat");
    tiktoken_free(enc);
}

static void test_roundtrip_utf8(void) {
    TEST("roundtrip: utf-8 multibyte through the full pipeline");
    TiktokenEncoding *enc = make_encoding();
    // Two-byte, three-byte and four-byte sequences in one string.
    check_roundtrip(enc, "cafe\xcc\x81 \xe6\x97\xa5\xe6\x9c\xac \xf0\x9f\x99\x82");
    tiktoken_free(enc);
}

static void test_roundtrip_whitespace_runs(void) {
    TEST("roundtrip: runs of spaces are preserved");
    TiktokenEncoding *enc = make_encoding();
    check_roundtrip(enc, "a   b");
    tiktoken_free(enc);
}

static void test_roundtrip_empty(void) {
    TEST("roundtrip: empty input");
    TiktokenEncoding *enc = make_encoding();

    TokenVec tokens = tiktoken_encode_ordinary(enc, "", 0);
    ASSERT_EQ(tokens.len, 0u);

    Bytes out = tiktoken_decode(enc, tokens.items, tokens.len);
    ASSERT_EQ(out.len, 0u);

    bytes_free(&out);
    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

// ── Properties of the surrounding API ─────────────────────────────────

static void test_count_matches_encode(void) {
    TEST("count: agrees with encode length");
    TiktokenEncoding *enc = make_encoding();

    const char *cases[] = { "", "a", "the the the", "a   b", "abcabcabc" };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        size_t len = strlen(cases[i]);
        TokenVec t = tiktoken_encode_ordinary(enc, cases[i], len);
        size_t counted = tiktoken_count(enc, cases[i], len);
        size_t encoded = t.len;
        tokvec_free(&t);
        ASSERT_EQ(counted, encoded);
    }

    tiktoken_free(enc);
    PASS();
}

static void test_decode_unknown_token(void) {
    TEST("decode: an unknown token id contributes nothing");
    TiktokenEncoding *enc = make_encoding();

    // 60000 is neither a byte, nor a merge, nor the special token.
    const uint32_t tokens[] = { (uint32_t)'a', 60000u, (uint32_t)'b' };
    Bytes out = tiktoken_decode(enc, tokens, 3);

    ASSERT_EQ(out.len, 2u);
    ASSERT_EQ(memcmp(out.data, "ab", 2), 0);

    bytes_free(&out);
    tiktoken_free(enc);
    PASS();
}

static void test_special_allow_vs_ignore(void) {
    TEST("special: ALLOW emits the id, IGNORE encodes the text");
    TiktokenEncoding *enc = make_encoding();
    const char *text = "a<|end|>b";
    size_t len = strlen(text);

    TokenVec allowed = tiktoken_encode(enc, text, len,
                                       TIKTOKEN_SPECIAL_ALLOW);
    bool found = false;
    for (size_t i = 0; i < allowed.len; i++) {
        if (allowed.items[i] == SPECIAL_ID) found = true;
    }
    ASSERT_TRUE(found);

    TokenVec ignored = tiktoken_encode(enc, text, len,
                                       TIKTOKEN_SPECIAL_IGNORE);
    for (size_t i = 0; i < ignored.len; i++) {
        ASSERT_FALSE(ignored.items[i] == SPECIAL_ID);
    }
    // Ignoring it means the literal characters are encoded, so it takes more
    // tokens than collapsing the whole marker into one.
    ASSERT_TRUE(ignored.len > allowed.len);

    tokvec_free(&allowed);
    tokvec_free(&ignored);
    tiktoken_free(enc);
    PASS();
}

static void test_special_roundtrip(void) {
    TEST("special: ALLOW round-trips back to the original text");
    TiktokenEncoding *enc = make_encoding();
    const char *text = "a<|end|>b";
    size_t len = strlen(text);

    TokenVec tokens = tiktoken_encode(enc, text, len,
                                      TIKTOKEN_SPECIAL_ALLOW);
    Bytes out = tiktoken_decode(enc, tokens.items, tokens.len);

    ASSERT_EQ(out.len, len);
    ASSERT_EQ(memcmp(out.data, text, len), 0);

    bytes_free(&out);
    tokvec_free(&tokens);
    tiktoken_free(enc);
    PASS();
}

static void test_null_arguments(void) {
    TEST("api: null and zero-length arguments are handled");
    TiktokenEncoding *enc = make_encoding();

    TokenVec a = tiktoken_encode_ordinary(nullptr, "abc", 3);
    ASSERT_EQ(a.len, 0u);
    tokvec_free(&a);

    TokenVec b = tiktoken_encode_ordinary(enc, nullptr, 3);
    ASSERT_EQ(b.len, 0u);
    tokvec_free(&b);

    Bytes c = tiktoken_decode(enc, nullptr, 5);
    ASSERT_EQ(c.len, 0u);
    bytes_free(&c);

    ASSERT_EQ(tiktoken_count(enc, "", 0), 0u);

    tiktoken_free(enc);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Round-trip and property tests\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_every_byte_survives_bpe();
    test_bpe_prefers_longer_merges();
    test_bpe_empty_input();

    test_roundtrip_ascii();
    test_roundtrip_utf8();
    test_roundtrip_whitespace_runs();
    test_roundtrip_empty();

    test_count_matches_encode();
    test_decode_unknown_token();
    test_special_allow_vs_ignore();
    test_special_roundtrip();
    test_null_arguments();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
