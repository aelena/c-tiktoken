// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 1 tests: Base64 Decoding
//
// A minimal test harness using only C23 and the standard library.
// No external test framework — we build our own from scratch.

#include "tiktoken/base64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Minimal test framework ─────────────────────────────────────────────
//
// We define a few macros that give us readable assertions and automatic
// pass/fail counting. This is intentionally simple — a real project
// might use a framework, but for a tutorial, seeing how the sausage is
// made is more valuable.

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

#define ASSERT_EQ_INT(a, b)                                 \
    do {                                                    \
        if ((a) != (b)) {                                   \
            FAIL("expected " #a " == " #b);                 \
        }                                                   \
    } while (0)

#define ASSERT_EQ_MEM(a, b, len)                            \
    do {                                                    \
        if (memcmp((a), (b), (len)) != 0) {                 \
            FAIL("memory mismatch: " #a " != " #b);         \
        }                                                   \
    } while (0)

// ── Helper: decode a base64 string and verify against expected bytes ───

static void check_decode(const char    *name,
                          const char    *input,
                          const uint8_t *expected,
                          size_t         expected_len)
{
    TEST(name);

    size_t input_len = strlen(input);
    size_t buf_size  = b64_decoded_size(input_len);

    uint8_t *buf = malloc(buf_size);
    if (buf == nullptr) {
        FAIL("malloc failed");
    }

    size_t decoded_len = 0;
    enum b64_status status = b64_decode(input, input_len, buf, &decoded_len);

    if (status != B64_OK) {
        free(buf);
        FAIL("b64_decode returned error");
    }

    if (decoded_len != expected_len) {
        free(buf);
        FAIL("decoded length mismatch");
    }

    if (memcmp(buf, expected, expected_len) != 0) {
        free(buf);
        FAIL("decoded bytes mismatch");
    }

    free(buf);
    PASS();
}

// ── Test cases ─────────────────────────────────────────────────────────

// RFC 4648 test vectors.
static void test_rfc4648_vectors(void) {
    // ""       → ""
    check_decode("RFC 4648: empty string",
                 "", (const uint8_t *)"", 0);

    // "Zg=="   → "f"
    check_decode("RFC 4648: \"f\"",
                 "Zg==", (const uint8_t *)"f", 1);

    // "Zm8="   → "fo"
    check_decode("RFC 4648: \"fo\"",
                 "Zm8=", (const uint8_t *)"fo", 2);

    // "Zm9v"   → "foo"
    check_decode("RFC 4648: \"foo\"",
                 "Zm9v", (const uint8_t *)"foo", 3);

    // "Zm9vYg==" → "foob"
    check_decode("RFC 4648: \"foob\"",
                 "Zm9vYg==", (const uint8_t *)"foob", 4);

    // "Zm9vYmE=" → "fooba"
    check_decode("RFC 4648: \"fooba\"",
                 "Zm9vYmE=", (const uint8_t *)"fooba", 5);

    // "Zm9vYmFy" → "foobar"
    check_decode("RFC 4648: \"foobar\"",
                 "Zm9vYmFy", (const uint8_t *)"foobar", 6);
}

// Unpadded input (no '=' suffix) — we accept this leniently.
static void test_unpadded(void) {
    check_decode("unpadded: \"f\"",
                 "Zg", (const uint8_t *)"f", 1);

    check_decode("unpadded: \"fo\"",
                 "Zm8", (const uint8_t *)"fo", 2);

    check_decode("unpadded: \"foob\"",
                 "Zm9vYg", (const uint8_t *)"foob", 4);
}

// Binary data (non-ASCII output bytes).
static void test_binary_data(void) {
    // Base64 "AAE=" decodes to bytes {0x00, 0x01}
    {
        const uint8_t expected[] = {0x00, 0x01};
        check_decode("binary: 0x00 0x01",
                     "AAE=", expected, 2);
    }

    // Base64 "/w==" decodes to byte {0xFF}
    {
        const uint8_t expected[] = {0xFF};
        check_decode("binary: 0xFF",
                     "/w==", expected, 1);
    }

    // A longer binary sequence: all 256 byte values would be 344 base64
    // chars. Test a 4-byte sequence: {0xDE, 0xAD, 0xBE, 0xEF} → "3q2+7w=="
    {
        const uint8_t expected[] = {0xDE, 0xAD, 0xBE, 0xEF};
        check_decode("binary: 0xDEADBEEF",
                     "3q2+7w==", expected, 4);
    }
}

// Error cases.
static void test_errors(void) {
    uint8_t buf[64];
    size_t  decoded_len = 0;

    // Single character remainder is invalid.
    TEST("error: single trailing char");
    {
        enum b64_status s = b64_decode("A", 1, buf, &decoded_len);
        ASSERT_EQ_INT(s, B64_INVALID_LENGTH);
    }
    PASS();

    // Invalid character in input.
    TEST("error: invalid character '!'");
    {
        enum b64_status s = b64_decode("Zm9!", 4, buf, &decoded_len);
        ASSERT_EQ_INT(s, B64_INVALID_CHAR);
    }
    PASS();

    // Null pointer arguments.
    TEST("error: null input pointer");
    {
        enum b64_status s = b64_decode(nullptr, 0, buf, &decoded_len);
        ASSERT_EQ_INT(s, B64_INVALID_LENGTH);
    }
    PASS();

    TEST("error: null output pointer");
    {
        enum b64_status s = b64_decode("Zg==", 4, nullptr, &decoded_len);
        ASSERT_EQ_INT(s, B64_INVALID_LENGTH);
    }
    PASS();
}

// A string representative of tiktoken vocabulary data.
// "hello world" → base64 "aGVsbG8gd29ybGQ="
static void test_tiktoken_like(void) {
    check_decode("tiktoken-like: \"hello world\"",
                 "aGVsbG8gd29ybGQ=",
                 (const uint8_t *)"hello world", 11);

    // A multi-byte UTF-8 token: "ñ" is 0xC3 0xB1 → base64 "w7E="
    {
        const uint8_t expected[] = {0xC3, 0xB1};
        check_decode("tiktoken-like: UTF-8 'ñ'",
                     "w7E=", expected, 2);
    }
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 1: Base64 Decoding\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_rfc4648_vectors();
    test_unpadded();
    test_binary_data();
    test_errors();
    test_tiktoken_like();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
