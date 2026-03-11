// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 5 tests: Regex Pre-tokenization

#include "tiktoken/regex.h"

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

// Helper: extract a match as a new string (for comparison).
static char *extract_match(const char *text, RegexMatch m) {
    char *s = malloc(m.len + 1);
    memcpy(s, text + m.start, m.len);
    s[m.len] = '\0';
    return s;
}

// Helper: check that match i equals expected string.
static bool match_eq(const char *text, RegexMatchVec *matches,
                     size_t i, const char *expected)
{
    if (i >= matches->len) return false;
    char *got = extract_match(text, matches->items[i]);
    bool eq = strcmp(got, expected) == 0;
    if (!eq) {
        printf("\n    match[%zu]: expected \"%s\", got \"%s\"",
               i, expected, got);
    }
    free(got);
    return eq;
}

// ── Tests ──────────────────────────────────────────────────────────────

static void test_compile_valid(void) {
    TEST("regex: compile valid pattern");
    Regex *re = regex_compile("\\w+");
    ASSERT_TRUE(re != nullptr);
    regex_free(re);
    PASS();
}

static void test_compile_invalid(void) {
    TEST("regex: compile invalid pattern returns nullptr");
    Regex *re = regex_compile("[invalid((");
    ASSERT_TRUE(re == nullptr);
    PASS();
}

static void test_simple_words(void) {
    TEST("regex: simple word splitting");
    Regex *re = regex_compile("\\w+");
    ASSERT_TRUE(re != nullptr);

    const char *text = "hello world foo";
    RegexMatchVec matches = regex_find_all(re, text, strlen(text));

    ASSERT_EQ(matches.len, 3u);
    ASSERT_TRUE(match_eq(text, &matches, 0, "hello"));
    ASSERT_TRUE(match_eq(text, &matches, 1, "world"));
    ASSERT_TRUE(match_eq(text, &matches, 2, "foo"));

    regexmatchvec_free(&matches);
    regex_free(re);
    PASS();
}

static void test_cl100k_basic(void) {
    TEST("regex: cl100k pattern splits English text");
    const char *pattern = tiktoken_pattern_cl100k();
    Regex *re = regex_compile(pattern);
    ASSERT_TRUE(re != nullptr);

    const char *text = "hello world";
    RegexMatchVec matches = regex_find_all(re, text, strlen(text));

    // "hello" and " world" (note the leading space on "world")
    ASSERT_EQ(matches.len, 2u);
    ASSERT_TRUE(match_eq(text, &matches, 0, "hello"));
    ASSERT_TRUE(match_eq(text, &matches, 1, " world"));

    regexmatchvec_free(&matches);
    regex_free(re);
    PASS();
}

static void test_cl100k_contractions(void) {
    TEST("regex: cl100k handles contractions");
    const char *pattern = tiktoken_pattern_cl100k();
    Regex *re = regex_compile(pattern);
    ASSERT_TRUE(re != nullptr);

    const char *text = "I'm don't";
    RegexMatchVec matches = regex_find_all(re, text, strlen(text));

    // Expected: "I", "'m", " don", "'t"
    ASSERT_TRUE(matches.len >= 4u);
    ASSERT_TRUE(match_eq(text, &matches, 0, "I"));
    ASSERT_TRUE(match_eq(text, &matches, 1, "'m"));
    ASSERT_TRUE(match_eq(text, &matches, 2, " don"));
    ASSERT_TRUE(match_eq(text, &matches, 3, "'t"));

    regexmatchvec_free(&matches);
    regex_free(re);
    PASS();
}

static void test_cl100k_numbers(void) {
    TEST("regex: cl100k splits numbers into 1-3 digit chunks");
    const char *pattern = tiktoken_pattern_cl100k();
    Regex *re = regex_compile(pattern);
    ASSERT_TRUE(re != nullptr);

    const char *text = "12345";
    RegexMatchVec matches = regex_find_all(re, text, strlen(text));

    // "123" "45" (3-digit + 2-digit)
    ASSERT_EQ(matches.len, 2u);
    ASSERT_TRUE(match_eq(text, &matches, 0, "123"));
    ASSERT_TRUE(match_eq(text, &matches, 1, "45"));

    regexmatchvec_free(&matches);
    regex_free(re);
    PASS();
}

static void test_cl100k_punctuation(void) {
    TEST("regex: cl100k handles punctuation");
    const char *pattern = tiktoken_pattern_cl100k();
    Regex *re = regex_compile(pattern);
    ASSERT_TRUE(re != nullptr);

    const char *text = "hello, world!";
    RegexMatchVec matches = regex_find_all(re, text, strlen(text));

    // "hello", ",", " world", "!"
    ASSERT_TRUE(matches.len >= 3u);
    ASSERT_TRUE(match_eq(text, &matches, 0, "hello"));

    regexmatchvec_free(&matches);
    regex_free(re);
    PASS();
}

static void test_empty_input(void) {
    TEST("regex: empty input returns no matches");
    Regex *re = regex_compile("\\w+");
    ASSERT_TRUE(re != nullptr);

    RegexMatchVec matches = regex_find_all(re, "", 0);
    ASSERT_EQ(matches.len, 0u);

    regexmatchvec_free(&matches);
    regex_free(re);
    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Chapter 5: Regex Pre-tokenization\n");
    printf("──────────────────────────────────────────────────────────\n");

    test_compile_valid();
    test_compile_invalid();
    test_simple_words();
    test_cl100k_basic();
    test_cl100k_contractions();
    test_cl100k_numbers();
    test_cl100k_punctuation();
    test_empty_input();

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
