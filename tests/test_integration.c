// SPDX-License-Identifier: MIT
//
// c-tiktoken — Integration tests: Compare against official tiktoken library
//
// This test suite validates the C implementation against the official
// Python tiktoken library by:
// 1. Encoding text with the C implementation
// 2. Calling the Python reference script to get expected results
// 3. Comparing the token IDs

#include "tiktoken/tiktoken.h"
#include "tiktoken/vocab.h"
#include "tiktoken/regex.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

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

// ── Helper: Find Python script path ───────────────────────────────────

static char *find_python_script(void) {
    // Try to find the Python script relative to the test executable
    // This works if tests are run from the build directory
    const char *candidates[] = {
        "tests/tiktoken_reference.py",
        "../tests/tiktoken_reference.py",
        "../../tests/tiktoken_reference.py",
        "tiktoken_reference.py",
        nullptr
    };
    
    for (int i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "r");
        if (f != nullptr) {
            fclose(f);
            char *path = malloc(strlen(candidates[i]) + 1);
            if (path != nullptr) {
                strcpy(path, candidates[i]);
                return path;
            }
        }
    }
    
    return nullptr;
}

// ── Helper: Call Python script and get token IDs ──────────────────────

static bool call_python_reference(const char *encoding_name,
                                   const char *text,
                                   bool allow_special,
                                   uint32_t **tokens_out,
                                   size_t *n_tokens_out)
{
    char *script_path = find_python_script();
    if (script_path == nullptr) {
        fprintf(stderr, "Warning: Could not find tiktoken_reference.py\n");
        return false;
    }
    
    // Build command: python3 script.py encoding_name "text" [allow_special]
    char cmd[2048];
    int allow_flag = allow_special ? 1 : 0;
    
#ifdef _WIN32
    // On Windows, use python instead of python3
    snprintf(cmd, sizeof(cmd),
             "python \"%s\" %s \"%s\" %d",
             script_path, encoding_name, text, allow_flag);
#else
    snprintf(cmd, sizeof(cmd),
             "python3 \"%s\" %s \"%s\" %d",
             script_path, encoding_name, text, allow_flag);
#endif
    
    FILE *pipe = popen(cmd, "r");
    if (pipe == nullptr) {
        free(script_path);
        return false;
    }
    
    // Read JSON output
    char buffer[4096];
    size_t total = 0;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        total += strlen(buffer);
        if (total >= sizeof(buffer) - 1) break;
    }
    
    int status = pclose(pipe);
    free(script_path);
    
    if (status != 0) {
        return false;
    }
    
    // Parse JSON array (simple parser for [1, 2, 3] format)
    // This is a minimal JSON parser - in production you might use a library
    char *json = buffer;
    while (*json && (*json == ' ' || *json == '\n' || *json == '\r' || *json == '\t')) json++;
    if (*json != '[') return false;
    json++;
    
    // Count tokens by counting commas + 1
    size_t count = 0;
    const char *p = json;
    bool in_number = false;
    while (*p && *p != ']') {
        if (*p >= '0' && *p <= '9') {
            if (!in_number) {
                count++;
                in_number = true;
            }
        } else if (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            in_number = false;
        }
        p++;
    }
    
    if (count == 0) {
        *tokens_out = nullptr;
        *n_tokens_out = 0;
        return true;
    }
    
    // Allocate and parse tokens
    uint32_t *tokens = malloc(count * sizeof(uint32_t));
    if (tokens == nullptr) return false;
    
    p = json;
    size_t idx = 0;
    in_number = false;
    while (*p && *p != ']' && idx < count) {
        if (*p >= '0' && *p <= '9') {
            if (!in_number) {
                char *end;
                unsigned long val = strtoul(p, &end, 10);
                if (val > UINT32_MAX) {
                    free(tokens);
                    return false;
                }
                tokens[idx] = (uint32_t)val;
                idx++;
                p = end;
                in_number = true;
            } else {
                p++;
            }
        } else {
            in_number = false;
            p++;
        }
    }
    
    *tokens_out = tokens;
    *n_tokens_out = count;
    return true;
}

// ── Helper: Compare token arrays ───────────────────────────────────────

static bool compare_tokens(const uint32_t *c_tokens, size_t c_len,
                           const uint32_t *py_tokens, size_t py_len)
{
    if (c_len != py_len) {
        printf("\n    Length mismatch: C=%zu, Python=%zu", c_len, py_len);
        return false;
    }
    
    for (size_t i = 0; i < c_len; i++) {
        if (c_tokens[i] != py_tokens[i]) {
            printf("\n    Token[%zu]: C=%u, Python=%u", i, c_tokens[i], py_tokens[i]);
            return false;
        }
    }
    
    return true;
}

// ── Test helper: Compare C vs Python ──────────────────────────────────

static void test_compare(const char *test_name,
                         TiktokenEncoding *enc,
                         const char *text,
                         bool allow_special)
{
    TEST(test_name);
    
    // Encode with C implementation
    TokenVec c_tokens;
    if (allow_special) {
        c_tokens = tiktoken_encode(enc, text, strlen(text), TIKTOKEN_SPECIAL_ALLOW);
    } else {
        c_tokens = tiktoken_encode_ordinary(enc, text, strlen(text));
    }
    
    // Get expected results from Python
    uint32_t *py_tokens = nullptr;
    size_t py_len = 0;
    bool py_ok = call_python_reference(enc->name, text, allow_special, &py_tokens, &py_len);
    
    if (!py_ok) {
        printf("[SKIP] Python reference unavailable\n");
        tokvec_free(&c_tokens);
        return;
    }
    
    // Compare results
    bool match = compare_tokens(c_tokens.items, c_tokens.len, py_tokens, py_len);
    
    free(py_tokens);
    tokvec_free(&c_tokens);
    
    if (!match) {
        FAIL("Token mismatch with Python reference");
    }
    
    PASS();
}

// ── Tests ──────────────────────────────────────────────────────────────

static void test_simple_text(void) {
    // Skip if vocabulary file not available
    const char *vocab_path = "data/cl100k_base.tiktoken";
    FILE *f = fopen(vocab_path, "r");
    if (f == nullptr) {
        vocab_path = "../data/cl100k_base.tiktoken";
        f = fopen(vocab_path, "r");
    }
    if (f == nullptr) {
        vocab_path = "../../data/cl100k_base.tiktoken";
        f = fopen(vocab_path, "r");
    }
    if (f == nullptr) {
        printf("  %-50s[SKIP] cl100k_base.tiktoken not found\n", "integration: simple text");
        return;
    }
    fclose(f);
    
    const SpecialToken *special;
    size_t n_special = tiktoken_cl100k_special(&special);
    
    VocabResult vocab = vocab_load_file(vocab_path);
    if (!vocab.ok) {
        printf("  %-50s[SKIP] Failed to load vocabulary\n", "integration: simple text");
        return;
    }
    
    Regex *pattern = regex_compile(tiktoken_pattern_cl100k());
    if (pattern == nullptr) {
        vocab_free(&vocab);
        printf("  %-50s[SKIP] Failed to compile regex\n", "integration: simple text");
        return;
    }
    
    // Copy special tokens
    SpecialToken *special_copy = nullptr;
    if (n_special > 0 && special != nullptr) {
        special_copy = malloc(n_special * sizeof(SpecialToken));
        if (special_copy == nullptr) {
            regex_free(pattern);
            vocab_free(&vocab);
            printf("  %-50s[SKIP] Out of memory\n", "integration: simple text");
            return;
        }
        memcpy(special_copy, special, n_special * sizeof(SpecialToken));
    }
    
    // Create encoding with correct name for Python reference
    TiktokenEncoding *enc = tiktoken_new("cl100k_base", vocab, pattern, special_copy, n_special);
    
    if (enc == nullptr) {
        printf("  %-50s[SKIP] Failed to create encoding\n", "integration: simple text");
        return;
    }
    
    test_compare("integration: simple text", enc, "Hello, world!", false);
    test_compare("integration: empty string", enc, "", false);
    test_compare("integration: single word", enc, "hello", false);
    test_compare("integration: numbers", enc, "12345", false);
    test_compare("integration: punctuation", enc, "Hello, world! How are you?", false);
    test_compare("integration: contractions", enc, "I'm don't won't", false);
    test_compare("integration: unicode", enc, "Hello 世界 🌍", false);
    test_compare("integration: newlines", enc, "Line 1\nLine 2\r\nLine 3", false);
    test_compare("integration: special token text", enc, "<|endoftext|>", true);
    test_compare("integration: mixed special", enc, "Hello<|endoftext|>world", true);
    
    tiktoken_free(enc);
}

// ── Main ───────────────────────────────────────────────────────────────

int main(void) {
    printf("Integration Tests: C vs Python tiktoken\n");
    printf("──────────────────────────────────────────────────────────\n");
    
    test_simple_text();
    
    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
