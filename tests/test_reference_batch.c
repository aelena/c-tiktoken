// SPDX-License-Identifier: MIT
//
// c-tiktoken, Volume comparison against the official Python tiktoken
//
// test_integration.c compares ten curated strings. This compares hundreds of
// generated ones, which is a different job: the curated cases check that the
// obvious things work, and this checks the inputs nobody thought to write down.
//
// Everything here goes through one Python process. get_encoding() loads a 1.6 MB
// vocabulary and building the ranks costs over a second, so a process per case
// would put this test out of reach. The request file is hex-encoded rather than
// quoted, which is what lets the generated cases contain quotes, backslashes,
// newlines and NUL bytes without a shell mangling them on the way.
//
// The generator is a fixed-seed xorshift64*, so a failure reproduces exactly on
// any machine. On a mismatch the case index and the input bytes are printed, which
// is what you need to lift that one string into test_integration.c as a permanent
// regression test.

#define _POSIX_C_SOURCE 200809L

#include "tiktoken/tiktoken.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXIT_SKIP 77

static int cases_run = 0;
static int cases_passed = 0;
static int cases_failed = 0;

#define MUST(expr)                                                      \
    do {                                                               \
        if (!(expr)) {                                                 \
            fprintf(stderr, "fixture failed: %s (%s:%d)\n",            \
                    #expr, __FILE__, __LINE__);                        \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// ── Deterministic generator ───────────────────────────────────────────

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;

static uint32_t rnd(uint32_t bound) {
    // xorshift64*, chosen because it is four lines and reproducible everywhere.
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t v = rng_state * 0x2545F4914F6CDD1DULL;
    return bound ? (uint32_t)((v >> 33) % bound) : 0;
}

static const char *const WORDS[] = {
    "the", "cat", "sat", "on", "mat", "architecture", "idempotent", "retry",
    "don't", "won't", "I'm", "they're", "vocabulary", "token", "encode",
    "42", "1024", "3.14", "007", "2026", "a", "I", "x",
};
#define N_WORDS (sizeof(WORDS) / sizeof(WORDS[0]))

// Appends one valid UTF-8 code point of the requested byte length.
static size_t append_utf8(uint8_t *out, size_t cap, size_t len, int nbytes) {
    if (len + 4 >= cap) return len;
    switch (nbytes) {
    case 2: {  // U+0080 to U+07FF: accents, Cyrillic, Greek
        uint32_t cp = 0x80 + rnd(0x780);
        out[len++] = (uint8_t)(0xC0 | (cp >> 6));
        out[len++] = (uint8_t)(0x80 | (cp & 0x3F));
        break;
    }
    case 3: {  // U+0800 to U+FFFF, skipping the surrogate range
        uint32_t cp = 0x800 + rnd(0xF000);
        if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0x4E00 + rnd(0x100);
        out[len++] = (uint8_t)(0xE0 | (cp >> 12));
        out[len++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[len++] = (uint8_t)(0x80 | (cp & 0x3F));
        break;
    }
    default: {  // U+10000 and up: emoji and the rest of the astral planes
        uint32_t cp = 0x10000 + rnd(0x10000);
        out[len++] = (uint8_t)(0xF0 | (cp >> 18));
        out[len++] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
        out[len++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[len++] = (uint8_t)(0x80 | (cp & 0x3F));
        break;
    }
    }
    return len;
}

// Six shapes, because the cl100k pattern branches on exactly these distinctions:
// contractions, letter runs, digit runs of at most three, punctuation runs,
// whitespace ending in a newline, and trailing whitespace.
static size_t gen_case(uint8_t *out, size_t cap, uint32_t kind) {
    size_t len = 0;
    switch (kind % 6) {
    case 0:  // words and spaces
        for (int i = 0; i < 1 + (int)rnd(12); i++) {
            const char *w = WORDS[rnd(N_WORDS)];
            size_t wl = strlen(w);
            if (len + wl + 1 >= cap) break;
            if (len) out[len++] = ' ';
            memcpy(out + len, w, wl);
            len += wl;
        }
        break;
    case 1:  // printable ASCII soup, quotes and backslashes included
        for (int i = 0; i < 1 + (int)rnd(40); i++) {
            if (len + 1 >= cap) break;
            out[len++] = (uint8_t)(0x20 + rnd(0x5F));
        }
        break;
    case 2:  // multi-byte UTF-8 at all three widths
        for (int i = 0; i < 1 + (int)rnd(20); i++) {
            len = append_utf8(out, cap, len, 2 + (int)rnd(3));
        }
        break;
    case 3: {  // whitespace shapes: runs, tabs, LF and CRLF
        static const char *ws[] = { " ", "  ", "   ", "\t", "\n", "\r\n", " \n" };
        for (int i = 0; i < 1 + (int)rnd(10); i++) {
            const char *w = WORDS[rnd(N_WORDS)];
            const char *s = ws[rnd(7)];
            size_t wl = strlen(w), sl = strlen(s);
            if (len + wl + sl >= cap) break;
            memcpy(out + len, w, wl); len += wl;
            memcpy(out + len, s, sl); len += sl;
        }
        break;
    }
    case 4:  // digits, which the pattern splits in runs of at most three
        for (int i = 0; i < 1 + (int)rnd(30); i++) {
            if (len + 1 >= cap) break;
            out[len++] = (uint8_t)('0' + rnd(10));
            if (rnd(4) == 0 && len + 1 < cap) out[len++] = ' ';
        }
        break;
    default:  // mixed, and long enough to matter for an O(n squared) merge loop
        for (int i = 0; i < 40 + (int)rnd(200); i++) {
            if (len + 8 >= cap) break;
            switch (rnd(4)) {
            case 0: {
                const char *w = WORDS[rnd(N_WORDS)];
                size_t wl = strlen(w);
                memcpy(out + len, w, wl); len += wl;
                break;
            }
            case 1: out[len++] = ' '; break;
            case 2: out[len++] = (uint8_t)(0x21 + rnd(0x40)); break;
            default: len = append_utf8(out, cap, len, 2 + (int)rnd(3)); break;
            }
        }
        break;
    }
    return len;
}

// ── The comparison ────────────────────────────────────────────────────

#define N_CASES 1000
#define MAX_TEXT 4096

static uint8_t texts[N_CASES][MAX_TEXT];
static size_t text_len[N_CASES];

static char *find_script(void) {
    static char buf[512];
    const char *candidates[] = {
        "tests/tiktoken_reference.py",
        "../tests/tiktoken_reference.py",
        "../../tests/tiktoken_reference.py",
        nullptr,
    };
    for (int i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "r");
        if (f != nullptr) {
            fclose(f);
            snprintf(buf, sizeof(buf), "%s", candidates[i]);
            return buf;
        }
    }
    return nullptr;
}

static const char *find_vocab(void) {
    const char *candidates[] = {
        "data/cl100k_base.tiktoken",
        "../data/cl100k_base.tiktoken",
        "../../data/cl100k_base.tiktoken",
        nullptr,
    };
    for (int i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

static TiktokenEncoding *load_encoding(const char *vocab_path) {
    VocabResult vocab = vocab_load_file(vocab_path);
    if (!vocab.ok) return nullptr;

    Regex *pattern = regex_compile(tiktoken_pattern_cl100k());
    if (pattern == nullptr) {
        vocab_free(&vocab);
        return nullptr;
    }

    const SpecialToken *special;
    size_t n_special = tiktoken_cl100k_special(&special);
    SpecialToken *copy = malloc(n_special * sizeof(SpecialToken));
    if (copy == nullptr) {
        regex_free(pattern);
        vocab_free(&vocab);
        return nullptr;
    }
    memcpy(copy, special, n_special * sizeof(SpecialToken));

    return tiktoken_new("cl100k_base", vocab, pattern, copy, n_special);
}

int main(void) {
    printf("Volume comparison against Python tiktoken\n");
    printf("──────────────────────────────────────────────────────────\n");

    char *script = find_script();
    const char *vocab_path = find_vocab();
    if (script == nullptr || vocab_path == nullptr) {
        printf("  %-50s[SKIP] %s\n", "reference: batch",
               script == nullptr ? "tiktoken_reference.py not found"
                                 : "cl100k_base.tiktoken not found");
        return EXIT_SKIP;
    }

    TiktokenEncoding *enc = load_encoding(vocab_path);
    if (enc == nullptr) {
        printf("  %-50s[SKIP] could not build the encoding\n", "reference: batch");
        return EXIT_SKIP;
    }

    // Write the requests.
    const char *req = "reference_requests.txt";
    const char *res = "reference_responses.txt";
    FILE *rf = fopen(req, "w");
    MUST(rf != nullptr);
    for (int i = 0; i < N_CASES; i++) {
        text_len[i] = gen_case(texts[i], MAX_TEXT, (uint32_t)i);
        fputs("cl100k_base 0 ", rf);
        for (size_t j = 0; j < text_len[i]; j++) fprintf(rf, "%02x", texts[i][j]);
        fputc('\n', rf);
    }
    fclose(rf);

    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "python3 \"%s\" --batch \"%s\" \"%s\"",
             script, req, res);
    FILE *pipe = popen(cmd, "r");
    if (pipe == nullptr) {
        printf("  %-50s[SKIP] could not run python3\n", "reference: batch");
        tiktoken_free(enc);
        return EXIT_SKIP;
    }
    char drain[512];
    while (fgets(drain, sizeof(drain), pipe) != nullptr) { /* stderr passes through */ }
    int status = pclose(pipe);

    FILE *sf = fopen(res, "r");
    if (status != 0 || sf == nullptr) {
        printf("  %-50s[SKIP] the reference did not produce output\n",
               "reference: batch");
        if (sf != nullptr) fclose(sf);
        tiktoken_free(enc);
        return EXIT_SKIP;
    }

    // Compare, case by case.
    char *line = nullptr;
    size_t line_cap = 0;
    uint32_t expected[MAX_TEXT];
    int first_failure = -1;

    for (int i = 0; i < N_CASES; i++) {
        if (getline(&line, &line_cap, sf) < 0) {
            printf("  reference returned %d of %d lines\n", i, N_CASES);
            cases_failed++;
            break;
        }
        cases_run++;

        if (strncmp(line, "ERROR", 5) == 0) {
            cases_failed++;
            if (first_failure < 0) first_failure = i;
            continue;
        }

        size_t n_expected = 0;
        for (char *p = line; *p; ) {
            while (*p == ' ' || *p == '\n' || *p == '\r') p++;
            if (*p == '\0') break;
            char *end;
            unsigned long v = strtoul(p, &end, 10);
            if (end == p) break;
            MUST(n_expected < MAX_TEXT);
            expected[n_expected++] = (uint32_t)v;
            p = end;
        }

        TokenVec got = tiktoken_encode_ordinary(enc, (const char *)texts[i],
                                                text_len[i]);
        bool same = (got.len == n_expected);
        for (size_t j = 0; same && j < got.len; j++) {
            if (got.items[j] != expected[j]) same = false;
        }

        if (same) {
            cases_passed++;
        } else {
            cases_failed++;
            if (first_failure < 0) {
                first_failure = i;
                printf("  case %d differs: %zu tokens from us, %zu from the reference\n",
                       i, got.len, n_expected);
                printf("    input hex: ");
                for (size_t j = 0; j < text_len[i] && j < 60; j++) {
                    printf("%02x", texts[i][j]);
                }
                printf("%s\n", text_len[i] > 60 ? "..." : "");
            }
        }
        tokvec_free(&got);
    }

    free(line);
    fclose(sf);
    remove(req);
    remove(res);
    tiktoken_free(enc);

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d compared\n",
           cases_passed, cases_failed, cases_run);
    if (cases_failed > 0) {
        printf("First failing case index: %d. The generator is seeded, so it "
               "reproduces.\n", first_failure);
        return EXIT_FAILURE;
    }
    if (cases_run == 0) return EXIT_SKIP;
    return EXIT_SUCCESS;
}
