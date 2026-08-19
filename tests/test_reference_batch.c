// SPDX-License-Identifier: MIT
//
// c-tiktoken, Volume comparison against the official Python tiktoken
//
// test_integration.c compares ten curated strings against cl100k_base. This
// compares a thousand generated ones against cl100k_base, o200k_base and
// p50k_base, which is a different job: the curated cases check that the obvious
// things work, and this checks the inputs nobody thought to write down, on the
// two encodings nothing was checking at all.
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

// Code point ranges to draw from, one group per UTF-8 encoded length.
//
// These are curated rather than random, and the reason is a bug this test had on
// its first CI run. Picking arbitrary values inside U+0080..U+10FFFF produces
// code points that are *unassigned*: U+12A90, for one, whose Unicode category is
// Cn. Whether \p{L} matches an unassigned code point depends on the Unicode
// version compiled into the regex engine, so PCRE2 on the runner classified it
// one way and the Rust engine inside tiktoken classified it the other, and two
// of three thousand cases disagreed depending on which machine ran them.
//
// Real text does not contain unassigned code points, and a test that fails based
// on the host's PCRE2 build is a test that gets ignored. So these ranges are all
// long-assigned and stable across versions. The portability caveat itself is
// worth knowing and is written up in Chapter 5, but it does not belong in an
// assertion.
typedef struct { uint32_t lo, hi; } CpRange;

static const CpRange CP2[] = {   // two bytes
    { 0x00C0, 0x00FF },          // Latin-1 letters
    { 0x0391, 0x03C9 },          // Greek
    { 0x0410, 0x044F },          // Cyrillic
};
static const CpRange CP3[] = {   // three bytes
    { 0x4E00, 0x9FFF },          // CJK Unified Ideographs
    { 0x3041, 0x3096 },          // Hiragana
    { 0xAC00, 0xD7A3 },          // Hangul syllables
};
static const CpRange CP4[] = {   // four bytes
    { 0x1F300, 0x1F5FF },        // symbols and pictographs
    { 0x1F600, 0x1F64F },        // emoticons
};

// Appends one valid, assigned UTF-8 code point of the requested byte length.
static size_t append_utf8(uint8_t *out, size_t cap, size_t len, int nbytes) {
    if (len + 4 >= cap) return len;

    const CpRange *set;
    size_t n_set;
    switch (nbytes) {
    case 2:  set = CP2; n_set = sizeof(CP2) / sizeof(CP2[0]); break;
    case 3:  set = CP3; n_set = sizeof(CP3) / sizeof(CP3[0]); break;
    default: set = CP4; n_set = sizeof(CP4) / sizeof(CP4[0]); break;
    }

    const CpRange *r = &set[rnd((uint32_t)n_set)];
    uint32_t cp = r->lo + rnd(r->hi - r->lo + 1);

    if (cp < 0x800) {
        out[len++] = (uint8_t)(0xC0 | (cp >> 6));
        out[len++] = (uint8_t)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out[len++] = (uint8_t)(0xE0 | (cp >> 12));
        out[len++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[len++] = (uint8_t)(0x80 | (cp & 0x3F));
    } else {
        out[len++] = (uint8_t)(0xF0 | (cp >> 18));
        out[len++] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
        out[len++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[len++] = (uint8_t)(0x80 | (cp & 0x3F));
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

static const char *find_vocab(const char *encoding_name) {
    static char buf[512];
    const char *dirs[] = { "data", "../data", "../../data", nullptr };
    for (int i = 0; dirs[i] != nullptr; i++) {
        snprintf(buf, sizeof(buf), "%s/%s.tiktoken", dirs[i], encoding_name);
        FILE *f = fopen(buf, "rb");
        if (f != nullptr) {
            fclose(f);
            return buf;
        }
    }
    return nullptr;
}

static TiktokenEncoding *load_encoding(const char *name, const char *vocab_path,
                                      const char *pattern_str) {
    VocabResult vocab = vocab_load_file(vocab_path);
    if (!vocab.ok) return nullptr;

    Regex *pattern = regex_compile(pattern_str);
    if (pattern == nullptr) {
        vocab_free(&vocab);
        return nullptr;
    }

    // Special tokens differ per encoding. Only the cl100k set ships with the
    // library, and these cases are all ordinary text anyway, so the comparison
    // runs with none: it is the pattern and the ranks under test here.
    return tiktoken_new(name, vocab, pattern, nullptr, 0);
}

typedef struct {
    const char *name;
    const char *(*pattern)(void);
} Encoding;

// Returns 0 if every case matched, 1 on a difference, 77 if it could not run.
static int compare_encoding(const Encoding *e, const char *script) {
    const char *vocab_path = find_vocab(e->name);
    if (vocab_path == nullptr) {
        printf("  %-28s [SKIP] %s.tiktoken not found\n", e->name, e->name);
        return EXIT_SKIP;
    }

    TiktokenEncoding *enc = load_encoding(e->name, vocab_path, e->pattern());
    if (enc == nullptr) {
        printf("  %-28s [SKIP] could not build the encoding\n", e->name);
        return EXIT_SKIP;
    }

    char req[256], res[256];
    snprintf(req, sizeof(req), "ref_req_%s.txt", e->name);
    snprintf(res, sizeof(res), "ref_res_%s.txt", e->name);

    FILE *rf = fopen(req, "w");
    MUST(rf != nullptr);
    rng_state = 0x2545F4914F6CDD1DULL;  // same thousand cases for every encoding
    for (int i = 0; i < N_CASES; i++) {
        text_len[i] = gen_case(texts[i], MAX_TEXT, (uint32_t)i);
        fprintf(rf, "%s 0 ", e->name);
        for (size_t j = 0; j < text_len[i]; j++) fprintf(rf, "%02x", texts[i][j]);
        fputc('\n', rf);
    }
    fclose(rf);

    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "python3 \"%s\" --batch \"%s\" \"%s\"",
             script, req, res);
    FILE *pipe = popen(cmd, "r");
    if (pipe == nullptr) {
        printf("  %-28s [SKIP] could not run python3\n", e->name);
        tiktoken_free(enc);
        return EXIT_SKIP;
    }
    char drain[512];
    while (fgets(drain, sizeof(drain), pipe) != nullptr) { /* stderr passes through */ }
    int status = pclose(pipe);

    FILE *sf = fopen(res, "r");
    if (status != 0 || sf == nullptr) {
        printf("  %-28s [SKIP] the reference produced no output\n", e->name);
        if (sf != nullptr) fclose(sf);
        tiktoken_free(enc);
        return EXIT_SKIP;
    }

    char *line = nullptr;
    size_t line_cap = 0;
    static uint32_t expected[MAX_TEXT];
    int passed = 0, failed = 0, first_failure = -1;

    for (int i = 0; i < N_CASES; i++) {
        if (getline(&line, &line_cap, sf) < 0) break;

        size_t n_expected = 0;
        if (strncmp(line, "ERROR", 5) != 0) {
            for (char *q = line; *q; ) {
                while (*q == ' ' || *q == '\n' || *q == '\r') q++;
                if (*q == '\0') break;
                char *end;
                unsigned long v = strtoul(q, &end, 10);
                if (end == q) break;
                MUST(n_expected < MAX_TEXT);
                expected[n_expected++] = (uint32_t)v;
                q = end;
            }
        }

        TokenVec got = tiktoken_encode_ordinary(enc, (const char *)texts[i],
                                                text_len[i]);
        bool same = (strncmp(line, "ERROR", 5) != 0) && (got.len == n_expected);
        for (size_t j = 0; same && j < got.len; j++) {
            if (got.items[j] != expected[j]) same = false;
        }

        if (same) {
            passed++;
        } else {
            failed++;
            if (first_failure < 0) {
                first_failure = i;
                printf("  %-28s case %d differs: %zu tokens from us, %zu expected\n",
                       e->name, i, got.len, n_expected);
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

    cases_run += passed + failed;
    cases_passed += passed;
    cases_failed += failed;

    printf("  %-28s %d passed, %d failed\n", e->name, passed, failed);
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(void) {
    printf("Volume comparison against Python tiktoken\n");
    printf("──────────────────────────────────────────────────────────\n");

    char *script = find_script();
    if (script == nullptr) {
        printf("  %-28s [SKIP] tiktoken_reference.py not found\n", "all");
        return EXIT_SKIP;
    }

    // All three patterns the library exposes. Until now only cl100k had ever
    // been compared against anything, and the o200k pattern in this repo was a
    // hand-written approximation that did not match the real one.
    static const Encoding ENCODINGS[] = {
        { "cl100k_base", tiktoken_pattern_cl100k },
        { "o200k_base",  tiktoken_pattern_o200k  },
        { "p50k_base",   tiktoken_pattern_p50k   },
    };

    int ran = 0, failed = 0;
    for (size_t i = 0; i < sizeof(ENCODINGS) / sizeof(ENCODINGS[0]); i++) {
        int r = compare_encoding(&ENCODINGS[i], script);
        if (r == EXIT_SKIP) continue;
        ran++;
        if (r != EXIT_SUCCESS) failed++;
    }

    printf("──────────────────────────────────────────────────────────\n");
    printf("Results: %d passed, %d failed, %d compared across %d encoding(s)\n",
           cases_passed, cases_failed, cases_run, ran);

    if (failed > 0) {
        printf("The generator is seeded, so every difference above reproduces.\n");
        return EXIT_FAILURE;
    }
    if (ran == 0) return EXIT_SKIP;
    return EXIT_SUCCESS;
}
