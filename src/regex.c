// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 5: Regex Pre-tokenization (implementation)
//
// Wraps PCRE2 for Unicode-aware regex matching.

// ── PCRE2 configuration ───────────────────────────────────────────────
//
// PCRE2 is a code-unit-width library: you must define the code unit
// size before including the header. We use 8-bit (UTF-8).
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "tiktoken/regex.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

// ── The Regex struct ──────────────────────────────────────────────────
//
// We hide the PCRE2 types behind an opaque struct so that callers don't
// need to include pcre2.h. This is a common C pattern for encapsulation:
// the header declares `typedef struct Regex Regex;` (incomplete type),
// and only this .c file knows what's inside.

struct Regex {
    pcre2_code       *code;
    pcre2_match_data *match_data;
};

Regex *regex_compile(const char *pattern) {
    if (pattern == nullptr) return nullptr;

    int errcode = 0;
    PCRE2_SIZE erroffset = 0;

    // ── PCRE2 compilation flags ────────────────────────────────────
    //
    // PCRE2_UTF      — interpret the pattern and subjects as UTF-8.
    // PCRE2_UCP      — use Unicode properties for \w, \d, \s, and
    //                   crucially for \p{L} (Unicode letters) and
    //                   \p{N} (Unicode numbers).
    // PCRE2_CASELESS — the (?i:...) group in the pattern enables
    //                   case-insensitive matching for contractions,
    //                   but we set it at the group level in the
    //                   pattern itself, not globally here.

    pcre2_code *code = pcre2_compile(
        (PCRE2_SPTR)pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errcode,
        &erroffset,
        nullptr      // default compile context
    );

    if (code == nullptr) {
        // Compilation failed. In a production library you'd report the
        // error. For the tutorial, we just return nullptr.
        return nullptr;
    }

    // JIT-compile for speed. This is optional — if JIT isn't available,
    // PCRE2 falls back to the interpreter.
    pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);

    // Create reusable match data (sized for the pattern's capture count).
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(
        code, nullptr
    );

    if (match_data == nullptr) {
        pcre2_code_free(code);
        return nullptr;
    }

    Regex *re = malloc(sizeof(Regex));
    if (re == nullptr) {
        pcre2_match_data_free(match_data);
        pcre2_code_free(code);
        return nullptr;
    }

    re->code       = code;
    re->match_data = match_data;
    return re;
}

void regex_free(Regex *re) {
    if (re == nullptr) return;
    pcre2_match_data_free(re->match_data);
    pcre2_code_free(re->code);
    free(re);
}

// ── Dynamic match vector ──────────────────────────────────────────────

static bool matchvec_push(RegexMatchVec *v, RegexMatch m) {
    if (v->len >= v->cap) {
        size_t new_cap;
        if (v->cap == 0) {
            new_cap = 64;
        } else if (v->cap > SIZE_MAX / 2) {
            // Can't double without overflow
            return false;
        } else {
            new_cap = v->cap * 2;
        }
        RegexMatch *new_items = realloc(v->items, new_cap * sizeof(RegexMatch));
        if (new_items == nullptr) return false;
        v->items = new_items;
        v->cap   = new_cap;
    }
    v->items[v->len++] = m;
    return true;
}

void regexmatchvec_free(RegexMatchVec *v) {
    if (v == nullptr) return;
    free(v->items);
    *v = (RegexMatchVec){};
}

RegexMatchVec regex_find_all(const Regex *re,
                             const char  *text,
                             size_t       text_len)
{
    RegexMatchVec result = {};

    if (re == nullptr || text == nullptr) {
        return result;
    }

    PCRE2_SIZE offset = 0;

    while (offset < (PCRE2_SIZE)text_len) {
        // ── Match attempt ──────────────────────────────────────────
        //
        // pcre2_match returns the number of capture groups matched
        // (including group 0 = the whole match), or a negative error
        // code. PCRE2_ERROR_NOMATCH means no more matches.

        int rc = pcre2_match(
            re->code,
            (PCRE2_SPTR)text,
            (PCRE2_SIZE)text_len,
            offset,
            0,                    // no special match options
            re->match_data,
            nullptr               // default match context
        );

        if (rc < 0) {
            break;  // no more matches (or error)
        }

        // Get the match boundaries.
        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(re->match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end   = ovector[1];

        // Guard against zero-length matches to avoid infinite loops.
        if (match_end == match_start) {
            offset = match_end + 1;
            continue;
        }

        matchvec_push(&result, (RegexMatch){
            .start = (size_t)match_start,
            .len   = (size_t)(match_end - match_start),
        });

        offset = match_end;
    }

    return result;
}

// ── Predefined patterns ───────────────────────────────────────────────
//
// These are the exact regex patterns used by tiktoken's Python
// implementation. They look intimidating, but each alternative handles
// a specific category of text chunk:
//
// For cl100k_base (GPT-4):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)  — English contractions
//   [^\r\n\p{L}\p{N}]?\p{L}+      — optional non-letter/digit + letters
//   \p{N}{1,3}                      — 1–3 digit numbers
//    ?[^\s\p{L}\p{N}]++[\r\n]*     — punctuation with trailing newlines
//   \s*[\r\n]                       — whitespace ending in newline
//   \s+(?!\S)                       — trailing whitespace (not before non-ws)
//   \s+                             — remaining whitespace

const char *tiktoken_pattern_cl100k(void) {
    return
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
        "|\\p{N}{1,3}"
        "| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*"
        "|\\s*[\\r\\n]"
        "|\\s+(?!\\S)"
        "|\\s+";
}

const char *tiktoken_pattern_o200k(void) {
    // o200k_base uses a more complex pattern with additional Unicode
    // handling. This is a simplified version that matches the key
    // behavior.
    return
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
        "|\\p{N}{1,3}"
        "| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*"
        "|\\s*[\\r\\n]"
        "|\\s+(?!\\S)"
        "|\\s+";
}

const char *tiktoken_pattern_p50k(void) {
    return
        "'s|'t|'re|'ve|'m|'ll|'d"
        "| ?\\p{L}+"
        "| ?\\p{N}+"
        "| ?[^\\s\\p{L}\\p{N}]+"
        "|\\s+(?=[\\S])"
        "|\\s+";
}
