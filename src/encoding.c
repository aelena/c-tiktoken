// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 7: High-Level Encoding API (implementation)

#include "tiktoken/encoding.h"

#include <stdlib.h>
#include <string.h>

// ── Lifecycle ──────────────────────────────────────────────────────────

TiktokenEncoding *tiktoken_new(const char       *name,
                               VocabResult       vocab,
                               Regex            *pattern,
                               SpecialToken     *special_tokens,
                               size_t            n_special)
{
    TiktokenEncoding *enc = malloc(sizeof(TiktokenEncoding));
    if (enc == nullptr) return nullptr;

    enc->name           = name;
    enc->vocab          = vocab;
    enc->pattern        = pattern;
    enc->special_tokens = special_tokens;
    enc->n_special      = n_special;

    return enc;
}

void tiktoken_free(TiktokenEncoding *enc) {
    if (enc == nullptr) return;
    vocab_free(&enc->vocab);
    regex_free(enc->pattern);
    free(enc->special_tokens);
    free(enc);
}

// ── Special token search ──────────────────────────────────────────────
//
// Scan the text for the earliest occurrence of any special token.
// Returns the index into the special_tokens array, or -1 if none found.
// Sets *pos to the position in the text where the special token starts.

static int find_next_special(const TiktokenEncoding *enc,
                             const char *text, size_t text_len,
                             size_t start, size_t *pos)
{
    int    best_idx = -1;
    size_t best_pos = text_len;

    for (size_t i = 0; i < enc->n_special; i++) {
        const char *needle = enc->special_tokens[i].text;
        size_t needle_len  = enc->special_tokens[i].text_len;

        if (needle_len == 0 || needle_len > text_len - start) continue;

        // Simple substring search. For a production tokenizer with many
        // special tokens, you'd use Aho-Corasick. For tiktoken's ~5
        // special tokens, linear scan is fine.
        for (size_t j = start; j + needle_len <= text_len; j++) {
            if (memcmp(text + j, needle, needle_len) == 0) {
                if (j < best_pos) {
                    best_pos = j;
                    best_idx = (int)i;
                }
                break;  // found earliest occurrence of this token
            }
        }
    }

    if (best_idx >= 0) {
        *pos = best_pos;
    }
    return best_idx;
}

// ── Encode a segment (no special tokens) ──────────────────────────────
//
// This is the core encoding pipeline for a segment of ordinary text:
//   1. Split with regex into chunks
//   2. BPE-encode each chunk
//   3. Collect all token IDs

static TokenVec encode_segment(const TiktokenEncoding *enc,
                               const char *text, size_t text_len)
{
    TokenVec result = tokvec_new();

    if (text_len == 0) return result;

    // Step 1: regex splitting.
    RegexMatchVec matches = regex_find_all(enc->pattern, text, text_len);

    // Step 2 & 3: BPE-encode each match and collect tokens.
    for (size_t i = 0; i < matches.len; i++) {
        const uint8_t *chunk = (const uint8_t *)(text + matches.items[i].start);
        size_t chunk_len = matches.items[i].len;

        TokenVec chunk_tokens = bpe_encode(&enc->vocab.ranks,
                                           chunk, chunk_len);

        tokvec_extend(&result, chunk_tokens.items, chunk_tokens.len);
        tokvec_free(&chunk_tokens);
    }

    regexmatchvec_free(&matches);
    return result;
}

// ── Public API ─────────────────────────────────────────────────────────

TokenVec tiktoken_encode(const TiktokenEncoding *enc,
                         const char             *text,
                         size_t                  text_len,
                         enum TiktokenSpecialMode mode)
{
    TokenVec result = tokvec_new();

    if (enc == nullptr || text == nullptr || text_len == 0) {
        return result;
    }

    // ── IGNORE mode: treat everything as ordinary text ─────────────
    if (mode == TIKTOKEN_SPECIAL_IGNORE || enc->n_special == 0) {
        tokvec_free(&result);
        return encode_segment(enc, text, text_len);
    }

    // ── ALLOW mode: scan for special tokens and handle them ────────
    //
    // Walk through the text. At each position, check if a special token
    // starts here. If so, encode the ordinary text before it, then add
    // the special token's ID directly.

    size_t cursor = 0;

    while (cursor < text_len) {
        size_t special_pos = 0;
        int special_idx = find_next_special(enc, text, text_len,
                                            cursor, &special_pos);

        if (special_idx < 0) {
            // No more special tokens — encode the rest as ordinary.
            TokenVec seg = encode_segment(enc, text + cursor,
                                          text_len - cursor);
            tokvec_extend(&result, seg.items, seg.len);
            tokvec_free(&seg);
            break;
        }

        if (mode == TIKTOKEN_SPECIAL_DISALLOW) {
            // Special token found but disallowed — for now, we treat
            // DISALLOW the same as ALLOW. A stricter implementation
            // would return an error here.
        }

        // Encode ordinary text before the special token.
        if (special_pos > cursor) {
            TokenVec seg = encode_segment(enc, text + cursor,
                                          special_pos - cursor);
            tokvec_extend(&result, seg.items, seg.len);
            tokvec_free(&seg);
        }

        // Add the special token directly.
        tokvec_push(&result, enc->special_tokens[special_idx].token_id);

        cursor = special_pos + enc->special_tokens[special_idx].text_len;
    }

    return result;
}

TokenVec tiktoken_encode_ordinary(const TiktokenEncoding *enc,
                                  const char             *text,
                                  size_t                  text_len)
{
    return tiktoken_encode(enc, text, text_len, TIKTOKEN_SPECIAL_IGNORE);
}

Bytes tiktoken_decode(const TiktokenEncoding *enc,
                      const uint32_t         *tokens,
                      size_t                  n_tokens)
{
    if (enc == nullptr || tokens == nullptr || n_tokens == 0) {
        return (Bytes){};
    }

    Bytes result = {};

    for (size_t i = 0; i < n_tokens; i++) {
        // Check special tokens first.
        bool found_special = false;
        for (size_t j = 0; j < enc->n_special; j++) {
            if (enc->special_tokens[j].token_id == tokens[i]) {
                bytes_append(&result,
                             (const uint8_t *)enc->special_tokens[j].text,
                             enc->special_tokens[j].text_len);
                found_special = true;
                break;
            }
        }

        if (!found_special) {
            Bytes token_bytes = {};
            if (i2b_get(&enc->vocab.ranks.decoder, tokens[i], &token_bytes)) {
                bytes_append(&result, token_bytes.data, token_bytes.len);
            }
        }
    }

    return result;
}

size_t tiktoken_count(const TiktokenEncoding *enc,
                      const char             *text,
                      size_t                  text_len)
{
    TokenVec tokens = tiktoken_encode(enc, text, text_len,
                                      TIKTOKEN_SPECIAL_IGNORE);
    size_t count = tokens.len;
    tokvec_free(&tokens);
    return count;
}
