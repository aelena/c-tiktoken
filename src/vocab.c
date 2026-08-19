// SPDX-License-Identifier: MIT
//
// c-tiktoken, Chapter 6: Vocabulary Loading (implementation)

#include "tiktoken/vocab.h"
#include "tiktoken/base64.h"
#include "tiktoken/bytes.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── C23 feature note: strdup / strndup ─────────────────────────────────
//
// strdup() and strndup() were POSIX extensions that many C programmers
// used but were technically not part of the C standard. C23 adds them
// to the standard library (<string.h>). We don't use them directly here,
// but it's worth noting: one fewer reason to depend on POSIX headers.

// ── Estimate vocabulary size ──────────────────────────────────────────
//
// Count newlines in the file to estimate the number of entries.
// This lets us pre-size the hash maps and arena to avoid repeated
// reallocations during loading.

static size_t count_lines(const char *data, size_t len) {
    size_t count = 0;
    for (size_t i = 0; i < len; i++) {
        if (data[i] == '\n') count++;
    }
    // If the file doesn't end with a newline, the last line still counts.
    if (len > 0 && data[len - 1] != '\n') count++;
    return count;
}

// ── Parse one line of a .tiktoken file ────────────────────────────────
//
// Format: <base64_token> <rank_integer>\n
//
// Returns true on success, writing the decoded bytes into the arena
// and populating *out_bytes and *out_rank.

static bool parse_line(const char *line, size_t line_len,
                       Arena *arena,
                       Bytes *out_bytes, uint32_t *out_rank)
{
    // Find the space separator.
    const char *space = nullptr;
    for (size_t i = 0; i < line_len; i++) {
        if (line[i] == ' ') {
            space = &line[i];
            break;
        }
    }
    if (space == nullptr) return false;

    size_t b64_len  = (size_t)(space - line);
    const char *rank_str = space + 1;
    size_t rank_len = line_len - b64_len - 1;

    // Strip trailing whitespace from rank string.
    while (rank_len > 0 && (rank_str[rank_len - 1] == '\r'
                         || rank_str[rank_len - 1] == '\n'
                         || rank_str[rank_len - 1] == ' ')) {
        rank_len--;
    }

    if (b64_len == 0 || rank_len == 0) return false;

    // ── Decode base64 token ────────────────────────────────────────

    size_t decode_buf_size = b64_decoded_size(b64_len);
    uint8_t *decode_buf = malloc(decode_buf_size);
    if (decode_buf == nullptr) return false;

    size_t decoded_len = 0;
    enum b64_status status = b64_decode(line, b64_len, decode_buf, &decoded_len);
    if (status != B64_OK) {
        free(decode_buf);
        return false;
    }

    // Copy decoded bytes into the arena (long-term storage).
    uint8_t *arena_data = arena_push_bytes(arena, decode_buf, decoded_len);
    free(decode_buf);

    if (arena_data == nullptr) return false;

    *out_bytes = (Bytes){
        .data = arena_data,
        .len  = decoded_len,
        .cap  = 0,   // arena-owned, don't free individually
    };

    // ── Parse rank integer ─────────────────────────────────────────
    //
    // This used to call strtoul with a null endptr and check errno, on the
    // assumption that errno reports a failed conversion. It does not: strtoul
    // sets errno for ERANGE and for nothing else. Given "q" it returned 0,
    // left errno untouched, and this function reported success with a rank of
    // zero. A fuzzer found that in nineteen executions, using a two-line file
    // whose second rank was the letter q, and the consequence was two tokens
    // claiming rank 0.
    //
    // Every character has to be a digit. That is stricter than strtoul and it
    // is exactly what the file format allows: no sign, no whitespace, no hex.

    if (rank_len >= 32) return false;
    for (size_t i = 0; i < rank_len; i++) {
        if (rank_str[i] < '0' || rank_str[i] > '9') return false;
    }

    char rank_buf[32];
    memcpy(rank_buf, rank_str, rank_len);
    rank_buf[rank_len] = '\0';

    errno = 0;
    char *end = nullptr;
    unsigned long rank_val = strtoul(rank_buf, &end, 10);
    if (errno != 0 || end != rank_buf + rank_len || rank_val > UINT32_MAX) {
        return false;
    }

    *out_rank = (uint32_t)rank_val;
    return true;
}

// ── Load from memory buffer ───────────────────────────────────────────

VocabResult vocab_load_mem(const char *data, size_t data_len) {
    VocabResult result = { .ok = false };

    if (data == nullptr || data_len == 0) {
        return result;
    }

    size_t estimated_entries = count_lines(data, data_len);

    // Pre-size data structures.
    // Arena: ~20 bytes average per token × entry count.
    result.arena  = arena_new(estimated_entries * 20);
    result.ranks.encoder    = b2i_new(estimated_entries * 2);
    result.ranks.decoder    = i2b_new(estimated_entries * 2);
    result.ranks.vocab_size = 0;

    // Parse line by line.
    const char *cursor = data;
    const char *end    = data + data_len;

    while (cursor < end) {
        // Find end of line.
        const char *eol = cursor;
        while (eol < end && *eol != '\n') eol++;

        size_t line_len = (size_t)(eol - cursor);

        // Skip empty lines.
        if (line_len > 0 && !(line_len == 1 && cursor[0] == '\r')) {
            Bytes token_bytes = {};
            uint32_t rank = 0;

            if (parse_line(cursor, line_len, &result.arena,
                           &token_bytes, &rank)) {
                // Both directions or neither. A vocabulary missing entries
                // still loads and then mistokenises every input that touches
                // the gap, silently and forever, so an insertion failure
                // has to fail the whole load, not one line of it.
                size_t before_enc = b2i_len(&result.ranks.encoder);
                size_t before_dec = i2b_len(&result.ranks.decoder);

                if (!b2i_insert(&result.ranks.encoder, token_bytes, rank)
                    || !i2b_insert(&result.ranks.decoder, rank, token_bytes)) {
                    vocab_free(&result);
                    return result;
                }

                // Both maps overwrite on a duplicate key rather than growing,
                // so a repeated rank or a repeated token used to leave the two
                // maps and vocab_size disagreeing, with ok still true. The
                // decoder would have lost an entry and nothing would say so.
                // A duplicate means the file is malformed, and a malformed
                // vocabulary has to fail rather than load at the wrong size.
                if (b2i_len(&result.ranks.encoder) == before_enc
                    || i2b_len(&result.ranks.decoder) == before_dec) {
                    vocab_free(&result);
                    return result;
                }

                result.ranks.vocab_size++;
            }
        }

        cursor = (eol < end) ? eol + 1 : end;
    }

    result.ok = (result.ranks.vocab_size > 0);
    return result;
}

// ── Load from file ────────────────────────────────────────────────────

VocabResult vocab_load_file(const char *path) {
    VocabResult result = { .ok = false };

    if (path == nullptr) return result;

    FILE *f = fopen(path, "rb");
    if (f == nullptr) return result;

    // Get file size.
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        return result;
    }

    // Validate file size fits in size_t and is reasonable.
    if ((unsigned long)file_size > SIZE_MAX) {
        fclose(f);
        return result;
    }

    size_t file_size_t = (size_t)file_size;

    // Read entire file into memory.
    char *buf = malloc(file_size_t);
    if (buf == nullptr) {
        fclose(f);
        return result;
    }

    size_t read_size = fread(buf, 1, file_size_t, f);
    fclose(f);

    // Verify we read the expected amount.
    if (read_size != file_size_t) {
        free(buf);
        return result;
    }

    // Parse from memory.
    result = vocab_load_mem(buf, read_size);
    free(buf);

    return result;
}

// ── Cleanup ───────────────────────────────────────────────────────────

void vocab_free(VocabResult *v) {
    if (v == nullptr) return;
    b2i_free(&v->ranks.encoder);
    i2b_free(&v->ranks.decoder);
    arena_free(&v->arena);
    *v = (VocabResult){};
}
