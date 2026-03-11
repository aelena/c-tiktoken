// SPDX-License-Identifier: MIT
//
// c-tiktoken — Example: Encode and Decode Round-Trip
//
// Demonstrates encoding text into tokens, then decoding back to text.
//
// Usage:
//     ./encode_decode <vocab_file> "text to tokenize"

#include <tiktoken/tiktoken.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <vocab_file> \"text\"\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *vocab_path = argv[1];
    const char *text       = argv[2];
    size_t text_len        = strlen(text);

    // Load vocabulary.
    const SpecialToken *special = nullptr;
    size_t n_special = tiktoken_cl100k_special(&special);

    TiktokenEncoding *enc = tiktoken_from_file(
        vocab_path,
        tiktoken_pattern_cl100k(),
        special,
        n_special
    );

    if (enc == nullptr) {
        fprintf(stderr, "Error: failed to load vocabulary from '%s'\n",
                vocab_path);
        return EXIT_FAILURE;
    }

    // Encode.
    printf("── Encoding ──────────────────────────────────\n");
    printf("Input:  \"%s\" (%zu bytes)\n", text, text_len);

    TokenVec tokens = tiktoken_encode_ordinary(enc, text, text_len);
    printf("Tokens: %zu\n", tokens.len);
    printf("IDs:    [");
    for (size_t i = 0; i < tokens.len; i++) {
        if (i > 0) printf(", ");
        printf("%u", tokens.items[i]);
    }
    printf("]\n\n");

    // Show individual token bytes.
    printf("── Token Details ─────────────────────────────\n");
    for (size_t i = 0; i < tokens.len; i++) {
        Bytes tb = {};
        if (i2b_get(&enc->vocab.ranks.decoder, tokens.items[i], &tb)) {
            printf("  [%zu] ID=%u  bytes=", i, tokens.items[i]);
            for (size_t j = 0; j < tb.len; j++) {
                if (tb.data[j] >= 0x20 && tb.data[j] < 0x7F) {
                    printf("%c", (char)tb.data[j]);
                } else {
                    printf("\\x%02x", tb.data[j]);
                }
            }
            printf("\n");
        }
    }
    printf("\n");

    // Decode.
    printf("── Decoding ──────────────────────────────────\n");
    Bytes decoded = tiktoken_decode(enc, tokens.items, tokens.len);
    printf("Output: \"");
    for (size_t i = 0; i < decoded.len; i++) {
        if (decoded.data[i] >= 0x20 && decoded.data[i] < 0x7F) {
            putchar((char)decoded.data[i]);
        } else {
            printf("\\x%02x", decoded.data[i]);
        }
    }
    printf("\" (%zu bytes)\n", decoded.len);

    // Verify round-trip.
    bool match = (decoded.len == text_len &&
                  memcmp(decoded.data, text, text_len) == 0);
    printf("Match:  %s\n", match ? "YES" : "NO");

    bytes_free(&decoded);
    tokvec_free(&tokens);
    tiktoken_free(enc);
    return EXIT_SUCCESS;
}
