// SPDX-License-Identifier: MIT
//
// c-tiktoken — Example: Count Tokens
//
// Usage:
//     ./count_tokens <vocab_file> "text to tokenize"
//
// Example:
//     ./count_tokens data/cl100k_base.tiktoken "Hello, world!"

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

    // Load vocabulary and create encoding.
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

    // Count tokens.
    size_t count = tiktoken_count(enc, text, text_len);
    printf("Text:   \"%s\"\n", text);
    printf("Tokens: %zu\n", count);

    // Also show the actual token IDs.
    TokenVec tokens = tiktoken_encode_ordinary(enc, text, text_len);
    printf("IDs:    [");
    for (size_t i = 0; i < tokens.len; i++) {
        if (i > 0) printf(", ");
        printf("%u", tokens.items[i]);
    }
    printf("]\n");

    tokvec_free(&tokens);
    tiktoken_free(enc);
    return EXIT_SUCCESS;
}
