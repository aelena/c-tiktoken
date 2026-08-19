// SPDX-License-Identifier: MIT
//
// c-tiktoken, Fuzz target: the vocabulary parser
//
// vocab_load_mem is handed the whole contents of a .tiktoken file. In this repo
// that file comes from OpenAI over HTTPS, but the function does not know that,
// and anyone building on this library will eventually point it at a file that
// came from somewhere else. It parses lines, splits on a space, base64-decodes
// the left half, and parses an integer out of the right half. Four chances to
// trust the input.
//
// Build and run:
//
//     cmake -S . -B fz -DCMAKE_C_COMPILER=clang-19 -DTIKTOKEN_FUZZ=ON
//     cmake --build fz
//     ./fz/fuzz_vocab -max_total_time=60 fuzz/corpus/vocab
//
// The properties asserted here are the ones a caller relies on. A load either
// succeeds or it does not, and if it says it succeeded then the two maps agree
// with each other and with the reported size. A parser that half-succeeds is
// worse than one that fails, because the caller carries on.

#include "tiktoken/vocab.h"
#include "tiktoken/hash.h"

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > (1u << 20)) return 0;

    VocabResult v = vocab_load_mem((const char *)data, size);

    if (v.ok) {
        // ok means at least one entry parsed.
        assert(v.ranks.vocab_size > 0);

        // Both directions were populated for every entry that counted, so the
        // three numbers have to agree. They disagreed before this branch: an
        // insertion failure used to increment vocab_size anyway.
        assert(b2i_len(&v.ranks.encoder) == v.ranks.vocab_size);
        assert(i2b_len(&v.ranks.decoder) == v.ranks.vocab_size);
    } else {
        // A failed load must leave nothing behind to look up.
        assert(v.ranks.vocab_size == 0);
    }

    // Freeing must be safe in both cases, and must be safe exactly once.
    vocab_free(&v);

    // vocab_free zeroes the struct, so a second free is a no-op rather than a
    // double free. Worth asserting, because callers do this.
    vocab_free(&v);

    return 0;
}
