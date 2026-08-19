// SPDX-License-Identifier: MIT
//
// c-tiktoken, Fuzz target: the base64 decoder
//
// b64_decode takes bytes from a file nobody in this project wrote. Every line of
// a .tiktoken vocabulary is attacker-controlled as far as this decoder is
// concerned, and the decoder is the first thing that touches them.
//
// Build and run:
//
//     cmake -S . -B fz -DCMAKE_C_COMPILER=clang-19 -DTIKTOKEN_FUZZ=ON
//     cmake --build fz
//     ./fz/fuzz_base64 -max_total_time=60 fuzz/corpus/base64
//
// The interesting property is not "does it crash on valid input". It is that for
// *any* input, valid or not, the decoder must either refuse or write no more than
// b64_decoded_size(len) bytes. The assert below is what makes a violation a
// crash rather than a silent overflow, and ASan catches the overflow itself.

#include "tiktoken/base64.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > (1u << 20)) return 0;  // a megabyte of base64 proves nothing more

    size_t cap = b64_decoded_size(size);

    // A zero-length allocation is implementation-defined, so keep one spare byte
    // and let ASan guard the boundary either way.
    uint8_t *out = malloc(cap + 1);
    if (out == nullptr) return 0;

    // Poison the buffer so a decoder that reports more bytes than it wrote is
    // visible downstream rather than reading zeros that happen to look plausible.
    memset(out, 0xA5, cap + 1);

    size_t written = SIZE_MAX;
    enum b64_status st = b64_decode((const char *)data, size, out, &written);

    if (st == B64_OK) {
        // The contract is that written never exceeds the advertised upper bound.
        assert(written <= cap);
        // And that the decoder set it at all.
        assert(written != SIZE_MAX);
    }

    free(out);
    return 0;
}
