// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 1: Base64 Decoding
//
// Decodes base64-encoded strings into raw bytes. Tiktoken vocabulary files
// store each token as a base64-encoded byte sequence, so this is the first
// building block we need.

#ifndef TIKTOKEN_BASE64_H
#define TIKTOKEN_BASE64_H

#include <stddef.h>
#include <stdint.h>

// ── C23 feature: fixed-width enum ──────────────────────────────────────
//
// C23 lets us specify the underlying integer type of an enum with `: type`.
// This guarantees the size and signedness — no more surprises about whether
// your enum is an int, unsigned, or something else. It also makes the ABI
// explicit, which matters for libraries.

enum b64_status : int {
    B64_OK              =  0,
    B64_INVALID_CHAR    = -1,
    B64_INVALID_LENGTH  = -2,
};

// ── C23 feature: [[nodiscard]] ─────────────────────────────────────────
//
// The [[nodiscard]] attribute tells the compiler to warn if the caller
// ignores the return value. We use it on functions whose return value
// indicates success or failure — silently ignoring an error is a bug.

// Returns the number of bytes that `input_len` bytes of base64 will
// decode into (upper bound — ignores padding for simplicity).
[[nodiscard]]
size_t b64_decoded_size(size_t input_len);

// Decodes `input_len` bytes of base64 from `input` into `output`.
// `*output_len` is set to the number of bytes actually written.
// `output` must have room for at least b64_decoded_size(input_len) bytes.
//
// Returns B64_OK on success, or a negative error code.
[[nodiscard]]
enum b64_status b64_decode(const char    *input,
                           size_t         input_len,
                           uint8_t       *output,
                           size_t        *output_len);

#endif // TIKTOKEN_BASE64_H
