// SPDX-License-Identifier: MIT
//
// c-tiktoken — Chapter 1: Base64 Decoding (implementation)

#include "tiktoken/base64.h"

// ── C23 feature: bool is a keyword ─────────────────────────────────────
//
// In C23, `bool`, `true`, and `false` are keywords — no need to include
// <stdbool.h>. This is a small but welcome change that brings C in line
// with C++ and eliminates a common source of confusion for beginners.

// ── C23 feature: nullptr ───────────────────────────────────────────────
//
// C23 introduces `nullptr` as the canonical null-pointer constant, with
// its own type `nullptr_t`. Unlike the traditional `NULL` (which is
// typically `(void *)0` or just `0`), `nullptr` is unambiguously a
// pointer — it won't silently convert to an integer. We use it throughout.

// ── C23 feature: constexpr ─────────────────────────────────────────────
//
// `constexpr` declares an object whose value is a compile-time constant.
// Unlike `const` (which merely means "I promise not to modify this"),
// `constexpr` guarantees the value is known at compile time. The compiler
// can use this for optimizations and can place the data in read-only
// memory with no runtime initialization.
//
// Here we use it for the base64 decode lookup table: 256 entries mapping
// each possible byte value to its 6-bit decoded value (or 0xFF for
// invalid characters). This table is fully determined at compile time.

// Sentinel value for bytes that are not valid base64 characters.
constexpr uint8_t B64_INV = 0xFF;

// The base64 alphabet: A-Z a-z 0-9 + /
// Index 0x00–0xFF → 6-bit value or B64_INV.
//
// We lay this out as a full 256-entry table so that decoding is a single
// array lookup per character — no branches, no range checks.
//
// C23 feature: constexpr on an array.
// The entire table is a compile-time constant. In C17 you would use
// `static const`, which is "const at runtime" but not necessarily
// evaluated at compile time. `constexpr` is stronger: the compiler
// must be able to compute every element during compilation.

static constexpr uint8_t DECODE_TABLE[256] = {
    // 0x00–0x0F: control characters
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    // 0x10–0x1F
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    // 0x20–0x2F:  space ! " # $ % & ' ( ) * + , - . /
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV,      62, B64_INV, B64_INV, B64_INV,      63,
    // 0x30–0x3F:  0-9 : ; < = > ?
         52,      53,      54,      55,      56,      57,      58,      59,
         60,      61, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    // 0x40–0x4F:  @ A-O
    B64_INV,       0,       1,       2,       3,       4,       5,       6,
          7,       8,       9,      10,      11,      12,      13,      14,
    // 0x50–0x5F:  P-Z [ \ ] ^ _
         15,      16,      17,      18,      19,      20,      21,      22,
         23,      24,      25, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    // 0x60–0x6F:  ` a-o
    B64_INV,      26,      27,      28,      29,      30,      31,      32,
         33,      34,      35,      36,      37,      38,      39,      40,
    // 0x70–0x7F:  p-z { | } ~ DEL
         41,      42,      43,      44,      45,      46,      47,      48,
         49,      50,      51, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    // 0x80–0xFF: high bytes — all invalid in base64
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
    B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV, B64_INV,
};

// ── C23 feature: static_assert without a message ──────────────────────
//
// C11 required: static_assert(expr, "message");
// C23 allows:   static_assert(expr);
//
// The message was always awkward — what do you say that the expression
// doesn't already say? C23 makes it optional.

static_assert(sizeof(DECODE_TABLE) == 256);

size_t b64_decoded_size(size_t input_len) {
    // Every 4 base64 characters encode 3 bytes.
    // This is an upper-bound: padding ('=') reduces the actual count,
    // but we use this only for buffer sizing so over-estimating is safe.
    return (input_len / 4) * 3 + 3;
}

// Internal helper: decode one base64 character via table lookup.
// Returns B64_INV (0xFF) for invalid characters.
static inline uint8_t b64_char_value(unsigned char c) {
    return DECODE_TABLE[c];
}

enum b64_status b64_decode(const char    *input,
                           size_t         input_len,
                           uint8_t       *output,
                           size_t        *output_len)
{
    // Null-pointer checks using nullptr.
    if (input == nullptr || output == nullptr || output_len == nullptr) {
        return B64_INVALID_LENGTH;
    }

    // Strip trailing whitespace and padding so we can handle both padded
    // and unpadded base64 (tiktoken files use standard padded base64, but
    // being lenient costs us nothing).
    while (input_len > 0 && (input[input_len - 1] == '='
                          || input[input_len - 1] == '\n'
                          || input[input_len - 1] == '\r'
                          || input[input_len - 1] == ' ')) {
        input_len--;
    }

    // Base64 works in groups of 4 characters → 3 bytes.
    // A truncated final group of 2 chars → 1 byte, 3 chars → 2 bytes.
    // A final group of 1 char is invalid (only 6 bits — less than a byte).
    size_t full_quads  = input_len / 4;
    size_t remainder   = input_len % 4;

    if (remainder == 1) {
        return B64_INVALID_LENGTH;
    }

    size_t out_pos = 0;

    // ── Decode full 4-character groups ──────────────────────────────
    //
    // Each group of 4 base64 characters encodes 3 bytes:
    //
    //   +--------+--------+--------+--------+
    //   | char 0 | char 1 | char 2 | char 3 |   4 × 6 = 24 bits
    //   +--------+--------+--------+--------+
    //         ↓        ↓        ↓
    //   +----------+----------+----------+
    //   |  byte 0  |  byte 1  |  byte 2  |       3 × 8 = 24 bits
    //   +----------+----------+----------+
    //
    // We combine the four 6-bit values into a 24-bit integer, then
    // extract 3 bytes from it.

    for (size_t i = 0; i < full_quads; i++) {
        size_t base = i * 4;

        uint8_t a = b64_char_value((unsigned char)input[base + 0]);
        uint8_t b = b64_char_value((unsigned char)input[base + 1]);
        uint8_t c = b64_char_value((unsigned char)input[base + 2]);
        uint8_t d = b64_char_value((unsigned char)input[base + 3]);

        if (a == B64_INV || b == B64_INV || c == B64_INV || d == B64_INV) {
            return B64_INVALID_CHAR;
        }

        // Pack four 6-bit values into a 24-bit integer.
        uint32_t triple = ((uint32_t)a << 18)
                        | ((uint32_t)b << 12)
                        | ((uint32_t)c <<  6)
                        | ((uint32_t)d <<  0);

        output[out_pos++] = (uint8_t)(triple >> 16);
        output[out_pos++] = (uint8_t)(triple >>  8);
        output[out_pos++] = (uint8_t)(triple >>  0);
    }

    // ── Decode the remainder (0, 2, or 3 trailing characters) ──────
    //
    // 2 chars → 12 bits → 1 output byte (top 8 bits; bottom 4 must be 0)
    // 3 chars → 18 bits → 2 output bytes (top 16 bits; bottom 2 must be 0)

    if (remainder == 2) {
        size_t base = full_quads * 4;

        uint8_t a = b64_char_value((unsigned char)input[base + 0]);
        uint8_t b = b64_char_value((unsigned char)input[base + 1]);

        if (a == B64_INV || b == B64_INV) {
            return B64_INVALID_CHAR;
        }

        uint32_t pair = ((uint32_t)a << 6) | (uint32_t)b;
        output[out_pos++] = (uint8_t)(pair >> 4);
    }
    else if (remainder == 3) {
        size_t base = full_quads * 4;

        uint8_t a = b64_char_value((unsigned char)input[base + 0]);
        uint8_t b = b64_char_value((unsigned char)input[base + 1]);
        uint8_t c = b64_char_value((unsigned char)input[base + 2]);

        if (a == B64_INV || b == B64_INV || c == B64_INV) {
            return B64_INVALID_CHAR;
        }

        uint32_t triple = ((uint32_t)a << 12)
                        | ((uint32_t)b <<  6)
                        | ((uint32_t)c <<  0);

        output[out_pos++] = (uint8_t)(triple >> 10);
        output[out_pos++] = (uint8_t)(triple >>  2);
    }

    *output_len = out_pos;
    return B64_OK;
}
