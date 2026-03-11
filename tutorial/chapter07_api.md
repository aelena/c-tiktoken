# Chapter 7 — The Encoding API

## Putting It All Together

We've built every piece of the tokenizer in isolation. Now we combine
them into a single, coherent API. This chapter is about **integration
design** — how individual modules compose into a system.

## The Encoding Pipeline

When you call `tiktoken_encode(enc, "Hello, world!", ...)`, here's what
happens:

```
Input text: "Hello, world!"
    │
    ▼  1. Special token scan
    Split at special token boundaries (if any)
    │
    ▼  2. Regex pre-tokenization (Chapter 5)
    ["Hello", ",", " world", "!"]
    │
    ▼  3. BPE encoding (Chapter 4) for each chunk
    [15339]  [11]  [1917]  [0]
    │
    ▼  4. Concatenate
    [15339, 11, 1917, 0]
```

In code:

```c
static TokenVec encode_segment(const TiktokenEncoding *enc,
                               const char *text, size_t text_len)
{
    TokenVec result = tokvec_new();

    // 1. Regex splitting
    RegexMatchVec matches = regex_find_all(enc->pattern, text, text_len);

    // 2. BPE encoding for each match
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
```

This is clean and simple because each module has a well-defined
interface. The encoding function doesn't know about hash maps, arenas,
or base64 — it only knows about regex matches and BPE encoding.

## Special Token Handling

Special tokens like `<|endoftext|>` are not processed by BPE. They
have fixed token IDs and are matched literally in the input text.

```c
enum TiktokenSpecialMode : int {
    TIKTOKEN_SPECIAL_DISALLOW = 0,
    TIKTOKEN_SPECIAL_ALLOW,
    TIKTOKEN_SPECIAL_IGNORE,
};
```

### C23 Feature: Fixed-Width Enum (Revisited)

We first saw `enum : type` in Chapter 1. Here it ensures the mode
parameter has a known size regardless of platform:

```c
enum TiktokenSpecialMode : int { ... };
```

This matters for the public API because callers might store the mode
in a struct or pass it across FFI boundaries. With a fixed underlying
type, the size is guaranteed.

### The Special Token Scan

When `TIKTOKEN_SPECIAL_ALLOW` is set, we scan the text for special
tokens before regex splitting:

```c
while (cursor < text_len) {
    // Find the next special token in the remaining text.
    int special_idx = find_next_special(enc, text, text_len,
                                        cursor, &special_pos);

    if (special_idx < 0) {
        // No more special tokens — encode the rest normally.
        encode_segment(enc, text + cursor, text_len - cursor);
        break;
    }

    // Encode the ordinary text before the special token.
    if (special_pos > cursor) {
        encode_segment(enc, text + cursor, special_pos - cursor);
    }

    // Add the special token's ID directly (no BPE).
    tokvec_push(&result, enc->special_tokens[special_idx].token_id);

    cursor = special_pos + enc->special_tokens[special_idx].text_len;
}
```

The scan uses simple `memcmp`-based substring search. With tiktoken's
~5 special tokens, this is fast. For many special tokens, you'd use
Aho-Corasick multi-pattern search.

## Decoding

Decoding is straightforward: look up each token ID and concatenate:

```c
Bytes tiktoken_decode(const TiktokenEncoding *enc,
                      const uint32_t *tokens, size_t n_tokens)
{
    Bytes result = {};
    for (size_t i = 0; i < n_tokens; i++) {
        // Check special tokens first.
        // Then look up in the vocabulary.
        // Append bytes to result.
    }
    return result;
}
```

Note that decoding always succeeds (assuming valid token IDs). The
output is raw bytes — not necessarily valid UTF-8. A full
implementation would optionally validate UTF-8 and handle errors.

## The `TiktokenEncoding` Structure

```c
typedef struct {
    const char      *name;           // "cl100k_base"
    VocabResult      vocab;          // vocabulary + arena
    Regex           *pattern;        // compiled regex
    SpecialToken    *special_tokens; // array of special tokens
    size_t           n_special;      // count of special tokens
} TiktokenEncoding;
```

Ownership is clear:
- `vocab` owns the arena and hash maps (freed by `vocab_free`)
- `pattern` owns the PCRE2 code (freed by `regex_free`)
- `special_tokens` is a `malloc`'d array (freed by `free`)
- `name` is a string literal (not freed)

`tiktoken_free` destroys everything in the right order.

## API Design Principles

### 1. Caller owns the results

```c
TokenVec tokens = tiktoken_encode(enc, text, len, mode);
// ... use tokens ...
tokvec_free(&tokens);
```

The caller allocates nothing upfront and frees the result when done.
This is the simplest ownership model — no ambiguity about who frees
what.

### 2. `[[nodiscard]]` everywhere

Every function that returns an allocated resource is marked
`[[nodiscard]]`. The compiler warns if you ignore the return value,
preventing memory leaks.

### 3. Null-safe

All public functions handle `nullptr` gracefully — they return empty
results rather than crashing. This makes the API robust against
programmer error.

### 4. `size_t` length parameters

Text length is always explicit — no `strlen` inside the library.
This lets callers pass substrings, pre-computed lengths, or text
with embedded nulls.

## Token Counting

A convenience function that encodes and counts without returning the
tokens:

```c
size_t tiktoken_count(const TiktokenEncoding *enc,
                      const char *text, size_t text_len)
{
    TokenVec tokens = tiktoken_encode(enc, text, text_len,
                                      TIKTOKEN_SPECIAL_IGNORE);
    size_t count = tokens.len;
    tokvec_free(&tokens);
    return count;
}
```

This is slightly wasteful (it allocates and frees the token array),
but the API is clean. A production implementation could short-circuit
and count without storing, but for correctness and maintainability,
reusing `tiktoken_encode` is better.

## Testing

The tests use the same tiny vocabulary as Chapter 6, plus a simple
`\S+| ` regex pattern. Test cases cover:

- Single-word encoding
- Text with spaces (regex splitting)
- Decode single and multiple tokens
- Encode→decode roundtrip
- Special token in ALLOW mode
- Special token decoding
- Token counting
- Empty input

## Building

Final `CMakeLists.txt` additions:

```cmake
add_library(tiktoken
    src/base64.c
    src/bytes.c
    src/arena.c
    src/hash.c
    src/bpe.c
    src/regex.c
    src/vocab.c
    src/encoding.c     # NEW
)

add_executable(test_encoding tests/test_encoding.c)
target_link_libraries(test_encoding PRIVATE tiktoken)
add_test(NAME encoding COMMAND test_encoding)
```

## What's Next

In [Chapter 8](chapter08_integration.md), we'll create the convenience
header, preset encoding constructors (like `tiktoken_cl100k_base()`),
example programs, and a complete build guide. We'll also summarize all
the C23 features we've used throughout the project.

## Summary of C23 Features Used

| Feature | Usage |
|---------|-------|
| `enum : int` | `TiktokenSpecialMode` with fixed underlying type |
| `[[nodiscard]]` | All public API functions returning resources |
| `nullptr` | Null checks in all public functions |
| `= {}` | Zero-initializing empty results |

This chapter introduces no new C23 features — it's about composition.
The value is in seeing how the features from Chapters 1–6 work together
in a cohesive API.
