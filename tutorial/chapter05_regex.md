# Chapter 5 — Regex Pre-tokenization

## Why Pre-tokenize?

If we fed the entire input text directly to BPE, every possible byte
sequence would be a candidate for merging. The tokenizer would need a
vocabulary entry for every word, every word-with-punctuation, every
word-with-space, and so on. The vocabulary would be enormous.

Instead, tiktoken first **splits** the text into smaller chunks using a
regex pattern. Each chunk is then independently encoded with BPE. This
has three benefits:

1. **Smaller vocabulary.** Tokens don't cross chunk boundaries, so the
   vocabulary only needs entries for within-chunk byte sequences.

2. **Better behavior.** Splitting at word boundaries prevents the
   tokenizer from creating tokens that span words (like "hello w"
   becoming a single token), which would hurt the language model.

3. **Consistent behavior.** The regex ensures that spaces, punctuation,
   and numbers are always handled the same way.

## The cl100k_base Pattern

GPT-4 uses the cl100k_base encoding with this regex pattern:

```regex
(?i:'s|'t|'re|'ve|'m|'ll|'d)
|[^\r\n\p{L}\p{N}]?\p{L}+
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]++[\r\n]*
|\s*[\r\n]
|\s+(?!\S)
|\s+
```

Each alternative (separated by `|`) matches a different type of text
chunk. Let's break them down:

### `(?i:'s|'t|'re|'ve|'m|'ll|'d)` — Contractions

The `(?i:...)` group enables case-insensitive matching. This captures
English contractions: "don**'t**", "I**'m**", "they**'re**", etc.

By matching contractions first, we prevent them from being split in
unexpected ways. Without this, "don't" might become ["don", "'", "t"].

### `[^\r\n\p{L}\p{N}]?\p{L}+` — Words

`\p{L}` matches any Unicode letter (Latin, Cyrillic, Chinese, etc.).
`\p{N}` matches any Unicode number.

This matches sequences of letters, optionally preceded by a single
non-letter, non-number character (like a space or punctuation). The
effect is that words get their leading space attached: `" hello"` is
one chunk, not `" "` + `"hello"`.

### `\p{N}{1,3}` — Numbers

Matches 1 to 3 digits at a time. This means `12345` becomes `["123",
"45"]` — numbers are tokenized in chunks of at most 3 digits. This
prevents very long numbers from becoming single tokens.

### ` ?[^\s\p{L}\p{N}]++[\r\n]*` — Punctuation

Matches punctuation sequences (optionally preceded by a space), with
any trailing newlines. The `++` is a **possessive quantifier** — it
doesn't backtrack. This is important for performance and is why we
need PCRE2 (standard POSIX regex doesn't support possessive quantifiers).

### `\s*[\r\n]` — Newlines

Matches whitespace ending in a newline. This keeps newlines attached
to preceding whitespace.

### `\s+(?!\S)` and `\s+` — Whitespace

The first matches trailing whitespace (whitespace not followed by a
non-whitespace character). The second matches any remaining whitespace.

## Why PCRE2?

We need PCRE2 for three features that POSIX regex doesn't support:

1. **`\p{L}` and `\p{N}`** — Unicode property classes. POSIX has
   `[:alpha:]` but it's locale-dependent and doesn't reliably cover
   all Unicode.

2. **`++`** — Possessive quantifiers. Without them, the regex engine
   may backtrack exponentially on certain inputs.

3. **`(?!...)`** — Negative lookahead. The `\s+(?!\S)` construct
   matches whitespace only when it's not followed by non-whitespace.

PCRE2 is the *only* external dependency in the entire project. It's
mature, fast, widely available, and is the de facto standard for
"real" regex in C.

## The Opaque Type Pattern

Our header declares:

```c
typedef struct Regex Regex;   // incomplete (opaque) type
```

And only the `.c` file defines what's inside:

```c
struct Regex {
    pcre2_code       *code;
    pcre2_match_data *match_data;
};
```

This is C's version of encapsulation. Callers can only interact with
`Regex` through pointers and the public functions — they can't access
the PCRE2 internals. Benefits:

- **Callers don't need `#include <pcre2.h>`** — they're isolated from
  the dependency.
- **We can change the implementation** (e.g., switch to a different
  regex library) without breaking any callers.
- **Compile times are better** — changes to the regex internals don't
  trigger recompilation of files that include `regex.h`.

## PCRE2 Integration

### Compilation

```c
pcre2_code *code = pcre2_compile(
    (PCRE2_SPTR)pattern,
    PCRE2_ZERO_TERMINATED,
    PCRE2_UTF | PCRE2_UCP,     // UTF-8 mode + Unicode properties
    &errcode,
    &erroffset,
    NULL
);
```

The key flags:
- **`PCRE2_UTF`** — treat the pattern and subjects as UTF-8 (not raw
  bytes). This means `.` matches a Unicode code point, not a single byte.
- **`PCRE2_UCP`** — use Unicode properties for `\w`, `\d`, `\s` and
  (crucially) `\p{L}`, `\p{N}`.

### JIT Compilation

```c
pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);
```

PCRE2 can JIT-compile patterns into native machine code for a 2–10x
speedup. This is optional — if the platform doesn't support JIT (e.g.,
some embedded systems), PCRE2 silently falls back to the interpreter.

For tiktoken, JIT compilation is worthwhile because we'll match the
same pattern against thousands of text segments.

### The Match Loop

```c
while (offset < text_len) {
    int rc = pcre2_match(code, text, text_len, offset, 0,
                         match_data, NULL);
    if (rc < 0) break;

    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    size_t match_start = ovector[0];
    size_t match_end   = ovector[1];

    // Store the match...

    offset = match_end;    // advance past this match
}
```

PCRE2 returns match boundaries in an "ovector" (output vector). For
group 0 (the whole match), `ovector[0]` is the start and `ovector[1]`
is the end (exclusive).

### Zero-Length Match Protection

```c
if (match_end == match_start) {
    offset = match_end + 1;
    continue;
}
```

Some regex patterns can match zero characters (e.g., `^` or `\b`).
Without this guard, the match loop would spin forever at the same
position. We skip ahead by one byte when we encounter a zero-length
match.

## Memory: Zero-Copy Match Results

The `RegexMatch` struct stores `(start, len)` pairs — offsets into the
original input string:

```c
typedef struct {
    size_t start;
    size_t len;
} RegexMatch;
```

No strings are copied. The match results are valid as long as the
original input text is alive. This is the same ownership pattern we
used for `Bytes` slices in Chapter 2 — non-owning views into existing
data.

## Testing Without PCRE2

If PCRE2 isn't installed, the tests won't compile. The tutorial notes
how to install PCRE2 on common platforms:

```bash
# Ubuntu/Debian
sudo apt install libpcre2-dev

# macOS
brew install pcre2

# Windows (vcpkg)
vcpkg install pcre2

# Windows (MSYS2)
pacman -S mingw-w64-x86_64-pcre2
```

## Building

Updated `CMakeLists.txt`:

```cmake
add_library(tiktoken
    src/base64.c
    src/bytes.c
    src/arena.c
    src/hash.c
    src/bpe.c
    src/regex.c        # NEW
)

# PCRE2 dependency
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
target_link_libraries(tiktoken PUBLIC PkgConfig::PCRE2)

add_executable(test_regex tests/test_regex.c)
target_link_libraries(test_regex PRIVATE tiktoken)
add_test(NAME regex COMMAND test_regex)
```

## What's Next

We can now split text into chunks. In [Chapter 6](chapter06_vocab.md),
we'll load a real tiktoken vocabulary file — combining base64 decoding
(Chapter 1), hash maps (Chapter 3), and the arena allocator (Chapter 3)
to build the `BpeRanks` structure that the BPE algorithm (Chapter 4)
needs.

## Summary of C23 Features Discussed

This chapter is primarily about PCRE2 integration rather than new C23
features. However, we use several C23 features introduced in previous
chapters:

| Feature | Usage in This Chapter |
|---------|----------------------|
| `nullptr` | Null checks throughout, returned on failure |
| `[[nodiscard]]` | On all functions returning allocated resources |
| `= {}` empty init | Zero-initializing `RegexMatchVec` |
| Opaque types | `typedef struct Regex Regex` — C's encapsulation |

The opaque type pattern isn't a C23 feature per se (it works in any C
version), but it's an important C idiom worth highlighting. C23 doesn't
add access modifiers like C++, so opaque types remain the primary
encapsulation mechanism.
