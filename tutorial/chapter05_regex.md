# Chapter 5: Regex Pre-tokenization

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
'(?i:[sdmt]|ll|ve|re)
|[^\r\n\p{L}\p{N}]?+\p{L}++
|\p{N}{1,3}+
| ?[^\s\p{L}\p{N}]++[\r\n]*+
|\s++$
|\s*[\r\n]
|\s+(?!\S)
|\s
```

This is the pattern verbatim, copied out of the library rather than retyped.
That matters more than it sounds: an earlier draft of this chapter carried a
tidied version that read as though it did the same thing, and the accompanying
code shipped a tidied `o200k_base` pattern that disagreed with the reference on
roughly half of all inputs. A tokenizer pattern is a specification, not a
description. If you find yourself simplifying one for readability, keep the
original and compare against it.

Each alternative (separated by `|`) matches a different type of text
chunk. Let's break them down:

### `'(?i:[sdmt]|ll|ve|re)`: Contractions

The apostrophe sits outside the group, and the group itself is a character
class plus three two-letter alternatives: `s`, `d`, `m`, `t`, `ll`, `ve`, `re`.
Spelled out, that covers `'s`, `'d`, `'m`, `'t`, `'ll`, `'ve`, `'re`. The
`(?i:...)` makes it case-insensitive, so `DON'T` splits the same way as
`don't`.

Matching contractions first prevents them from being split in unexpected
ways. Without this branch, "don't" might become `["don", "'", "t"]`.

### `[^\r\n\p{L}\p{N}]?+\p{L}++`: Words

`\p{L}` matches any Unicode letter (Latin, Cyrillic, Chinese, etc.).
`\p{N}` matches any Unicode number.

This matches sequences of letters, optionally preceded by a single
non-letter, non-number character (like a space or punctuation). The
effect is that words get their leading space attached: `" hello"` is
one chunk, not `" "` + `"hello"`.

### `\p{N}{1,3}+`: Numbers

Matches 1 to 3 digits at a time. This means `12345` becomes `["123",
"45"]`: numbers are tokenized in chunks of at most 3 digits. This
prevents very long numbers from becoming single tokens.

### ` ?[^\s\p{L}\p{N}]++[\r\n]*+`: Punctuation

Matches punctuation sequences (optionally preceded by a space), with
any trailing newlines. The `++` is a **possessive quantifier**: it
doesn't backtrack. This is important for performance and is why we
need PCRE2 (standard POSIX regex doesn't support possessive quantifiers).

### `\s*[\r\n]`: Newlines

Matches whitespace ending in a newline. This keeps newlines attached
to preceding whitespace.

### `\s++$`, `\s+(?!\S)` and `\s`: Whitespace

Three branches, and the order is doing work.

`\s++$` takes whitespace that runs to the end of the input. `\s+(?!\S)`
takes a whitespace run that is not followed by a non-whitespace character,
which is the same idea one position earlier. The last branch is a bare `\s`,
a *single* whitespace character, not `\s+`.

That final detail is easy to get wrong and hard to notice. If you write `\s+`
there, a run of spaces in the middle of a line collapses into one chunk instead
of being consumed one character at a time by the earlier branches, and the token
ids come out different. It will still look like a tokenizer. It will just not be
this one.

## The three encodings, and why they differ

The library exposes three patterns, because OpenAI has shipped three
generations of tokenizer and each one changed the pre-tokenization rules.

**`p50k_base`** is the GPT-3 and Codex era. Its shape is the simplest of the
three: contractions, then a letter run, then a digit run, then a punctuation
run, each optionally taking one leading space.

```regex
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
```

Note ` ?\p{N}++`: digits run without limit. `"1234567"` is one chunk here.

**`cl100k_base`** is GPT-3.5-turbo and GPT-4, and it is the one this book
implements end to end. Two changes matter. Digits are capped at three per chunk
(`\p{N}{1,3}+`), so long numbers are broken up rather than swallowed whole.
And the leading-space rule moved: instead of ` ?` in front of each category, a
word may absorb one preceding character that is neither a letter nor a digit
(`[^\r\n\p{L}\p{N}]?+`), which is a wider net than a space.

**`o200k_base`** is GPT-4o, and it is where the pattern stops being tidy:

```regex
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]+[\r\n/]*
|\s*[\r\n]+
|\s+(?!\S)
|\s+
```

Two things changed, and both are about languages that are not English.

The single "word" branch became two, and both are built from explicit Unicode
letter categories rather than the blanket `\p{L}`: uppercase (`Lu`), titlecase
(`Lt`), modifier (`Lm`), other (`Lo`), lowercase (`Ll`), and combining marks
(`M`). One branch handles a capitalised word, the other an all-caps run, and
each may take a trailing contraction. The effect is that `"McDonald"`,
`"HTTP"` and `"iPhone"` chunk differently from each other, and that scripts
with combining marks, such as Devanagari, Arabic and Hebrew, stop being cut in the
middle of a grapheme.

The other change is small and easy to miss: the punctuation branch is
`[\r\n/]*`, not `[\r\n]*`. A forward slash now attaches to a punctuation
run, which is a URL and file-path optimisation.

### Why this is worth belabouring

Because the code accompanying this book got it wrong. The `o200k_base` pattern
here was, until recently, a hand-simplified version with one plain `\p{L}+`
word branch and no case distinction. It compiled, it looked plausible, it
tokenized text, and it disagreed with the official library on **490 of 1000
generated inputs**. Nothing caught it for months, because the test suite only
ever compared `cl100k_base`.

The lesson is not "be careful". It is that a pre-tokenization pattern cannot be
verified by reading it. The only check that works is running it against the
reference implementation on inputs you did not choose by hand, which is what
`tests/test_reference_batch.c` now does for all three.

### One thing that is not portable

That comparison found something else on its first run in CI, and it is worth
knowing before you ship a tokenizer anywhere.

Two of three thousand cases disagreed with the reference. The same two passed on
the machine they were written on. The input responsible contained `U+12A90`, and
`U+12A90` is unassigned: its Unicode category is `Cn`, not a letter, not a symbol,
not anything.

Whether `\p{L}` matches an unassigned code point depends on the Unicode version
compiled into the regex engine doing the matching. PCRE2 on one machine had a
different answer from the Rust engine inside the Python package, so the same bytes
pre-tokenized into different chunks and the token ids diverged.

Three things follow.

The first is practical: if you tokenize text that might contain unassigned code
points, and you need bit-identical output to a reference implementation, the
Unicode version of your regex engine is part of your contract. It is not usually
in anyone's dependency pinning, and it changes when the distribution updates.

The second is about testing. A generated-input test that wanders into unassigned
code points is a test that fails depending on which machine ran it, and a test
that fails intermittently is a test people learn to ignore. The generator in
`test_reference_batch.c` draws from curated, long-assigned ranges for exactly that
reason, and the comment above the ranges says so, because otherwise the next
person to read it will widen them again.

The third is that real text does not contain unassigned code points, so none of
this shows up until a fuzzer or a generator finds it. Which is the argument for
having one.


## Why PCRE2?

We need PCRE2 for three features that POSIX regex doesn't support:

1. **`\p{L}` and `\p{N}`**. Unicode property classes. POSIX has
   `[:alpha:]` but it's locale-dependent and doesn't reliably cover
   all Unicode.

2. **`++`**. Possessive quantifiers. Without them, the regex engine
   may backtrack exponentially on certain inputs.

3. **`(?!...)`**. Negative lookahead. The `\s+(?!\S)` construct
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
`Regex` through pointers and the public functions; they can't access
the PCRE2 internals. Benefits:

- **Callers don't need `#include <pcre2.h>`**. They're isolated from
  the dependency.
- **We can change the implementation** (e.g., switch to a different
  regex library) without breaking any callers.
- **Compile times are better**. Changes to the regex internals don't
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
- **`PCRE2_UTF`**. Treat the pattern and subjects as UTF-8 (not raw
  bytes). This means `.` matches a Unicode code point, not a single byte.
- **`PCRE2_UCP`**. Use Unicode properties for `\w`, `\d`, `\s` and
  (crucially) `\p{L}`, `\p{N}`.

### JIT Compilation

```c
pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);
```

PCRE2 can JIT-compile patterns into native machine code for a 2–10x
speedup. This is optional: if the platform doesn't support JIT (e.g.,
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

The `RegexMatch` struct stores `(start, len)` pairs, offsets into the
original input string:

```c
typedef struct {
    size_t start;
    size_t len;
} RegexMatch;
```

No strings are copied. The match results are valid as long as the
original input text is alive. This is the same ownership pattern we
used for `Bytes` slices in Chapter 2, non-owning views into existing
data.

## Testing Without PCRE2

If PCRE2 isn't installed, the tests won't compile. On Debian and Ubuntu,
which is what this book was written and tested on:

```bash
sudo apt install libpcre2-dev
```

Other platforms package it under some spelling of `pcre2`. This book does not
give the command, because it has not been run there.

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
we'll load a real tiktoken vocabulary file, combining base64 decoding
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
| Opaque types | `typedef struct Regex Regex`, C's encapsulation |

The opaque type pattern isn't a C23 feature per se (it works in any C
version), but it's an important C idiom worth highlighting. C23 doesn't
add access modifiers like C++, so opaque types remain the primary
encapsulation mechanism.
