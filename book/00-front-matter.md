# Preface

Every developer working with language models has typed `len(tokens)` and moved on. The tokenizer is the least examined component in the entire stack, a black box that turns text into integers, priced per thousand, and otherwise ignored.

That incuriosity has a cost. Tokenization is where a surprising number of production problems actually live. Why does this prompt cost 40% more in Spanish than in English. Why does the model split this identifier into six pieces and lose the plot. Why does the context window fill faster than the character count suggests. Why does trimming a string by characters corrupt the encoding. None of these are model problems. They are tokenizer problems, and you cannot reason about them from the outside.

This book builds OpenAI's `tiktoken` from scratch, in C, so that the box stops being black.

## Why C

Not for performance, though the result is fast. C is the constraint that forces you to see the thing.

In Python you would `dict[key]` and never think about it. In C you have to build the hash map, choose the hash function, size the table, decide what happens on collision, and free the memory. Every data structure the algorithm needs becomes a decision you make explicitly rather than a facility you inherit. That is the whole pedagogical point: you cannot skim a component you have to allocate.

The same applies to text. Python hands you `str` and hides the encoding. C hands you bytes, and byte pair encoding turns out to be exactly what its name says: an algorithm about bytes, which is much easier to understand once the language stops pretending otherwise.

## What you will build

A complete, working tokenizer, compatible with OpenAI's `cl100k_base` encoding and verified against the reference Python implementation. By the end you will have:

- A base64 decoder, because the vocabulary file is base64-encoded.
- A byte-string type that survives embedded nulls, which C strings do not.
- An arena allocator and an open-addressing hash map, because the vocabulary is 100,000 entries and `malloc` per entry is not a plan.
- The byte pair encoding merge loop, on an indexed linked list.
- Regex pre-tokenization with PCRE2, against the actual `cl100k_base` pattern.
- Vocabulary loading, a public API, and an integration test suite that checks token-for-token agreement with Python's `tiktoken`.

Roughly 3,000 lines of C23, built in eight steps, each of which runs and passes tests before the next one starts.

## Who this is for

You should be comfortable reading C. You do not need to be fluent (the book explains the C23 features it uses, including the ones that are genuinely new), but pointer arithmetic should not frighten you.

You do not need any background in machine learning. Nothing in a tokenizer is machine learning. It is string processing, a hash table, and a greedy merge loop, and that is precisely why it is a good place to start if the rest of the stack feels like magic.

## How to read it

Each chapter is self-contained enough to read alone, and ordered so that each one uses what the previous built. The code for every chapter is in the companion repository:

**github.com/aelena/c-tiktoken**

Clone it, build it, and break it. The tests are the interesting part: a tokenizer either agrees with the reference implementation exactly or it is wrong, which makes it an unusually honest thing to learn on. There is no partial credit and no plausible-looking output. Either the token IDs match or they do not.

Start with Chapter 1. It is about base64, which sounds like a detour and is not: the vocabulary file cannot be read without it, and it is the gentlest possible introduction to thinking in bytes.

\newpage

# Table of Contents

1. **Base64 Decoding**. The lookup-table approach, and why the vocabulary file needs it
2. **Byte Strings**. What C strings cannot represent, and the type that replaces them
3. **Hash Map and Arena Allocator**. Storing 100,000 vocabulary entries without dying
4. **The BPE Algorithm**. The merge loop, and the indexed linked list that makes it tractable
5. **Regex Pre-tokenization**. The `cl100k_base` pattern, and why PCRE2
6. **Vocabulary Loading**. The `.tiktoken` file format and the loading pipeline
7. **The Encoding API**. Assembling the pieces, and handling special tokens
8. **Putting It All Together**. The library, the tests, and agreement with the reference

\newpage
