#!/usr/bin/env python3
"""
Reference implementation using the official tiktoken library.

Two modes:

  Single case, one process per string:

      tiktoken_reference.py <encoding_name> <text> [allow_special]
      -> prints the token ids as a JSON array

  Batch, one process for thousands of strings:

      tiktoken_reference.py --batch <requests> <responses>

The batch mode exists because get_encoding() loads a 1.6 MB vocabulary and
building the ranks takes over a second. At one process per case, a few hundred
comparisons cost minutes; batched, they cost one load.

The request file is deliberately not JSON. Each line is:

      <encoding_name> <allow_special:0|1> <hex-encoded UTF-8 bytes>

Hex, because the C side has to write this file and the text under test contains
quotes, backslashes, newlines and embedded NULs — exactly the inputs a tokenizer
has to get right. Passing any of that through argv or a shell command line either
breaks or silently mangles it, which is why the single-case mode above can only
test well-behaved strings.

The response file has one line per request: the token ids, space separated, in
the same order. An empty line means zero tokens. A line reading ERROR means that
case failed and the C side should report it rather than skip it.
"""

import sys

import tiktoken

_CACHE = {}


def _encoding(name):
    if name not in _CACHE:
        _CACHE[name] = tiktoken.get_encoding(name)
    return _CACHE[name]


def encode_text(encoding_name: str, text: str, allow_special: bool = False) -> list[int]:
    """Encode text with the official library. Special tokens per the flag."""
    enc = _encoding(encoding_name)
    if allow_special:
        return enc.encode(text, allowed_special="all")
    return enc.encode(text, allowed_special=set())


def run_batch(request_path: str, response_path: str) -> None:
    out = []
    with open(request_path, "r", encoding="ascii") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                name, allow, payload = raw.split(" ", 2)
            except ValueError:
                # No payload means the empty string, which is a case worth having.
                parts = raw.split(" ")
                if len(parts) != 2:
                    out.append("ERROR")
                    continue
                name, allow, payload = parts[0], parts[1], ""
            try:
                text = bytes.fromhex(payload).decode("utf-8")
                tokens = encode_text(name, text, allow == "1")
                out.append(" ".join(str(t) for t in tokens))
            except Exception as exc:  # noqa: BLE001 - the C side needs the line, not the type
                print(f"case failed: {exc}", file=sys.stderr)
                out.append("ERROR")

    with open(response_path, "w", encoding="ascii") as f:
        f.write("\n".join(out))
        f.write("\n")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--batch":
        if len(sys.argv) != 4:
            print("Usage: tiktoken_reference.py --batch <requests> <responses>",
                  file=sys.stderr)
            sys.exit(1)
        run_batch(sys.argv[2], sys.argv[3])
        return

    if len(sys.argv) < 3:
        print("Usage: tiktoken_reference.py <encoding_name> <text> [allow_special]",
              file=sys.stderr)
        sys.exit(1)

    import json

    encoding_name = sys.argv[1]
    text = sys.argv[2]
    allow_special = len(sys.argv) > 3 and sys.argv[3] == "1"

    try:
        tokens = encode_text(encoding_name, text, allow_special)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(tokens))


if __name__ == "__main__":
    main()
