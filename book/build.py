#!/usr/bin/env python3
"""Assemble the tutorial chapters into a single PDF.

The chapters in ../tutorial/ are the single source of truth. This script only
concatenates and typesets them, so the book and the repository cannot drift.

    pip install -r requirements.txt
    python build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

TITLE = "Build a Tokenizer in C"
SUBTITLE = "OpenAI's BPE, from base64 to a working cl100k_base encoder"
AUTHOR = "Antonio Elena"

BOOK_DIR = Path(__file__).parent
REPO_DIR = BOOK_DIR.parent
TUTORIAL_DIR = REPO_DIR / "tutorial"
OUTPUT = BOOK_DIR / "Build-a-Tokenizer-in-C.pdf"

# WeasyPrint has no \newpage; the front matter uses it as a marker and we
# translate it into a page-breaking element.
PAGE_BREAK_MARKER = "\\newpage"


def collect_markdown() -> str:
    """Front matter, then the chapters in filename order, then the colophon."""
    parts: list[str] = []

    front = BOOK_DIR / "00-front-matter.md"
    if not front.exists():
        sys.exit(f"Missing {front}")
    parts.append(front.read_text(encoding="utf-8"))

    chapters = sorted(TUTORIAL_DIR.glob("chapter*.md"))
    if not chapters:
        sys.exit(f"No chapters found in {TUTORIAL_DIR}")
    for chapter in chapters:
        parts.append(PAGE_BREAK_MARKER)
        parts.append(chapter.read_text(encoding="utf-8"))
    print(f"  {len(chapters)} chapters")

    colophon = BOOK_DIR / "99-colophon.md"
    if colophon.exists():
        parts.append(colophon.read_text(encoding="utf-8"))

    return "\n\n".join(parts)


CSS = """
@page {
    size: A4;
    margin: 22mm 20mm 20mm 20mm;
    @top-center {
        content: "Build a Tokenizer in C";
        font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', serif;
        font-size: 8.5pt;
        color: #8a8a8a;
    }
    @bottom-center {
        content: counter(page);
        font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', serif;
        font-size: 9pt;
        color: #8a8a8a;
    }
}

/* The cover carries no running head or folio. */
@page :first {
    margin: 0;
    @top-center { content: none; }
    @bottom-center { content: none; }
}

body {
    font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    hyphens: auto;
}

/* Cover: the aelena.com register: cream ground, near-black ink, one rule. */
.cover {
    page-break-after: always;
    height: 297mm;
    background: #fafaf8;
    color: #1a1a1a;
    padding: 70mm 22mm 0 22mm;
    box-sizing: border-box;
}
.cover .title {
    font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', serif;
    font-size: 40pt;
    font-weight: normal;
    line-height: 1.1;
    margin: 0 0 10mm 0;
    letter-spacing: -0.5pt;
}
.cover .rule {
    border: none;
    border-top: 1px solid #1a1a1a;
    width: 40mm;
    margin: 0 0 10mm 0;
}
.cover .subtitle {
    font-family: Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace;
    font-size: 11pt;
    color: #555;
    line-height: 1.5;
    margin: 0 0 55mm 0;
}
.cover .author {
    font-family: Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace;
    font-size: 10pt;
    letter-spacing: 1pt;
    text-transform: uppercase;
    color: #1a1a1a;
}

.page-break { page-break-after: always; }

h1 {
    font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', serif;
    font-size: 21pt;
    font-weight: normal;
    color: #1a1a1a;
    page-break-before: always;
    page-break-after: avoid;
    margin: 0 0 8mm 0;
    padding-bottom: 3mm;
    border-bottom: 1px solid #d8d8d4;
}

h2 {
    font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', serif;
    font-size: 14pt;
    font-weight: bold;
    color: #1a1a1a;
    margin: 9mm 0 3mm 0;
    page-break-after: avoid;
}

h3 {
    font-family: Georgia, 'Liberation Serif', 'DejaVu Serif', serif;
    font-size: 11.5pt;
    font-weight: bold;
    color: #333;
    margin: 6mm 0 2mm 0;
    page-break-after: avoid;
}

p { margin: 0 0 3.2mm 0; orphans: 3; widows: 3; }

code {
    font-family: Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace;
    font-size: 9pt;
    background: #f2f1ee;
    padding: 0.5mm 1.2mm;
    border-radius: 2px;
}

pre {
    background: #f7f6f3;
    border-left: 2px solid #c8c6c0;
    padding: 3mm 4mm;
    margin: 4mm 0;
    page-break-inside: avoid;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-wrap: break-word;
    white-space: pre-wrap;
}
pre code { background: none; padding: 0; font-size: 8.5pt; }

blockquote {
    margin: 4mm 0 4mm 6mm;
    padding-left: 4mm;
    border-left: 2px solid #c8c6c0;
    color: #555;
    font-style: italic;
}

ul, ol { margin: 0 0 3.2mm 0; padding-left: 6mm; }
li { margin: 0 0 1.2mm 0; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 4mm 0;
    font-size: 9pt;
    page-break-inside: avoid;
}
th, td { border: 1px solid #d8d8d4; padding: 1.6mm 2.4mm; text-align: left; }
th { background: #f2f1ee; font-weight: bold; }

a { color: #1a1a1a; text-decoration: none; border-bottom: 1px solid #c8c6c0; }

hr { border: none; border-top: 1px solid #d8d8d4; margin: 6mm 0; }

img { max-width: 100%; page-break-inside: avoid; }
"""


def main() -> None:
    try:
        from markdown import markdown
        from weasyprint import CSS as WeasyCSS
        from weasyprint import HTML
    except ImportError:
        sys.exit(
            "Missing dependencies.\n"
            "    pip install -r requirements.txt\n"
            "WeasyPrint also needs system libraries (pango, cairo); see its install docs."
        )

    print(f"Assembling from {TUTORIAL_DIR}...")
    md = collect_markdown()

    html_body = markdown(
        md,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )
    # Translate the front matter's page-break markers.
    html_body = html_body.replace(
        f"<p>{PAGE_BREAK_MARKER}</p>", '<div class="page-break"></div>'
    ).replace(PAGE_BREAK_MARKER, "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{TITLE}</title></head>
<body>
  <section class="cover">
    <h1 class="title">{TITLE}</h1>
    <hr class="rule">
    <p class="subtitle">{SUBTITLE}</p>
    <p class="author">{AUTHOR}</p>
  </section>
  {html_body}
</body>
</html>"""

    print("Rendering...")
    HTML(string=html).write_pdf(OUTPUT, stylesheets=[WeasyCSS(string=CSS)])
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"  {OUTPUT.name}, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
