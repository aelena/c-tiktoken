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

# The title, encoded by the encoder this book builds. Real cl100k_base output,
# not decoration: note that "Tokenizer" costs two tokens and "C" costs one.
# Regenerate with:
#     python -c "import tiktoken; e=tiktoken.get_encoding('cl100k_base'); #                t=e.encode('Build a Tokenizer in C'); #                print([(e.decode([x]), x) for x in t])"
TITLE_TOKENS = [
    ("Build", 11313),
    (" a", 264),
    (" Token", 9857),
    ("izer", 3213),
    (" in", 304),
    (" C", 356),
]

BOOK_DIR = Path(__file__).parent
REPO_DIR = BOOK_DIR.parent
TUTORIAL_DIR = REPO_DIR / "tutorial"
OUTPUT = BOOK_DIR / "Build-a-Tokenizer-in-C.pdf"
COVER = BOOK_DIR / "cover.png"

# Gumroad shows a square image in its library, discover and profile pages, at
# 600x600 or more, so the portrait page alone is not enough.
COVER_SQUARE = BOOK_DIR / "cover-square.png"

# Anything darker than the cream ground counts as ink when locating the design.
INK_THRESHOLD = 240

# A4 at 200 dpi is 1654 x 2339, which is above what Gumroad and LinkedIn
# downscale from and still a small file, because the page is nearly flat colour.
COVER_DPI = 200

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
    border-bottom: none;
    padding-bottom: 0;
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
    margin: 0 0 22mm 0;
}
.cover .author {
    font-family: Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace;
    font-size: 10pt;
    letter-spacing: 1pt;
    text-transform: uppercase;
    color: #1a1a1a;
}

.cover .tokens {
    border-collapse: collapse;
    width: auto;
    margin: 0 0 38mm 0;
    font-family: Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace;
}
.cover .tokens td {
    border: none;
    padding: 0 3mm;
    white-space: pre;
    text-align: center;
    vertical-align: baseline;
}
.cover .tokens__piece td {
    font-size: 10.5pt;
    color: #93938e;
    padding-bottom: 2mm;
}
.cover .tokens__id td {
    font-size: 10.5pt;
    color: #1a1a1a;
    border-top: 1px solid #cfcec9;
    padding-top: 2mm;
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


def square_crop(page):
    """A square version of the page, centred on the design instead of the paper.

    The page is A4 and its lower third is empty, so centring on the paper would
    leave the design in the top half with a band of nothing under it. Centring on
    the ink keeps the full page width, which means the left margin and the
    horizontal composition survive untouched.

    Written as a paste onto a square canvas rather than a crop so that a cover
    taller than it is wide is padded instead of being silently clipped.
    """
    from PIL import Image

    width, height = page.size
    ink = page.convert("L").point(
        lambda v: 255 if v < INK_THRESHOLD else 0
    ).getbbox() or (0, 0, width, height)

    side = max(min(width, height), ink[3] - ink[1])
    centre = (ink[1] + ink[3]) / 2
    top = max(0, min(round(centre - side / 2), max(0, height - side)))

    band = page.crop((0, top, width, min(height, top + side)))
    canvas = Image.new("RGB", (side, side), page.getpixel((0, 0)))
    canvas.paste(band, ((side - band.width) // 2, (side - band.height) // 2))
    return canvas


def render_cover(pdf_path: Path) -> None:
    """Rasterise page 1 of the finished PDF into a cover image.

    Deliberately taken from the PDF rather than re-rendered from the cover HTML.
    The image a shop or a social card shows is then the same pixels as page 1 by
    construction, and cannot drift from the book when the cover changes.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print(f"  {COVER.name} skipped: pypdfium2 is not installed.")
        print("    pip install -r requirements.txt")
        return

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        image = doc[0].render(scale=COVER_DPI / 72).to_pil()
    finally:
        doc.close()

    image.save(COVER)
    width, height = image.size
    size_kb = COVER.stat().st_size / 1024
    print(f"  {COVER.name}, {width}x{height}, {size_kb:.0f} KB")

    square = square_crop(image)
    square.save(COVER_SQUARE)
    size_kb = COVER_SQUARE.stat().st_size / 1024
    print(f"  {COVER_SQUARE.name}, {square.width}x{square.height}, {size_kb:.0f} KB")


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

    # tiktoken is not a build dependency, so the ids above cannot be recomputed
    # here. This is the next best thing: if the title ever changes and the token
    # list does not, the pieces stop spelling it and the build says so.
    if "".join(piece for piece, _ in TITLE_TOKENS) != TITLE:
        sys.exit(
            "TITLE_TOKENS no longer spells TITLE. Re-encode the title with "
            "cl100k_base; the command is in the comment above TITLE_TOKENS."
        )

    piece_cells = "".join(
        f"<td>{piece}</td>" for piece, _ in TITLE_TOKENS
    )
    id_cells = "".join(
        f"<td>{tok}</td>" for _, tok in TITLE_TOKENS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{TITLE}</title></head>
<body>
  <section class="cover">
    <h1 class="title">{TITLE}</h1>
    <hr class="rule">
    <p class="subtitle">{SUBTITLE}</p>
    <table class="tokens">
      <tr class="tokens__piece">{piece_cells}</tr>
      <tr class="tokens__id">{id_cells}</tr>
    </table>
    <p class="author">{AUTHOR}</p>
  </section>
  {html_body}
</body>
</html>"""

    print("Rendering...")
    HTML(string=html).write_pdf(OUTPUT, stylesheets=[WeasyCSS(string=CSS)])
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"  {OUTPUT.name}, {size_kb:.0f} KB")

    render_cover(OUTPUT)


if __name__ == "__main__":
    main()
