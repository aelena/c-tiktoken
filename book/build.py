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

# Gumroad's own advice: buyers read on phones and e-readers, and a PDF is a
# fixed page size on both. EPUB reflows.
EPUB = BOOK_DIR / "Build-a-Tokenizer-in-C.epub"

# A stable identifier, so a reader that already has an earlier copy treats a new
# build as the same book rather than a second one in the library.
EPUB_ID = "urn:uuid:6f1c4d18-2a3b-4f5e-9c0d-7b8a1e2f3c4d"

# Anything darker than the cream ground counts as ink when locating the design.
INK_THRESHOLD = 240

# A4 at 200 dpi is 1654 x 2339, which is above what Gumroad and LinkedIn
# downscale from and still a small file, because the page is nearly flat colour.
COVER_DPI = 200

# WeasyPrint has no \newpage; the front matter uses it as a marker and we
# translate it into a page-breaking element.
PAGE_BREAK_MARKER = "\\newpage"


def first_heading(markdown_text: str, fallback: str) -> str:
    """The section's own H1, which is what an EPUB table of contents needs."""
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def collect_sections() -> list[tuple[str, str, bool]]:
    """Front matter, then the chapters in filename order, then the colophon.

    Returns (title, markdown, break_before) so that both outputs can be built
    from one traversal. The flag exists because the PDF puts a page break before
    every chapter and none before the colophon, and the EPUB needs the same
    boundaries as separate files rather than as markers in one string.
    """
    sections: list[tuple[str, str, bool]] = []

    front = BOOK_DIR / "00-front-matter.md"
    if not front.exists():
        sys.exit(f"Missing {front}")
    front_text = front.read_text(encoding="utf-8")
    sections.append((first_heading(front_text, "Front matter"), front_text, False))

    chapters = sorted(TUTORIAL_DIR.glob("chapter*.md"))
    if not chapters:
        sys.exit(f"No chapters found in {TUTORIAL_DIR}")
    for chapter in chapters:
        text = chapter.read_text(encoding="utf-8")
        sections.append((first_heading(text, chapter.stem), text, True))
    print(f"  {len(chapters)} chapters")

    colophon = BOOK_DIR / "99-colophon.md"
    if colophon.exists():
        text = colophon.read_text(encoding="utf-8")
        sections.append((first_heading(text, "Colophon"), text, False))

    return sections


def assemble_markdown(sections: list[tuple[str, str, bool]]) -> str:
    """One string for the typesetter, with the page-break markers put back."""
    parts: list[str] = []
    for _, text, break_before in sections:
        if break_before:
            parts.append(PAGE_BREAK_MARKER)
        parts.append(text)
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


EPUB_CSS = """
/* Deliberately not the print stylesheet. That one is built on millimetres and
   @page rules, which mean nothing on a device whose page size is decided by the
   reader's font setting.

   Every colour here is a translucent neutral, and nothing sets a background
   without letting the text colour come from the reader. An earlier version gave
   code blocks an opaque near-white background and left the text inherited: in
   Calibre's dark theme the reader supplied its light foreground, the background
   stayed near-white, and every code block became unreadable. A translucent grey
   composes over whatever the reader's page colour is, so it darkens a light page
   and lightens a dark one and can never fight the theme. */
body {
    font-family: Georgia, serif;
    line-height: 1.5;
    margin: 0 1em;
}
h1 {
    font-size: 1.5em;
    font-weight: normal;
    margin: 1em 0 0.6em;
    padding-bottom: 0.2em;
    border-bottom: 1px solid rgba(128, 128, 128, 0.4);
}
h2 { font-size: 1.2em; margin: 1.4em 0 0.4em; }
h3 { font-size: 1.05em; margin: 1.1em 0 0.3em; }
p { margin: 0 0 0.7em; }
code {
    font-family: "DejaVu Sans Mono", Consolas, monospace;
    font-size: 0.85em;
    background: rgba(128, 128, 128, 0.16);
    padding: 0 0.15em;
    border-radius: 0.15em;
}
pre {
    font-family: "DejaVu Sans Mono", Consolas, monospace;
    font-size: 0.75em;
    line-height: 1.35;
    background: rgba(128, 128, 128, 0.12);
    border-left: 2px solid rgba(128, 128, 128, 0.45);
    padding: 0.6em 0.8em;
    margin: 0.8em 0;
    /* Code cannot reflow, so let it wrap rather than crop on a narrow screen. */
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
/* Inside a block the tint would stack on itself and the padding would double. */
pre code { background: none; padding: 0; font-size: 1em; }
blockquote {
    margin: 0.8em 0 0.8em 1em;
    padding-left: 0.8em;
    border-left: 2px solid rgba(128, 128, 128, 0.45);
    font-style: italic;
}
table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 0.8em 0; }
th, td { border: 1px solid rgba(128, 128, 128, 0.4); padding: 0.25em 0.4em; text-align: left; }
th { background: rgba(128, 128, 128, 0.14); }
hr { border: none; border-top: 1px solid rgba(128, 128, 128, 0.4); margin: 1.2em 0; }
"""


def render_epub(sections, to_html) -> None:
    """Write the EPUB, one file per section, with cover.png as the cover.

    A PDF is a fixed page size, which on a phone means pinching and panning, and
    on an e-reader means a page that does not match the screen. EPUB reflows.
    The cover is the PNG rasterised from page 1, so all three artifacts show the
    same cover by construction.
    """
    try:
        from ebooklib import epub
    except ImportError:
        print(f"  {EPUB.name} skipped: ebooklib is not installed.")
        print("    pip install -r requirements.txt")
        return

    book = epub.EpubBook()
    book.set_identifier(EPUB_ID)
    book.set_title(TITLE)
    book.set_language("en")
    book.add_author(AUTHOR)
    book.add_metadata("DC", "description", SUBTITLE)

    if COVER.exists():
        book.set_cover("cover.png", COVER.read_bytes())
    else:
        print(f"  {EPUB.name}: no {COVER.name} to use as a cover")

    css = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS,
    )
    book.add_item(css)

    items = []
    for index, (title, markdown_text, _) in enumerate(sections, start=1):
        # The markdown already opens with its own H1, so the title is metadata
        # for the table of contents and is not repeated in the text.
        body = to_html(markdown_text.replace(PAGE_BREAK_MARKER, ""))
        item = epub.EpubHtml(
            title=title,
            file_name=f"section{index:02d}.xhtml",
            lang="en",
        )
        item.content = body
        item.add_item(css)
        book.add_item(item)
        items.append(item)

    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["cover", "nav"] + items

    epub.write_epub(str(EPUB), book)
    size_kb = EPUB.stat().st_size / 1024
    print(f"  {EPUB.name}, {len(items)} sections, {size_kb:.0f} KB")


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

    # Locating the design by what is darker than the page assumes a light page
    # with dark marks on it, which is true of this cover and not of a full-bleed
    # one. On a dark full-bleed cover the background itself is below the
    # threshold, the whole page reads as ink, the ink is taller than the page is
    # wide, and the padding branch below fires: it returns a canvas wider than
    # the page with the design floating between two bars of whatever colour the
    # corner pixel happens to be. Measured on a 1700x2200 page with a 16px paper
    # margin, that was a 2169x2169 image with 250px of white down each side.
    #
    # So detect the case and centre on the paper instead, which is the right
    # answer when the design covers all of it. A cream cover is unaffected: it
    # produces the same crop as before.
    ink_area = (ink[2] - ink[0]) * (ink[3] - ink[1])
    full_bleed = ink_area > 0.9 * width * height
    if full_bleed:
        ink = (0, 0, width, height)

    side = (min(width, height) if full_bleed
            else max(min(width, height), ink[3] - ink[1]))
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
    sections = collect_sections()
    md = assemble_markdown(sections)

    def to_html(text: str) -> str:
        return markdown(text, extensions=["extra", "tables", "fenced_code", "sane_lists"])

    html_body = to_html(md)
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
    render_epub(sections, to_html)


if __name__ == "__main__":
    main()
