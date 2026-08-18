# book/

Builds the eight tutorial chapters into a single typeset PDF, *Build a Tokenizer in C*.

The chapters in [`../tutorial/`](../tutorial/) are the single source of truth. This directory adds front matter, a colophon and a stylesheet, and nothing else, so the book and the repository cannot drift apart.

## Build

```bash
pip install -r requirements.txt
python build.py
```

Output: `Build-a-Tokenizer-in-C.pdf`, around 54 pages.

WeasyPrint needs system libraries beyond the Python packages (pango, cairo, gdk-pixbuf). On Debian/Ubuntu:

```bash
sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 shared-mime-info
```

See the [WeasyPrint install docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) for macOS and Windows.

### Fonts

The stylesheet asks for Georgia and Consolas, falling back to Liberation and DejaVu. Building on Windows or macOS gives you the intended pair; building on a bare Linux box gives you the fallbacks, which are close enough that only the author will notice.

## Layout

| File | Purpose |
|---|---|
| `00-front-matter.md` | Preface, why C, what you will build, who it is for, contents |
| `99-colophon.md` | Repository pointer, about the author, other titles |
| `build.py` | Assembly and typesetting. The stylesheet lives here as a string |
| `requirements.txt` | Python dependencies |

Chapters are picked up automatically from `../tutorial/chapter*.md` in filename order. Adding a chapter means adding a file; nothing here needs editing except the contents list in the front matter.

## Why the PDF is not committed

The tutorial is free and stays free; it is the reason anyone finds this repository. The PDF is the typeset convenience version, sold at [aelena74.gumroad.com](https://aelena74.gumroad.com), and building it yourself from these sources is both possible and permitted. If you would rather spend five minutes than the price of a coffee, the instructions above are complete and not deliberately hobbled.
