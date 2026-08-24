# book/

Builds the eight tutorial chapters into a single typeset PDF, *Build a Tokenizer in C*.

The chapters in [`../tutorial/`](../tutorial/) are the single source of truth. This directory adds front matter, a colophon and a stylesheet, and nothing else, so the book and the repository cannot drift apart.

## Build

```bash
pip install -r requirements.txt
python build.py
```

Output: `Build-a-Tokenizer-in-C.pdf`, 62 pages, and `cover.png`, which is
page 1 of that PDF rasterised at 200 dpi (1654 x 2339), plus
`cover-square.png` at 1654 x 1654 for shops that want a square thumbnail.
Gumroad asks for one at 600 x 600 or more for its library, discover and
profile pages.

Both come from the finished PDF rather than being rendered separately, so the
picture a shop or a social card shows is the same pixels as page 1 and cannot
drift from the book. The square one is centred on the design rather than on
the paper: the page is A4 and its lower third is empty, so centring on the
paper would leave the title up top with a band of nothing under it. It keeps
the full page width, so the left margin and the horizontal composition are
untouched.

WeasyPrint needs system libraries beyond the Python packages (pango, cairo, gdk-pixbuf). On Debian/Ubuntu:

```bash
sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 shared-mime-info
```

See the [WeasyPrint install docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) for macOS and Windows.

### Building on Windows

Getting those libraries onto Windows means installing a GTK runtime and having
it on `PATH`, and WeasyPrint fails at import if it is missing:

```
OSError: cannot load library 'libgobject-2.0-0'
```

Every release of this book so far was built in a container instead, which needs
nothing installed on the host beyond Docker:

```bash
docker run --rm -v "$PWD:/work" -w /work/book python:3.12-slim sh -c '
  apt-get update -qq
  apt-get install -y -qq libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b       libcairo2 libgdk-pixbuf-2.0-0 fonts-dejavu
  pip install -q -r requirements.txt
  python build.py'
```

Prefix it with `MSYS_NO_PATHCONV=1` under Git Bash, or the `-v` and `-w` paths
get rewritten on the way in.

The container has no Georgia or Consolas, so the fallbacks are what ship: the
released PDF embeds DejaVu Serif and DejaVu Sans Mono. That is worth knowing
before switching to a native build, because it would change the typeface of
every page including the cover.

### Fonts

The stylesheet asks for Georgia and Consolas and falls back to Liberation and DejaVu. A native build on Windows or macOS would give you the intended pair, but the container build above is what actually ships, so the released PDF is set in DejaVu. The two are close enough that only the author will notice, and the fallbacks are the reference: if you rebuild natively, expect the pages to reflow slightly.

## Layout

| File | Purpose |
|---|---|
| `00-front-matter.md` | Preface, why C, what you will build, who it is for, contents |
| `99-colophon.md` | Repository pointer, about the author, other titles |
| `build.py` | Assembly and typesetting. The stylesheet lives here as a string |
| `requirements.txt` | Python dependencies |
| `cover.png` | Generated. Page 1 at 200 dpi, for shop listings and social cards |
| `cover-square.png` | Generated. The same page cropped square, for shop thumbnails |

Chapters are picked up automatically from `../tutorial/chapter*.md` in filename order. Adding a chapter means adding a file; nothing here needs editing except the contents list in the front matter.

## Why the PDF is not committed

The tutorial is free and stays free; it is the reason anyone finds this repository. The PDF is the typeset convenience version, sold at [aelena74.gumroad.com](https://aelena74.gumroad.com), and building it yourself from these sources is both possible and permitted. If you would rather spend five minutes than the price of a coffee, the instructions above are complete and not deliberately hobbled.
