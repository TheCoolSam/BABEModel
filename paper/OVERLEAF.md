# Compiling the JASSS manuscript

This tree uses the official Bravo **JASSS** LaTeX class (v0.6).

## Files

| File | Role |
|------|------|
| `main.tex` | Author-identified manuscript (`\reviewcopy{false}`) |
| `main_blind.tex` | Anonymised review copy (`\reviewcopy{true}`) |
| `JASSS.cls`, `jasssbib.bst`, `bar.png`, `jassslogo.png`, `SectionIcon.png` | Class assets |
| `sections/*.tex`, `references.bib` | Content |
| `jasss/pkg/` | Vendor copy of the original journal zip |

Figures live in `../figures/` (repo root) or `./figures/` in the submission zip. `\graphicspath{{../}{./}}` covers both.

## Windows naming note

On case-insensitive filesystems, `jasss.bst` clashes with `JASSS.cls`. This tree uses **`jasssbib.bst`** and **`jassslogo.png`** (class patched accordingly).

## Build (local)

Requires pdfLaTeX, BibTeX, and Source Sans Pro (TeX Live / MiKTeX). A `tools/tectonic` binary can also compile.

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main

pdflatex main_blind
bibtex main_blind
pdflatex main_blind
pdflatex main_blind
```

## Build (Overleaf)

1. Upload `paper/` plus repo-root `figures/`, or upload the `latex/` folder from `JASSS_submission.zip`.
2. Set main document to `main.tex` or `main_blind.tex`.
3. Compiler: pdfLaTeX.

Official template: https://www.overleaf.com/latex/templates/jasss-article-template/pnrdvncxsjmn

## Still needed from authors

- ~100-word biographies ×4 (replace `TODO` in `main.tex`)
- Corresponding author email (`\email{...}`)
- CoMSES reviewer/deposit URL (replace “intend to deposit” in `main.tex`, `main_blind.tex`, and `sections/02_introduction.tex`)
