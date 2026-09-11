#!/usr/bin/env bash
# Build draft three in the Kirin visual system: markdown body -> LaTeX (pandoc + kirin-boxes.lua) -> XeLaTeX.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_v3 && rm -f build_v3/paper_style && ln -sfn /home/moloch/Documents/Research/paper_style paper_style && ln -sfn ../figures_v3 build_v3/figures_v3
pandoc JLens_Ouro_Third_Draft.md --from markdown+raw_tex+fenced_divs --to latex --lua-filter kirin-boxes.lua --citeproc --bibliography references.bib -o build_v3/body.tex
cp JLens_Ouro_Third_Draft.tex build_v3/main.tex
cd build_v3 && xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && echo "built build_v3/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
