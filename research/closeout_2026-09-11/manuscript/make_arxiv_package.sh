#!/usr/bin/env bash
# Build the arXiv source package for draft three.
#   1. rebuild body.tex + main.pdf with build_v3.sh (pandoc + XeLaTeX)
#   2. stage a self-contained source tree: main.tex, body.tex, the kirin style flattened
#      (its ../paper_style inputs rewritten), the five figure PDFs, 00README.json
#   3. compile the staged tree in isolation (empty HOME, no system fontconfig) with XeLaTeX,
#      so the fonts must come from TeX Live by file name, as on arXiv
#   4. write arxiv/JLens_Ouro_arXiv_v3.tar.gz (+ sha256) and arxiv/SUBMISSION_NOTES.md
set -euo pipefail
cd "$(dirname "$0")"
PS=/home/moloch/Documents/Research/paper_style
OUT=arxiv; STAGE=$OUT/JLens_Ouro_arXiv_src; TAR=$OUT/JLens_Ouro_arXiv_v3.tar.gz

bash build_v3.sh >/dev/null
REF_PAGES=$(pdfinfo build_v3/main.pdf | awk '/^Pages/{print $2}')

rm -rf "$STAGE"; mkdir -p "$STAGE/figures_v3"
sed -e 's|\\input{\.\./paper_style/palette\.tex}|\\input{kirin-palette.tex}|' \
    -e 's|\\input{\.\./paper_style/boxes\.tex}|\\input{kirin-boxes.tex}|' \
    -e 's|\\input{\.\./paper_style/tables\.tex}|\\input{kirin-tables.tex}|' \
    -e 's|\\input{\.\./paper_style/titlepage\.tex}|\\input{kirin-titlepage.tex}|' \
    "$PS/kirin-papers.sty" > "$STAGE/kirin-papers.sty"
grep -q 'paper_style' "$STAGE/kirin-papers.sty" && { echo "unrewritten path in sty"; exit 1; }
cp "$PS/palette.tex" "$STAGE/kirin-palette.tex"; cp "$PS/boxes.tex" "$STAGE/kirin-boxes.tex"
cp "$PS/tables.tex" "$STAGE/kirin-tables.tex"; cp "$PS/titlepage.tex" "$STAGE/kirin-titlepage.tex"
sed -e 's|\\usepackage\[paper1\]{\.\./paper_style/kirin-papers}|\\usepackage[paper1]{kirin-papers}|' JLens_Ouro_Third_Draft.tex > "$STAGE/main.tex"
cp build_v3/body.tex "$STAGE/body.tex"
for f in fig1_discovery fig2_confirmation fig3_same_band fig4_contrasts fig5_local_exit; do cp "figures_v3/$f.pdf" "$STAGE/figures_v3/"; done
grep -o 'figures_v3/[A-Za-z0-9_]*\.pdf' "$STAGE/body.tex" | sort -u | while read -r f; do [ -f "$STAGE/$f" ] || { echo "missing $f"; exit 1; }; done
cat > "$STAGE/00README.json" <<'JSON'
{
  "spec_version": 1,
  "texlive_version": 2025,
  "process": { "compiler": "xelatex" },
  "sources": [ { "filename": "main.tex", "usage": "toplevel" } ]
}
JSON

# isolated compile: no system fonts, no user config, only TeX Live
T=$(mktemp -d); cp -r "$STAGE" "$T/src"; mkdir -p "$T/home"
printf '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig><cachedir>%s/fc</cachedir></fontconfig>\n' "$T/home" > "$T/home/fonts.conf"
run() { ( cd "$T/src" && env -i PATH="$PATH" HOME="$T/home" FONTCONFIG_FILE="$T/home/fonts.conf" XDG_DATA_HOME="$T/home" LANG=C.UTF-8 xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 ); }
run && run || { echo "isolated XeLaTeX compile FAILED"; grep -m3 -A3 '^!' "$T/src/main.log"; exit 1; }
PAGES=$(pdfinfo "$T/src/main.pdf" | awk '/^Pages/{print $2}')
[ "$PAGES" = "$REF_PAGES" ] || { echo "page count differs: isolated $PAGES vs build $REF_PAGES"; exit 1; }
cmp -s <(pdftotext "$T/src/main.pdf" - 2>/dev/null) <(pdftotext build_v3/main.pdf - 2>/dev/null) || { echo "text differs from build_v3"; exit 1; }
FONTS=$(pdffonts "$T/src/main.pdf" 2>/dev/null | awk 'NR>2{print $1}' | sed 's/^[A-Z]*+//; s/-Identity-H//' | sort -u | tr '\n' ' ')
cp "$T/src/main.pdf" "$OUT/JLens_Ouro_arXiv_v3_compiled_check.pdf"
rm -rf "$T"

rm -f "$TAR"; tar -C "$STAGE" -czf "$TAR" --owner=0 --group=0 --numeric-owner --mtime='2026-09-12 00:00Z' --sort=name .
SHA=$(sha256sum "$TAR" | cut -d' ' -f1); SIZE=$(stat -c%s "$TAR")
ABSTRACT=$(/home/moloch/ouro_project/venv/bin/python -B -c "
import re; s=open('JLens_Ouro_Third_Draft.md',encoding='utf-8').read()
a=re.search(r'::: abstractbox\s*\n(.*?)\n:::',s,re.S).group(1).strip(); a=re.sub(r'\*\*(.+?)\*\*',r'\1',a); print(a)")
cat > "$OUT/SUBMISSION_NOTES.md" <<NOTES
# arXiv submission package — draft three

Built $(date -u +%Y-%m-%dT%H:%MZ) by \`make_arxiv_package.sh\` from commit $(git rev-parse --short HEAD).

- Archive: \`$(basename "$TAR")\`, $SIZE bytes, sha256 \`$SHA\`.
- Compiler to select on arXiv: **XeLaTeX** (declared in \`00README.json\`, TeX Live 2025; built locally with $(xelatex --version | head -1 | sed 's/ (.*//') on TeX Live 2026).
- Contents (flat, \`main.tex\` at the root): \`main.tex\`, \`body.tex\`, \`kirin-papers.sty\`, \`kirin-palette.tex\`, \`kirin-boxes.tex\`, \`kirin-tables.tex\`, \`kirin-titlepage.tex\`, \`figures_v3/fig1_discovery.pdf\` … \`fig5_local_exit.pdf\`, \`00README.json\`. No bibliography files are needed: the references are typeset inline by pandoc's citeproc in \`body.tex\`.
- Fonts: Libertinus and Source Sans Pro are loaded by file name through the TeX Live packages \`libertinus\` and \`sourcesanspro\`, which is the lookup arXiv requires for XeLaTeX. Nothing is bundled.
- Verification: the staged tree compiled twice with XeLaTeX in an isolated environment (empty HOME, fontconfig pointed at an empty configuration, so no system font was visible): $PAGES pages, text identical to \`build_v3/main.pdf\`, embedded fonts: $FONTS. The compiled check copy is \`JLens_Ouro_arXiv_v3_compiled_check.pdf\` (not part of the archive).

## Metadata for the submission form

- Title: Final-Target J-Lens in Ouro: Early-Pass Deficits and a Confirmed Late-Pass Advantage
- Author: Jan Kirin
- Comments: $PAGES pages, 5 figures, 6 tables. Code and records: https://github.com/VykosMolt/JLens-Ouro
- Abstract (${#ABSTRACT} characters; arXiv's limit is 1,920):

$ABSTRACT

## Before uploading

- The title block still says "Research paper · working draft v3" and the footnote line "Working draft v3 (12 September 2026)…"; change or keep as you prefer (\`main.tex\`, the \`\\KirinTitleBlock\` call).
- Category and license are yours to choose on the form (the work is machine-learning interpretability; arXiv's default license is the non-exclusive distribution license).
- arXiv adds its identifier stamp in the left margin; \`stamp\` is left at the default.
NOTES
echo "package: $TAR ($SIZE bytes) sha256 $SHA; isolated compile $PAGES pages; fonts: $FONTS"
