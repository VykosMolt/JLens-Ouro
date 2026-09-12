#!/usr/bin/env bash
# Build the TMLR versions of draft three: anonymous submission and named preprint.
#   pandoc (natbib citations, tmlr-filter.lua) -> body.tex; pdflatex + bibtex with the official tmlr.sty/tmlr.bst.
# Outputs: tmlr/JLens_Ouro_TMLR_submission.pdf, tmlr/JLens_Ouro_TMLR_preprint.pdf, tmlr/main_*.tex, tmlr/body_*.tex
set -euo pipefail
cd "$(dirname "$0")/.."
T=tmlr
pandoc JLens_Ouro_Third_Draft.md --from markdown+raw_tex+fenced_divs --to latex --natbib --top-level-division=section \
       --lua-filter $T/tmlr-filter.lua -o $T/body_preprint.tex
# anonymous variant: withhold the repository link
sed -e 's|\\url{https://github.com/VykosMolt/JLens-Ouro}|[URL withheld for double-blind review]|' $T/body_preprint.tex > $T/body_submission.tex
grep -q 'github.com' $T/body_submission.tex && { echo "submission body still names the repository"; exit 1; }
grep -q 'URL withheld' $T/body_submission.tex || { echo "anonymization sed did not match"; exit 1; }

preamble() {  # $1 = tmlr option ("" or "preprint")
cat <<PRE
\\documentclass[10pt]{article}
\\usepackage${1:+[$1]}{tmlr}
\\usepackage{amsmath,amssymb}
\\usepackage{graphicx,booktabs,array,calc,longtable}
\\usepackage[table]{xcolor}
\\usepackage[section]{placeins}
\\usepackage{capt-of}
\\raggedbottom
\\usepackage{newunicodechar}
\\newunicodechar{−}{\\ensuremath{-}}
\\newunicodechar{→}{\\ensuremath{\\rightarrow}}
\\usepackage{hyperref}
\\usepackage{url}
\\hypersetup{colorlinks=true,linkcolor=black,citecolor=blue!55!black,urlcolor=blue!55!black}
\\providecommand{\\tightlist}{\\setlength{\\itemsep}{0pt}\\setlength{\\parskip}{0pt}}
\\newcommand{\\statusEstablished}{\\textsc{Established}}
\\newcommand{\\statusBounded}{\\textsc{Bounded screen}}
\\newcommand{\\statusUnresolved}{\\textsc{Unresolved}}
\\newcommand{\\statusDiagnostic}{\\textsc{Diagnostic}}
\\newcommand{\\statusRetracted}{\\textsc{Retracted}}
\\newcommand{\\statusHistorical}{\\textsc{Historical}}
\\newcommand{\\statusCorrected}{\\textsc{Corrected}}
\\title{Final-Target J-Lens in Ouro: Early-Pass Deficits and a Confirmed Late-Pass Advantage}
PRE
}
{
  preamble ""
  cat <<'DOC'
% Double-blind submission: tmlr.sty prints "Anonymous authors" and ignores \author.
\author{\name Anonymous}
\begin{document}
\maketitle
\input{body.tex}
\end{document}
DOC
} > $T/main_submission.tex
{
  preamble preprint
  cat <<'DOC'
\author{\name Jan Kirin \email TODO@example.org \\
      \addr TODO affiliation}
\begin{document}
\maketitle
\input{body.tex}
\end{document}
DOC
} > $T/main_preprint.tex

for v in submission preprint; do
  B=$T/build_$v; rm -rf "$B"; mkdir -p "$B"
  cp $T/main_$v.tex "$B/main.tex"; cp $T/body_$v.tex "$B/body.tex"; cp references.bib "$B/"; cp $T/tmlr.sty $T/tmlr.bst "$B/"
  cp -r figures_v3 "$B/figures_v3"; rm -f "$B"/figures_v3/*.png "$B"/figures_v3/*.py
  ( cd "$B" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null ) \
    || { echo "== $v: LaTeX failed"; grep -m5 -A4 '^!' "$B/main.log"; exit 1; }
  cp "$B/main.pdf" $T/JLens_Ouro_TMLR_$v.pdf
  echo "== $v: $(pdfinfo "$B/main.pdf" | awk '/^Pages/{print $2}') pages; undefined refs/cites: $(grep -c -i 'undefined' "$B/main.log"); multiply defined: $(grep -c -i 'multiply' "$B/main.log"); overfull hbox: $(grep -c 'Overfull \\hbox' "$B/main.log"); bibtex warnings: $(grep -c -i 'warning' "$B/main.blg")"
done
