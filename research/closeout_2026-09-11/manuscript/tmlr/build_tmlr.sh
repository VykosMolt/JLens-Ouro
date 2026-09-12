#!/usr/bin/env bash
# Build the TMLR versions of draft three: anonymous submission and named preprint.
#   pandoc (natbib citations, tmlr-filter.lua) -> body; postprocess_body.py; pdflatex + bibtex with the official tmlr.sty/tmlr.bst.
# Outputs: tmlr/JLens_Ouro_TMLR_submission.pdf, tmlr/JLens_Ouro_TMLR_preprint.pdf, tmlr/main_*.tex, tmlr/body_*.tex
set -euo pipefail
cd "$(dirname "$0")/.."
T=tmlr; PY=/home/moloch/ouro_project/venv/bin/python
pandoc JLens_Ouro_Third_Draft.md --from markdown+raw_tex+fenced_divs --to latex --natbib --top-level-division=section \
       --lua-filter $T/tmlr-filter.lua -o $T/body_raw.tex
$PY -B $T/postprocess_body.py $T/body_raw.tex $T/body_preprint.tex preprint
$PY -B $T/postprocess_body.py $T/body_raw.tex $T/body_submission.tex submission
rm -f $T/body_raw.tex

AI_FOOTNOTE='\begingroup\renewcommand{\thefootnote}{}\footnotetext{\textbf{AI assistance.} Claude Fable 5.1 and GPT 6 Astra assisted with code and draft review, editing, bug fixes, result verification and archiving, and with the implementation, analysis, figures and drafting of the follow-up experiments. Every number, table and statement was verified by the author.}\endgroup'

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
  cat <<DOC
% Double-blind submission: tmlr.sty prints "Anonymous authors" and ignores \\author.
\\author{\\name Anonymous}
\\begin{document}
\\maketitle
% AI-assistance statement as an unnumbered first-page footnote.
$AI_FOOTNOTE
\\input{body.tex}
\\end{document}
DOC
} > $T/main_submission.tex
{
  preamble preprint
  cat <<DOC
\\author{\\name Jan Kirin \\email TODO@example.org \\\\
      \\addr TODO affiliation}
\\begin{document}
\\maketitle
$AI_FOOTNOTE
\\input{body.tex}
\\end{document}
DOC
} > $T/main_preprint.tex

for v in submission preprint; do
  B=$T/build_$v; rm -rf "$B"; mkdir -p "$B"
  cp $T/main_$v.tex "$B/main.tex"; cp $T/body_$v.tex "$B/body.tex"; cp references.bib "$B/"; cp $T/tmlr.sty $T/tmlr.bst "$B/"
  mkdir -p "$B/figures_v3"; cp figures_v3/*.pdf "$B/figures_v3/"
  ( cd "$B" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null ) \
    || { echo "== $v: LaTeX failed"; grep -m5 -A4 '^!' "$B/main.log"; exit 1; }
  cp "$B/main.pdf" $T/JLens_Ouro_TMLR_$v.pdf
  echo "== $v: $(pdfinfo "$B/main.pdf" | awk '/^Pages/{print $2}') pages; undefined: $(grep -c -i 'undefined' "$B/main.log"); multiply defined: $(grep -c -i 'multiply' "$B/main.log"); overfull hbox: $(grep -c 'Overfull \\hbox' "$B/main.log"); bibtex warnings: $(grep -c -i '^Warning' "$B/main.blg")"
done

# checks: numbering of hard-coded cross-references, footnote on page 1, anonymity of the submission
$PY -B - <<'CHK'
import re, subprocess, sys
from pathlib import Path
ok = True
aux = Path('tmlr/build_submission/main.aux').read_text(); body = Path('tmlr/build_submission/body.tex').read_text()
labels = {m.group(1): m.group(2) for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', aux)}
norm = lambda t: re.sub(r'\s+', ' ', t).strip()
sec = [(norm(t), l) for t, l in re.findall(r'\\(?:sub)?section\{(.+?)\}\\label\{([^}]+)\}', body, re.S)]
md = Path('JLens_Ouro_Third_Draft.md').read_text()
for num, title in re.findall(r'^#{1,2} (\d+(?:\.\d+)?)\.? (.+)$', md, re.M):
    hit = [l for t, l in sec if t == norm(title)]; got = labels.get(hit[0]) if hit else None
    if got != num: ok = False; print('  section mismatch', num, got, title)
for k in range(1, 7):
    if labels.get(f'tab:{k}') != str(k): ok = False; print('  table mismatch', k, labels.get(f'tab:{k}'))
for k in range(1, 6):
    if labels.get(f'fig:{k}') != str(k): ok = False; print('  figure mismatch', k, labels.get(f'fig:{k}'))
for L in 'ABC':  # appendices
    if labels.get(f'app:{L}') != L: ok = False; print('  appendix mismatch', L, labels.get(f'app:{L}'))
for v in ('submission', 'preprint'):
    p1 = subprocess.run(['pdftotext', '-f', '1', '-l', '1', f'tmlr/JLens_Ouro_TMLR_{v}.pdf', '-'], capture_output=True, text=True).stdout
    if 'AI assistance.' not in p1: ok = False; print(f'  {v}: AI-assistance footnote not on page 1')
full = subprocess.run(['pdftotext', 'tmlr/JLens_Ouro_TMLR_submission.pdf', '-'], capture_output=True, text=True).stdout
scrub = re.sub(r'Kirin \(2026\)|\(Kirin, 2026\)|Jan Kirin\.\s+Operational proto-introspection', '', full, flags=re.I)  # third-person self-citation is allowed
leaks = re.findall(r'Kirin|Vykos|github|illja|esterhazy|MATS', scrub, re.I)
if leaks: ok = False; print('  submission PDF leaks:', sorted(set(leaks)))
if 'Appendix D' in full: ok = False; print('  submission PDF still mentions Appendix D')
print('checks passed: cross-reference numbering, first-page footnote, anonymity' if ok else 'CHECKS FAILED'); sys.exit(0 if ok else 1)
CHK
