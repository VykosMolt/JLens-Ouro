# arXiv submission package — draft three

Built 2026-09-12T18:06Z by `make_arxiv_package.sh` from commit 6166a17.

- Archive: `JLens_Ouro_arXiv_v3.tar.gz`, 128040 bytes, sha256 `32d910117a616e261f16452fc15fec1b3148db16930b17b3d5cf46be7f0f8f94`.
- Compiler to select on arXiv: **XeLaTeX** (declared in `00README.json`, TeX Live 2025; built locally with XeTeX 3.141592653-2.6-0.999998 on TeX Live 2026).
- Contents (flat, `main.tex` at the root): `main.tex`, `body.tex`, `kirin-papers.sty`, `kirin-palette.tex`, `kirin-boxes.tex`, `kirin-tables.tex`, `kirin-titlepage.tex`, `figures_v3/fig1_discovery.pdf` … `fig5_local_exit.pdf`, `00README.json`. No bibliography files are needed: the references are typeset inline by pandoc's citeproc in `body.tex`.
- Fonts: Libertinus and Source Sans Pro are loaded by file name through the TeX Live packages `libertinus` and `sourcesanspro`, which is the lookup arXiv requires for XeLaTeX. Nothing is bundled.
- Verification: the staged tree compiled twice with XeLaTeX in an isolated environment (empty HOME, fontconfig pointed at an empty configuration, so no system font was visible): 14 pages, text identical to `build_v3/main.pdf`, embedded fonts: LibertinusMath-Regular LibertinusMono-Regular LibertinusSerif-Bold LibertinusSerif-Italic LibertinusSerif-Regular SourceSansPro-Regular SourceSansPro-Semibold . The compiled check copy is `JLens_Ouro_arXiv_v3_compiled_check.pdf` (not part of the archive).

## Metadata for the submission form

- Title: J-Lens in Ouro: A Confirmed Late-Pass Band and a Target-Dependent Early-Pass Deficit
- Author: Jan Kirin
- Comments: 14 pages, 5 figures, 6 tables. Code and records: https://github.com/VykosMolt/JLens-Ouro
- Abstract (1917 characters; arXiv's limit is 1,920):

Does a Jacobian-based readout reveal intermediate content that a model's own output head misses during recurrent computation? I compare J-Lens with the raw logit lens in frozen Ouro-2.6B, which applies 48 shared blocks over four recurrent passes. Layerwise discovery analysis found a positive J-Lens region in pass 4 whose location and size five independent calibration fits reproduced. I then froze one estimator (fit01) and that band, physical layers 26–37 of pass 4, before evaluating 160 new two-hop questions with 80 new intermediate concepts. In the band, J-Lens improves excess hit@10 by 23.19 percentage points (95% dependency-group bootstrap interval 16.29–32.90); intended-concept recovery is 37.92% against 14.43% for the raw lens. Under the same final-target estimator, the same physical band shows the opposite sign in passes 1–3 (−6.5, −8.6 and −11.1 points; post-confirmation, simultaneous intervals excluding zero), so the pattern is pass-dependent at fixed depth. The late advantage remains positive with within-domain controls (20.8 points, 9.1–32.5) and under an exhaustive tokenized-input audit, FP64 reference scoring and a cross-hardware state replay (changes of at most 0.42 points), but it depends strongly on the estimator: changing the derivative target or the position aggregation substantially reduces it. In a post-confirmation analysis on the same states, initial-study banks targeting each pass's own exit reverse the deficit: in the band they exceed the final-target lens of their family by 22.6, 26.5 and 34.2 points and the raw lens by 16.4, 18.3 and 23.2 points (passes 1–3; simultaneous intervals exclude zero), so the early deficit belongs to the estimator's target, not to the passes. The result supports a localized, estimator- and target-specific improvement in concept recovery, not general J-Lens superiority, improved answer accuracy, or causal use of the recovered content.

## Before uploading

- The title block still says "Research paper · working draft v3" and the footnote line "Working draft v3 (12 September 2026)…"; change or keep as you prefer (`main.tex`, the `\KirinTitleBlock` call).
- Category and license are yours to choose on the form (the work is machine-learning interpretability; arXiv's default license is the non-exclusive distribution license).
- arXiv adds its identifier stamp in the left margin; `stamp` is left at the default.
