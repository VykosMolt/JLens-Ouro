# Build the application

The PDF uses Jan's saved `kirin-papers` LaTeX package from Documents/Research. A local copy is in the sibling `paper_style` directory; the title block links to Jan's GitHub profile. The figure is a vector PDF.

From this directory, run twice:

```sh
xelatex -interaction=nonstopmode -halt-on-error MATS12_application_Jan_Kirin.tex
```

The reader-facing PDF is also copied one directory up. Form answers are in `../FORM_ANSWERS.md` and are separate from the research write-up.
