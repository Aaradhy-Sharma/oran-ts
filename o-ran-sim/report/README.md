# Mid-term Report (LaTeX)

## Structure
- `midterm_report.tex`: IEEE-style paper
- `references.bib`: bibliography

## Compile
On macOS with MacTeX installed:

```zsh
# From repo root
cd report
pdflatex midterm_report.tex
bibtex midterm_report
pdflatex midterm_report.tex
pdflatex midterm_report.tex
open midterm_report.pdf
```

If you don't have TeX installed, install MacTeX (large) or BasicTeX (lightweight) from https://tug.org/mactex/ and ensure `pdflatex` and `bibtex` are on your PATH.

## Notes
- Citations are in `references.bib`. Add more as needed.
- The paper intentionally avoids disclosing sensitive details from `work.pdf` and instead documents the agents and code structure.
