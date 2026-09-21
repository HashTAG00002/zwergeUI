#!/bin/bash
# Build script for main paper + supplementary material
# Handles cross-document references by injecting main-paper labels into supplementary
# Key: inject AFTER first pdflatex (which creates .aux) but BEFORE the final pdflatex
# (which reads .aux). LaTeX preserves existing \newlabel entries when it rewrites .aux.

set -e
cd "$(dirname "$0")"

echo "=== Building main paper ==="
rm -f AnonymousSubmission2027.aux AnonymousSubmission2027.bbl
pdflatex -interaction=nonstopmode AnonymousSubmission2027.tex > /dev/null 2>&1
bibtex AnonymousSubmission2027 > /dev/null 2>&1
pdflatex -interaction=nonstopmode AnonymousSubmission2027.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode AnonymousSubmission2027.tex > /dev/null 2>&1

MAIN_PAGES=$(grep "Output written" AnonymousSubmission2027.log | tail -1 | grep -oP '\d+ pages' | head -1)
echo "Main paper: ${MAIN_PAGES}"

echo ""
echo "=== Building supplementary material ==="
rm -f supplementary.aux supplementary.bbl
# First pass: creates supplementary.aux with local labels
pdflatex -interaction=nonstopmode supplementary.tex > /dev/null 2>&1
bibtex supplementary > /dev/null 2>&1
# Second pass: resolves internal references
pdflatex -interaction=nonstopmode supplementary.tex > /dev/null 2>&1

# Inject main-paper labels into supplementary .aux
# This MUST happen between the second and third pdflatex passes.
# LaTeX reads .aux at \begin{document} and preserves existing \newlabel when rewriting.
python3 inject_labels.py

# Third pass: LaTeX reads the augmented .aux and resolves cross-document references
pdflatex -interaction=nonstopmode supplementary.tex > /dev/null 2>&1

SUPP_PAGES=$(grep "Output written" supplementary.log | tail -1 | grep -oP '\d+ pages' | head -1)
echo "Supplementary: ${SUPP_PAGES}"

echo ""
echo "=== Done ==="
echo "Main paper:    AnonymousSubmission2027.pdf (${MAIN_PAGES})"
echo "Supplementary: supplementary.pdf (${SUPP_PAGES})"
