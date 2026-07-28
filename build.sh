#!/bin/bash
# Build the SenseAct ICLR 2026 paper.
# Usage: bash build.sh
# Prefers a user-space TeX Live at ~/texlive/2025 if present, otherwise falls
# back to the system TeX Live installed via `yum install texlive-scheme-basic ...`
# (see /usr/bin/pdflatex, TeX Live 2013 from EPEL/base).
set -e
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code/docs/our_paper_tex
if [ -d "$HOME/texlive/2025/bin/x86_64-linux" ]; then
  export PATH="$HOME/texlive/2025/bin/x86_64-linux:$PATH"
fi
echo "=== pdflatex: $(which pdflatex) ==="
pdflatex -interaction=nonstopmode AnonymousSubmission2027.tex < /dev/null
bibtex AnonymousSubmission2027 < /dev/null
pdflatex -interaction=nonstopmode AnonymousSubmission2027.tex < /dev/null
pdflatex -interaction=nonstopmode AnonymousSubmission2027.tex < /dev/null
echo "=== done ==="
echo "=== undefined citations remaining: $(grep -c 'Citation.*undefined' AnonymousSubmission2027.log || true) ==="
ls -la AnonymousSubmission2027.pdf
