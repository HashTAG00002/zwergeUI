#!/bin/bash
# One-shot setup for compiling docs/our_paper_tex/*.tex on this CodeLab
# container (CentOS 7, EPEL/base yum repos, no internet access to CTAN/
# VSCode Marketplace). Installs a system-wide TeX Live 2013 via yum plus
# inotify-tools for the save-triggers-compile watcher.
#
# Usage:
#   bash setup_latex_env.sh
#
# Idempotent: safe to re-run; yum will just say "already installed".
# Requires: sudo yum install permission (no conda / Python env needed).

set -e

echo "=== [1/3] Installing base TeX Live scheme + LaTeX package collections ==="
sudo yum install -y \
  texlive-scheme-basic \
  texlive-collection-latex \
  texlive-collection-latexrecommended \
  texlive-collection-fontsrecommended \
  texlive-bibtex-bin \
  texlive-bibtex

echo "=== [2/3] Installing extra packages needed by this paper's .tex/.sty ==="
sudo yum install -y \
  texlive-iftex \
  texlive-newtx \
  texlive-comment \
  texlive-environ \
  texlive-makecell \
  texlive-trimspaces \
  texlive-subfigure \
  texlive-placeins \
  texlive-tcolorbox \
  texlive-cjk \
  texlive-multirow

echo "=== [3/3] Installing inotify-tools (for watch_build.sh auto-compile-on-save) ==="
sudo yum install -y inotify-tools

echo ""
echo "=== Verifying ==="
echo "pdflatex: $(which pdflatex)"
pdflatex --version | head -1
echo "bibtex:   $(which bibtex)"
bibtex --version | head -1
echo "inotifywait: $(which inotifywait)"

echo ""
echo "=== Done. Next steps: ==="
echo "  bash build.sh                 # compile once"
echo "  nohup bash watch_build.sh > /tmp/watch_build.log 2>&1 &   # auto-compile on save"
