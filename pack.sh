#!/usr/bin/env bash
# pack.sh — 一键打包当前 HEAD 的全部 git tracked 文件为 zip（放仓库根目录 zwerge/code）
#
# 原理：git archive 只打包 git 跟踪的文件，天然等于 ".gitignore 排除后剩下的全部内容"
#       （docs/references/ 竞品原文、docs/our_tex/scripts/ + .env.local 凭据、
#        wandb/、results/、__pycache__、LaTeX 构建产物、docs/our_tex/*.pdf 一律不进包）。
# 用法：
#   bash pack.sh                 # 输出 ZwerGeUI_main_<shortSHA>.zip 到仓库根目录
#   bash pack.sh myname.zip      # 自定义输出名
set -euo pipefail
cd "$(dirname "$0")"

SHA="$(git rev-parse --short HEAD)"
OUT="${1:-ZwerGeUI_main_${SHA}.zip}"

rm -rf *.zip
git archive --format=zip -o "$OUT" HEAD

# 计数用 unzip -Z1（一行一个条目）而不是 unzip -l | awk NF==4——后者是两个
# 假账源：①表头分隔线 "---------  ---------- -----   ----" 恰好 NF==4 被
# 多算一行；②文件名带空格会撑成 NF==5 被漏数。zwerge 当前 184 个 tracked
# 文件无空格名，但此计数法沿用 a2ui/pack.sh 的实证结论（2026-08-29：zip
# 内容与 tracked 逐字节一致，当时 684/685 纯属 awk 计数 bug），对任何仓库稳健。
N_FILES="$(unzip -Z1 "$OUT" | grep -cv '/$')"
N_TRACKED="$(git ls-files | wc -l)"
echo "packed : ${OUT}"
echo "sha    : ${SHA} ($(git rev-parse HEAD))"
echo "files  : ${N_FILES} in zip / ${N_TRACKED} tracked"
[ "${N_FILES}" -eq "${N_TRACKED}" ] || { echo "MISMATCH: zip files != tracked files"; exit 1; }
