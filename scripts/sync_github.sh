#!/usr/bin/env bash
# ==============================================================
# 推送仓库总代码到 GitHub（https://github.com/HashTAG00002/zwergeUI）
#
# 用法：
#   bash scripts/sync_github.sh            # push 当前分支到 origin
#   bash scripts/sync_github.sh master     # 显式指定分支
#
# 凭据：git credential.helper=store，有效 token 存于 ~/.git-credentials
#   （⚠ 不要使用 .git/config 里曾内嵌的 PAT——已过期 401；
#     remote URL 应保持为无内嵌凭据的干净形式，由 credential store 接管）
# 代理：~/.gitconfig 全局 http.proxy=http://10.70.16.106:3128（2026-09-22 修复，
#   原仓库本地 .git/config 内嵌死代理 10.70.11.143:8412 已删），裸 git 命令即可联网；
#   下方 -c 覆盖保留为双保险（-c 优先级最高）。
# ==============================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
PROXY="http://10.70.16.106:3128"

cd "${REPO_ROOT}"

# 防御：remote URL 若被写成内嵌凭据形式，重置为干净 URL（credential store 接管）
CLEAN_URL="https://github.com/HashTAG00002/zwergeUI.git"
if [ "$(git remote get-url origin)" != "${CLEAN_URL}" ]; then
  echo ">> 修正 origin URL 为无内嵌凭据形式"
  git remote set-url origin "${CLEAN_URL}"
fi

echo ">> push ${BRANCH} -> origin (github.com/HashTAG00002/zwergeUI)"
git -c http.proxy="${PROXY}" push origin "${BRANCH}"
echo "✅ GitHub 推送完成"
