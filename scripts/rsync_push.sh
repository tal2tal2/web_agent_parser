#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER_HOST="oz.diamond@lambda.cs.technion.ac.il"
REMOTE_PATH="/home/oz.diamond/Projects/web_agent_parser"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_github}"

EXCLUDES=(
  ".venv/"
  "data/"
  "__pycache__/"
  "*.pyc"
  ".pytest_cache/"
  ".mypy_cache/"
  ".ruff_cache/"
  ".git/"
  ".DS_Store"
)

RSYNC_EXCLUDES=()
for item in "${EXCLUDES[@]}"; do
  RSYNC_EXCLUDES+=(--exclude "$item")
done

RSYNC_DELETE_FLAG=()
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_DELETE_FLAG=(--delete)
fi

rsync -az "${RSYNC_EXCLUDES[@]}" "${RSYNC_DELETE_FLAG[@]}" -e "ssh -i ${SSH_KEY_PATH}" ./ "${REMOTE_USER_HOST}:${REMOTE_PATH}/"
