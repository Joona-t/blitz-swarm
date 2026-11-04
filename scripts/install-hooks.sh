#!/bin/sh
# Point this checkout (or worktree) at the versioned hooks. Run once:
#   sh scripts/install-hooks.sh [/path/to/venv/bin/python]
set -eu
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath scripts/hooks
if [ "${1:-}" != "" ]; then git config blitz.python "$1"; fi
echo "hooks installed: $(git config core.hooksPath)"
