#!/bin/sh
# Install the repo's git hooks.
#
# git does not track .git/hooks, so a fresh clone has none. Point core.hooksPath
# at the versioned directory instead of copying, so hook edits take effect
# without a reinstall.
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

chmod +x scripts/git-hooks/*
git config core.hooksPath scripts/git-hooks

printf '%s\n' "core.hooksPath -> scripts/git-hooks"
printf '%s\n' "installed: $(ls scripts/git-hooks | tr '\n' ' ')"
