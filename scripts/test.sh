#!/usr/bin/env bash
# Run a test suite in the container at low priority.
#
# Why this exists: the suites run on the same host as the live PurrView
# services (and frigate, jellyfin, …). A full Playwright run saturates all 16
# cores — each Chromium test costs roughly two — and the app becomes visibly
# unresponsive while it does. The tests have no deadline; the app does. `nice`
# and `ionice` express exactly that, per process rather than per container, so
# the live services keep winning the scheduler without the tests being capped
# when the box is otherwise idle.
#
# This does NOT slow the suite down in practice: the losing side of that
# contention was wall time nobody was waiting on, and worker count is already
# capped at 8 in pytest.ini for the same reason (see the comment there).
#
# The container requirement is not optional — see CLAUDE.md. This wrapper is
# also how you get it right by default.
#
# Usage:
#   scripts/test.sh                       # the unit suite (tests/)
#   scripts/test.sh playwright            # the Playwright suite, correct flags
#   scripts/test.sh both                  # both, sequentially; non-zero if either fails
#   scripts/test.sh tests/db_server/ -x   # any pytest args, passed straight through
#   scripts/test.sh playwright playwright_tests/test_tabs.py
#
# Anything after the suite word goes to pytest verbatim, so `-n 0`, `-k`, `-x`
# and friends all work and override the defaults.

set -uo pipefail

CONTAINER=purrview_mono_dev
# Lowering priority never needs privileges; raising it would.
PRIORITY=(nice -n 19 ionice -c3)

# Near-miss tracing (playwright_tests/waits.py). Exported here rather than left to the
# caller because `docker exec` does not inherit the host environment — a bare
# `FLAKE_TRACE=… scripts/test.sh` would silently trace nothing.
#
#   FLAKE_TRACE=/project/near-misses.txt scripts/test.sh playwright
#
# records every `wait_until` that burned more than FLAKE_TRACE_RATIO (default 0.5) of
# its timeout. Those waits passed — they are the flakes that have not happened yet.
TRACE_ENV=()
[ -n "${FLAKE_TRACE:-}" ] && TRACE_ENV+=(-e "FLAKE_TRACE=$FLAKE_TRACE")
[ -n "${FLAKE_TRACE_RATIO:-}" ] && TRACE_ENV+=(-e "FLAKE_TRACE_RATIO=$FLAKE_TRACE_RATIO")

run() {
  echo "==> pytest $*"
  docker exec "${TRACE_ENV[@]+"${TRACE_ENV[@]}"}" -w /project "$CONTAINER" "${PRIORITY[@]}" pytest "$@"
}

# True when the args contain a test path rather than only flags.
#
# This matters more than it looks: pytest.ini sets `testpaths = tests`, so a
# command that ends up with no path at all silently runs the *unit* suite. Under
# `playwright` that also hands it `-p no:asyncio`, which turns off
# `asyncio_mode = auto` and errors out every async test in the repo. `test.sh
# playwright -q` did exactly that before this check existed.
has_path() {
  for a in "$@"; do
    case "$a" in -*) ;; *) return 0 ;; esac
  done
  return 1
}

case "${1:-unit}" in
  playwright|pw)
    shift
    # playwright_tests must disable the asyncio plugin; the unit suite must not.
    if has_path "$@"; then
      run "$@" -p no:asyncio
    else
      run playwright_tests/ "$@" -p no:asyncio
    fi
    ;;
  both)
    shift
    run tests/ "$@"
    unit=$?
    run playwright_tests/ -p no:asyncio "$@"
    pw=$?
    echo
    echo "==> unit: $([ $unit -eq 0 ] && echo PASS || echo FAIL)   playwright: $([ $pw -eq 0 ] && echo PASS || echo FAIL)"
    [ $unit -eq 0 ] && [ $pw -eq 0 ]
    ;;
  unit)
    shift
    run tests/ "$@"
    ;;
  *)
    # A path or pytest flag: infer the plugin flag from the first argument, and
    # never leave pytest without a path (see has_path).
    case "$1" in
      playwright_tests*) run "$@" -p no:asyncio ;;
      -*)                run tests/ "$@" ;;
      *)                 run "$@" ;;
    esac
    ;;
esac
