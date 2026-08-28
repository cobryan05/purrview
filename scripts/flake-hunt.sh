#!/usr/bin/env bash
# Hunt intermittent failures by running a suite repeatedly and diffing which
# tests failed in *some* runs but not all.
#
# Why this exists: this repo's Playwright suite runs on the same host as
# frigate / yolo2mqtt / jellyfin, whose CPU use is bursty. A test that passes
# on a quiet box can miss a fixed `wait_for_timeout` when frigate is mid
# detection. That failure is invisible to a single green run, so catching it
# needs repetition, ideally with the box under load.
#
# Usage:
#   scripts/flake-hunt.sh [-n RUNS] [-s] [-- PYTEST_ARGS...]
#
#   -n RUNS   how many times to run the suite (default 5)
#   -s        stress: pin the spare cores busy while the suite runs, to
#             emulate a frigate burst. Use this — most flakes will not
#             reproduce on an idle box.
#
# Examples:
#   scripts/flake-hunt.sh -n 5 -s                       # whole Playwright suite
#   scripts/flake-hunt.sh -n 10 -s -- playwright_tests/test_settings.py
#   scripts/flake-hunt.sh -n 5 -- tests/               # the unit suite instead
#
# Exit status is 1 if any test was flaky, so CI can gate on it.

set -uo pipefail

CONTAINER=encounterwatch_mono_dev
RUNS=5
STRESS=0

while getopts ":n:s" opt; do
  case $opt in
    n) RUNS=$OPTARG ;;
    s) STRESS=1 ;;
    *) echo "unknown flag -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift

TARGET=("$@")
if [ ${#TARGET[@]} -eq 0 ]; then
  TARGET=(playwright_tests/)
fi

# playwright_tests needs -p no:asyncio; the unit suite must not get it.
EXTRA=()
case "${TARGET[0]}" in
  playwright_tests*) EXTRA=(-p no:asyncio) ;;
esac

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

HOGS=()
if [ "$STRESS" = "1" ]; then
  # Leave a couple of cores free so the run still makes progress.
  n=$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))
  echo "==> starting $n CPU hogs to emulate a frigate burst"
  for _ in $(seq "$n"); do ( while :; do :; done ) & HOGS+=($!); done
  trap 'kill "${HOGS[@]}" 2>/dev/null; rm -rf "$OUT"' EXIT
fi

echo "==> ${RUNS} runs of: ${TARGET[*]}"
for i in $(seq "$RUNS"); do
  # Low priority for the same reason as scripts/test.sh: this runs the suite N
  # times over on the box serving the live app.
  docker exec -w /project "$CONTAINER" \
    nice -n 19 ionice -c3 pytest "${TARGET[@]}" "${EXTRA[@]}" -q > "$OUT/run$i.txt" 2>&1
  # `-q` prints one FAILED/ERROR line per failure in the short summary.
  grep -oE '^(FAILED|ERROR) [^ ]+' "$OUT/run$i.txt" | awk '{print $2}' | sort -u > "$OUT/fail$i.txt"
  printf '   run %-3s %-46s %s failed\n' "$i" "$(tail -1 "$OUT/run$i.txt")" \
    "$(wc -l < "$OUT/fail$i.txt")"
done

cat "$OUT"/fail*.txt | sort | uniq -c | sort -rn > "$OUT/tally.txt"

if [ ! -s "$OUT/tally.txt" ]; then
  echo
  echo "==> no failures in $RUNS runs."
  [ "$STRESS" = "0" ] && echo "    Re-run with -s; an idle box hides most timing flakes."
  exit 0
fi

echo
echo "==> results over $RUNS runs"
flaky=0
while read -r count name; do
  if [ "$count" -eq "$RUNS" ]; then
    printf '    BROKEN  %s (failed every run)\n' "$name"
  else
    printf '    FLAKY   %s (failed %s/%s)\n' "$name" "$count" "$RUNS"
    flaky=1
  fi
done < "$OUT/tally.txt"

if [ "$flaky" = "1" ]; then
  echo
  echo "==> first traceback for each flaky test:"
  while read -r count name; do
    [ "$count" -eq "$RUNS" ] && continue
    for i in $(seq "$RUNS"); do
      if grep -q "$name" "$OUT/fail$i.txt" 2>/dev/null; then
        echo "--- $name (run $i) ---"
        awk "/^_+ .*${name##*::}/,/^(_{5,}|=|FAILED)/" "$OUT/run$i.txt" | head -40
        break
      fi
    done
  done < "$OUT/tally.txt"
fi

exit "$flaky"
