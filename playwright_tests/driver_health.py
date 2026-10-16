"""Telling "this worker's Playwright is dead" apart from "that page is closed".

When the node driver process behind a worker's Playwright connection dies, the worker's
*Python* side stays perfectly healthy and keeps being handed tests. Every one of them
then fails in the `page` fixture with the same error, so one dead process became **1100
fixture errors across twelve minutes** in GitHub Actions run 38898595092 — a 332k-line
log in which nothing names the cause. See FLAKY_TESTS.md, 2026-10-16.

Two separate problems; this file is about the second:

1. *Why* the driver dies. Unknown — a kernel OOM kill of `node` is the standing
   hypothesis for a 4-vCPU runner hosting four chromiums, and nothing records an exit
   reason because the kill happens inside the container.
2. The **cascade**, which is a defect on its own. It costs the rest of the run and
   buries the first sighting under a thousand identical ones.

So the run is stopped at the first sighting, loudly, with the memory numbers taken *at
that moment* — which is the evidence problem (1) needs and which nothing was capturing.

⚠️ **The predicate must stay narrow.** Ordinary page-closed errors are routine and
expected: a test that navigates away, a context torn down at module scope, a `p.close()`
racing its own teardown. Matching those would abort healthy runs, which is a worse
failure than the cascade — so this matches only the message Playwright uses for the
*driver connection* itself, never the one for a closed page or context.
"""

from pathlib import Path

#: Messages that mean the driver connection is gone, so every later call must fail too.
#:
#: Playwright raises `TargetClosedError` for both a closed page and a dead driver, and
#: only the message separates them. The first is what run 38898595092 reported 1100
#: times; the second is the same condition surfaced from inside the transport.
#:
#: Deliberately NOT included: "Target page, context or browser has been closed" and
#: "Target closed" — those are the ordinary, recoverable closures above.
DEAD_DRIVER_MARKERS = (
    "Playwright connection closed",
    "Connection closed while reading from the driver",
)


def is_dead_driver(exc: BaseException) -> bool:
    """Whether this exception means the worker's Playwright driver has died.

    Matched on the message rather than the type: `TargetClosedError` covers a closed
    page too, and closing a page is something the suite does on purpose after every
    test.
    """
    return any(marker in str(exc) for marker in DEAD_DRIVER_MARKERS)


def report_says_dead_driver(text: str) -> bool:
    """The same question asked of a *report*, which is all the controller gets.

    Under xdist the exception lives in the worker process; the controller sees only the
    formatted report. So the stop has to be decided from text — and it is decided on the
    controller, because that is the only side that can stop the other workers. A worker
    cannot: `pytest.exit` from inside a test makes xdist's `worker_workerfinished`
    assert on the crashed item and the whole session dies with an INTERNALERROR and "no
    tests ran" — the same shape already logged against this suite in FLAKY_TESTS.md.
    """
    return any(marker in (text or "") for marker in DEAD_DRIVER_MARKERS)


def memory_snapshot() -> str:
    """Memory and browser-process counts as they are *now*, for the abort banner.

    Taken at the moment of the failure rather than afterwards: a run that has already
    collapsed has released whatever it was holding, so a reading from the end of the job
    cannot say whether the box was out of memory when the driver died.

    Never raises. This runs on the way out of an already-failing run, and a diagnostic
    that can fail the teardown it is explaining is worse than no diagnostic.
    """
    parts = []
    try:
        meminfo = dict(
            (line.split(":", 1)[0], line.split(":", 1)[1].strip())
            for line in Path("/proc/meminfo").read_text().splitlines()
            if ":" in line
        )
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if key in meminfo:
                parts.append(f"{key}={meminfo[key]}")
    except Exception as e:
        parts.append(f"meminfo unavailable ({e})")
    try:
        # Counted from /proc rather than by shelling out to ps: the point of this
        # function is to work on a box that may be too short of memory to fork.
        names = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                names.append((proc / "comm").read_text().strip())
            except OSError:
                continue  # exited between the listing and the read
        parts.append(f"chrome_procs={sum(1 for n in names if 'chrome' in n)}")
        parts.append(f"node_procs={sum(1 for n in names if n == 'node')}")
    except Exception as e:
        parts.append(f"process counts unavailable ({e})")
    return " ".join(parts)


def abort_banner(where: str, exc: BaseException) -> str:
    """The one message that has to survive in a log nobody will read twice."""
    return (
        "\n"
        "=================== PLAYWRIGHT DRIVER DIED — ABORTING RUN ===================\n"
        f"  where: {where}\n"
        f"  error: {type(exc).__name__}: {exc}\n"
        f"  memory: {memory_snapshot()}\n"
        "\n"
        "  The node driver process behind this worker's Playwright connection has\n"
        "  exited. Its Python side is still healthy and would keep accepting tests,\n"
        "  failing every one of them in the `page` fixture — which is how one dead\n"
        "  process became 1100 fixture errors in CI run 38898595092.\n"
        "\n"
        "  The run is stopped here so this message is the last thing in the log\n"
        "  rather than the first of a thousand copies. See FLAKY_TESTS.md,\n"
        "  'one worker's Playwright driver died', for what is known.\n"
        "============================================================================\n"
    )
