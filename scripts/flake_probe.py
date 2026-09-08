"""Temporary diagnostic plugin (not part of the suite — delete when done).

The xdist worker deaths leave no evidence: the master reports "node down: Not
properly terminated" and names whichever test was in flight, which need not be
the cause. Two channels are missing, and this adds both, per worker, to FILES
rather than to stderr — pytest owns fd 2, so anything written there when the
process is dying can be captured and discarded.

  faulthandler -> /tmp/flake_probe/fh-<pid>.log
      A native crash (SIGSEGV/SIGABRT/SIGBUS/SIGFPE) dumps a C-level traceback
      here. A file that exists means the worker crashed in native code.

  signal log   -> /tmp/flake_probe/sig-<pid>.log
      SIGTERM/SIGINT/SIGHUP are catchable, so a line here means something asked
      the worker to stop. SIGKILL cannot be caught, so its signature is the
      absence of BOTH files next to a vanished pid.

So: fh file => native crash. sig file => something signalled it. Neither, and
the process gone => SIGKILL from outside the process.
"""

import atexit
import faulthandler
import os
import signal
import time

_DIR = "/tmp/flake_probe"


def pytest_configure(config):
    os.makedirs(_DIR, exist_ok=True)
    pid = os.getpid()
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")

    # Kept open for the life of the process on purpose: faulthandler writes to
    # this fd from a signal handler, when opening a file is not an option.
    fh = open(os.path.join(_DIR, f"fh-{worker}-{pid}.log"), "w")
    faulthandler.enable(file=fh, all_threads=True)

    sig_path = os.path.join(_DIR, f"sig-{worker}-{pid}.log")

    def _record(signum, frame):
        with open(sig_path, "a") as f:
            f.write(f"{time.time():.3f} received signal {signum} ({signal.Signals(signum).name})\n")
            f.flush()
        # Re-raise with the default handler so behaviour is unchanged.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(sig, _record)
        except (ValueError, OSError):
            pass  # not the main thread, or not settable here

    def _clean_exit():
        # A worker that finished normally leaves a marker, so "no files at all"
        # is never confused with "the probe did not run".
        with open(os.path.join(_DIR, f"exit-{worker}-{pid}.log"), "w") as f:
            f.write("clean\n")

    atexit.register(_clean_exit)
