"""Guard the other invariant the image build imposes on every server's schema:

    No operationId may appear twice in a spec a client is generated from.

openapi-generator validates the spec before it generates a client and refuses a
repeated operationId outright, so a duplicate is not a warning — it fails the
docker build at the `openapi-builder` stage, several minutes in, with a stack
trace that names the path but not the route that produced it.

The way to write one by accident is a single route serving several methods under
one explicit `operation_id`: FastAPI stamps that id onto every method of the
route. Two decorators on the same handler, each with its own id, is the fix.

image_tagging_server is deliberately out of scope. Only the combined spec is built
from it, and that one generator call passes `--skip-validate-spec`, so its proxy
routes — which serve several methods under one auto-generated id and have repeated
ids throughout — cannot fail a build. Asserting on them would be asserting on
something nothing checks.

Run in a fresh interpreter for the same reason as the torch-free guard: this is
what the builder does, in the order it does it.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_CHILD = r"""
import os, json
os.environ["GENERATING_OPENAPI_CLIENTS"] = "1"
from apps.db_server.main import app as db
from apps.stream_server.main import app as relay
from apps.inference_server.main import app as inference
from apps.training_server.main import app as training
from apps.tasks_server.main import app as tasks

# Exactly the specs generate_openapi_modules.sh generates a client from without
# --skip-validate-spec.
repeated = {}
for name, app in [("db", db), ("relay", relay), ("inference", inference),
                  ("training", training), ("tasks", tasks)]:
    seen = {}
    for path, item in app.openapi()["paths"].items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if op_id is None:
                continue
            seen.setdefault(op_id, []).append(f"{method.upper()} {path}")
    for op_id, where in seen.items():
        if len(where) > 1:
            repeated[f"{name}: {op_id}"] = where
print(json.dumps(repeated))
"""


def test_no_server_repeats_an_operation_id():
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"schema generation failed:\nSTDERR:\n{proc.stderr}"
    repeated = json.loads(proc.stdout.strip().splitlines()[-1])
    assert repeated == {}, (
        "Repeated operationIds would fail openapi-generator's spec validation during "
        f"the docker build: {repeated}. A route serving several methods under one "
        "explicit operation_id is the usual cause — give each method its own decorator "
        "and its own id."
    )
