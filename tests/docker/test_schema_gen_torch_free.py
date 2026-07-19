"""Guard the invariant the slim `app` image and the torch-free `openapi-builder`
stage depend on:

    Generating the OpenAPI schemas for every server (what the builder does at
    image-build time) must NOT import the heavy training stack (torch,
    ultralytics, libreyolo, timm, sklearn, albumentations).

Those are imported lazily by trainers at registration time, guarded behind
GENERATING_OPENAPI_CLIENTS. If someone adds an eager top-level import of one of
them into a module reachable from a server `main` (or its `.openapi()`), the
slim/CPU image build breaks (ModuleNotFound) — this test fails first instead.

Run in a fresh interpreter (subprocess) because other tests in the session may
have already imported torch.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Heavy training-only modules that must never load during schema generation.
FORBIDDEN = ["torch", "ultralytics", "libreyolo", "timm", "sklearn", "albumentations"]

_CHILD = r"""
import os, sys, json
os.environ["GENERATING_OPENAPI_CLIENTS"] = "1"
# Mirror the modules generate_openapi_modules.sh introspects, in dependency order.
from apps.db_server.main import app as a1
from apps.relay_buffer_server.main import app as a2
from apps.inference_server.main import app as a3
from apps.training_server.main import app as a4
from apps.tasks_server.main import app as a5
from apps.image_tagging_server.main import app as a6
for a in (a1, a2, a3, a4, a5, a6):
    a.openapi()
forbidden = ["torch", "ultralytics", "libreyolo", "timm", "sklearn", "albumentations"]
loaded = [m for m in forbidden if m in sys.modules]
print(json.dumps(loaded))
"""


def test_schema_generation_does_not_import_torch_stack():
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"schema generation failed:\nSTDERR:\n{proc.stderr}"
    loaded = eval(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        "Heavy training modules loaded during OpenAPI schema generation: "
        f"{loaded}. This breaks the torch-free slim `app` image / openapi-builder. "
        "Keep trainer/torch imports lazy (see tests/docker/test_schema_gen_torch_free.py)."
    )
