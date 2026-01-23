"""Retrieve the openapi.json files for generating the API clients"""

import json
import os
import sys
os.environ["GENERATING_OPENAPI_CLIENTS"] = "1"

if len(sys.argv) < 2:
    print("Usage: python -m apps.generate_openapi_jsons <targets...> [output_dir]")
    sys.exit(1)

args = sys.argv[1:]
output_dir = "."
if args[-1].startswith(("/", ".")):
    output_dir = args.pop()

targets = [a.lower() for a in args]

# simple ifs for each known target
if "db" in targets:
    from .db_server.main import app as db_server_app
    with open(os.path.join(output_dir, "db_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(db_server_app.openapi(), f, indent=2)

if "relay" in targets:
    from .relay_buffer_server.main import app as relay_server_app
    with open(os.path.join(output_dir, "relay_buffer_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(relay_server_app.openapi(), f, indent=2)

if "inference" in targets:
    from .inference_server.main import app as inference_server_app
    with open(os.path.join(output_dir, "inference_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(inference_server_app.openapi(), f, indent=2)

if "tasks" in targets:
    from .tasks_server.main import app as tasks_server_app
    with open(os.path.join(output_dir, "tasks_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(tasks_server_app.openapi(), f, indent=2)

if "image_tagging" in targets or "frontend" in targets:
    from .image_tagging_server.main import app as image_tagging_server_app
    with open(os.path.join(output_dir, "image_tagging_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(image_tagging_server_app.openapi(), f, indent=2)
