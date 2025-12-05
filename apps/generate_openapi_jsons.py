"""Retrieve the openapi.json files for generating the API clients"""

import json
import os
import sys
os.environ["GENERATING_OPENAPI_CLIENTS"] = "1"

generate_backend = False
generate_frontend = False

if len(sys.argv) < 2:
    print("Usage: python -m apps.generate_openapi_jsons [frontend|backend]")
    sys.exit(1)

arg = sys.argv[1].lower()
if arg == "backend":
    generate_backend = True
elif arg == "frontend":
    generate_frontend = True
else:
    print(f"Unknown option: {arg}")
    print("Usage: python -m apps.generate_openapi_jsons [frontend|backend]")
    sys.exit(1)

output_dir = sys.argv[2] if len(sys.argv) >= 3 else "."

if generate_backend:
    from .relay_buffer_server.main import app as relay_server_app
    with open(os.path.join(output_dir, "relay_buffer_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(relay_server_app.openapi(), f, indent=2)

    from .inference_server.main import app as inference_server_app
    with open(os.path.join(output_dir, "inference_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(inference_server_app.openapi(), f, indent=2)

    from .tasks_server.main import app as tasks_server_app
    with open(os.path.join(output_dir, "tasks_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(tasks_server_app.openapi(), f, indent=2)

    from .db_server.main import app as db_server_app
    with open(os.path.join(output_dir, "db_server_openapi.json"), "w", encoding="utf-8") as f:
        json.dump(db_server_app.openapi(), f, indent=2)

if generate_frontend:
    from .image_tagging_server.main import app as image_tagging_server_app
    with open(os.path.join(output_dir, "image_tagging_server_openapi.json"), "w", encoding="utf-8") as f:
            json.dump(image_tagging_server_app.openapi(), f, indent=2)
