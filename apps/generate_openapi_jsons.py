"""Retrieve the openapi.json files for generating the API clients"""
import json
from .relay_buffer_server.main import app as relay_server_app

with open("relay_buffer_server_openapi.json", "w") as f:
    json.dump(relay_server_app.openapi(), f, indent=2)