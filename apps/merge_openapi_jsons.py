import json
import os
import sys
import glob

def merge(output_dir, output_filename="combined_openapi.json"):
    # Pattern to find all openapi.json files
    search_pattern = os.path.join(output_dir, "*openapi.json")
    files = glob.glob(search_pattern)

    # Exclude the output file from the input list to prevent recursive nesting
    files = [f for f in files if os.path.basename(f) != output_filename]

    if not files:
        print(f"No OpenAPI files found in {output_dir}", file=sys.stderr)
        return

    combined = {
        "openapi": "3.1.0",
        "info": {"title": "Combined API", "version": "1.0.0"},
        "paths": {},
        "components": {"schemas": {}}
    }

    for path in files:
        print(f"Merging: {path}", file=sys.stderr)
        with open(path, "r") as f:
            try:
                spec = json.load(f)
                # Merge schemas
                schemas = spec.get("components", {}).get("schemas", {})
                combined["components"]["schemas"].update(schemas)
                # Merge paths
                paths = spec.get("paths", {})
                combined["paths"].update(paths)
            except Exception as e:
                print(f"Error parsing {path}: {e}", file=sys.stderr)

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Successfully created: {output_path}", file=sys.stderr)

if __name__ == "__main__":
    # Usage: python -m apps.merge_openapi_jsons [dir] [filename]
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    filename = sys.argv[2] if len(sys.argv) > 2 else "combined_openapi.json"

    merge(target_dir, filename)
