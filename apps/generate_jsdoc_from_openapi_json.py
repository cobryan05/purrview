# ChatGPT-generated to convert openapi.json to api-types jsdoc.
import json
from pathlib import Path
import sys


def json_schema_to_jsdoc(name: str, schema: dict) -> str:
    """
    Convert a JSON Schema object to a JSDoc @typedef string.
    Only handles basic types, arrays, objects, and references.
    """
    lines = ["/**", f" * @typedef {{Object}} {name}"]
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for prop_name, prop_schema in props.items():
        js_type = json_type_to_js(prop_schema)
        if prop_name in required:
            lines.append(f" * @property {{{js_type}}} {prop_name}")
        else:
            lines.append(f" * @property {{{js_type}}} [{prop_name}]")
    lines.append(" */\n")
    return "\n".join(lines)


def json_type_to_js(prop_schema: dict) -> str:
    if "$ref" in prop_schema:
        return prop_schema["$ref"].split("/")[-1]

    if "oneOf" in prop_schema:
        return " | ".join(json_type_to_js(s) for s in prop_schema["oneOf"])
    if "anyOf" in prop_schema:
        return " | ".join(json_type_to_js(s) for s in prop_schema["anyOf"])

    if "allOf" in prop_schema:
        return " & ".join(json_type_to_js(s) for s in prop_schema["allOf"])

    if prop_schema.get("nullable"):
        base = json_type_to_js({k: v for k, v in prop_schema.items() if k != "nullable"})
        return f"{base} | null"

    t = prop_schema.get("type")
    if isinstance(t, list):
        return " | ".join(json_type_to_js({"type": x}) for x in t)

    if t == "integer":
        return "number"
    if t == "number":
        return "number"
    if t == "string":
        return "string"
    if t == "boolean":
        return "boolean"

    if t == "array":
        items = prop_schema.get("items", {})
        return f"{json_type_to_js(items)}[]"

    if t == "object":
        return "Object"

    return "any"


def generate_jsdoc(openapi_path: str, output_path: str):
    """Generate JSDoc typedefs from OpenAPI JSON."""
    with open(openapi_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    schemas = spec.get("components", {}).get("schemas", {})

    output_lines = ["// Auto-generated JSDoc typedefs from OpenAPI JSON\n"]

    for name, schema in schemas.items():
        output_lines.append(json_schema_to_jsdoc(name, schema))

    output_lines.append("export { };")

    outpath = Path(output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Wrote {len(schemas)} typedefs to {outpath}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m apps.generate_jsdoc_from_openapi_json <openapi.json> <outputPath>")
        sys.exit(1)

    openapi_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_jsdoc(openapi_path, output_path)


if __name__ == "__main__":
    main()
