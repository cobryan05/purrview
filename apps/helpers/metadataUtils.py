"""Utility functions for models and json metadata"""

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import onnx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def get_model_metadata(model_path: str, create_json_if_missing: bool = True) -> Dict[str, Any]:
    create_json = create_json_if_missing
    try:
        metadata = get_model_json_metadata(model_path)
        create_json = False
    except FileNotFoundError:
        metadata = {}
        onnx_metadata = get_onnx_metadata(model_path)
        description = onnx_metadata.get("description", "")
        # Case insensitive check for Yolov7 or Yolov8 in description.
        # If in description set yoloVer = 7 (or 8) as appropriate, else none
        yoloVer = 5
        if "yolov7" in description.lower():
            yoloVer = 7
        elif "yolov8" in description.lower():
            yoloVer = 8
        else:
            logger.warning(f"onnx metadata not explicit. Assuming YoloV5 for [{model_path}]")
        metadata["yolo"] = yoloVer

        classes: Dict[int, str] = onnx_metadata.get("names", {})
        if isinstance(classes, str):
            try:
                classes = ast.literal_eval(classes)
            except Exception as e:
                raise ValueError(f"Failed to parse 'names' metadata: {e}")

        class_list = [name for _, name in sorted(classes.items())]
        metadata["classes"] = class_list

    if create_json and metadata:
        json_path = Path(model_path).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def get_onnx_metadata(model_path: str) -> Dict[str, Any]:
    model = onnx.load(model_path)
    metadata_dict = {prop.key: prop.value for prop in model.metadata_props}
    return metadata_dict


def get_model_json_metadata(model_path: str) -> Dict[str, Any]:
    json_path = Path(model_path).with_suffix(".json")
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata
