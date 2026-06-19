"""Shared types"""

from __future__ import annotations

import base64
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.alias_generators import to_camel
from typing_extensions import Annotated

from .consts import JsonValues

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Payload(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class StatusResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    status: str = JsonValues.SUCCESS
    message: Optional[str] = None


def validate_raw_json_dict(v: Any) -> Dict[str, str]:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return {}
    return v if isinstance(v, dict) else {}


RawJsonDict = Annotated[Dict[str, str], BeforeValidator(validate_raw_json_dict)]


def validate_base64_img(v: Any) -> str:
    if isinstance(v, np.ndarray):
        try:
            import cv2

            _, v = cv2.imencode(".png", v)
        except Exception:
            pass

    if isinstance(v, bytes):
        return base64.b64encode(v).decode("utf-8")

    return v if isinstance(v, str) else None


Base64Image = Annotated[Optional[str], BeforeValidator(validate_base64_img)]


# ---------------------------------------------------------------------------
# Bounding-box / inference transport types
# ---------------------------------------------------------------------------


@dataclass
class BoundingBoxMetadata:
    uuid: str
    x: float
    y: float
    width: float
    height: float
    label_uuid: Optional[str] = None
    class_str: Optional[str] = None
    tag_uuids: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)


class BoundingBoxMetadataModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    uuid: str = Field(validation_alias=AliasChoices("bbox_uuid", "uuid"), serialization_alias="uuid")
    x: float
    y: float
    width: float
    height: float
    label_uuid: Optional[str] = None
    class_str: Optional[str] = None
    tag_uuids: List[str] = Field(default_factory=list)
    extra: RawJsonDict = Field(
        default_factory=dict,
        validation_alias=AliasChoices("bbox_meta", "extra"),
        serialization_alias="extra",
    )


@dataclass
class DetectionResult:
    bbox: BoundingBoxMetadata
    confidence: float


class DetectionResultModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    bbox: BoundingBoxMetadataModel
    confidence: float


@dataclass
class InferenceResult:
    source_width: int
    source_height: int
    detections: List[DetectionResult]
    inference_time: Optional[float] = None
    annotated_image: Optional[np.ndarray] = None
    source_image: Optional[np.ndarray] = None


class InferenceResultModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    source_width: int
    source_height: int
    detections: List[DetectionResultModel] = []
    inference_time: Optional[float] = None
    annotated_image: Optional[Base64Image] = None
    source_image: Optional[Base64Image] = None


# ---------------------------------------------------------------------------
# File-system entry types  (transport only — no DB table)
# ---------------------------------------------------------------------------


class FileEntryBase(BaseModel):
    """Core filesystem fields shared across all representations."""

    name: str
    path: str  # relative path from FILES_ROOT
    is_dir: bool


class FileEntryRead(FileEntryBase):
    """File entry as returned by the API, optionally enriched with DB info."""

    model_config = ConfigDict(
        from_attributes=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    uuid: Optional[str] = None  # DB image UUID if this path is registered; None if not in DB
    bbox_count: Optional[int] = None  # number of bounding boxes on this image; None if not in DB
    size: int = 0
    modified_at: Optional[datetime] = None
    mime_type: Optional[str] = None
