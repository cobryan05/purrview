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
    # None (omitted) and {} mean different things on the update path: omitted =
    # "I'm not telling you about metadata, keep whatever is stored", {} = "clear
    # it". The canvas save never sends this field, so defaulting it to {} made
    # every save wipe metadata_json on every bbox of the image — taking
    # source_model / source_confidence / run_model_added with it, and
    # relabel_prev_label_uuid, which "revert filter-chain relabel" needs.
    extra: Optional[RawJsonDict] = Field(
        default=None,
        validation_alias=AliasChoices("bbox_meta", "extra"),
        serialization_alias="extra",
    )


@dataclass
class ClassScore:
    """One runner-up class on a detection — the model's next-best guess.

    `label_uuid` is filled in by the inference manager from the model's sidecar
    class map (the same resolution the winning class gets), so consumers never
    have to know a model's class ordering.
    """

    class_id: int
    confidence: float
    class_str: Optional[str] = None
    label_uuid: Optional[str] = None


class ClassScoreModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    class_id: int
    confidence: float
    class_str: Optional[str] = None
    label_uuid: Optional[str] = None


@dataclass
class DetectionResult:
    bbox: BoundingBoxMetadata
    confidence: float
    # Next-best classes for THIS detection, best-first, excluding the winner.
    # Populated only when the caller asked for them (`top_k` on the request);
    # empty otherwise, and always empty for NMS-free/E2E models — see
    # `alternates_available` on InferenceResult.
    alternates: List["ClassScore"] = field(default_factory=list)


class DetectionResultModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    bbox: BoundingBoxMetadataModel
    confidence: float
    alternates: List[ClassScoreModel] = Field(default_factory=list)


@dataclass
class InferenceResult:
    source_width: int
    source_height: int
    detections: List[DetectionResult]
    inference_time: Optional[float] = None
    annotated_image: Optional[np.ndarray] = None
    source_image: Optional[np.ndarray] = None
    # False when this model architecturally cannot produce runner-up classes —
    # NMS-free/E2E exports (YOLOv10, YOLO26) bake the argmax into the graph, so
    # the per-class scores never leave the ONNX file.
    #
    # ⚠️ This is NOT the same as "no alternates were found". A caller collecting
    # confusable detections must treat False as "this model cannot answer the
    # question" and say so, or a multi-day run silently flags nothing.
    alternates_available: bool = True


class InferenceResultModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    source_width: int
    source_height: int
    detections: List[DetectionResultModel] = []
    inference_time: Optional[float] = None
    annotated_image: Optional[Base64Image] = None
    source_image: Optional[Base64Image] = None
    alternates_available: bool = True


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
