"""Shared types"""

from __future__ import annotations

import base64
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)
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


T = TypeVar("T")
M = TypeVar("M", bound="DataclassMapper")


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
        except (Exception):
            pass

    if isinstance(v, bytes):
        return base64.b64encode(v).decode("utf-8")

    return v if isinstance(v, str) else None


Base64Image = Annotated[Optional[str], BeforeValidator(validate_base64_img)]


class DataclassMapper(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    @classmethod
    def from_dataclass(cls: Type[M], dc: T) -> M:
        """
        Converts a standard Python dataclass (and its nested children)
        into this Pydantic model tree.
        """
        return cls.model_validate(dc)

@dataclass
class DetectionResult:
    bbox: BoundingBoxMetadata
    confidence: float

class DetectionResultModel(DataclassMapper):
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

class InferenceResultModel(DataclassMapper):
    source_width: int
    source_height: int
    detections: List[DetectionResultModel] = []
    inference_time: Optional[float] = None
    annotated_image: Optional[Base64Image] = None
    source_image: Optional[Base64Image] = None

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


class BoundingBoxMetadataModel(DataclassMapper):
    uuid: str = Field(validation_alias=AliasChoices("bbox_uuid", "uuid"), serialization_alias="uuid")
    x: float
    y: float
    width: float
    height: float
    label_uuid: Optional[str] = None
    class_str: Optional[str] = None
    tag_uuids: List[str] = Field(default_factory=list)
    extra: RawJsonDict = Field(default_factory=dict, validation_alias=AliasChoices("bbox_meta", "extra"), serialization_alias="extra")


@dataclass
class FileEntry:
    name: str
    type: str
    path: str


class FileEntryModel(DataclassMapper):
    name: str
    type: str  # "file" or "dir"
    path: str  # relative path from root


@dataclass
class SourceMetadata:
    name: str
    typename: str
    params: dict[str, Any]
    uuid: str


class SourceMetadataModel(DataclassMapper):
    name: str
    typename: str
    params: dict[str, Any]
    uuid: str = Field(validation_alias=AliasChoices("src_uuid", "uuid"), serialization_alias="uuid")


@dataclass
class TaskConfigMetadata:
    uuid: str
    typename: str
    params: dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TaskConfigMetadataModel(DataclassMapper):
    uuid: str = Field(validation_alias=AliasChoices("task_cfg_uuid", "uuid"), serialization_alias="uuid")
    typename: str
    params: dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class TaskInstanceMetadata:
    uuid: str
    config_uuid: str
    typename: str
    status: str
    resume_data: Optional[dict[str, str]] = None
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None


class TaskInstanceMetadataModel(DataclassMapper):
    uuid: str = Field(validation_alias=AliasChoices("task_uuid", "uuid"), serialization_alias="uuid")
    config_uuid: str
    typename: str
    status: str
    resume_data: Optional[dict[str, str]] = None
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None


@dataclass
class TaskResult:
    data: dict[str, str] = field(default_factory=dict)


class TaskResultModel(DataclassMapper):
    data: dict[str, str]
