from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Annotated

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

T = TypeVar("T")
M = TypeVar("M", bound="DataclassMapper")


def validate_json(v: Any) -> Dict[str, str]:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return {}
    return v if isinstance(v, dict) else {}


RawJsonDict = Annotated[Dict[str, str], BeforeValidator(validate_json)]


class DataclassMapper(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_dataclass(cls: Type[M], dc: T) -> M:
        """
        Converts a standard Python dataclass (and its nested children)
        into this Pydantic model tree.
        """
        return cls.model_validate(dc)


@dataclass
class ClassMetadata:
    uuid: str
    name: str
    color: str
    parent_uuid: Optional[str] = None


class ClassMetadataModel(DataclassMapper):
    uuid: str = Field(AliasChoices("cls_uuid", "uuid"))
    name: str
    color: str
    parent_uuid: Optional[str] = None


@dataclass
class ClassData:
    metadata: Optional[ClassMetadata] = None
    children: List[ClassData] = field(default_factory=list)


class ClassDataModel(DataclassMapper):
    metadata: Optional[ClassMetadataModel] = None
    children: List[ClassDataModel] = []


class TagKinds:
    GENERIC = "generic"
    SYSTEM = "system"


@dataclass
class TagMetadata:
    uuid: str
    name: str
    color: str
    protected: bool = False
    kind: Optional[str] = None
    exclusive_group: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TagMetadataModel(DataclassMapper):
    uuid: str = Field(validation_alias=AliasChoices("tag_uuid", "uuid"))
    name: str
    color: str
    protected: bool = False
    kind: Optional[str] = None
    exclusive_group: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class BoundingBoxMetadata:
    uuid: str
    class_uuid: str
    x: float
    y: float
    width: float
    height: float
    class_str: str = field(default_factory=str)
    tag_uuids: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)


class BoundingBoxMetadataModel(DataclassMapper):
    uuid: str = Field(validation_alias=AliasChoices("bbox_uuid", "uuid"))
    class_uuid: str
    x: float
    y: float
    width: float
    height: float
    class_str: str = Field(default_factory=str)
    tag_uuids: List[str] = Field(default_factory=list)
    extra: RawJsonDict = Field(default_factory=dict, validation_alias=AliasChoices("bbox_meta", "extra"))

    class Config:
        populate_by_name = True


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
class ImageMetadata:
    filename: str
    uuid: Optional[str] = None
    last_updated: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)
    boxes: List[BoundingBoxMetadata] = field(default_factory=list)

    @classmethod
    def from_model(cls, model: ImageMetadataModel) -> ImageMetadata:
        ret = ImageMetadata(**model.model_dump())
        ret.boxes = [BoundingBoxMetadata(**box.model_dump()) for box in model.boxes]
        return ret


class ImageMetadataModel(DataclassMapper):
    filename: str
    uuid: Optional[str] = Field(default=None, validation_alias=AliasChoices("img_uuid", "uuid"))
    last_updated: Optional[str] = None
    extra: RawJsonDict = Field(default_factory=dict, validation_alias=AliasChoices("img_meta", "extra"))
    boxes: List[BoundingBoxMetadataModel] = Field(default_factory=list)

    class Config:
        populate_by_name = True


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
    uuid: str = Field(validation_alias=AliasChoices("src_uuid", "uuid"))


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
    uuid: str = Field(validation_alias=AliasChoices("task_cfg_uuid", "uuid"))
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
    uuid: str = Field(validation_alias=AliasChoices("task_uuid", "uuid"))
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
