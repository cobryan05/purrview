from __future__ import annotations
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
from uuid import uuid4
from dataclasses import field, dataclass
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")
M = TypeVar("M", bound="DataclassMapper")


class DataclassMapper(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_dataclass(cls: Type[M], dc: T) -> M:
        """
        Converts a standard Python dataclass (and its nested children)
        into this Pydantic model tree.
        """
        return cls.model_validate(dc)


class ClassMetadata(BaseModel):
    uuid: str
    name: str
    color: str
    parent_uuid: Optional[str] = None


class ClassMetadataModel(DataclassMapper):
    uuid: str
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
    uuid: str
    name: str
    color: str
    protected: bool = False
    kind: Optional[str] = None
    exclusive_group: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class BoundingBoxMetadata:
    class_uuid: str
    x: float
    y: float
    width: float
    height: float
    uuid: str = field(default_factory=lambda: str(uuid4()))
    tag_uuids: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)


class BoundingBoxMetadataModel(DataclassMapper):
    class_uuid: str
    x: float
    y: float
    width: float
    height: float
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    tag_uuids: List[str] = Field(default_factory=list)
    extra: Dict[str, str] = Field(default_factory=dict)


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


class ImageMetadataModel(DataclassMapper):
    filename: str
    uuid: Optional[str] = None
    last_updated: Optional[str] = None
    extra: Dict[str, str] = Field(default_factory=dict)
    boxes: List[BoundingBoxMetadataModel] = Field(default_factory=list)


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
    uuid: str


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
    uuid: str
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

@dataclass
class TaskResult:
    data: dict[str, str] = field(default_factory=dict)

class TaskResultModel(DataclassMapper):
    data: dict[str, str]