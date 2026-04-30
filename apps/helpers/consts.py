"""Constants for JSON keys and values"""

from enum import Enum


class JsonKeys(str, Enum):
    STATUS = "status"
    MESSAGE = "message"
    CONFIG = "config"
    REQUEST = "request"
    DATA = "data"
    RESULT = "result"
    ERROR = "error"

class JsonValues(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"

class ApiTags(str, Enum):
    LABELS = "labels"
    MODELS = "models"
    INFERENCE = "inference"
    IMAGES = "images"
    SOURCES = "sources"
    TAGS = "tags"
    TASKS = "tasks"

class TaskStatus(str, Enum):
    NEW = "new"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"