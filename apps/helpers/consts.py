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
    DATASETS = "datasets"
    MODEL_CONFIGS = "model_configs"
    TRAINING_CONFIGS = "training_configs"
    FILTER_CHAINS = "filter_chains"
    TRAINING_COLLECTORS = "training_collectors"
    PREVIEW_RUNS = "preview_runs"
    MODEL_GROUPS = "model_groups"
    SETTINGS = "settings"
    WATCH_PROFILES = "watch_profiles"
    RECORDINGS = "recordings"


class TaskAccent(str, Enum):
    """Named card accents a task type may claim (Task.CARD_ACCENT).

    Names, not colours, deliberately. The Jobs pane is dark-only and Pico paints
    the surfaces under these cards, so a hex chosen inside a task class cannot
    know what it will sit on — the actual values live with every other colour in
    static/css/tokens.css (`--task-accent-*`) and are picked for separation
    against that surface and from each other.

    Adding a member means adding the matching token and the
    `[data-task-accent="..."]` rule in app.css; test_task_accents pins the two
    sides together, since a name with no rule fails silently as "no colour".
    """

    BLUE = "blue"
    GREEN = "green"
    AMBER = "amber"
    VIOLET = "violet"
    TEAL = "teal"
    SLATE = "slate"


class TaskStatus(str, Enum):
    NEW = "new"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
