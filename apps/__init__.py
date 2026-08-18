import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass
class AppConfig:
    name: str
    host: str
    port: int


class APPS(str, Enum):
    IMAGE_TAGGING_SERVER = "image_tagging_server"
    INFERENCE_SERVER = "inference_server"
    TASKS_SERVER = "tasks_server"
    DB_SERVER = "db_server"
    STREAM_SERVER = "stream_server"
    TRAINING_SERVER = "training_server"
    MEDIA_MTX_RTSP = "media_mtx_rtsp"
    MEDIA_MTX_API = "media_mtx_api"


_port_offset = int(os.environ.get("PORT_OFFSET", "0"))

# Base ports before any PORT_OFFSET is applied. Import this if you need to
# compute ports for a different offset (e.g. per-worker test isolation).
APP_BASE_PORTS = {
    APPS.IMAGE_TAGGING_SERVER: 7999,
    APPS.INFERENCE_SERVER: 8001,
    APPS.TASKS_SERVER: 8002,
    APPS.DB_SERVER: 8003,
    APPS.STREAM_SERVER: 8004,
    APPS.TRAINING_SERVER: 8005,
}

APPS_CONFIG = {
    APPS.IMAGE_TAGGING_SERVER: AppConfig(
        "Image Tagging Server",
        os.environ.get("IMAGE_TAGGING_SERVER_HOST", "localhost"),
        APP_BASE_PORTS[APPS.IMAGE_TAGGING_SERVER] + _port_offset,
    ),
    APPS.INFERENCE_SERVER: AppConfig(
        "Inference Server",
        os.environ.get("INFERENCE_SERVER_HOST", "localhost"),
        APP_BASE_PORTS[APPS.INFERENCE_SERVER] + _port_offset,
    ),
    APPS.TASKS_SERVER: AppConfig(
        "Tasks Server",
        os.environ.get("TASKS_SERVER_HOST", "localhost"),
        APP_BASE_PORTS[APPS.TASKS_SERVER] + _port_offset,
    ),
    APPS.DB_SERVER: AppConfig(
        "DB Server",
        os.environ.get("DB_SERVER_HOST", "localhost"),
        APP_BASE_PORTS[APPS.DB_SERVER] + _port_offset,
    ),
    APPS.STREAM_SERVER: AppConfig(
        "Stream Server",
        os.environ.get("STREAM_SERVER_HOST", "localhost"),
        APP_BASE_PORTS[APPS.STREAM_SERVER] + _port_offset,
    ),
    APPS.TRAINING_SERVER: AppConfig(
        "Training Server",
        os.environ.get("TRAINING_SERVER_HOST", "localhost"),
        APP_BASE_PORTS[APPS.TRAINING_SERVER] + _port_offset,
    ),
    APPS.MEDIA_MTX_RTSP: AppConfig(
        "MediaMTXRtsp",
        os.environ.get("MEDIA_MTX_RTSP_HOST", "localhost"),
        8554,
    ),
    APPS.MEDIA_MTX_API: AppConfig(
        "MediaMTXApi",
        os.environ.get("MEDIA_MTX_API_HOST", "localhost"),
        9997,
    ),
}

DB_DIR = Path(os.environ.get("DB_DIR", "/data/db/"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/data/cache/"))
FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
# Maximum thumbnail cache size in MB. Set to 0 to disable the limit.
THUMB_CACHE_MAX_MB = int(os.environ.get("THUMB_CACHE_MAX_MB", "100"))

# In-RAM rolling buffer that MediaMTX records stream segments into, giving the
# stream watcher a rewind window to cut event clips from. Backed by a dedicated
# tmpfs mount (see docker-compose.yaml) — deliberately NOT /dev/shm, which is
# 64 MB on the `app` container and shared with the PyTorch DataLoader on `mono`.
STREAM_BUFFER_DIR = Path(os.environ.get("STREAM_BUFFER_DIR", "/buffer"))
# The tmpfs cap, in the same docker size syntax the mount option uses ("512m").
# Passed in as an env var as well as a mount option so the relay can refuse to
# arm a source whose committed bytes would breach the mount, instead of only
# discovering the limit by failing a write.
STREAM_BUFFER_SIZE = os.environ.get("STREAM_BUFFER_SIZE", "512m")
# Default seconds of rewind a recorded path keeps. Deliberately under a minute: the
# buffer is RAM, and the design trades depth for costing ~nothing while idle. Lives
# here rather than in stream_server so db_server can validate a watch profile's
# pre-roll against it without importing another service.
STREAM_BUFFER_RETENTION_S = float(os.environ.get("STREAM_BUFFER_RETENTION_S", "45"))

# How long a finished (completed/error) task instance is kept before the tasks
# server prunes it, and how many are kept regardless of age. Both bounds apply —
# whichever is reached first — because they answer different questions: the age
# limit is "how far back do I still care?", the count limit is what actually caps
# storage when a persistent config churns out instances faster than they age out.
# Set either to 0 to disable that bound.
TASK_RETENTION_DAYS = int(os.environ.get("TASK_RETENTION_DAYS", "15"))
TASK_RETENTION_MAX_COMPLETED = int(os.environ.get("TASK_RETENTION_MAX_COMPLETED", "200"))
