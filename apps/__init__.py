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
    RELAY_BUFFER_SERVER = "relay_buffer_server"
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
    APPS.RELAY_BUFFER_SERVER: 8004,
}

APPS_CONFIG = {
    APPS.IMAGE_TAGGING_SERVER: AppConfig(
        "Image Tagging Server", "localhost", APP_BASE_PORTS[APPS.IMAGE_TAGGING_SERVER] + _port_offset
    ),
    APPS.INFERENCE_SERVER: AppConfig(
        "Inference Server", "localhost", APP_BASE_PORTS[APPS.INFERENCE_SERVER] + _port_offset
    ),
    APPS.TASKS_SERVER: AppConfig("Tasks Server", "localhost", APP_BASE_PORTS[APPS.TASKS_SERVER] + _port_offset),
    APPS.DB_SERVER: AppConfig("DB Server", "localhost", APP_BASE_PORTS[APPS.DB_SERVER] + _port_offset),
    APPS.RELAY_BUFFER_SERVER: AppConfig(
        "RTSP/Relay Buffer Server", "localhost", APP_BASE_PORTS[APPS.RELAY_BUFFER_SERVER] + _port_offset
    ),
    APPS.MEDIA_MTX_RTSP: AppConfig("MediaMTXRtsp", "localhost", 8554),
    APPS.MEDIA_MTX_API: AppConfig("MediaMTXApi", "localhost", 9997),
}

DB_DIR = Path(os.environ.get("DB_DIR", "/data/db/"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/data/cache/"))
# Maximum thumbnail cache size in MB. Set to 0 to disable the limit.
THUMB_CACHE_MAX_MB = int(os.environ.get("THUMB_CACHE_MAX_MB", "100"))
