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

APPS_CONFIG = {
    APPS.IMAGE_TAGGING_SERVER: AppConfig("Image Tagging Server", "localhost", 7999 + _port_offset),
    APPS.INFERENCE_SERVER: AppConfig("Inference Server", "localhost", 8001 + _port_offset),
    APPS.TASKS_SERVER: AppConfig("Tasks Server", "localhost", 8002 + _port_offset),
    APPS.DB_SERVER: AppConfig("DB Server", "localhost", 8003 + _port_offset),
    APPS.RELAY_BUFFER_SERVER: AppConfig("RTSP/Relay Buffer Server", "localhost", 8004 + _port_offset),
    APPS.MEDIA_MTX_RTSP: AppConfig("MediaMTXRtsp", "localhost", 8554),
    APPS.MEDIA_MTX_API: AppConfig("MediaMTXApi", "localhost", 9997),
}

DB_DIR = Path(os.environ.get("DB_DIR", "/data/db/"))
