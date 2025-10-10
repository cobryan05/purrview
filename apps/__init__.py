from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class AppConfig:
    name: str
    host: str
    port: int


APPS_CONFIG = {
    "image_tagging_server": AppConfig("Image Tagging Server", "localhost", 8000),
    "inference_server": AppConfig("Inference Server", "localhost", 8001),
    "tasks_server": AppConfig("Tasks Server", "localhost", 8002),
    "db_server": AppConfig("DB Server", "localhost", 8003),
    "relay_buffer_server": AppConfig("RTSP/Relay Buffer Server", "localhost", 8004),
    "media_mtx_rtsp": AppConfig("MediaMTXRtsp", "localhost", 8554),
    "media_mtx_api": AppConfig("MediaMTXApi", "localhost", 9997),
}

DB_DIR = Path(os.environ.get("DB_DIR", "/data/db/"))
