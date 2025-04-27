from mediamtx_client.api_client import ApiClient
from mediamtx_client.api.paths_api import PathsApi
from mediamtx_client.api.configuration_api import ConfigurationApi
from mediamtx_client.models.path_conf import PathConf
from mediamtx_client.models.path import Path as MtxPath
from apps.helpers.delayedStreamer import DelayedStreamer
from apps.helpers.ffmpegStreamerIn import FFmpegStreamerIn
from apps.helpers.ffmpegStreamerOut import FFmpegStreamerOut
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Dict
import sys


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    @dataclass
    class StreamInfo:
        name: str
        url: Optional[str] = None
        active: bool = False
        owned: bool = False  # True if Manager created this
        streamer: Optional[DelayedStreamer] = None

    POLLING_INTERVAL: float = 30.0  # Interval in seconds for periodic tasks

    def __init__(self, api_client: ApiClient):
        """Initialize the Manager with an API client"""
        self._api_client: ApiClient = api_client
        self._streams: Dict[str, Manager.StreamInfo] = {}
        self._hostname: str = "localhost"
        self._api_port: int = 9997
        self._rtsp_port: int = 8554
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def create_rtsp_relay_stream(self, rtsp_url: str, stream_name: str, overwrite: bool = True) -> None:
        """Create a new stream with the given source and name, overwriting any existing stream"""
        if stream_name in self._streams:
            if overwrite:
                await self.destroy_stream(stream_name)
            else:
                raise Exception(f"Stream named {stream_name} already exists")
        path_conf: PathConf = PathConf(name=stream_name, source=rtsp_url)  # Define the stream configuration
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        # Add the stream configuration using the API
        self._streams[stream_name] = Manager.StreamInfo(stream_name)
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        await self.refresh_streams()  # Update stream info

    async def create_delay_stream(self, rtsp_url: str, stream_name: str, delay: float, overwrite: bool = True) -> None:
        """Create a new stream that replays source_stream with a delay, overwriting any existing stream"""
        await self.create_new_publish_stream(stream_name, overwrite)
        input = FFmpegStreamerIn(rtsp_url)
        delayed_stream = self._streams[stream_name]
        output = FFmpegStreamerOut(delayed_stream.url)
        delayed_stream.streamer = DelayedStreamer(input, output, delay)
        delayed_stream.streamer.start()

    async def create_new_publish_stream(self, stream_name: str, overwrite: bool = True):
        """Create a new stream with the given source and name, overwriting any existing stream"""
        if stream_name in self._streams:
            if overwrite:
                await self.destroy_stream(stream_name)
            else:
                raise Exception(f"Stream named {stream_name} already exists")

        await self.destroy_stream(stream_name)

        path_conf: PathConf = PathConf(name=stream_name, source="publisher")  # Define the stream configuration
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        # Add the stream configuration using the API
        self._streams[stream_name] = Manager.StreamInfo(stream_name)
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        await self.refresh_streams()  # Update stream info

    async def destroy_stream(self, stream_name: str) -> None:
        """Destroy a stream with the given name"""
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        if stream_name not in self._streams:
            logger.warning(f"Destroying unknown stream {stream_name}")
        else:
            self._streams[stream_name].active = False  # Mark as pending
        try:
            await asyncio.to_thread(api.config_paths_delete, name=stream_name)
        except Exception as e:
            logger.warning(f"Failed to destroy stream: {e}")
        await self.refresh_streams()

    def get_streams(self) -> Dict[str, StreamInfo]:
        return dict(self._streams)

    async def refresh_streams(self) -> None:
        """Retrieve the current list of streams, updating the cached list"""
        api: PathsApi = PathsApi(self._api_client)
        streams = await asyncio.to_thread(api.paths_list)  # Fetch the list of streams
        if streams.items is None:
            raise Exception("Failed to get stream list from server")
        stream_dict: Dict[str, MtxPath] = {str(stream.name): stream for stream in streams.items}

        self._update_streams(stream_dict)

    async def get_config(self) -> str:
        """Retrieve the global configuration as a string"""
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        config = await asyncio.to_thread(api.config_global_get)  # Fetch the global configuration
        return config.to_str()  # Convert the configuration to a string

    def _update_streams(self, streams: Dict[str, MtxPath]):
        """Update the internal stream list to match reported streams"""
        # Add new streams
        for name, stream in streams.items():
            if name not in self._streams:
                logger.info(f"Unexpected new stream on server: {name}")
                self._streams[name] = Manager.StreamInfo(name, owned=False)  # Add the new stream to self._streams
            stream_info = self._streams[name]
            if not stream_info.active:
                logger.info(f"Stream {name} marked active")
                stream_info.active = True
                stream_info.url = f"rtsp://{self._hostname}:{self._rtsp_port}/{name}"

        # Remove unexpectedly closed streams
        for name in list(self._streams.keys()):
            if name not in streams:
                stream = self._streams[name]
                if stream.active:
                    logger.warning(f"Stream unexpectedly removed: {name}")
                else:
                    logger.info(f"Confirming removal of {name}")
                self._streams.pop(name)

    def _init(self):
        config_api: ConfigurationApi = ConfigurationApi(self._api_client)
        config = config_api.config_global_get()
        host_url = self._api_client.configuration.host
        host = host_url.split("://")[1]
        host_name, host_port = host.split(':')
        self._hostname = host_name
        self._api_port = host_port
        self._rtsp_port = config.rtsp_address.split(':')[1]


    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            self._init()
            while True:
                # Placeholder for periodic operations (e.g., monitoring or maintenance tasks)
                await self.refresh_streams()
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            print("Periodic task cleanup.")  # Perform cleanup when the task is stopped
