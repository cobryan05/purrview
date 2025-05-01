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
        streamer: Optional[DelayedStreamer] = None
        mtx_path: Optional[MtxPath] = None

        active: bool = False
        expected_active: Optional[bool] = None
        owned: bool = False  # True if Manager created this

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, api_client: ApiClient):
        """Initialize the Manager with an API client"""
        self._api_client: ApiClient = api_client
        self._streams: Dict[str, Manager.StreamInfo] = {}
        self._hostname: str = "localhost"
        self._api_port: int = 9997
        self._rtsp_port: int = 8554
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations

    async def create_delay_stream(self, rtsp_url: str, stream_name: str, delay: float, overwrite: bool = True) -> None:
        """Create a new stream that replays source_stream with a delay"""
        await self.create_new_publish_stream(stream_name, overwrite)
        input_stream = FFmpegStreamerIn(rtsp_url)
        delayed_stream = self._streams[stream_name]
        if delayed_stream.url is None:
            raise ValueError("Publish URL is missing. Failed to create publish stream?")
        output = FFmpegStreamerOut(delayed_stream.url)
        delayed_stream.streamer = DelayedStreamer(input_stream, output, delay)
        delayed_stream.streamer.start()


    async def create_new_publish_stream(self, stream_name: str, overwrite: bool = True):
        """Create a new stream with the given source and name"""
        if stream_name in self._streams:
            if overwrite:
                await self.destroy_stream(stream_name)
            else:
                raise KeyError(f"Stream named {stream_name} already exists")

        await self.destroy_stream(stream_name)

        path_conf: PathConf = PathConf(name=stream_name, source="publisher", sourceOnDemand=False)  # Define the stream configuration
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        # Add the stream configuration using the API
        self._streams[stream_name] = Manager.StreamInfo(stream_name)
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        await self.refresh_streams()  # Update stream info

    async def create_rtsp_relay_stream(self, rtsp_url: str, stream_name: str, overwrite: bool = True) -> None:
        """Create a new stream with the given source and name"""
        if stream_name in self._streams:
            if overwrite:
                await self.destroy_stream(stream_name)
            else:
                raise KeyError(f"Stream named {stream_name} already exists")
        path_conf: PathConf = PathConf(name=stream_name, source=rtsp_url)  # Define the stream configuration
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        # Add the stream configuration using the API
        self._streams[stream_name] = Manager.StreamInfo(stream_name)
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        await self.refresh_streams()  # Update stream info


    async def destroy_all_streams(self) -> None:
        """Destroys all streams on the server"""
        for name in list(self.get_streams().keys()):
            await self.destroy_stream(name)

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

    async def get_config(self) -> str:
        """Retrieve the global configuration as a string"""
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        config = await asyncio.to_thread(api.config_global_get)  # Fetch the global configuration
        return config.to_str()  # Convert the configuration to a string

    def get_streams(self) -> Dict[str, StreamInfo]:
        return dict(self._streams)

    async def refresh_streams(self) -> None:
        """Retrieve the current list of streams, updating the cached list"""
        api: PathsApi = PathsApi(self._api_client)
        streams = await asyncio.to_thread(api.paths_list)  # Fetch the list of streams
        if streams.items is None:
            raise Exception("Failed to get stream list from server")
        stream_dict: Dict[str, MtxPath] = {str(stream.name): stream for stream in streams.items}

        # Add new streams
        for name, mtx_path in stream_dict.items():
            if name not in self._streams:
                logger.info(f"Unexpected new stream on server: {name}")
                self._streams[name] = Manager.StreamInfo(name, owned=False)  # Add the new stream to self._streams
            stream_info = self._streams[name]

            stream_info.mtx_path = mtx_path
            stream_info.url = f"rtsp://{self._hostname}:{self._rtsp_port}/{name}"
            if mtx_path.ready != stream_info.active:
                logger.info(f"Stream {name} set to url: {stream_info.url}  active: {mtx_path.ready}")
                stream_info.active = mtx_path.ready
                if stream_info.expected_active is None and stream_info.active:
                    # Once active expect it to stay active
                    stream_info.expected_active = True
                elif not stream_info.active and stream_info.expected_active:
                    logger.warning(f"Stream {name} expected active but is not!")

        # Remove unexpectedly closed streams
        for name in list(self._streams.keys()):
            if name not in stream_dict:
                mtx_path = self._streams[name]
                if mtx_path.active:
                    logger.warning(f"Stream unexpectedly removed: {name}")
                else:
                    logger.info(f"Confirming removal of {name}")
                self._streams.pop(name)


    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        config_api: ConfigurationApi = ConfigurationApi(self._api_client)
        config = await asyncio.to_thread(config_api.config_global_get)
        host_url = self._api_client.configuration.host
        host = host_url.split("://")[1]
        host_name, host_port = host.split(":")
        self._hostname = host_name
        self._api_port = host_port
        self._rtsp_port = config.rtsp_address.split(":")[1]
        await self.refresh_streams()
        await self.destroy_all_streams()

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
                await self.refresh_streams()
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            print("Periodic task cleanup.")  # Perform cleanup when the task is stopped
