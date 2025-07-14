"""Manages streams on the MediaMTX server"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mediamtx_client.api.configuration_api import ConfigurationApi
from mediamtx_client.api.paths_api import PathsApi
from mediamtx_client.api_client import ApiClient
from mediamtx_client.models.path import Path as MtxPath
from mediamtx_client.models.path_conf import PathConf

from apps.helpers.streams.delayedStreamer import DelayedStreamer
from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn
from apps.helpers.streams.ffmpegStreamerOut import FFmpegStreamerOut

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


# TODO:
# When restarting streams should check if even still exists on mediamtx, maybe it was stopped?
#   If we didn't stop it (it was externally stopped) we should probably just start it back up


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
        source_url: Optional[str] = None  # Tracks the original RTSP source (if relay)

    @dataclass
    class SourceRelayInfo:
        master_stream_name: str  # Internal relay that actually connects to remote source
        refcount: int = 1

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, api_client: ApiClient):
        """Initialize the Manager with an API client"""
        self._api_client: ApiClient = api_client
        self._streams: Dict[str, Manager.StreamInfo] = {}
        self._source_relays: Dict[str, Manager.SourceRelayInfo] = {}
        self._master_stream_counter: int = 0
        self._hostname: str = "localhost"
        self._api_port: int = 9997
        self._rtsp_port: int = 8554
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations

    async def _get_or_create_master_stream(self, rtsp_url: str) -> str:
        """Ensure a single MediaMTX stream exists for a given RTSP URL, return master stream name."""
        relay_info = self._source_relays.get(rtsp_url)
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        if relay_info:
            relay_info.refcount += 1
            return relay_info.master_stream_name

        master_name = f"relay_master_{self._master_stream_counter}"
        self._master_stream_counter += 1

        path_conf = PathConf(name=master_name, source=rtsp_url, sourceOnDemand=False)
        await asyncio.to_thread(api.config_paths_add, name=master_name, path_conf=path_conf)

        self._source_relays[rtsp_url] = Manager.SourceRelayInfo(master_stream_name=master_name)

        self._streams[master_name] = Manager.StreamInfo(
            name=master_name,
            url=f"rtsp://{self._hostname}:{self._rtsp_port}/{master_name}",
            owned=True,
            source_url=rtsp_url,
        )

        logger.info(f"Created master stream {master_name} for source {rtsp_url}")
        await self.refresh_streams()
        return master_name

    async def create_delay_stream(self, rtsp_url: str, stream_name: str, delay: float, overwrite: bool = True) -> None:
        """Create a new stream that replays an RTSP stream with a delay"""
        await self.create_new_publish_stream(stream_name, overwrite)

        master_name = await self._get_or_create_master_stream(rtsp_url)
        relay_url = f"rtsp://{self._hostname}:{self._rtsp_port}/{master_name}"
        self._streams[stream_name].source_url = rtsp_url

        # Wait until master stream is active
        for _ in range(10):
            await self.refresh_streams()
            master_stream = self._streams.get(master_name)
            if master_stream and master_stream.active:
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError(f"Timed out waiting for master stream {master_name} to become active")

        input_stream = FFmpegStreamerIn(relay_url)
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

        path_conf: PathConf = PathConf(name=stream_name, source="publisher", sourceOnDemand=False)
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        self._streams[stream_name] = Manager.StreamInfo(name=stream_name)
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        await self.refresh_streams()

    async def create_rtsp_relay_stream(self, rtsp_url: str, stream_name: str, overwrite: bool = True) -> None:
        """Create a new named RTSP relay that proxies another rtsp url"""
        if stream_name in self._streams:
            if overwrite:
                await self.destroy_stream(stream_name)
            else:
                raise KeyError(f"Stream named {stream_name} already exists")

        master_name = await self._get_or_create_master_stream(rtsp_url)
        relay_url = f"rtsp://{self._hostname}:{self._rtsp_port}/{master_name}"

        path_conf = PathConf(name=stream_name, source=relay_url)
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)

        self._streams[stream_name] = Manager.StreamInfo(
            name=stream_name,
            url=f"rtsp://{self._hostname}:{self._rtsp_port}/{stream_name}",
            owned=True,
            source_url=rtsp_url,
        )

        await self.refresh_streams()

    async def destroy_stream(self, stream_name: str) -> None:
        """Destroy a stream with the given name, and possibly its master relay."""
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        stream_info = self._streams.get(stream_name)
        if stream_info is None:
            logger.warning(f"Destroying unknown stream {stream_name}")
            return

        stream_info.active = False

        source_url = stream_info.source_url
        if source_url:
            relay_info = self._source_relays.get(source_url)
            if relay_info:
                if stream_name == relay_info.master_stream_name:
                    if relay_info.refcount > 0:
                        logger.info(f"Cannot destroy master stream {stream_name}, refcount = {relay_info.refcount}")
                        return
                    logger.info(f"Destroying master stream {stream_name}")
                    try:
                        await asyncio.to_thread(api.config_paths_delete, name=stream_name)
                    except Exception as e:
                        logger.warning(f"Failed to destroy master stream {stream_name}: {e}")
                    del self._source_relays[source_url]
                    self._streams.pop(stream_name, None)
                    await self.refresh_streams()
                    return
                else:
                    relay_info.refcount -= 1
                    logger.info(f"Decremented refcount for {relay_info.master_stream_name} → {relay_info.refcount}")
                    if relay_info.refcount <= 0:
                        master_name = relay_info.master_stream_name
                        logger.info(f"No more users of {master_name}; removing")
                        try:
                            await asyncio.to_thread(api.config_paths_delete, name=master_name)
                        except Exception as e:
                            logger.warning(f"Failed to destroy master stream {master_name}: {e}")
                        del self._source_relays[source_url]
                        self._streams.pop(master_name, None)

        try:
            await asyncio.to_thread(api.config_paths_delete, name=stream_name)
        except Exception as e:
            logger.warning(f"Failed to destroy stream: {e}")

        self._streams.pop(stream_name, None)
        await self.refresh_streams()

    async def destroy_all_streams(self) -> None:
        """Destroys all streams on the server"""
        for name in list(self.get_streams().keys()):
            await self.destroy_stream(name)

    async def get_config(self) -> str:
        """Retrieve the global configuration as a string"""
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        config = await asyncio.to_thread(api.config_global_get)
        return config.to_str()

    def get_streams(self) -> Dict[str, StreamInfo]:
        return dict(self._streams)

    async def refresh_streams(self) -> None:
        """Retrieve the current list of streams, updating the cached list"""
        api: PathsApi = PathsApi(self._api_client)
        streams = await asyncio.to_thread(api.paths_list)
        if streams.items is None:
            raise Exception("Failed to get stream list from server")
        stream_dict: Dict[str, MtxPath] = {str(stream.name): stream for stream in streams.items}

        # Add new streams
        for name, mtx_path in stream_dict.items():
            if name not in self._streams:
                logger.info(f"Unexpected new stream on server: {name}")
                self._streams[name] = Manager.StreamInfo(name=name, owned=False)
            stream_info = self._streams[name]
            stream_info.mtx_path = mtx_path
            stream_info.url = f"rtsp://{self._hostname}:{self._rtsp_port}/{name}"
            if mtx_path.ready != stream_info.active:
                logger.info(f"Stream {name} set to url: {stream_info.url}  active: {mtx_path.ready}")
                stream_info.active = mtx_path.ready
                if stream_info.owned:
                    if stream_info.expected_active is None and stream_info.active:
                        stream_info.expected_active = True
                    elif stream_info.expected_active and not stream_info.active:
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
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        """Initialization that should run on event loop"""
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
                await asyncio.sleep(Manager.POLLING_INTERVAL)
                await self.refresh_streams()
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")
        finally:
            print("Periodic task cleanup.")
