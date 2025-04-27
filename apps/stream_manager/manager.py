from mediamtx_client.api_client import ApiClient
from mediamtx_client.api.paths_api import PathsApi
from mediamtx_client.api.configuration_api import ConfigurationApi
from mediamtx_client.models.path_conf import PathConf
from mediamtx_client.models.path import Path as MtxPath

import asyncio
import logging
from typing import Any, Optional, Dict
import sys


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    POLLING_INTERVAL: float = 30.0  # Interval in seconds for periodic tasks

    def __init__(self, api_client: ApiClient):
        """Initialize the Manager with an API client"""
        self._api_client: ApiClient = api_client
        self._streams: Dict[str, Any] = {}
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def create_relay_stream(self, src: str, stream_name: str, delay: float = 0, overwrite: bool = False) -> None:
        """Create a new stream with the given source and name
        If overwrite is True, destroy the existing stream with the same name before creating a new one"""
        await self.get_streams()
        stream_exists: bool = stream_name in self._streams
        if stream_exists:
            if overwrite:
                await self.destroy_stream(stream_name)
            else:
                raise Exception(f"A stream named {stream_name} already exists!")

        path_conf: PathConf = PathConf(name=stream_name, source=src)  # Define the stream configuration
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        # Add the stream configuration using the API
        await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        await self.get_streams()

        if delay > 0:

            # TODO: Create delayed stream
            pass

    async def destroy_stream(self, stream_name: str) -> None:
        """Destroy a stream with the given name"""
        # TODO: Do we have to deal with the delayed stream?
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        await asyncio.to_thread(api.config_paths_delete, name=stream_name)
        if stream_name not in self._streams:
            logger.warning(f"Destroying unknown stream {stream_name}")
        else:
            self._streams[stream_name] = None  # Mark as pending
        await self.get_streams()

    async def get_streams(self) -> Dict[str, MtxPath]:
        """Retrieve the current list of streams, updating the cached list"""
        api: PathsApi = PathsApi(self._api_client)
        streams = await asyncio.to_thread(api.paths_list)  # Fetch the list of streams
        ret_dict: Dict[str, MtxPath] = {}
        for stream in streams.items or {}:
            if stream.name is None:
                continue  # Skip streams without a name
            ret_dict[stream.name] = stream  # Add the stream to the dictionary

        # As long as we did the request might as well update our cache
        self._update_streams(ret_dict)
        return ret_dict

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
                logger.info(f"New stream detected: {name}")  # Log new streams
                self._streams[name] = stream  # Add the new stream to self._streams

        # Remove unexpectedly closed streams
        for name in list(self._streams.keys()):
            if name not in streams:
                stream = self._streams[name]
                if stream is not None:
                    logger.warning(f"Stream unexpectedly removed: {name}")
                else:
                    logger.info(f"Confirming removal of {name}")
                self._streams.pop(name)

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            while True:
                # Placeholder for periodic operations (e.g., monitoring or maintenance tasks)
                await self.get_streams()
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            print("Periodic task cleanup.")  # Perform cleanup when the task is stopped
