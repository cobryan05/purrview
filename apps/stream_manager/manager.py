import mediamtx_client
from mediamtx_client.api_client import ApiClient
from mediamtx_client.api.paths_api import PathsApi
from mediamtx_client.api.configuration_api import ConfigurationApi
from mediamtx_client.models.path_conf import PathConf
from mediamtx_client.models.path import Path as MtxPath
from mediamtx_client.exceptions import BadRequestException

import asyncio
from typing import Optional, Dict, List


class Manager:
    POLLING_INTERVAL: float = 30.0

    def __init__(self, api_client: ApiClient):
        self._api_client: ApiClient = api_client
        self._task: Optional[asyncio.Task] = None

    def start(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def create_stream(self, src: str, stream_name: str, delay: float = 0, overwrite: bool = False):
        stream_exists: bool = stream_name in await self.get_streams()
        if stream_exists:
            if overwrite:
                results = await self.destroy_stream(stream_name)
            else:
                raise Exception(f"A stream named {stream_name} already exists!")

        path_conf: PathConf = PathConf(name=stream_name, source=src)
        api: ConfigurationApi = ConfigurationApi(self._api_client)

        results = await asyncio.to_thread(api.config_paths_add, name=stream_name, path_conf=path_conf)
        return results

    async def destroy_stream(self, stream_name: str):
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        results = await asyncio.to_thread(api.config_paths_delete, name=stream_name)
        return results

    async def get_streams(self) -> Dict[str, MtxPath]:
        api: PathsApi = PathsApi(self._api_client)
        streams = await asyncio.to_thread(api.paths_list)
        if not streams.items:
            return {}

        ret_dict: Dict[str, MtxPath] = {}
        for stream in streams.items:
            if stream.name is None:
                continue
            ret_dict[stream.name] = stream
        return ret_dict

    async def get_config(self) -> str:
        api: ConfigurationApi = ConfigurationApi(self._api_client)
        config = await asyncio.to_thread(api.config_global_get)
        return config.to_str()

    async def _worker_task(self):
        try:
            while True:
                # print("Running periodic task...")
                streams = await self.get_streams()
                # await asyncio.sleep(Manager.POLLING_INTERVAL)
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")
        finally:
            print("Periodic task cleanup.")


# TODO: initialize ApiClient connection, get list of running tasks, poll for changes
