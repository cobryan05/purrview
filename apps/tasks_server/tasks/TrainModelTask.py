import asyncio
import json
import logging
from typing import Any, Optional

import httpx
import training_client
from training_client.api.training_api import TrainingApi
from training_client.models.submit_job_payload import SubmitJobPayload

from apps import APPS, APPS_CONFIG

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_task()
class TrainModelTask(Task):
    # Epochs can take minutes each — extend heartbeat tolerance.
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._trainer_type: str = str(params["trainer_type"])
        self._label_uuid: str = str(params.get("label_uuid", ""))
        self._model_name: str = str(params.get("model_name", ""))
        self._max_epochs: int = int(params.get("max_epochs", 50))
        self._image_source: Optional[str] = params.get("image_source")

        training_cfg = APPS_CONFIG[APPS.TRAINING_SERVER]
        self._training_base_url: str = f"http://{training_cfg.host}:{training_cfg.port}"

        tc_config = training_client.Configuration(host=self._training_base_url)
        self._api_client = training_client.ApiClient(tc_config)

        self._job_id: Optional[str] = resume_data.get("job_id") if resume_data else None

    async def _run(self) -> dict[str, Any]:
        # Submit a new job unless we're resuming a previous one.
        if not self._job_id:
            payload = SubmitJobPayload(
                trainer_type=self._trainer_type,
                label_uuid=self._label_uuid,
                model_name=self._model_name,
                max_epochs=self._max_epochs,
                image_source=self._image_source,
            )
            resp = await asyncio.to_thread(
                TrainingApi(self._api_client).submit_training_job,
                payload,
            )
            self._job_id = resp.job_id
            self.log(f"Submitted training job {self._job_id}")
            # Persist job_id immediately so a crash/cancel can still cancel the remote job.
            self._resume_data["job_id"] = self._job_id
            self._data_ready_flag.set()
        else:
            self.log(f"Resuming monitoring of training job {self._job_id}")

        # Send a fresh heartbeat before the SSE stream starts — epochs can be long.
        await self.heartbeat(force=True)

        return await self._consume_sse(self._job_id)

    async def _consume_sse(self, job_id: str) -> dict[str, Any]:
        """Open the SSE stream for *job_id* and relay events until done/error/cancel."""
        stream_url = f"{self._training_base_url}/api/training/jobs/{job_id}/stream"
        cancelled = False

        # Guard: if cancel was requested before we enter the SSE block, handle it now.
        if self._cancel_flag.is_set():
            cancelled = True
        else:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", stream_url) as response:
                    event_type: Optional[str] = None
                    async for raw_line in response.aiter_lines():
                        if self._cancel_flag.is_set():
                            cancelled = True
                            break

                        line = raw_line.strip()
                        if not line:
                            # Blank line = SSE event boundary; reset event_type.
                            event_type = None
                            continue
                        if line.startswith("event:"):
                            event_type = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            result = await self._handle_event(event_type, data)
                            if result is not None:
                                # Terminal event received; return immediately.
                                return result

        if cancelled:
            await self._cancel_remote_job(job_id)
            # Return a partial result rather than raising so the task finishes COMPLETED.
            return {"cancelled": True, "job_id": job_id}

        # Stream closed without a terminal event — treat as an error.
        raise RuntimeError(f"Training job {job_id} stream ended without a terminal event")

    async def _handle_event(self, event_type: Optional[str], data: dict) -> Optional[dict[str, Any]]:
        """Process one SSE event.  Returns a result dict if this is a terminal event, else None."""
        if event_type == "progress":
            progress = float(data.get("progress", 0.0))
            self._update_progress(progress * 100.0)
            log_msg = data.get("log", "")
            if log_msg:
                self.log(log_msg)
            await self.heartbeat()

        elif event_type == "status":
            status = data.get("status", "")
            if status:
                self.log(f"Status: {status}")

        elif event_type == "done":
            model_path = data.get("model_path")
            val_accuracy = data.get("val_accuracy")
            self.log(f"Training complete — model: {model_path}, val_accuracy: {val_accuracy}")
            result: dict[str, Any] = {"job_id": self._job_id}
            if model_path:
                result["model_path"] = model_path
            if val_accuracy is not None:
                result["val_accuracy"] = val_accuracy
            return result

        elif event_type == "error":
            msg = data.get("message", "unknown error")
            raise RuntimeError(f"Training job failed: {msg}")

        return None

    async def _cancel_remote_job(self, job_id: str) -> None:
        """Best-effort DELETE of the remote training job."""
        try:
            await asyncio.to_thread(
                TrainingApi(self._api_client).cancel_training_job,
                job_id,
            )
            self.log(f"Cancelled remote training job {job_id}")
        except Exception as exc:
            # Non-fatal — the training container will eventually time out.
            logger.warning("Could not cancel remote job %s: %s", job_id, exc)

    async def _deinit(self) -> None:
        pass

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "trainer_type": {
                "type": "str",
                "required": True,
                "default": "binary_image_classifier",
                "options": ["binary_image_classifier"],
                "help": "Which training algorithm to use",
            },
            "label_uuid": {
                "type": "str",
                "required": True,
                "help": "UUID of the label to train on (image_labels table)",
            },
            "model_name": {
                "type": "str",
                "required": True,
                "help": "Output model filename (without extension)",
            },
            "max_epochs": {
                "type": "int",
                "required": False,
                "default": 50,
                "help": "Maximum number of training epochs",
            },
            "image_source": {
                "type": "str",
                "required": False,
                "default": "filesystem",
                "options": ["filesystem", "http"],
                "help": "How the training server fetches images",
            },
        }
