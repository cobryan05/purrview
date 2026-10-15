"""Enumerate the host's GPUs, and map physical ids to CUDA ordinals.

Two numbering schemes are in play and they are not the same number:

* ``nvidia-smi`` reports a **physical** index. It ignores ``CUDA_VISIBLE_DEVICES``
  entirely — inside a container with ``CUDA_VISIBLE_DEVICES=1`` it still lists both
  cards as 0 and 1.
* torch and onnxruntime take a **CUDA ordinal**, which is a position within
  ``CUDA_VISIBLE_DEVICES``. With ``CUDA_VISIBLE_DEVICES=1`` the host's second card is
  ordinal 0, and asking for ordinal 1 names a device that does not exist.

So every number a human picks is physical, and every number stored in config is an
ordinal. This module owns that translation; nothing else should compute it.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

_QUERY = "index,name,pci.bus_id,memory.total,uuid"


@dataclass(frozen=True)
class GpuDevice:
    """One GPU as the driver reports it, plus where CUDA numbers it."""

    index: int
    name: str
    bus_id: str
    memory_mb: Optional[int]
    uuid: str
    # Position within CUDA_VISIBLE_DEVICES, or None when this card is not exposed to
    # this container at all. The value config stores.
    cuda_ordinal: Optional[int]

    @property
    def label(self) -> str:
        """Human-facing one-liner.

        Carries the PCI bus id because identical cards are the normal case — two of the
        same model have the same name and the same memory, and the bus id is the only
        short field that tells them apart (the UUID is unique but unreadable).
        """
        parts = [self.name]
        short_bus = self.bus_id.split(":", 1)[-1] if ":" in self.bus_id else self.bus_id
        if short_bus:
            parts.append(short_bus)
        if self.memory_mb:
            parts.append(f"{self.memory_mb // 1024} GB")
        return (
            f"GPU {self.index} — {parts[0]} ({', '.join(parts[1:])})"
            if len(parts) > 1
            else f"GPU {self.index} — {parts[0]}"
        )


def _visible_spec() -> Optional[List[str]]:
    """``CUDA_VISIBLE_DEVICES`` split into entries, or None when unset.

    Unset and empty are different states and must not be collapsed: unset means every
    card is visible, empty string means none are.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()] if raw.strip() else []


def _ordinal_for(device_index: int, device_uuid: str, spec: Optional[List[str]]) -> Optional[int]:
    """Where CUDA numbers this card, given a visibility spec."""
    if spec is None:
        return device_index
    for ordinal, entry in enumerate(spec):
        # The variable accepts UUIDs as well as indices, and a UUID entry may be a
        # prefix of the full one.
        if entry == str(device_index) or (entry.startswith("GPU-") and device_uuid.startswith(entry)):
            return ordinal
    return None


def list_gpus() -> List[GpuDevice]:
    """Every GPU the driver can see, in physical order.

    Shells out to ``nvidia-smi`` rather than using torch: this runs in
    ``inference_server`` too, and the CUDA inference image ships onnxruntime-gpu
    without torch. Returns [] on any failure — no driver, no binary, unparseable
    output — because "which GPUs are there" is never worth failing a request over.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception as e:
        logger.debug("Could not enumerate GPUs (%s)", e)
        return []

    spec = _visible_spec()
    devices: List[GpuDevice] = []
    for line in proc.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        try:
            memory_mb: Optional[int] = int(fields[3])
        except ValueError:
            memory_mb = None
        uuid = fields[4]
        devices.append(
            GpuDevice(
                index=index,
                name=fields[1],
                bus_id=fields[2],
                memory_mb=memory_mb,
                uuid=uuid,
                cuda_ordinal=_ordinal_for(index, uuid, spec),
            )
        )
    return devices


def visible_gpus() -> List[GpuDevice]:
    """Only the cards this container may actually use, in CUDA ordinal order."""
    exposed = [d for d in list_gpus() if d.cuda_ordinal is not None]
    return sorted(exposed, key=lambda d: d.cuda_ordinal or 0)
