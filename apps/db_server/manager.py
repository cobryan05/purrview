"""Manages the database"""

import asyncio
import fnmatch
import hashlib
import io
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple
from uuid import uuid4

import aiofiles
import jstyleson
from sqlalchemy.orm import make_transient

from apps import (
    CACHE_DIR,
    RECORDINGS_CLIP_DIR,
    RECORDINGS_ORPHAN_GRACE_S,
    RECORDINGS_SNAPSHOT_DIR,
    RECORDINGS_TIMELAPSE_DIR,
    STREAM_BUFFER_RETENTION_S,
    THUMB_CACHE_MAX_MB,
)
from apps.db_server.hash_index import HashIndex
from apps.helpers.actions.outputDir import has_placeholder, resolve_output_dir, static_prefix
from apps.helpers.actions.Registry import action_registry
from apps.helpers.actions.saveTimelapseAction import DEFAULT_TIMELAPSE_DIR
from apps.helpers.consts import TaskStatus
from apps.helpers.datasetExport import (
    EXPORT_FORMAT_IDS,
    TarStream,
    image_extension,
    make_writer,
    sanitize_stem,
    split_images,
)
from apps.helpers.db.db_client import (
    BBOX_VERIFICATION_STATE_TO_TAG_UUID,
    DbClient,
    TagKind,
    WatchEventPinnedError,
)
from apps.helpers.db.types import (
    MODEL_TYPE_ARCHITECTURES,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    BBox,
    BBoxConfidenceRange,
    BBoxUpdate,
    BufferMode,
    DatasetCreate,
    DatasetRead,
    DatasetUpdate,
    EventBlipRead,
    FilterChainCreate,
    FilterChainRead,
    FilterChainUpdate,
    ImageItemStatus,
    ImageLabelRead,
    ImageLabelWithFilename,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    ImageTagStatusRead,
    Label,
    LabelScope,
    LabelType,
    ModelConfigCreate,
    ModelConfigNodeWrite,
    ModelConfigRead,
    ModelConfigUpdate,
    ModelGroupCreate,
    ModelGroupRead,
    ModelGroupUpdate,
    PreviewApplyResult,
    PreviewCandidate,
    PreviewCandidateBulkUpdate,
    PreviewCandidateCreate,
    PreviewCandidateQuery,
    PreviewCandidateRead,
    PreviewConflictPolicy,
    PreviewDecision,
    PreviewDecisionAction,
    PreviewRunCreate,
    PreviewRunRead,
    PreviewRunUpdate,
    Recording,
    RecordingRead,
    RecordingStatus,
    RegionRole,
    RuleRelation,
    Source,
    SourceRegionRead,
    SourceRegionWrite,
    Tag,
    TagRead,
    TagUpdate,
    TaskConfig,
    TaskConfigRead,
    TaskInstance,
    TaskInstanceRead,
    TaskParams,
    TaskResultData,
    TaskResumeData,
    TrainingCollectorCreate,
    TrainingCollectorRead,
    TrainingCollectorUpdate,
    WatchEvent,
    WatchEventRead,
    WatchProfileCreate,
    WatchProfileRead,
    WatchProfileUpdate,
    confidence_in_band,
)
from apps.helpers.fileUtils import get_safe_path
from apps.helpers.hashUtils import (
    DEFAULT_BBOX_HASH_SIZE,
    DEFAULT_IMAGE_HASH_SIZE,
    DEFAULT_IMAGE_THRESHOLD,
    compute_bbox_phash,
    compute_image_phash,
    phash_from_numpy,
    phash_from_numpy_crop,
)
from apps.helpers.imageProviders.Registry import image_provider_registry
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.clipDrain import DRAIN_FRACTION, MIN_DRAIN_INTERVAL_S, SEGMENT_LAG_S
from apps.helpers.serviceStatus import row, section
from apps.helpers.timeUtils import ensure_utc
from apps.helpers.tracking.Registry import tracker_registry
from apps.helpers.training.collectorSpec import CollectorSpec, build_review_actions, task_tag_candidates
from apps.helpers.types import FileEntryRead

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)

#: Settings key holding how many days of event clips and card art to keep.
#: Mirrored in `app/settings/main.js` — keep the two in sync.
RECORDING_RETENTION_SETTING_KEY = "recording_retention_days"

#: Kept for a month by default. Long enough that "what happened while I was away" is
#: still answerable after a holiday, short enough that a busy watcher does not fill a
#: disk before anyone notices. 0 or less disables the sweep entirely.
DEFAULT_RECORDING_RETENTION_DAYS = 30

#: Settings key holding the ceiling on total recording bytes, in MB.
#: Mirrored in `app/settings/main.js` — keep the two in sync.
RECORDING_MAX_TOTAL_MB_SETTING_KEY = "recording_max_total_mb"

#: Both bounds apply, whichever is reached first, for the same reason the task-instance
#: bounds do: they answer different questions. The age limit is "how far back do I still
#: care?"; the size limit is what actually caps storage when a busy watcher produces
#: clips faster than they age out. 0 or less disables that bound.
DEFAULT_RECORDING_MAX_TOTAL_MB = 10240

#: Statuses whose rows are *expected* to have no file yet, so their absence is not a
#: deleted clip. Spelled as strings because the column stores the enum's value.
_RECORDING_IN_PROGRESS_STATUSES = frozenset({RecordingStatus.PENDING.value, RecordingStatus.WRITING.value})

#: Circuit breaker on the reconcile pass. When most rows lose their file at once the
#: cause is almost never "the user deleted almost everything" — it is a volume that did
#: not mount, and deleting the history of a month of events because a bind mount was
#: late is not a recoverable mistake. Below the row floor the fraction is meaningless
#: (one missing clip out of two is 50%), so both have to be exceeded to trip it.
RECONCILE_GUARD_MAX_FRACTION = 0.5
RECONCILE_GUARD_MIN_ROWS = 8

#: Settings keys for the timelapse bounds. Deliberately *separate* from the recording
#: ones, because the two answer different questions about different material: a clip is
#: kept because something happened, a timelapse frame is kept because nothing did. A
#: month of events is a modest folder; a month of 30-second frames is hundreds of GB per
#: camera, so one pair of numbers cannot serve both.
#: Mirrored in `app/settings/main.js` — keep the two in sync.
TIMELAPSE_RETENTION_SETTING_KEY = "timelapse_retention_days"
TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY = "timelapse_max_total_mb"

#: Shorter than the recording default: a timelapse writes ~2900 frames per camera per
#: day at the default 30 s interval, so the size cap is what normally binds and the day
#: window mostly exists to keep the folder navigable. 0 or less disables either bound.
DEFAULT_TIMELAPSE_RETENTION_DAYS = 14
DEFAULT_TIMELAPSE_MAX_TOTAL_MB = 20480

#: The action's registry key, spelled out rather than imported from the registry: this
#: module reads action rows out of the database and never instantiates one.
TIMELAPSE_ACTION_KIND = "save_timelapse"

#: Action kinds that put an event in the Events view — the mirror of `Action.logs_event()`
#: in `apps/helpers/actions/action.py`, which is what the watcher itself reads. Spelled
#: out for the same reason as above: this module never instantiates an action, and
#: importing the action package here would drag cv2 and an MQTT client into db_server for
#: the sake of one predicate. `tests/tasks_server/test_watch_event_logging.py` asserts the
#: two sides name the same kinds, so adding a logging action and forgetting this warning
#: is a test failure rather than a silently absent hint.
EVENT_LOGGING_ACTION_KINDS = frozenset({"record_clip", "record_event_timelapse", "save_event"})

#: What `SaveTimelapseAction` writes: `<camera>_HH-MM-SS.jpg`, inside a `YYYY-MM-DD`
#: directory. Both halves are matched before anything is deleted, and that pattern is the
#: whole safety story for this sweep. Unlike recordings, a timelapse lives under
#: FILES_ROOT — the same tree as the images a user has spent months annotating — and its
#: directory is one they chose. A sweep that deleted whatever it found in there would
#: eventually be pointed at a dataset folder and take the annotations with it.
TIMELAPSE_FRAME_RE = re.compile(r"^.+_\d{2}-\d{2}-\d{2}\.jpe?g$", re.IGNORECASE)

#: How far below a timelapse root the scan will walk. A templated directory puts the day
#: folder one or two levels down (`timelapse/{source}/{date}`), so the walk cannot stop at
#: the first level any more — but it must still not turn into a full recursive stat of
#: whatever tree the root happens to sit above. Four covers every layout the templates can
#: produce with room to spare.
TIMELAPSE_SCAN_MAX_DEPTH = 4

#: How often the sweep runs. The window it enforces is measured in days, so anything
#: tighter than hourly is filesystem work for no benefit.
RETENTION_SWEEP_INTERVAL_S = 3600.0
# The WAL only needs reclaiming on a slow burn, and a checkpoint that finds the
# file under threshold still costs a stat — so this is deliberately far coarser
# than the 5s poll, and far finer than the hourly retention sweep it rides beside.
WAL_CHECKPOINT_INTERVAL_S = 300.0

#: Events deleted per size-eviction pass before the total is re-measured. Re-walking is
#: how the total stays truthful (it counts snapshots, which carry no size column), and
#: batching keeps the number of walks down.
SIZE_EVICTION_BATCH = 50

#: Cap on eviction passes, so a mis-set cap or an undeletable file cannot spin the
#: worker. Whatever is left over is reported and picked up on the next sweep.
SIZE_EVICTION_MAX_PASSES = 20
logger.setLevel(LOG_LEVEL)

# Only bboxes explicitly carrying this tag produce PRESENT image-label rows.
_BBOX_VERIFIED_TAG = VERIFIED_TAG_UUID

# Sentinel for Manager.update_label's parent_uuid param: `None` is itself a
# meaningful value there (promote to root), so it can't also mean "not supplied".
_PARENT_UUID_NOT_SET = object()

# Bulk image actions that mutate image_labels, and therefore need the sidecar's
# image_labels slice rewritten once per batch after the action has been applied.
IMAGE_LABEL_BULK_ACTIONS = ("setPresent", "setAbsent", "setUncertain")

# Bulk image actions that drop the record entirely. Both take the batched path
# through Manager.remove_image_records_bulk; they differ only in whether the
# image file on disk goes with the record ("delete") or stays ("removeFromDb").
REMOVAL_IMAGE_BULK_ACTIONS = ("delete", "removeFromDb")

# Crop-thumbnail outline weight, as a fraction of the RENDERED width (3px on the
# 300px crops the review grid asks for). See Manager._render_outlined_crop for
# why it can't be a source-pixel constant.
_OUTLINE_WIDTH_RATIO = 0.01
_OUTLINE_MIN_PX = 2

# ── Thumbnail encoding ──────────────────────────────────────────────────────────
#
# JPEG, not PNG. These are photographic crops, and PNG was storing them at ~57 KB
# apiece — 1,833 files filling the whole 100 MB cache, which then sat permanently at
# its cap evicting on every write and regenerating whatever it had just dropped. The
# same images at this quality land around 8-12 KB, so the working set stops touching
# the cap at all and the thrash disappears rather than being tuned around.
_THUMB_FORMAT = "JPEG"
_THUMB_SUFFIX = ".jpg"
_THUMB_MEDIA_TYPE = "image/jpeg"
_THUMB_QUALITY = 82

# Every suffix the cache has ever written. New files are always _THUMB_SUFFIX, but the
# accounting and eviction still have to see the PNGs written before the switch —
# glob only the new suffix and the old set becomes invisible: uncounted by the storage
# readout and, worse, never evicted, so it would sit there forever.
_THUMB_SUFFIXES = (".jpg", ".png")

# Evict down to this fraction of the cap rather than to just under it. Trimming to
# exactly the limit guarantees the very next write trips the trim again, which is what
# turned a bounded cache into a continuous evict-and-regenerate loop; the headroom
# means one scan buys many writes.
_THUMB_TRIM_TARGET = 0.85

#: Candidate-geometry keys that describe where a proposal *came from* rather than where
#: its box sits. Matched by prefix rather than listed exactly, so a producer that starts
#: recording one more `source*` fact does not silently lose it at apply — the failure
#: mode an explicit list has, and an invisible one, because the box still applies fine.
_PROVENANCE_PREFIX = "source"
_PROVENANCE_KEYS = frozenset({"estimated"})


def _provenance_from(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the provenance out of a candidate's geometry, dropping the geometry itself.

    `x`/`y`/`width`/`height` are the box and belong to the bbox columns; everything a
    harvester recorded about *origin* — which event, which blip, where in the clip,
    whether the box was interpolated — belongs in the bbox's metadata, because that is
    the only copy that outlives the preview run.
    """
    return {
        key: value
        for key, value in geometry.items()
        if (key.startswith(_PROVENANCE_PREFIX) or key in _PROVENANCE_KEYS) and value is not None
    }


def _decode_image_bytes(data: bytes):
    """Decode uploaded bytes, or None if they are not a readable image.

    Deliberately decodes with the same library that will later read the file
    off disk, rather than trusting the extension or sniffing magic bytes: the
    question worth answering is not "is this a PNG?" but "can this app open
    it?", and those differ for truncated or subtly-corrupt files.

    Runs on a worker thread (see the caller) — decoding a large image is slow
    enough to stall the event loop.
    """
    import cv2
    import numpy as np

    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    except Exception:
        logger.exception("Uploaded bytes could not be decoded")
        return None


class Manager:
    """Manages interactions with the database"""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks
    BULK_APPLY_PAGE_SIZE: int = 200  # Keyset page size for bulk_apply_bboxes (class attr so tests can shrink it)

    def __init__(self, db_client: DbClient, files_root: Path):
        """Initialize the Manager with a database client"""
        self._db_client = db_client
        self._files_root: Path = files_root
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._config = {}
        self._image_hash_size: int = int(os.environ.get("IMAGE_HASH_SIZE", str(DEFAULT_IMAGE_HASH_SIZE)))
        self._bbox_hash_size: int = int(os.environ.get("BBOX_HASH_SIZE", str(DEFAULT_BBOX_HASH_SIZE)))
        self._hash_index = HashIndex(
            db_client,
            image_hash_size=self._image_hash_size,
            bbox_hash_size=self._bbox_hash_size,
        )
        self._rebuild_in_progress: bool = False
        self._hash_index_ready: bool = False  # False until initial build completes on startup
        self._build_phase: str | None = None  # "migrating", "loading", or "constructing" during startup
        self._build_img_loaded: int = 0  # running count during loading phase
        self._build_bbox_loaded: int = 0
        self._db_ready: bool = False  # True after init_db (alembic) succeeds; gates normal requests
        self._init_error: str | None = None  # Set if _init() fails; surfaced as an error notice
        # Short-lived cache for expensive count queries.  Key is an MD5 of the
        # filter predicates (sort/page fields excluded); value is (count, ts).
        # TTL is intentionally short so writes that happen while browsing are
        # reflected within a few seconds when the user hits Refresh.
        # key → (count, monotonic timestamp, db write epoch when it was taken)
        self._count_cache: dict[str, tuple[int, float, int]] = {}
        self._COUNT_CACHE_TTL = 60.0
        # Expired entries are only ever overwritten on an identical key, so a browsing
        # session that keeps trying new filter combinations would grow this dict
        # forever. Sweep expired keys once the dict crosses a soft ceiling.
        self._COUNT_CACHE_MAX_ENTRIES = 512
        # Thumbnail generation decodes the *full-resolution* source image (PIL
        # open + convert) even when the output is a small bbox crop. Dispatched
        # via asyncio.to_thread that runs on the default executor, so a grid view
        # firing dozens of thumbnail requests at once could hold
        # min(32, cpu+4) full-res decodes in memory simultaneously — hundreds of
        # MB of spike for 4K sources. Cap the concurrent decodes instead.
        self._thumb_decode_sem = asyncio.Semaphore(int(os.environ.get("THUMB_DECODE_CONCURRENCY", "4")))
        # Held reference to the in-flight cache-trim task: asyncio only keeps weak
        # references to tasks, so a bare create_task() can be garbage-collected
        # mid-run. Doubles as a coalescing flag — one trim at a time is plenty.
        self._thumb_trim_task: Optional[asyncio.Task] = None
        # The directories the retention sweep may walk and delete inside. An instance
        # attribute rather than a module constant read directly, so a test can point it
        # at a tmp dir: a sweep that walks a directory and unlinks must never be able to
        # run against the developer's own /data/recordings just because a test called it.
        self._recording_roots: Tuple[Path, ...] = (
            RECORDINGS_CLIP_DIR,
            RECORDINGS_SNAPSHOT_DIR,
            # Event timelapses. Included from the start rather than added once someone
            # noticed the folder growing: a root the orphan sweep does not know about is
            # one where a failed compile's leftovers accumulate with nothing to remove
            # them, and where the size cap silently under-measures the disk it caps.
            RECORDINGS_TIMELAPSE_DIR,
        )
        # When the retention sweep last ran, so it happens on a clock of its own rather
        # than on every worker tick. None means "not yet", which makes the first sweep
        # run shortly after startup — the point at which a long-stopped deployment has
        # the most expired material to shed.
        self._last_retention_sweep: Optional[float] = None
        self._last_wal_checkpoint: Optional[float] = None
        # Maximum seconds to wait for the count query before returning -1.
        # count_images_search does a full table scan with no LIMIT; on large
        # datasets (100k+ images) with complex multi-EXISTS filters it can take
        # many seconds.  The list query (LIMIT 51) completes quickly thanks to
        # index early-termination, so we cap the count independently.
        self._COUNT_QUERY_TIMEOUT = 3.0  # seconds

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        """Initialization that should run on event loop"""
        self._build_phase = "migrating"
        try:
            await self._db_client.init_db()
        except Exception as e:
            self._init_error = str(e)
            self._build_phase = None
            logger.error("Database initialization (alembic) failed: %s", e, exc_info=True)
            raise  # re-raise so _worker_task stops and doesn't enter the polling loop
        self._build_phase = None
        self._db_ready = True

        # Hash sizes are stored in DB settings so the review page, tasks, and the
        # db_server all reference the same value.  Read them here (after init_db
        # ensures the settings table exists) and override the env-var defaults set
        # in __init__.  Falls back silently to the env-var value when the setting
        # is absent (e.g. first run before the user has opened the settings page).
        raw_img = await self._db_client.get_setting("dedup_image_hash_size")
        raw_bbox = await self._db_client.get_setting("dedup_bbox_hash_size")
        if raw_img is not None:
            try:
                self._image_hash_size = int(raw_img)
            except (ValueError, TypeError):
                pass
        if raw_bbox is not None:
            try:
                self._bbox_hash_size = int(raw_bbox)
            except (ValueError, TypeError):
                pass
        # Propagate the resolved sizes to the HashIndex before building.
        self._hash_index.image_hash_size = self._image_hash_size
        self._hash_index.bbox_hash_size = self._bbox_hash_size

        def _on_phase(p: str) -> None:
            self._build_phase = p
            if p == "constructing":
                self._build_img_loaded = 0
                self._build_bbox_loaded = 0

        def _on_progress(img: int, bbox: int) -> None:
            self._build_img_loaded = img
            self._build_bbox_loaded = bbox

        await self._hash_index.build(on_phase=_on_phase, on_progress=_on_progress)
        self._build_phase = None
        self._build_img_loaded = 0
        self._build_bbox_loaded = 0
        self._hash_index_ready = True

    async def _maybe_checkpoint_wal(self) -> None:
        """Keep the write-ahead log from growing without bound.

        Runs here rather than in a request handler for the same reason the hash
        rebuild does: a TRUNCATE checkpoint wants a quiet moment, and asking for
        one from inside a live transaction is how you get a busy result every
        time. DbClient.checkpoint_wal() decides whether there is anything to do,
        so this only owns the schedule.

        Failures are logged and swallowed — the worker also rebuilds the hash
        index and runs retention, and an oversized WAL must not take those down.
        """
        now = time.monotonic()
        if self._last_wal_checkpoint is not None and now - self._last_wal_checkpoint < WAL_CHECKPOINT_INTERVAL_S:
            return
        self._last_wal_checkpoint = now
        try:
            await self._db_client.checkpoint_wal()
        except Exception:
            logger.exception("WAL checkpoint failed")

    async def _maybe_prune_expired_recordings(self) -> None:
        """Run the retention sweep at most once an hour, and never let it kill the loop.

        Hourly rather than per tick because the sweep touches the filesystem and the
        window it enforces is measured in days — running it every poll would be pure
        cost. Failures are logged and swallowed: the worker also rebuilds the hash
        index, and a retention error must not take that down with it.
        """
        now = time.monotonic()
        if self._last_retention_sweep is not None and now - self._last_retention_sweep < RETENTION_SWEEP_INTERVAL_S:
            return
        self._last_retention_sweep = now
        try:
            await self.prune_expired_recordings()
        except Exception:
            logger.exception("Retention sweep failed")
        # After the retention pass, not before: retention deletes rows *and* their files
        # together, so reconciling first would stat paths that are about to go anyway.
        try:
            await self.reconcile_missing_recording_files()
        except Exception:
            logger.exception("Recording reconcile failed")
        # Its own try, not a second statement inside the one above: the two sweeps share
        # a schedule and nothing else, and a recordings failure that also silently
        # skipped the timelapse pass would let a disk fill for a reason the logs blame
        # on the wrong subsystem.
        try:
            await self.prune_expired_timelapse()
        except Exception:
            logger.exception("Timelapse retention sweep failed")

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)
                # Opportunistically rebuild the hash index when the tombstone
                # ratio has crossed the threshold.  Doing it here (between
                # requests rather than inside a request handler) avoids opening
                # a concurrent DB session while another transaction is active.
                await self._hash_index.rebuild_if_needed()
                await self._maybe_prune_expired_recordings()
                await self._maybe_checkpoint_wal()
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")
        except Exception as e:
            # _init() already set _init_error and logged before re-raising.
            # The worker stops here; the error notice remains visible in /status.
            logger.error("Worker task terminated after initialization failure: %s", e)
        finally:
            print("Periodic task cleanup.")

    ################################################################################
    # LABELS API
    ################################################################################

    async def create_new_label(
        self,
        name: str,
        color: str,
        parent_uuid: Optional[str] = None,
        label_type: LabelType = LabelType.STANDARD,
        scope: LabelScope = LabelScope.OBJECT,
        publish_to_ha: bool = False,
        publish_per_source: bool = False,
    ) -> Label:
        """
        Adds a new label to the database

        Returns:
            Label: Label added to database
        """
        ret = await self._db_client.add_label(
            name,
            color,
            parent_uuid=parent_uuid,
            label_type=label_type,
            scope=scope,
            publish_to_ha=publish_to_ha,
            publish_per_source=publish_per_source,
        )
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))
        return ret

    async def delete_label(self, label_uuid: str) -> None:
        """
        Deletes a label from the database
        """
        # Check if the label has a parent
        child_uuids = await self._db_client.get_label_children(label_uuid)
        if child_uuids:
            raise ValueError(f"Cannot delete label '{label_uuid}' because it has child labels.")

        await self._db_client.delete_label(label_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def update_label(
        self,
        label_uuid: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        label_type: Optional[LabelType] = None,
        scope: Optional[LabelScope] = None,
        publish_to_ha: Optional[bool] = None,
        publish_per_source: Optional[bool] = None,
        parent_uuid: Any = _PARENT_UUID_NOT_SET,
    ) -> None:
        """
        Update a label's name, color, label_type, scope, and/or parent.

        Args:
            label_uuid (str): uuid of the label to update.
            name (Optional[str]): New name for the label.
            color (Optional[str]): New color for the label.
            label_type (Optional[LabelType]): New label type.
            scope (Optional[LabelScope]): New label scope.
            publish_to_ha (Optional[bool]): Whether this label gets a Home Assistant
                device at all. Does not affect detection, events or history.
            publish_per_source (Optional[bool]): Whether it also gets the finest
                (label x rule x source) counter tier. Off by default — see the column's
                comment in `types.py` for why this one is the cardinality risk.
            parent_uuid: New parent uuid ("move" the label), or None to promote
                the label to root. Uses the sentinel default `_PARENT_UUID_NOT_SET`
                (rather than `None`) because `None` is itself a meaningful value
                here (promote to root) distinct from "the caller didn't ask to
                change the parent at all".

        Raises:
            ValueError: if `parent_uuid` would make the label its own parent, or
                a descendant of itself (a cycle).
        """
        # `None` means "not supplied" for every key here, which is why the two booleans
        # can live in the same filtered dict: `False` is not `None`, so turning a flag
        # off survives the filter while leaving it alone does not.
        updates = {
            k: v
            for k, v in {
                "name": name,
                "color": color,
                "label_type": label_type,
                "scope": scope,
                "publish_to_ha": publish_to_ha,
                "publish_per_source": publish_per_source,
            }.items()
            if v is not None
        }

        if parent_uuid is not _PARENT_UUID_NOT_SET:
            if parent_uuid is not None:
                await self._assert_no_reparent_cycle(label_uuid, parent_uuid)
            updates["parent_uuid"] = parent_uuid

        await self._db_client.update_label(label_uuid=label_uuid, **updates)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def _assert_no_reparent_cycle(self, label_uuid: str, new_parent_uuid: str) -> None:
        """
        Walk up from `new_parent_uuid` toward the root. If we encounter
        `label_uuid` along the way (including immediately, i.e. new_parent_uuid
        == label_uuid), the move would make the label its own ancestor — reject it.

        Complexity: O(V) to build the uuid->label map (get_label_uuid_map already
        does this walk once) + O(depth) for the upward walk.
        """
        uuid_to_node = await self.get_label_uuid_map()
        current: Optional[str] = new_parent_uuid
        seen: Set[str] = set()
        while current is not None:
            if current == label_uuid:
                raise ValueError(
                    f"Cannot re-parent label '{label_uuid}': the target parent "
                    f"'{new_parent_uuid}' is the label itself or one of its own descendants."
                )
            if current in seen:
                break  # defensive: pre-existing cycle in stored data; don't loop forever
            seen.add(current)
            node = uuid_to_node.get(current)
            current = node.parent_uuid if node else None

    async def get_label_uuid_map(self) -> Dict[str, Label]:
        flat_list: List[Label] = await self._db_client.list_labels()

        # Manually create parent<->child relationships
        uuid_to_node: Dict[str, Label] = {}
        for x in flat_list:
            # Severe the SQL link and manually initialize children
            make_transient(x)
            x.children = []
            uuid_to_node[x.uuid] = x

        for node in uuid_to_node.values():
            if node.parent_uuid:
                parent_node = uuid_to_node.get(node.parent_uuid)
                if parent_node:
                    parent_node.children.append(node)

        return uuid_to_node

    async def get_labels(self) -> List[Label]:
        """
        Gets a list of all labels
        """
        return list((await self.get_label_uuid_map()).values())

    ################################################################################
    # SOURCES API
    ################################################################################

    async def get_avail_sources(self) -> List[Source]:
        """
        List all sources managed by the Manager.
        """
        sources = await self._db_client.get_sources()
        return sources

    async def get_source_ingest_tiers(self) -> Dict[str, Dict[str, Any]]:
        """Resolve every source's ingest tier: can it be rewound, and why or why not.

        Served rather than re-derived in the browser deliberately — `resolve_ingest_tier`
        is documented as the single function that decides this, because the same answer
        drives the relay's path config, the watcher's choice of clip backend and the UI
        badge, and those three disagreeing is a class of bug with no visible symptom.

        Pure and cheap (no I/O beyond the source list), so it is recomputed per request
        rather than cached: a source's URL can be edited at any time, and a stale
        "buffered" badge is exactly the lie this endpoint exists to prevent.
        """
        from apps.helpers.streams.ingest import resolve_ingest_tier

        tiers: Dict[str, Dict[str, Any]] = {}
        for source in await self._db_client.get_sources():
            provider_cls = image_provider_registry.get(source.typename)
            try:
                plan = resolve_ingest_tier(provider_cls, source.params or {})
            except Exception:
                # An unknown provider or an unparseable URL must not blank the badge for
                # every other source; it is reported as the unbuffered case it is.
                logger.exception("Could not resolve ingest tier for source %s", source.uuid)
                continue
            tiers[source.uuid] = plan.to_dict()
        return tiers

    async def create_new_source(self, image_provider: str, provider_params: Dict[str, Any], source_name: str) -> Source:
        """
        Create a new source from an image source as a preset image_provider/params

        Args:
            image_provider (str): The image provider for this source
            params (Dict[str, Any]): The parameters to pass to the image provider
            source_name (str): The name given for the new source
        """
        if image_provider not in image_provider_registry:
            raise ValueError(f"Unknown image provider: {image_provider}")

        ret: Source = await self._db_client.add_source(
            name=source_name, typename=image_provider, params=provider_params
        )
        return ret

    async def delete_sources(self, uuid_list: list[str]) -> None:
        """
        Delete sources by their UUIDs.

        Args:
            uuid_list (list[str]): List of source UUIDs to delete.
        """
        await self._db_client.delete_sources(uuid_list)

    async def update_source(
        self,
        source_uuid: str,
        image_provider: str,
        provider_params: Dict[str, Any],
        source_name: str,
    ) -> Source:
        """
        Updates sources by their UUIDs.

        Args:
            uuid_list (list[str]): List of source UUIDs to delete.
        """
        return await self._db_client.update_source(
            source_uuid=source_uuid,
            name=source_name,
            typename=image_provider,
            params=provider_params,
        )

    async def list_source_regions(
        self, source_uuid: Optional[str] = None, role: Optional[str] = None
    ) -> List[SourceRegionRead]:
        """Zones and masks for one source, or for all sources when `source_uuid` is None."""
        if role is not None and role not in {r.value for r in RegionRole}:
            raise ValueError(f"Unknown region role: {role}")
        return await self._db_client.list_source_regions(source_uuid, role)

    async def set_source_regions(
        self, source_uuid: str, regions: List[SourceRegionWrite]
    ) -> tuple[List[SourceRegionRead], List[str]]:
        """Replace a source's regions, reporting any rules the write orphaned.

        The orphan check runs *before* the write, while the removed uuids are still
        resolvable. It is a warning rather than a refusal: deleting a zone whose rule is
        also on its way out is legitimate, and blocking it would make the two edits
        impossible to sequence.
        """
        existing = {r.uuid for r in await self._db_client.list_source_regions(source_uuid)}
        surviving = {r.uuid for r in regions if r.uuid}
        orphaned = await self._db_client.find_rules_referencing_regions(existing - surviving)

        saved = await self._db_client.replace_source_regions(source_uuid, regions)
        return saved, sorted(set(orphaned))

    async def list_avail_image_providers(self) -> List[str]:
        """
        List all available image providers.
        """
        return list(image_provider_registry.keys())

    async def get_image_provider_schema(self, image_provider: str) -> Dict[str, Any]:
        """
        Get the schema for a specific image provider.

        Args:
            image_provider (str): The name of the image provider.

        Returns:
            Dict[str, Any]: The schema for the image provider.
        """
        if image_provider not in image_provider_registry:
            raise ValueError(f"Unknown image provider: {image_provider}")

        return image_provider_registry[image_provider].params_schema()

    ################################################################################
    # TAGS API
    ################################################################################
    async def create_new_tag(
        self,
        name: str,
        color: Optional[str] = None,
        icon: Optional[str] = None,
        protected: bool = False,
        kind: TagKind = TagKind.GENERIC,
        exclusive_group: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> Tag:
        """
        Adds a new tag to the database
        """
        ret = await self._db_client.add_tag(
            name=name,
            color=color,
            icon=icon,
            protected=protected,
            kind=kind,
            exclusive_group=exclusive_group,
            uuid=uuid,
        )
        return ret

    async def delete_tag(self, tag_uuid: str) -> bool:
        """
        Deletes a tag from the database.
        Returns True if deleted, False if not found.
        """
        return await self._db_client.delete_tag(tag_uuid)

    async def update_tag(self, tag_uuid: str, update_data: TagUpdate) -> Optional[Tag]:
        """
        Update a tag's fields.

        Args:
            tag_uuid: The "Who" - used to find the row.
            update_data: The "What" - the validated fields to change.
        """
        changes = update_data.model_dump(exclude_unset=True)

        if not changes:
            return await self.get_tag_by_uuid(tag_uuid)

        return await self._db_client.update_tag(tag_uuid, **changes)

    async def get_tag_by_uuid(self, uuid: str) -> Optional[Tag]:
        """Retrieve a tag by its primary key UUID."""
        return await self._db_client.get_tag_by_uuid(uuid)

    async def get_tag_uuid_map(self) -> Dict[str, Tag]:
        """
        Get a mapping of tag UUIDs to their SQLModel objects.
        """
        tags: List[Tag] = await self._db_client.list_tags()
        return {tag.uuid: tag for tag in tags}

    async def get_tags(self) -> List[Tag]:
        """
        Gets a list of all tags
        """
        return await self._db_client.list_tags()

    ################################################################################
    # USER SETTINGS API
    ################################################################################

    async def get_setting(self, key: str) -> Optional[str]:
        """Return stored value for key, or None if not set."""
        return await self._db_client.get_setting(key)

    async def set_setting(self, key: str, value: str) -> None:
        """Upsert a user setting."""
        await self._db_client.set_setting(key, value)

    ################################################################################
    # IMAGES API
    ################################################################################

    async def get_image_metadata(self, image_path: Path) -> Optional[ImageRecordRead]:
        """
        Get metadata for image by resolving image ID from filename.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Optional[ImageRecordRead]: Full metadata object or None.
        """
        safe_path = get_safe_path(self._files_root, image_path)
        if not safe_path or not safe_path.exists():
            return None
        image_record = await self._db_client.get_image_by_filename(safe_path)
        if image_record is None or len(image_record.bboxes) == 0:
            await self._db_client.sync_image_from_json(safe_path)
            image_record = await self._db_client.get_image_by_filename(safe_path)
        if image_record is None:
            return None
        return ImageRecordRead.model_validate(image_record)

    def _get_json_sidefile(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def _fixup_entry(self, entry):
        logger.info(f"TODO: Fixup entry {entry}")

    async def update_image_metadata(self, image_rel_path: str, update: ImageRecordUpdate) -> Optional[ImageRecordRead]:
        """
        Replace the bboxes for an image, persisting to both the DB and the JSON sidecar.

        The sidecar is written after re-reading the DB so that label and tag *names* are
        stored alongside UUIDs.  Names serve as fallback resolution keys if UUIDs change
        on a future DB rebuild.

        Args:
            image_rel_path (str): Relative image path.
            update (ImageRecordUpdate): New bbox data.

        Returns:
            The updated ImageRecordRead, or None if the image path is invalid.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if safe_path is None or not safe_path.exists():
            logger.warning(f"Path not found or inaccessible: {image_rel_path}")
            return None
        image = await self._db_client.add_image(safe_path, source_uuid=update.source_uuid)

        # Capture old bbox entries BEFORE the DB write so we can remove stale hash
        # index entries.  write_image_metadata_to_db does a full DELETE + INSERT, so
        # old phashes are gone once it returns.
        old_image_record = await self._db_client.get_image_by_uuid(image.uuid)
        old_bboxes = old_image_record.bboxes if old_image_record else []

        # Carry each bbox's stored metadata_json across the DELETE+INSERT unless
        # the caller explicitly supplied one. The canvas save posts only
        # geometry/label/tags, so without this every save silently wiped
        # source_model / source_confidence / run_model_added — and
        # relabel_prev_label_uuid, which "revert filter-chain relabel" reads.
        # `{}` still clears, so a caller can deliberately empty it.
        old_metadata = {b.uuid: (b.metadata_json or {}) for b in old_bboxes}
        for b in update.bboxes:
            if b.metadata_json is None:
                b.metadata_json = old_metadata.get(b.uuid, {})

        # Tombstone all previous bbox entries from the hash index (covers
        # deleted, moved, and resized boxes). This must happen outside the
        # phash-recompute conditional below: deleting the LAST bbox leaves
        # update.bboxes empty, skipping recompute entirely — but the old
        # entries still have to go, or deleted boxes keep matching in
        # similarity searches.
        for old_bbox in old_bboxes:
            if old_bbox.phash and old_bbox.label_uuid:
                self._hash_index.remove_bbox(old_bbox.uuid, old_bbox.label_uuid)

        # Hashes are computed BEFORE the write, not after it, so they can ride
        # the same transaction as the rows they describe. Decoding the frame and
        # cropping it is ~50ms of pure CPU; doing it while holding SQLite's
        # single writer lock would block every other service for that long, and
        # doing it after the write meant asking for the lock a second time —
        # which, behind a running collect task, cost far more than the work.
        #
        # Image phash is computed once (first write only — the file never changes).
        # Bbox phashes must always be (re-)computed because the user may have moved or
        # resized boxes since the last save, which changes the cropped pixel region.
        image_phash: Optional[str] = image.phash
        bbox_phash_list: List[Dict[str, str]] = []
        if image_phash is None or update.bboxes:
            try:
                image_bgr = await asyncio.to_thread(__import__("cv2").imread, str(safe_path))
                if image_bgr is not None:
                    if image_phash is None:
                        image_phash = await asyncio.to_thread(phash_from_numpy, image_bgr, self._image_hash_size)
                    # Always (re-)compute per-bbox phashes.
                    for b in update.bboxes:
                        try:
                            bp = await asyncio.to_thread(
                                phash_from_numpy_crop,
                                image_bgr,
                                b.x,
                                b.y,
                                b.width,
                                b.height,
                                self._bbox_hash_size,
                            )
                            b.phash = bp
                            bbox_phash_list.append({"uuid": b.uuid, "phash": bp})
                        except Exception:
                            logger.exception("bbox phash error for %s", b.uuid)
            except Exception:
                logger.exception("phash computation failed for %s", image_rel_path)
                image_phash = image.phash  # revert to previously-stored value on error

        # One write for the geometry, the tags and every hash that goes with them.
        await self._db_client.write_image_metadata_to_db(
            image.uuid,
            update,
            image_phash=image_phash if image_phash != image.phash else None,
        )

        # Update hash index (old entries were tombstoned above).
        if image.phash is None and image_phash:
            self._hash_index.add_image(image.uuid, image_phash)
        uuid_to_label = {b.uuid: b.label_uuid for b in update.bboxes}
        for bp_entry in bbox_phash_list:
            label_uuid = uuid_to_label.get(bp_entry["uuid"])
            if label_uuid:
                self._hash_index.add_bbox(bp_entry["uuid"], label_uuid, bp_entry["phash"])

        # Re-read from DB; get_image_by_filename eager-loads bboxes → label + tags,
        # so we can include names in the sidecar for rebuild robustness.
        updated = await self._db_client.get_image_by_filename(safe_path)
        if updated:
            enriched_update = ImageRecordUpdate(
                bboxes=[
                    BBoxUpdate(
                        uuid=b.uuid,
                        label_uuid=b.label_uuid,
                        label_name=b.label.name if b.label else None,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        tag_uuids=[t.uuid for t in (b.tags or [])],
                        tag_names=[t.name for t in (b.tags or [])],
                        phash=b.phash,
                    )
                    for b in updated.bboxes
                ]
            )
            await asyncio.to_thread(
                self._db_client.write_image_metadata_to_json,
                self._get_json_sidefile(safe_path),
                image.uuid,
                image.filename,
                enriched_update,
                updated.phash,
            )
            await self._recompute_image_labels(image.uuid, enriched_update)
        else:
            # DB re-read failed; fall back to the raw caller payload for label recomputation.
            await self._recompute_image_labels(image.uuid, update)
        return ImageRecordRead.model_validate(updated) if updated else None

    async def update_single_bbox(
        self,
        bbox_uuid: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update one bbox's label and tags, then sync sidecar and image_labels."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None or result[1] is None:
            return False
        bbox, image_record = result
        old_label_uuid = bbox.label_uuid
        await self._db_client.update_single_bbox(bbox_uuid, label_uuid, tag_uuids, metadata_json=metadata_json)
        # Move bbox in hash index if label changed and bbox has a phash.
        if label_uuid and label_uuid != old_label_uuid and bbox.phash and old_label_uuid:
            self._hash_index.move_bbox(bbox_uuid, old_label_uuid, label_uuid, bbox.phash)
        await self._sync_image_state(image_record)
        return True

    async def add_single_bbox(
        self,
        image_rel_path: str,
        label_uuid: Optional[str],
        x: float,
        y: float,
        width: float,
        height: float,
        tag_uuids: List[str],
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Insert one new bbox on an image (leaving siblings untouched), then sync
        its phash / hash-index entry / JSON sidecar / image_labels. Returns the new
        bbox uuid, or None if the image path is invalid.

        Unlike update_image_metadata (a full bbox replace that re-writes and
        re-phashes every box on the image), this is a targeted single-row insert —
        the right primitive for programmatic callers such as the filter-chain
        add_bbox action.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if safe_path is None or not safe_path.exists():
            logger.warning(f"Path not found or inaccessible: {image_rel_path}")
            return None
        image = await self._db_client.add_image(safe_path)
        bbox_uuid = str(uuid4())
        await self._db_client.add_single_bbox(
            image.uuid, bbox_uuid, label_uuid, x, y, width, height, tag_uuids, metadata_json=metadata_json
        )

        # Compute the new crop's phash and register it in the hash index (mirrors
        # the per-bbox phash bookkeeping in update_image_metadata).
        try:
            image_bgr = await asyncio.to_thread(__import__("cv2").imread, str(safe_path))
            if image_bgr is not None:
                bp = await asyncio.to_thread(
                    phash_from_numpy_crop, image_bgr, x, y, width, height, self._bbox_hash_size
                )
                await self._db_client.update_phashes(image.uuid, image.phash, [{"uuid": bbox_uuid, "phash": bp}])
                if label_uuid and bp:
                    self._hash_index.add_bbox(bbox_uuid, label_uuid, bp)
        except Exception:
            logger.exception("phash computation failed for new bbox on %s", image_rel_path)

        image_record = await self._db_client.get_image_by_uuid(image.uuid)
        if image_record is not None:
            await self._sync_image_state(image_record)
        return bbox_uuid

    async def delete_single_bbox(self, bbox_uuid: str) -> bool:
        """Delete one bbox, then sync sidecar and image_labels."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None or result[1] is None:
            return False
        bbox, image_record = result
        # Tombstone in hash index before DB deletion.
        if bbox.label_uuid:
            self._hash_index.remove_bbox(bbox_uuid, bbox.label_uuid)
        await self._db_client.delete_single_bbox(bbox_uuid)
        await self._sync_image_state(image_record)
        return True

    async def bulk_apply_bboxes(
        self,
        search_params: ImageSearchParams,
        action: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        bbox_tag_filter: List[str],
        bbox_label_filter: List[str],
        show_all_bboxes: bool = False,
        bbox_verification: Optional[str] = None,
        bbox_confidence: Optional[BBoxConfidenceRange] = None,
    ) -> Tuple[int, int]:
        """Apply action to every bbox matching the filter, across all pages.

        Pages through list_images_search in batches of 200, applies the requested
        action to each matching bbox, and calls _sync_image_state once per image.

        bbox_tag_filter / bbox_label_filter / bbox_verification mirror the
        client-side bbox subset logic:
          - bbox_tag_filter: bbox must carry at least one of these tag UUIDs
            (ignored when show_all_bboxes is True or the list is empty)
          - bbox_label_filter: bbox label_uuid must be in this set
            (ignored when the list is empty)
          - bbox_verification: bbox must carry the matching system verification
            tag. ANDed with bbox_tag_filter rather than folded into it — that
            list is OR-semantics, so merging the verification tag in would
            widen the scope instead of narrowing it.
          - bbox_confidence: bbox's stored model confidence must fall in the
            band. Same reason it is a bbox-level argument rather than only a
            filter term: the term picks images that HAVE an in-band bbox, which
            would still let the action hit their out-of-band boxes.

        Returns (modified_count, fail_count).
        """
        tag_filter_set: Optional[Set[str]] = (
            None if show_all_bboxes else (set(bbox_tag_filter) if bbox_tag_filter else None)
        )
        label_filter_set: Optional[Set[str]] = set(bbox_label_filter) if bbox_label_filter else None
        # 'any' is the UI's no-op tri-state value; treat it like an omitted filter.
        verification_tag_uuid: Optional[str] = (
            BBOX_VERIFICATION_STATE_TO_TAG_UUID.get(bbox_verification) if bbox_verification else None
        )
        confidence_band = bbox_confidence if (bbox_confidence and bbox_confidence.is_active()) else None

        # Paginate by uuid regardless of the caller's requested sort. The action
        # mutates image.updated_at, so keyset-paginating by updated_at (asc) would
        # bump every processed row past the cursor and re-match it on the next
        # page — an infinite loop that never terminates. Traversal order is
        # irrelevant to the action (each matching bbox is touched exactly once).
        search_params = search_params.model_copy(update={"sort_by": "uuid", "sort_order": "asc"})

        modified_count = 0
        fail_count = 0
        _PAGE_SIZE = self.BULK_APPLY_PAGE_SIZE
        cursor_uuid: Optional[str] = None
        cursor_sort_value = None

        while True:
            page_params = search_params.model_copy(
                update={
                    "limit": _PAGE_SIZE + 1,
                    "after_uuid": cursor_uuid,
                    "after_sort_value": cursor_sort_value,
                }
            )
            batch = await self._db_client.list_images_search(page_params)
            has_more = len(batch) > _PAGE_SIZE
            batch = batch[:_PAGE_SIZE]

            if not batch:
                break

            for img in batch:
                # Replicate the client-side bbox subset filter.
                matching = []
                for bbox in img.bboxes or []:
                    if tag_filter_set is not None and not any(t.uuid in tag_filter_set for t in bbox.tags or []):
                        continue
                    if label_filter_set is not None and bbox.label_uuid not in label_filter_set:
                        continue
                    if verification_tag_uuid is not None and not any(
                        t.uuid == verification_tag_uuid for t in bbox.tags or []
                    ):
                        continue
                    if confidence_band is not None and not confidence_in_band(
                        # The generated column, already loaded with the row —
                        # identical to re-parsing metadata_json, minus the parse.
                        bbox.confidence,
                        confidence_band.min,
                        confidence_band.max,
                        confidence_band.include_missing,
                    ):
                        continue
                    matching.append(bbox)

                if not matching:
                    continue

                any_modified = False
                for bbox in matching:
                    try:
                        await self._apply_bbox_bulk_action(bbox, action, label_uuid, tag_uuids)
                        modified_count += 1
                        any_modified = True
                    except Exception as e:
                        logger.error("bulk_apply_bboxes: bbox %s failed: %s", bbox.uuid, e)
                        fail_count += 1

                if any_modified:
                    try:
                        await self._sync_image_state(img)
                    except Exception as e:
                        logger.error(
                            "bulk_apply_bboxes: _sync_image_state failed for %s: %s",
                            img.uuid,
                            e,
                        )

            if not has_more:
                break

            last = batch[-1]
            sort_by = search_params.sort_by
            cursor_uuid = last.uuid
            cursor_sort_value = (
                (last.updated_at if sort_by == "updated_at" else last.created_at)
                if sort_by not in ("uuid", "filename")
                else None
            )

        return modified_count, fail_count

    async def count_bboxes_matching(
        self,
        search_params: ImageSearchParams,
        bbox_tag_filter: List[str],
        bbox_label_filter: List[str],
        show_all_bboxes: bool = False,
        bbox_verification: Optional[str] = None,
        bbox_confidence: Optional[BBoxConfidenceRange] = None,
    ) -> int:
        """How many bboxes bulk_apply_bboxes would touch for this exact filter.

        Same argument shape as bulk_apply_bboxes so the number the UI shows and
        the number it then modifies cannot drift.
        """
        verification_tag_uuid = (
            BBOX_VERIFICATION_STATE_TO_TAG_UUID.get(bbox_verification) if bbox_verification else None
        )
        return await self._db_client.count_bboxes_search(
            search_params,
            bbox_tag_filter=bbox_tag_filter,
            bbox_label_filter=bbox_label_filter,
            show_all_bboxes=show_all_bboxes,
            bbox_verification_tag_uuid=verification_tag_uuid,
            bbox_confidence=bbox_confidence,
        )

    async def _apply_bbox_bulk_action(
        self,
        bbox: BBox,
        action: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        geometry: Optional[Dict[str, float]] = None,
        mark_verified: bool = True,
    ) -> None:
        """Apply one bulk action (relabel/delete/addTags/removeTags/revertRelabel) to a single bbox.

        Shared by bulk_apply_bboxes (filter-derived candidates) and
        bulk_apply_bboxes_by_uuids (explicit-list candidates) so the two entry
        points can't drift. Raises on failure — caller is responsible for
        catching and counting it.

        `mark_verified` governs the relabel branch only. A relabel is normally a
        human decision, so it promotes the box to VERIFIED; callers applying a
        *model's* suggestion ("Run Model on Results") pass False so the box keeps
        whatever verification state it already had and stays reviewable.
        """
        _VERIFICATION_UUIDS: Set[str] = {UNVERIFIED_TAG_UUID}
        current_tags = [t.uuid for t in bbox.tags or []]
        if action == "delete":
            if bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)
            await self._db_client.delete_single_bbox(bbox.uuid)
        elif action == "addTags":
            # Pass [existing, new] — db_client resolves exclusive-group conflicts
            # so new tags win any group conflict.
            await self._db_client.update_single_bbox(bbox.uuid, None, [*current_tags, *tag_uuids])
        elif action == "removeTags":
            remove_set = set(tag_uuids)
            new_tags = [u for u in current_tags if u not in remove_set]
            await self._db_client.update_single_bbox(bbox.uuid, None, new_tags)
        elif action == "revertRelabel":
            # Undo a filter-chain `relabel_bbox` step: restore the label stashed in
            # metadata_json by ApplyFilterChainTask and drop the review tag(s)
            # (tag_uuids — typically the caller's current tag filter).
            metadata = dict(bbox.metadata_json or {})
            prev_label = metadata.pop("relabel_prev_label_uuid", None)
            if prev_label is None:
                raise ValueError(f"bbox {bbox.uuid} has no relabel_prev_label_uuid to revert")
            remove_set = set(tag_uuids)
            new_tags = [u for u in current_tags if u not in remove_set]
            if bbox.label_uuid and prev_label != bbox.label_uuid and bbox.phash:
                self._hash_index.move_bbox(bbox.uuid, bbox.label_uuid, prev_label, bbox.phash)
            await self._db_client.update_single_bbox(bbox.uuid, prev_label, new_tags, metadata_json=metadata)
        else:  # relabel
            assigning = bool(label_uuid)
            new_label = label_uuid if assigning else bbox.label_uuid
            metadata_json = None
            if assigning:
                new_tags = (
                    [u for u in current_tags if u not in _VERIFICATION_UUIDS] + [VERIFIED_TAG_UUID]
                    if mark_verified
                    else current_tags
                )
                if bbox.label_uuid and label_uuid != bbox.label_uuid and bbox.phash:
                    self._hash_index.move_bbox(bbox.uuid, bbox.label_uuid, label_uuid, bbox.phash)
                # A manual relabel supersedes any prior filter-chain relabel: drop the
                # stashed original label so a later "revert filter-chain relabel" can't
                # silently restore a now-stale label over the user's manual choice.
                if bbox.metadata_json and "relabel_prev_label_uuid" in bbox.metadata_json:
                    metadata_json = {k: v for k, v in bbox.metadata_json.items() if k != "relabel_prev_label_uuid"}
            else:
                new_tags = current_tags
            await self._db_client.update_single_bbox(
                bbox.uuid, new_label, new_tags, metadata_json=metadata_json, geometry=geometry if assigning else None
            )

    async def bulk_apply_bboxes_by_uuids(
        self,
        bbox_uuids: List[str],
        action: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        bbox_geometry: Optional[Dict[str, Dict[str, float]]] = None,
        mark_verified: bool = True,
    ) -> Tuple[int, int]:
        """Apply action to an explicit list of bboxes in one server-side pass.

        Used by review modals whose candidate set has no server-side filter
        equivalent (e.g. similarity/diversity search results ranked by phash
        distance) — the caller already knows the exact UUID list and only needs
        the action applied without round-tripping one HTTP call per bbox.

        `bbox_geometry`, if given, maps bbox_uuid -> {"x","y","width","height"} —
        used by "Run Model on Results" relabels to replace a bbox's coordinates
        with the refinement model's suggested box instead of keeping the original
        (the model that ran is assumed to be more precise than whatever produced
        the original box).

        `mark_verified=False` relabels without promoting the box to VERIFIED —
        used when the relabel comes from a model suggestion rather than a human.

        Returns (modified_count, fail_count).
        """
        modified_count = 0
        fail_count = 0
        synced_images: Dict[str, ImageRecord] = {}

        for bbox_uuid in bbox_uuids:
            result = await self._db_client.get_bbox_with_image(bbox_uuid)
            if result is None or result[1] is None:
                fail_count += 1
                continue
            bbox, img = result
            try:
                geometry = bbox_geometry.get(bbox_uuid) if bbox_geometry else None
                await self._apply_bbox_bulk_action(
                    bbox, action, label_uuid, tag_uuids, geometry=geometry, mark_verified=mark_verified
                )
                modified_count += 1
                synced_images[img.uuid] = img
            except Exception as e:
                logger.error("bulk_apply_bboxes_by_uuids: bbox %s failed: %s", bbox_uuid, e)
                fail_count += 1

        for img in synced_images.values():
            try:
                await self._sync_image_state(img)
            except Exception as e:
                logger.error(
                    "bulk_apply_bboxes_by_uuids: _sync_image_state failed for %s: %s",
                    img.uuid,
                    e,
                )

        return modified_count, fail_count

    async def _apply_image_bulk_action(
        self,
        image_uuid: str,
        action: str,
        label_uuids: List[str],
        tag_uuids: List[str],
    ) -> str:
        """Apply one image-level bulk action to a single image uuid.

        Shared by bulk_apply_images_by_uuids (explicit-list candidates) and
        bulk_apply_images (filter-derived candidates) so the two entry points
        can't drift.

        Label actions accept several labels and apply to each in turn. The
        return value stays per-image so the counts callers report keep meaning
        "images", not "(image, label) pairs": "skipped" is returned only when
        *every* requested label was skipped, "modified" when at least one landed.

        Returns "modified" or "skipped" ("skipped" == setAbsent refused because
        the image already has a verified bbox for that label — a legitimate
        skip, not a failure). Raises on real failure — caller is responsible
        for catching and counting it as a fail.
        """
        if action == "delete":
            rel_path = await self.delete_image_permanently(image_uuid)
            if rel_path is None:
                raise ValueError(f"image {image_uuid} not found or already deleted")
        elif action == "removeFromDb":
            # Forget the image without touching the file: DB record + bboxes go
            # away, the sidecar goes with them (otherwise reconcile resurrects
            # the record), and the image file itself stays on disk.
            rel_path = await self.delete_image_db_only(image_uuid)
            if rel_path is None:
                raise ValueError(f"image {image_uuid} not found or already removed")
        elif action == "setUncertain":
            await self._db_client.delete_image_labels(image_uuid, list(label_uuids))
        elif action == "addTags":
            await self._db_client.add_image_tags(image_uuid, tag_uuids)
        elif action == "removeTags":
            await self._db_client.remove_image_tags(image_uuid, tag_uuids)
        elif action == "setTagAbsent":
            for tag_uuid in tag_uuids:
                await self._db_client.set_image_tag_status(image_uuid, tag_uuid, ImageItemStatus.ABSENT)
        elif action == "setTagUncertain":
            for tag_uuid in tag_uuids:
                await self._db_client.set_image_tag_status(image_uuid, tag_uuid, ImageItemStatus.UNCERTAIN)
        elif action == "setAbsent":
            # Mirror the single-image path's rule (set_image_label_status): ABSENT is
            # refused when a verified bbox already contradicts it. A bulk run treats
            # that as a skip, not a failure. Batched into one query + one upsert per
            # image regardless of label count, rather than a round trip per label.
            verified = await self._db_client.labels_with_verified_bbox(image_uuid, label_uuids)
            applicable = [label_uuid for label_uuid in label_uuids if label_uuid not in verified]
            if applicable:
                await self._db_client.set_image_labels_status(image_uuid, applicable, ImageItemStatus.ABSENT)
            if label_uuids and not applicable:
                return "skipped"
        elif action == "setPresent":
            await self._db_client.set_image_labels_status(image_uuid, label_uuids, ImageItemStatus.PRESENT)
        else:
            raise ValueError(f"Unknown bulk image action: {action}")
        return "modified"

    async def bulk_apply_images_by_uuids(
        self,
        image_uuids: List[str],
        action: str,
        label_uuids: Optional[List[str]] = None,
        tag_uuids: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        """Apply an image-level bulk action (setPresent/setAbsent/setUncertain/delete/
        addTags/removeTags/setTagAbsent/setTagUncertain) to an explicit list of images
        in one server-side pass — the image-kind counterpart of bulk_apply_bboxes_by_uuids.

        Returns (modified_count, fail_count, skipped_count).
        """
        tag_uuids = tag_uuids or []
        label_uuids = label_uuids or []
        if action in ("addTags", "removeTags", "setTagAbsent", "setTagUncertain") and not tag_uuids:
            # Nothing to do — don't report images as "modified" for a no-op.
            return 0, 0, 0
        if action in ("setPresent", "setAbsent", "setUncertain") and not label_uuids:
            # Same no-op guard as the tag actions above: a label action with no
            # labels would otherwise count every image as "modified".
            return 0, 0, 0
        if action in REMOVAL_IMAGE_BULK_ACTIONS:
            # One batched pass instead of a round trip per image. Chunked because
            # this list is caller-supplied and unbounded, and both the fetch and
            # the delete bind one SQL variable per uuid.
            delete_files = action == "delete"
            unique_uuids = list(dict.fromkeys(image_uuids))
            removed_total = 0
            fail_total = 0
            for i in range(0, len(unique_uuids), self.BULK_APPLY_PAGE_SIZE):
                chunk = unique_uuids[i : i + self.BULK_APPLY_PAGE_SIZE]
                records = await self._db_client.get_images_by_uuids(chunk)
                # A uuid the DB does not know about is a failure here exactly as
                # the per-image path's ValueError was.
                fail_total += len(chunk) - len(records)
                try:
                    removed, failed = await self.remove_image_records_bulk(records, delete_files=delete_files)
                except Exception as e:
                    logger.error("bulk_apply_images_by_uuids: chunk of %d removals failed: %s", len(records), e)
                    fail_total += len(records)
                    continue
                removed_total += removed
                fail_total += failed
            return removed_total, fail_total, 0

        modified_count = 0
        fail_count = 0
        skipped_count = 0
        touched: List[str] = []
        for image_uuid in image_uuids:
            try:
                result = await self._apply_image_bulk_action(image_uuid, action, label_uuids, tag_uuids)
                if result == "skipped":
                    skipped_count += 1
                else:
                    modified_count += 1
                    touched.append(image_uuid)
            except Exception as e:
                logger.error("bulk_apply_images_by_uuids: image %s failed: %s", image_uuid, e)
                fail_count += 1
        if action in IMAGE_LABEL_BULK_ACTIONS:
            await self._write_image_label_sidecars(touched)
        return modified_count, fail_count, skipped_count

    async def bulk_apply_images(
        self,
        search_params: ImageSearchParams,
        action: str,
        label_uuids: Optional[List[str]] = None,
        tag_uuids: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        """Apply an image-level action to every image matching the filter, across all pages.

        Pages through list_images_search in batches of self.BULK_APPLY_PAGE_SIZE and applies
        the requested action to each image — the image-kind counterpart of bulk_apply_bboxes.

        Returns (modified_count, fail_count, skipped_count).
        """
        tag_uuids = tag_uuids or []
        label_uuids = label_uuids or []
        if action in ("addTags", "removeTags", "setTagAbsent", "setTagUncertain") and not tag_uuids:
            return 0, 0, 0
        if action in ("setPresent", "setAbsent", "setUncertain") and not label_uuids:
            return 0, 0, 0

        # Paginate by uuid regardless of the caller's requested sort — mirrors
        # bulk_apply_bboxes: the action can mutate rows the cursor depends on
        # (e.g. updated_at), so keyset-paginating by anything but uuid risks
        # re-matching an already-processed row on the next page. Traversal
        # order is irrelevant to the action (each matching image is touched
        # exactly once).
        search_params = search_params.model_copy(update={"sort_by": "uuid", "sort_order": "asc"})

        modified_count = 0
        fail_count = 0
        skipped_count = 0
        _PAGE_SIZE = self.BULK_APPLY_PAGE_SIZE
        cursor_uuid: Optional[str] = None
        cursor_sort_value = None

        while True:
            page_params = search_params.model_copy(
                update={
                    "limit": _PAGE_SIZE + 1,
                    "after_uuid": cursor_uuid,
                    "after_sort_value": cursor_sort_value,
                }
            )
            batch = await self._db_client.list_images_search(page_params)
            has_more = len(batch) > _PAGE_SIZE
            batch = batch[:_PAGE_SIZE]

            if not batch:
                break

            if action in REMOVAL_IMAGE_BULK_ACTIONS:
                # The page query already eager-loaded these records with their
                # bboxes and tags, so the batch path needs no further reads at
                # all — the per-image path re-fetched every one of them.
                try:
                    removed, failed = await self.remove_image_records_bulk(batch, delete_files=(action == "delete"))
                    modified_count += removed
                    fail_count += failed
                except Exception as e:
                    logger.error("bulk_apply_images: page of %d removals failed: %s", len(batch), e)
                    fail_count += len(batch)
                if not has_more:
                    break
                cursor_uuid = batch[-1].uuid
                cursor_sort_value = None
                continue

            touched: List[str] = []
            for img in batch:
                try:
                    result = await self._apply_image_bulk_action(img.uuid, action, label_uuids, tag_uuids)
                    if result == "skipped":
                        skipped_count += 1
                    else:
                        modified_count += 1
                        touched.append(img.uuid)
                except Exception as e:
                    logger.error("bulk_apply_images: image %s failed: %s", img.uuid, e)
                    fail_count += 1

            # Per page, not per run: this walks every matching image, so holding
            # the full touched set to the end would grow without bound.
            if action in IMAGE_LABEL_BULK_ACTIONS:
                await self._write_image_label_sidecars(touched)

            if not has_more:
                break

            last = batch[-1]
            sort_by = search_params.sort_by
            cursor_uuid = last.uuid
            # sort_by is forced to "uuid" above, so this is always None — kept in
            # this shape (rather than hardcoding None) to mirror bulk_apply_bboxes
            # exactly, in case that forcing is ever relaxed for either function.
            cursor_sort_value = (
                (last.updated_at if sort_by == "updated_at" else last.created_at)
                if sort_by not in ("uuid", "filename")
                else None
            )

        return modified_count, fail_count, skipped_count

    async def _sync_image_state(self, image_record: ImageRecord) -> None:
        """Rebuild JSON sidecar and image_labels after a single-bbox mutation."""
        safe_path = get_safe_path(self._files_root, image_record.filename)
        # get_bboxes_for_image eager-loads both tags and label so names are available.
        bboxes = await self._db_client.get_bboxes_for_image(image_record.uuid)
        update = ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=b.uuid,
                    label_uuid=b.label_uuid,
                    label_name=b.label.name if b.label else None,
                    x=b.x,
                    y=b.y,
                    width=b.width,
                    height=b.height,
                    tag_uuids=[t.uuid for t in (b.tags or [])],
                    tag_names=[t.name for t in (b.tags or [])],
                )
                for b in bboxes
            ]
        )
        if safe_path is not None:
            await asyncio.to_thread(
                self._db_client.write_image_metadata_to_json,
                self._get_json_sidefile(safe_path),
                image_record.uuid,
                image_record.filename,
                update,
            )
        else:
            logger.warning(f"Skipping sidecar write for {image_record.filename!r}: path not under FILES_ROOT")
        await self._recompute_image_labels(image_record.uuid, update)

    async def _recompute_image_labels(self, image_uuid: str, update: ImageRecordUpdate) -> None:
        """Sync PRESENT rows in image_labels from reviewed bboxes.

        Only PRESENT rows are written/deleted here.  ABSENT rows are always
        set explicitly by the user via set_image_label_status and are never touched.

        Bboxes tagged UNVERIFIED are excluded from PRESENT tracking, as they have
        not been confirmed as true annotations.

        The stale-PRESENT sweep below only applies to object-scope labels, whose
        PRESENT state is owned by bbox sync (derived from verified bboxes). Image-scope
        labels can never have a bbox — the UI short-circuits the draw tool for them — so
        their PRESENT state is user-asserted, exactly like ABSENT, and must never be
        treated as stale here. A candidate whose label no longer exists (deleted) is not
        exempt: it's an orphan and is still swept.
        """
        present_label_uuids = {
            b.label_uuid for b in update.bboxes if b.label_uuid and _BBOX_VERIFIED_TAG in b.tag_uuids
        }

        existing = await self._db_client.get_image_labels(image_uuid)
        stale_candidates = [
            r.label_uuid
            for r in existing
            if r.status == ImageItemStatus.PRESENT and r.label_uuid not in present_label_uuids
        ]
        stale_present = stale_candidates
        if stale_candidates:
            labels = await self._db_client.get_labels_by_uuids(stale_candidates)
            stale_present = [
                u for u in stale_candidates if labels.get(u) is None or labels[u].scope != LabelScope.IMAGE
            ]
        # One write for both halves: the upserts used to be a transaction each.
        await self._db_client.sync_image_present_labels(image_uuid, sorted(present_label_uuids), stale_present)
        await self._sync_image_labels_sidecar(image_uuid)

    async def _sync_image_labels_sidecar(self, image_uuid: str) -> None:
        """Mirror this image's whole-image label assessments into its JSON sidecar.

        ABSENT rows ("I checked; this label is not in this image") and
        image-scope PRESENT rows are asserted by the user and derivable from
        nothing else — not from `boxes`, not from the label vocabulary. Without
        this they would live only in the database, which is exactly the failure
        the sidecars exist to survive.

        A batch of one: the single-image and bulk paths must agree on when a
        sidecar is created, and having two implementations of that rule is how
        they came to disagree in the first place.
        """
        await self._write_image_label_sidecars([image_uuid])

    @staticmethod
    def _sidecar_bbox_update(bboxes: List[BBox]) -> ImageRecordUpdate:
        """Build the `boxes` half of a sidecar from DB rows.

        Only used when a sidecar has to be created from scratch; the normal
        bbox path already has the payload it just wrote.
        """
        return ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=b.uuid,
                    label_uuid=b.label_uuid,
                    label_name=b.label.name if b.label else None,
                    x=b.x,
                    y=b.y,
                    width=b.width,
                    height=b.height,
                    tag_uuids=[t.uuid for t in (b.tags or [])],
                    tag_names=[t.name for t in (b.tags or [])],
                    phash=b.phash,
                )
                for b in bboxes
            ]
        )

    async def _write_image_label_sidecars(self, image_uuids: List[str]) -> Dict[str, int]:
        """Write the image_labels slice of the sidecar for a batch of images.

        A few queries for the whole batch rather than a few per image. The bulk
        actions go through here precisely so their DB cost stays independent of
        how many images were selected — a per-image sync turns a 5000-image bulk
        apply into 10000 extra queries.

        A missing sidecar is created when — and only when — there is something
        to record in it. Sidecars are otherwise only ever produced by a bbox
        write, so an image that has never had a box has no file, and "I checked,
        this label is not here" is exactly the assessment such an image carries;
        skipping those left them in SQLite only, which is the loss the sidecars
        exist to prevent. But an image with no assessments and no boxes has
        nothing to say beyond "this file exists", which the file itself already
        says — writing one there would litter the dataset with empty JSON.

        What does get created is a *complete* sidecar (uuid, filename, boxes,
        phash), because the reconcile path skips a sidecar with no uuid and an
        overwrite sync reads a missing `boxes` key as "this image has none".

        Returns counts keyed 'written', 'created', 'skipped_no_data',
        'skipped_no_path'.
        """
        stats = {"written": 0, "created": 0, "skipped_no_data": 0, "skipped_no_path": 0}
        if not image_uuids:
            return stats

        batch = await self._db_client.get_image_label_sidecar_batch(image_uuids)

        targets: List[Tuple[str, str, Optional[str], Path, List[Dict[str, Any]]]] = []
        for image_uuid in image_uuids:
            filename, phash, rows = batch.get(image_uuid, (None, None, []))
            safe_path = get_safe_path(self._files_root, filename) if filename else None
            if safe_path is None:
                stats["skipped_no_path"] += 1
                continue
            targets.append((image_uuid, filename, phash, self._get_json_sidefile(safe_path), rows))
        if not targets:
            return stats

        # One thread hop for every stat(): exists() is a blocking syscall, and a
        # bulk apply over a few thousand images would otherwise run one of them
        # per image directly on the event loop.
        existing = await asyncio.to_thread(lambda: {t[3] for t in targets if t[3].exists()})
        missing = [t[0] for t in targets if t[3] not in existing]
        # Boxes are read from the DB rather than written as [] so that a sidecar
        # recreated after deletion doesn't claim the image has no boxes — an
        # overwrite sync would then delete the real ones.
        bboxes_by_image = await self._db_client.get_bboxes_for_images(missing) if missing else {}
        # Convert while the ORM rows are still session-fresh, not inside the thread.
        updates = {uuid: self._sidecar_bbox_update(bboxes_by_image.get(uuid, [])) for uuid in missing}

        def _write_all() -> Tuple[int, int, int]:
            written = created = skipped = 0
            for image_uuid, filename, phash, sidecar_path, rows in targets:
                if sidecar_path in existing:
                    # An existing sidecar is refreshed even when `rows` is empty:
                    # there, [] is the user having cleared every assessment, and
                    # an overwrite sync has to see that to honour it.
                    self._db_client.write_image_labels_to_json(sidecar_path, rows)
                    written += 1
                    continue
                if not rows and not updates[image_uuid].bboxes:
                    # Nothing to preserve — don't leave an empty file behind.
                    skipped += 1
                    continue
                self._db_client.write_image_metadata_to_json(
                    sidecar_path,
                    image_uuid,
                    filename,
                    updates[image_uuid],
                    image_phash=phash,
                    image_labels=rows,
                )
                created += 1
            return written, created, skipped

        # The writes themselves are one hop too, rather than one per image.
        stats["written"], stats["created"], stats["skipped_no_data"] = await asyncio.to_thread(_write_all)
        return stats

    async def backfill_image_label_sidecars(self, batch_size: int = 500) -> Dict[str, int]:
        """Write existing image_labels rows into sidecars that predate the feature.

        One-shot repair for databases annotated before sidecars carried label
        assessments. Idempotent — rerunning it rewrites the same content — so it
        is safe to run again if it is interrupted.

        Batched deliberately: the live DB has tens of thousands of assessment
        rows, and loading every row plus every filename in one query to build a
        dict would hold the lot in memory at once for no benefit.

        Returns counts keyed 'written', 'created', 'skipped_no_data',
        'skipped_no_path'.
        """
        image_uuids = await self._db_client.get_image_uuids_with_labels()
        stats = {"written": 0, "created": 0, "skipped_no_data": 0, "skipped_no_path": 0}

        for start in range(0, len(image_uuids), batch_size):
            batch_stats = await self._write_image_label_sidecars(image_uuids[start : start + batch_size])
            for key, value in batch_stats.items():
                stats[key] += value

        logger.info("backfill_image_label_sidecars: %s", stats)
        return stats

    async def _resolve_image_uuid(self, image_path: str) -> Optional[str]:
        """Resolve an image path to its DB uuid, or None if not found."""
        safe_path = get_safe_path(self._files_root, image_path)
        if safe_path is None:
            return None
        return await self._db_client.get_image_uuid_by_filename(safe_path)

    async def set_image_label_status(self, image_path: str, label_uuid: str, status: ImageItemStatus) -> str:
        """Explicitly set a label state for an image.

        PRESENT and ABSENT create/update the row.  UNCERTAIN clears an existing
        row (deletes it so the label reverts to the implicit default).

        ABSENT is blocked when the image already has a verified bbox with the
        same label — that would be contradictory evidence, not a failure. This
        mirrors the bulk paths' "setAbsent" handling in _apply_image_bulk_action:
        same predicate (image_has_verified_bbox_for_label), same "skipped" outcome
        vocabulary, so a single-image save and a bulk action never disagree about
        what counts as a legitimate skip vs. a real error.

        Returns "modified", "skipped" (verified-bbox conflict), or "not_found".
        """
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return "not_found"
        if status == ImageItemStatus.ABSENT:
            if await self._db_client.image_has_verified_bbox_for_label(image_uuid, label_uuid):
                return "skipped"
        if status == ImageItemStatus.UNCERTAIN:
            await self._db_client.delete_image_labels(image_uuid, [label_uuid])
        else:
            await self._db_client.set_image_label_status(image_uuid, label_uuid, status)
        await self._sync_image_labels_sidecar(image_uuid)
        return "modified"

    async def get_image_labels(self, image_path: str) -> Optional[List[ImageLabelRead]]:
        """Return all label statuses for an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        rows = await self._db_client.get_image_labels(image_uuid)
        return [ImageLabelRead.model_validate(r) for r in rows]

    async def get_image_labels_bulk(self, image_uuids: List[str]) -> Dict[str, List[ImageLabelRead]]:
        """Return {image_uuid: [label rows]} for a batch of image UUIDs.

        Images with no rows are omitted; unknown UUIDs are silently ignored.
        Lets the review grid resolve whole-image label chips for a page of
        cards in one request instead of one GET per card.
        """
        rows = await self._db_client.get_image_labels_bulk(image_uuids)
        out: Dict[str, List[ImageLabelRead]] = {}
        for r in rows:
            out.setdefault(r.image_uuid, []).append(ImageLabelRead.model_validate(r))
        return out

    async def list_images_for_label(
        self,
        label_uuid: str,
        statuses: List[ImageItemStatus],
    ) -> List["ImageLabelWithFilename"]:
        """Return image-label rows (with filename) for the given label + status filter."""
        return await self._db_client.list_images_for_label(label_uuid, statuses)

    async def set_image_tag(self, image_path: str, tag_uuid: str, action: str) -> bool:
        """Add or remove a whole-image tag. action is 'add' or 'remove'."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return False
        if action == "remove":
            await self._db_client.remove_image_tag(image_uuid, tag_uuid)
        else:
            await self._db_client.add_image_tag(image_uuid, tag_uuid)
        return True

    async def set_image_tag_status(self, image_path: str, tag_uuid: str, status: ImageItemStatus) -> bool:
        """Explicitly set PRESENT, ABSENT, or UNCERTAIN for a whole-image tag.

        Unlike labels, all three states are user-settable here — tags have no
        derived-PRESENT concept to protect.
        """
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return False
        await self._db_client.set_image_tag_status(image_uuid, tag_uuid, status)
        return True

    async def get_image_tag_statuses(self, image_path: str) -> Optional[List[ImageTagStatusRead]]:
        """Return all tag statuses (PRESENT and ABSENT) for an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        rows = await self._db_client.get_image_tag_statuses(image_uuid)
        return [ImageTagStatusRead.model_validate(r) for r in rows]

    async def get_image_tags(self, image_path: str) -> Optional[List[TagRead]]:
        """Return all tags linked to an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None:
            return None
        return [TagRead.model_validate(t) for t in image.tags]

    def _image_rel_path(self, image_filename: str) -> str:
        """Convert an absolute stored filename back to a FILES_ROOT-relative path."""
        try:
            return str(Path(image_filename).relative_to(self._files_root))
        except ValueError:
            return image_filename

    async def _count_images_bounded(self, params: ImageSearchParams) -> int:
        """Run count_images_search with a timeout; return -1 if it exceeds the limit.

        Uses asyncio.shield so the inner DB coroutine is NOT cancelled on timeout —
        it continues running in the background and properly closes its session when
        done.  This avoids leaving dangling SQLite connections while still letting
        the caller return quickly with -1 instead of blocking for minutes on large
        datasets.
        """
        t0 = time.monotonic()
        inner: asyncio.Task[int] = asyncio.ensure_future(self._db_client.count_images_search(params))
        try:
            result = await asyncio.wait_for(asyncio.shield(inner), timeout=self._COUNT_QUERY_TIMEOUT)
            logger.info(
                "[review] count query completed in %.3fs → %d",
                time.monotonic() - t0,
                result,
            )
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[review] count query timed out after %.3fs (timeout=%.1fs); "
                "returning -1, background query continues",
                elapsed,
                self._COUNT_QUERY_TIMEOUT,
            )
            # Consume any future exception so Python doesn't warn about
            # "Task exception was never retrieved" when the task eventually ends.
            inner.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return -1

    async def get_review_queue(self, params: ImageSearchParams) -> tuple[List[ImageRecordRead], bool, int]:
        """Return (images, has_more, total_count) for the review queue."""
        t0 = time.monotonic()
        logger.info(
            "[review] request: terms=%s require_bboxes=%s after_uuid=%s",
            params.terms,
            params.require_bboxes,
            params.after_uuid,
        )
        fetch_params = params.model_copy(update={"limit": params.limit + 1})

        if params.after_uuid is None:
            cache_key = hashlib.md5(
                params.model_dump_json(
                    exclude={
                        "sort_by",
                        "sort_order",
                        "limit",
                        "after_uuid",
                        "after_sort_value",
                    }
                ).encode(),
                usedforsecurity=False,
            ).hexdigest()
            now = time.monotonic()
            cached = self._count_cache.get(cache_key)
            # The TTL alone let a cached count outlive the rows it counted: a
            # bulk delete emptied the grid while the toolbar kept saying
            # "Page 1 of 2" and "Apply to Filtered Images (500)" for up to a
            # minute. The write epoch makes that exact rather than eventual —
            # any write to a table the count is derived from retires it now.
            epoch = self._db_client.write_epoch
            if cached is not None and cached[2] == epoch and (now - cached[1]) < self._COUNT_CACHE_TTL:
                logger.info(
                    "[review] count cache hit (key=%s, age=%.1fs)",
                    cache_key[:8],
                    now - cached[1],
                )
                t1 = time.monotonic()
                batch = await self._db_client.list_images_search(fetch_params)
                logger.info(
                    "[review] list query completed in %.3fs → %d rows",
                    time.monotonic() - t1,
                    len(batch),
                )
                total_count = cached[0]
            else:
                logger.info(
                    "[review] count cache miss (key=%s) — running list+count in parallel",
                    cache_key[:8],
                )
                t1 = time.monotonic()
                batch, total_count = await asyncio.gather(
                    self._db_client.list_images_search(fetch_params),
                    self._count_images_bounded(params),
                )
                logger.info(
                    "[review] gather done in %.3fs: list=%d rows, total_count=%d",
                    time.monotonic() - t1,
                    len(batch),
                    total_count,
                )
                if total_count >= 0:
                    # Re-read the epoch: a write that landed while the count was
                    # in flight must not be cached as though it were included.
                    self._count_cache[cache_key] = (total_count, now, self._db_client.write_epoch)
                    self._evict_expired_counts(now)
        else:
            t1 = time.monotonic()
            batch = await self._db_client.list_images_search(fetch_params)
            logger.info(
                "[review] list query (paginated) completed in %.3fs → %d rows",
                time.monotonic() - t1,
                len(batch),
            )
            total_count = -1

        has_more = len(batch) > params.limit
        images_page = batch[: params.limit]

        results = []
        for img in images_page:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)

        logger.info(
            "[review] total handler time %.3fs → %d images, has_more=%s",
            time.monotonic() - t0,
            len(results),
            has_more,
        )
        return results, has_more, total_count

    async def list_files(
        self,
        rel_path: str = "/",
        patterns: List[str] = ["*"],
        exclude_patterns: List[str] = ["*.json"],
        recursive: bool = False,
        include_db_status: bool = False,
        offset: int = 0,
        limit: int = 200,
    ) -> tuple[List[FileEntryRead], bool]:
        """
        List files and directories in the given relative path under the root directory.

        Args:
            rel_path (str): Relative path from the root directory (default: "/").
            patterns (List[str]): Glob pattern(s) to include (default: "*").
            exclude_patterns (List[str]): Glob pattern(s) to exclude (default: "").
            recursive (bool): Whether to list files recursively (default: False).
            include_db_status (bool): When True, populate uuid on entries that are
                registered in the image database (default: False).
            offset (int): Number of entries to skip (default: 0).
            limit (int): Maximum number of entries to return (default: 200).

        Returns:
            Tuple of (entries page, has_more flag).
        """
        if not self._files_root or not self._files_root.is_dir():
            return [], False

        safe_path = get_safe_path(self._files_root, rel_path)
        if not safe_path or not safe_path.is_dir():
            return [], False

        search_pattern = "**/*" if recursive else None

        def _scan() -> tuple[List[FileEntryRead], bool]:
            # Recursive glob walks the whole tree synchronously — run in a
            # worker thread so a large FILES_ROOT doesn't freeze the event
            # loop (and every other request) for the duration of the walk.
            # Collect and sort all matching paths by name (string sort — no stat calls yet).
            raw = sorted(
                safe_path.glob(search_pattern) if recursive else safe_path.iterdir(),
                key=lambda p: p.name,
            )

            filtered = []
            for p in raw:
                if p.name.startswith("."):
                    continue
                rel_str = str(p.relative_to(self._files_root))
                is_matched = any(fnmatch.fnmatch(rel_str, pat) for pat in patterns)
                is_excluded = any(fnmatch.fnmatch(rel_str, pat) for pat in exclude_patterns)
                if is_matched and not is_excluded:
                    filtered.append(p)

            more = len(filtered) > offset + limit
            page = filtered[offset : offset + limit]

            # Stat calls (is_dir) and FileEntryRead construction only for the page slice.
            return [
                FileEntryRead(
                    name=p.name,
                    is_dir=p.is_dir(),
                    path=str(p.relative_to(self._files_root)),
                )
                for p in page
            ], more

        entries, has_more = await asyncio.to_thread(_scan)

        if include_db_status and entries:
            abs_paths = [str(self._files_root / e.path) for e in entries if not e.is_dir]
            in_db_map = await self._db_client.get_filenames_in_db(abs_paths)
            for e in entries:
                if not e.is_dir:
                    info = in_db_map.get(str(self._files_root / e.path))
                    if info:
                        e.uuid = info["uuid"]
                        e.bbox_count = info["bbox_count"]

        return entries, has_more

    async def create_directory(self, rel_path: str) -> None:
        """
        Create a directory under FILES_ROOT.

        Args:
            rel_path: Path relative to FILES_ROOT (must resolve inside it).

        Raises:
            ValueError: If the path resolves outside FILES_ROOT or already exists as a file.
        """
        if not self._files_root or not self._files_root.is_dir():
            raise ValueError("FILES_ROOT is not configured")
        safe_path = get_safe_path(self._files_root, rel_path)
        if safe_path is None:
            raise ValueError(f"Path is outside FILES_ROOT: {rel_path!r}")
        if safe_path.exists() and not safe_path.is_dir():
            raise ValueError(f"Path already exists as a file: {rel_path!r}")
        safe_path.mkdir(parents=True, exist_ok=True)

    # Extensions accepted on upload. An allowlist, not a denylist: the file is
    # written under FILES_ROOT where the browser and every scan will pick it up,
    # so "anything not obviously bad" is the wrong default.
    UPLOAD_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    async def save_uploaded_image(
        self,
        rel_dir: str,
        filename: str,
        data: bytes,
        *,
        overwrite: bool = False,
        source_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write an uploaded image under FILES_ROOT and register it.

        Registering here rather than leaving it to lazy registration is what
        makes an upload immediately findable in Review: a record only appears on
        its own once something writes annotations to the image, so an untouched
        upload would otherwise be invisible to every search that matters.

        Near-duplicates are reported, never rejected. Re-importing an image you
        already have is a real thing to want to do (a better scan of the same
        frame, say), so the caller is told and decides.

        Args:
            rel_dir: Destination directory, relative to FILES_ROOT.
            filename: Client-supplied name; only its basename is honoured.
            data: Raw file bytes.
            overwrite: Replace an existing file instead of uniquifying the name.

        Returns:
            {path, uuid, duplicate_of, duplicate_distance, renamed_from}

        Raises:
            ValueError: unusable destination, rejected extension, or undecodable image.
        """
        if not self._files_root or not self._files_root.is_dir():
            raise ValueError("FILES_ROOT is not configured")

        # Path components in the *filename* are the classic traversal vector, and
        # get_safe_path below would reject them anyway — but silently dropping
        # them here means "../../etc/passwd.jpg" uploads as "passwd.jpg" rather
        # than failing confusingly.
        base_name = Path(filename).name
        if not base_name or base_name in {".", ".."}:
            raise ValueError(f"Unusable filename: {filename!r}")

        suffix = Path(base_name).suffix.lower()
        if suffix not in self.UPLOAD_EXTENSIONS:
            allowed = ", ".join(sorted(self.UPLOAD_EXTENSIONS))
            raise ValueError(f"{base_name!r} is not an image this app reads (accepts: {allowed})")

        dest_dir = get_safe_path(self._files_root, rel_dir or "")
        if dest_dir is None:
            raise ValueError(f"Path is outside FILES_ROOT: {rel_dir!r}")
        if dest_dir.exists() and not dest_dir.is_dir():
            raise ValueError(f"Upload destination is a file, not a folder: {rel_dir!r}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        target = dest_dir / base_name
        renamed_from = None
        if target.exists() and not overwrite:
            # Uniquify rather than clobber: two cameras happily produce the same
            # "snapshot.jpg", and silently replacing the first one loses both the
            # file and every annotation already made against that path.
            stem, ext = Path(base_name).stem, Path(base_name).suffix
            for n in range(1, 1000):
                candidate = dest_dir / f"{stem}_{n}{ext}"
                if not candidate.exists():
                    renamed_from, target = base_name, candidate
                    break
            else:
                raise ValueError(f"Too many files named like {base_name!r} in {rel_dir!r}")

        # Decode before writing: an unreadable file under FILES_ROOT is worse
        # than a rejected upload, because every later scan trips over it.
        frame = await asyncio.to_thread(_decode_image_bytes, data)
        if frame is None:
            raise ValueError(f"{base_name!r} could not be decoded as an image")

        # An overwrite replaces a file that is already registered and may carry
        # annotations, so it cannot simply be deleted if the write half-fails.
        # Move it aside first (a rename, not a copy) and put it back on failure:
        # otherwise a failed re-upload costs the user the new file *and* the old
        # one. The dot-prefixed, non-image name keeps scans off it in between.
        backup: Optional[Path] = None
        if target.exists():
            backup = target.with_name(f".{target.name}.upload-bak")
            await asyncio.to_thread(os.replace, target, backup)

        try:
            await asyncio.to_thread(target.write_bytes, data)
            image = await self._db_client.add_image(target, source_uuid=source_uuid)
        except Exception:
            # Do not leave a file the DB does not know about — it would show in
            # the browser and be missing from every search.
            await asyncio.to_thread(target.unlink, True)
            if backup is not None:
                await asyncio.to_thread(os.replace, backup, target)
            raise

        if backup is not None:
            await asyncio.to_thread(backup.unlink, True)

        # An overwrite reuses the existing row (add_image returns it), so the
        # pHash on that row still describes the file that was just replaced.
        duplicate_of, duplicate_distance = await self._find_upload_duplicate(
            image.uuid, target, replaced=backup is not None
        )
        if backup is not None:
            # Thumbnails are keyed on the source file's mtime, so the new bytes
            # miss the cache on their own. The bbox crops are not: they are keyed
            # on bbox uuid and validated against bbox.updated_at, which nothing
            # here moves.
            bboxes = await self._db_client.get_bboxes_for_image(image.uuid)
            if bboxes:
                await self.invalidate_bbox_thumbnails([b.uuid for b in bboxes])
                await self._rehash_bboxes_after_replace(image.uuid, target, bboxes)

        return {
            "path": self._image_rel_path(str(target)),
            "uuid": image.uuid,
            "duplicate_of": duplicate_of,
            "duplicate_distance": duplicate_distance,
            "renamed_from": renamed_from,
        }

    async def _rehash_bboxes_after_replace(self, image_uuid: str, path: Path, bboxes: List[Any]) -> None:
        """Re-hash every bbox crop after the picture underneath it was replaced.

        The image pHash is refreshed by _find_upload_duplicate, but a box's hash
        describes a *crop*, and the crop changed too. Left alone, `bboxes.phash`
        and the per-label index entries keep describing content that is no longer
        on disk, so "find similar bboxes" matches the replaced picture forever —
        the same failure HashIndex.replace_image fixes one level up.

        Best-effort, like the duplicate check it follows: the file is already
        written and registered, and a hash that could not be recomputed must not
        turn a successful upload into an error.
        """
        entries: List[Dict[str, str]] = []
        for bbox in bboxes:
            try:
                new_phash = await asyncio.to_thread(
                    compute_bbox_phash,
                    path,
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    self._hash_index.bbox_hash_size,
                )
            except Exception:
                logger.exception("Could not re-hash bbox %s after its image was replaced", bbox.uuid)
                continue
            entries.append({"uuid": bbox.uuid, "phash": new_phash})
            if bbox.label_uuid:
                if bbox.phash:
                    self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)
                self._hash_index.add_bbox(bbox.uuid, bbox.label_uuid, new_phash)
        if not entries:
            return
        # update_phashes writes the image hash too, so it needs the current one —
        # passing a stale or empty value here would undo the refresh above it.
        record = await self._db_client.get_image_by_uuid(image_uuid)
        image_phash = getattr(record, "phash", None) or await asyncio.to_thread(
            compute_image_phash, path, self._hash_index.image_hash_size
        )
        await self._db_client.update_phashes(image_uuid, image_phash, entries)

    async def _find_upload_duplicate(
        self, image_uuid: str, path: Path, replaced: bool = False
    ) -> Tuple[Optional[str], Optional[int]]:
        """Nearest already-known image within the configured dedup threshold.

        Also records the hash it computed, in both places that have to agree: the
        in-memory index (which answers this query) and images.phash (which the
        index is rebuilt from on startup). Writing only the first meant every
        uploaded-but-not-yet-annotated image silently dropped out of dedup on the
        next db_server restart — update_image_metadata recomputes an image pHash
        only when it is None, but that runs on the first annotation save, which
        an imported-and-ignored image never gets.

        *replaced* says this path already had a different picture at it, which
        makes it an index *update* rather than an insert — see
        HashIndex.replace_image, without which later uploads keep matching
        content that is no longer on disk.

        Best-effort: a hash index that is cold, rebuilding, or sized differently
        must not fail the upload — the file is already written and registered,
        and "we could not check" is not the same as "it is a duplicate".
        """
        try:
            threshold_raw = await self._db_client.get_setting("dedup_image_threshold")
            threshold = int(threshold_raw) if threshold_raw is not None else DEFAULT_IMAGE_THRESHOLD
            phash = await asyncio.to_thread(compute_image_phash, path, self._hash_index.image_hash_size)
            if replaced:
                self._hash_index.replace_image(image_uuid, phash)
            else:
                self._hash_index.add_image(image_uuid, phash)
            await self._db_client.update_phashes(image_uuid, phash, [])
            for other_uuid, distance in self._hash_index.find_similar_images(phash, threshold):
                if other_uuid != image_uuid:
                    return other_uuid, distance
        except Exception:
            logger.exception("Duplicate check failed for %s; treating as not-a-duplicate", path)
        return None, None

    ################################################################################
    # Image Reconciliation API
    ################################################################################

    async def list_all_image_stubs(self) -> List[Dict[str, str]]:
        """Return [{uuid, filename}] for every image record without pagination."""
        return await self._db_client.get_all_image_stubs()

    async def list_image_stubs_needing_phash(self, image_hex_len: int, bbox_hex_len: int) -> List[Dict[str, str]]:
        """Return [{uuid, filename}] for images whose phash is missing or was computed at a different size."""
        return await self._db_client.get_image_stubs_needing_phash(image_hex_len, bbox_hex_len)

    async def reconcile_image(self, rel_path: str, sync_mode: str) -> Dict[str, Any]:
        """Move-aware reconcile of a single image file against the DB.

        1. Reads the sidecar to get the image UUID.
        2. If that UUID exists in the DB under a different filename and the old
           file is gone, updates the record's filename (move detection).
        3. Applies sync_mode: replace_db_with_json | replace_json_with_db |
           merge | keep_newer.

        Returns a dict with "action" and "rel_path".
        """
        image_path = get_safe_path(self._files_root, rel_path)
        if image_path is None:
            return {
                "action": "skipped",
                "reason": "path outside files_root",
                "rel_path": rel_path,
            }

        # Sidecars replace the image extension: image.jpg → image.json (matches _get_json_sidefile)
        sidecar_path = self._get_json_sidefile(image_path)

        if not sidecar_path.exists():
            logger.debug("reconcile_image: no sidecar for %s, skipping", rel_path)
            return {"action": "skipped", "reason": "no sidecar", "rel_path": rel_path}

        try:
            sidecar_data = jstyleson.loads(sidecar_path.read_text())
        except Exception as e:
            logger.warning("reconcile_image: failed to parse sidecar %s: %s", sidecar_path, e)
            return {"action": "error", "reason": str(e), "rel_path": rel_path}

        sidecar_uuid = sidecar_data.get("uuid")
        if not sidecar_uuid:
            return {
                "action": "skipped",
                "reason": "sidecar has no uuid",
                "rel_path": rel_path,
            }

        # Move detection: UUID exists but under a different absolute filename.
        # The DB stores absolute paths (e.g. /files_root/subdir/img.png).
        existing = await self._db_client.get_image(sidecar_uuid)
        action = "synced"
        if existing is not None and existing.filename != str(image_path):
            old_abs = (
                Path(existing.filename)
                if Path(existing.filename).is_absolute()
                else get_safe_path(self._files_root, existing.filename)
            )
            if old_abs is None or not old_abs.exists():
                # The old file is gone — this is a move.
                logger.info(
                    "reconcile_image: move detected %s → %s",
                    existing.filename,
                    image_path,
                )
                await self._db_client.update_image_filename(sidecar_uuid, str(image_path))
                action = "moved"

        # Apply sync_mode.
        # Every branch that mutates the DB calls _sync_image_state so that
        # image_labels stays consistent with the new bbox set.
        if sync_mode == "replace_db_with_json":
            record = await self._db_client.sync_image_from_json(image_path, overwrite=True)
            await self._sync_image_state(record)
        elif sync_mode == "replace_json_with_db":
            # DB wins: write current DB bboxes back to the sidecar.
            updated_record = await self._db_client.get_image(sidecar_uuid)
            if updated_record:
                await self._sync_image_state(updated_record)
        elif sync_mode == "merge":
            record = await self._db_client.sync_image_from_json(image_path, overwrite=False)
            # Write the merged result back to the sidecar.
            await self._sync_image_state(record)
        elif sync_mode == "keep_newer":
            updated_record = await self._db_client.get_image(sidecar_uuid)
            if updated_record and updated_record.updated_at:
                db_ts = ensure_utc(updated_record.updated_at)
                sidecar_mtime = datetime.fromtimestamp(sidecar_path.stat().st_mtime, tz=timezone.utc)
                if sidecar_mtime > db_ts:
                    # Sidecar is newer — import from JSON and rebuild image_labels.
                    record = await self._db_client.sync_image_from_json(image_path, overwrite=True)
                    await self._sync_image_state(record)
                else:
                    # DB is newer — write DB state back to the sidecar.
                    await self._sync_image_state(updated_record)
            else:
                # No DB record yet — import from JSON.
                record = await self._db_client.sync_image_from_json(image_path, overwrite=True)
                await self._sync_image_state(record)
        else:
            return {
                "action": "error",
                "reason": f"unknown sync_mode: {sync_mode}",
                "rel_path": rel_path,
            }

        return {"action": action, "rel_path": rel_path}

    async def prune_missing_images(self, known_disk_uuids: List[str]) -> int:
        """Delete DB records for images whose file is missing and whose UUID is not on disk.

        The known_disk_uuids set is the set of UUIDs seen in sidecars during the
        reconcile pass.  Any DB record whose file is absent AND whose UUID is not in
        that set is truly gone (not just moved) and can be safely deleted.

        Returns the number of records deleted.
        """
        known_set: Set[str] = set(known_disk_uuids)
        stubs = await self._db_client.get_all_image_stubs()
        deleted = 0
        for stub in stubs:
            uuid = stub["uuid"]
            filename = stub["filename"]
            if uuid in known_set:
                continue
            abs_path = get_safe_path(self._files_root, filename)
            if abs_path is not None and abs_path.exists():
                continue
            logger.info("prune_missing_images: deleting %s (%s)", filename, uuid)
            image_record = await self._db_client.get_image_by_uuid(uuid)
            if image_record:
                self._remove_from_hash_index(image_record)
            await self._db_client.delete_image(uuid)
            deleted += 1
        return deleted

    async def delete_images_by_directory(self, directories: List[str]) -> int:
        """Delete all DB image records (and their bboxes) whose path falls under any given directory.

        Each entry in *directories* is a path relative to FILES_ROOT.  The DB stores
        filenames as absolute paths, so each input directory is resolved to an absolute
        path before prefix-matching.  Hash-index entries are removed before deletion
        so the BK-tree stays consistent.

        Returns the number of image records deleted.
        """
        if not directories:
            return 0
        # Resolve to absolute paths — the DB stores filenames as absolute paths
        # (e.g. /app/image_datasets/timelapse_out2/frame001.jpg).
        norm_dirs = [str(p).rstrip("/") for d in directories if (p := get_safe_path(self._files_root, d))]
        if not norm_dirs:
            return 0

        # Two targeted queries instead of a full-table scan + N per-record fetches.
        image_rows, bbox_rows = await self._db_client.get_image_hash_data_by_prefixes(norm_dirs)
        if not image_rows:
            return 0

        # Update the in-memory hash index before bulk-deleting.
        for img_uuid, img_phash in image_rows:
            if img_phash:
                self._hash_index.remove_image(img_uuid)
        for bbox_uuid, bbox_phash, bbox_label_uuid in bbox_rows:
            if bbox_phash and bbox_label_uuid:
                self._hash_index.remove_bbox(bbox_uuid, bbox_label_uuid)

        image_uuids = [r[0] for r in image_rows]
        deleted = await self._db_client.delete_images_bulk(image_uuids)
        logger.info("delete_images_by_directory: deleted %d images under %s", deleted, norm_dirs)
        return deleted

    async def list_images_by_directory(self, directories: List[str]) -> List[Dict[str, str]]:
        """Every image record whose file lives under one of *directories*.

        Scoped exactly like `delete_images_by_directory` — same resolution, same prefix
        rule — because the two are asked the same question from opposite ends: "what
        would this directory take with it?". A relocation that saw a different set than
        the deletion would move some records and orphan others.

        Paths come back FILES_ROOT-relative, which is the spelling every caller of this
        API already speaks (the directory picker, the task's params). The DB's absolute
        form is an internal detail and stays one.
        """
        if not directories:
            return []
        norm_dirs = [str(p).rstrip("/") for d in directories if (p := get_safe_path(self._files_root, d))]
        if not norm_dirs:
            return []
        rows = await self._db_client.get_images_by_prefixes(norm_dirs)
        return [{"uuid": uuid, "path": self._image_rel_path(filename)} for uuid, filename in rows]

    async def relocate_images(self, moves: List[Dict[str, str]]) -> Dict[str, Any]:
        """Re-point image records at new paths. Moving the *files* is the caller's job.

        Deliberately DB-only. The caller has already copied the bytes and knows whether
        that succeeded; doing the file I/O here as well would mean a second service
        walking FILES_ROOT, and db_server is the one process whose thread pool the whole
        UI queues behind. It updates records and reports what it could not update.

        Two things are rejected rather than attempted, both because `images.filename` is
        UNIQUE and a mid-batch IntegrityError would leave the caller unable to say which
        half landed:

        * a destination outside FILES_ROOT, and
        * a destination already held by a *different* image record.

        A destination already held by the *same* record is a no-op success, so re-running
        an interrupted relocation is safe.

        Returns ``{"updated": int, "errors": [{"uuid", "reason"}]}``.
        """
        if not moves:
            return {"updated": 0, "errors": []}

        resolved: List[Tuple[str, str]] = []
        errors: List[Dict[str, str]] = []
        for move in moves:
            uuid = str(move.get("uuid") or "")
            new_path = str(move.get("new_path") or move.get("newPath") or "")
            if not uuid or not new_path:
                errors.append({"uuid": uuid, "reason": "missing uuid or new_path"})
                continue
            target = get_safe_path(self._files_root, new_path)
            if target is None:
                errors.append({"uuid": uuid, "reason": f"path outside FILES_ROOT: {new_path!r}"})
                continue
            resolved.append((uuid, str(target)))

        # One lookup for the whole batch — a per-record existence check would be N
        # queries against the table the relocation is about to write to.
        owners = await self._db_client.get_image_uuids_by_filenames([t for _u, t in resolved])

        updated = 0
        for uuid, target in resolved:
            holder = owners.get(target)
            if holder is not None and holder != uuid:
                errors.append({"uuid": uuid, "reason": f"destination already used by image {holder}"})
                continue
            if await self._db_client.update_image_filename(uuid, target):
                updated += 1
            else:
                errors.append({"uuid": uuid, "reason": "image not found"})

        logger.info("relocate_images: updated %d record(s), %d error(s)", updated, len(errors))
        return {"updated": updated, "errors": errors}

    def _resolve_image_path(self, filename: str) -> Optional[Path]:
        """Resolve a filename (possibly absolute or relative) to a Path.

        Returns an absolute Path, or None if ``get_safe_path`` rejects it.
        """
        _p = Path(filename)
        return _p if _p.is_absolute() else get_safe_path(self._files_root, filename)

    def _remove_from_hash_index(self, image_record) -> None:
        """Remove an image and all its bboxes from the in-memory hash index."""
        if image_record.phash:
            self._hash_index.remove_image(image_record.uuid)
        for bbox in image_record.bboxes or []:
            if bbox.phash and bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)

    async def delete_image_permanently(self, image_uuid: str) -> Optional[str]:
        """Delete a single image from the DB and from disk (file + JSON sidecar).

        Removes the image and its bboxes from the hash index before DB deletion
        so the BK-tree stays consistent.

        Returns the FILES_ROOT-relative path of the deleted image, or None if
        the image UUID was not found in the DB.
        """
        image_record = await self._db_client.get_image_by_uuid(image_uuid)
        if not image_record:
            return None

        filename = image_record.filename
        image_path = self._resolve_image_path(filename)

        # Delete the JSON sidecar FIRST and let any failure propagate BEFORE the
        # DB is touched. If the DB record were deleted first and the sidecar
        # deletion then failed (swallowed), the orphaned sidecar would cause the
        # next reconcile to silently re-import (resurrect) the deleted image.
        if image_path:
            sidecar_path = self._get_json_sidefile(image_path)
            if sidecar_path.exists():
                sidecar_path.unlink()  # let exception propagate; caller handles it

        # Remove from hash index before DB deletion.
        self._remove_from_hash_index(image_record)

        # Delete DB record (bboxes cascade).
        await self._db_client.delete_image(image_uuid)

        # Delete the image file last. A leftover image file without its sidecar
        # is harmless (reconcile only re-imports images that have a sidecar), so
        # this is best-effort and does not gate the delete.
        if image_path and image_path.exists():
            try:
                image_path.unlink()
            except Exception as e:
                logger.warning(
                    "delete_image_permanently: failed to delete file %s: %s",
                    image_path,
                    e,
                )

        rel_path = self._image_rel_path(filename)
        logger.info("delete_image_permanently: deleted %s (%s)", filename, image_uuid)
        return rel_path

    async def delete_image_db_only(self, image_uuid: str) -> Optional[str]:
        """Remove a single image from the DB (and hash index) without deleting the image file.

        The JSON sidecar is deleted so that a subsequent reconcile does not
        re-import the old record back into the DB.  The image file itself is
        left on disk.

        Returns the FILES_ROOT-relative path of the removed image, or None if
        the image UUID was not found in the DB.
        """
        image_record = await self._db_client.get_image_by_uuid(image_uuid)
        if not image_record:
            return None

        filename = image_record.filename

        # Delete the JSON sidecar FIRST so that if it cannot be removed we
        # abort before touching the DB.  A swallowed failure here would leave
        # the sidecar on disk after the DB record is gone, causing the next
        # reconcile run to silently re-import (resurrect) the deleted record.
        image_path = self._resolve_image_path(filename)
        if image_path:
            sidecar_path = self._get_json_sidefile(image_path)
            if sidecar_path.exists():
                sidecar_path.unlink()  # let exception propagate; caller handles it

        # Remove from hash index before DB deletion.
        self._remove_from_hash_index(image_record)

        # Delete DB record (bboxes cascade); image file is untouched.
        await self._db_client.delete_image(image_uuid)

        rel_path = self._image_rel_path(filename)
        logger.info("delete_image_db_only: removed from DB %s (%s)", filename, image_uuid)
        return rel_path

    async def remove_image_records_bulk(self, image_records: Sequence[Any], *, delete_files: bool) -> Tuple[int, int]:
        """Batch counterpart of delete_image_permanently / delete_image_db_only.

        Takes records the caller has *already loaded* and removes them in three
        passes instead of per-image round trips: one threaded sweep for the
        sidecars, one DELETE for the whole batch, one threaded sweep for the
        image files. At 200 images a page that is ~5 queries + 1 fsync each
        (~80ms/image measured on a 9k-image run) collapsed to a handful of
        statements and a single commit.

        The single-image ordering guarantees are preserved exactly, because
        both are correctness rules rather than incidental sequencing:

        * The sidecar is unlinked BEFORE the DB row goes. A record deleted while
          its sidecar survives is resurrected by the next reconcile, so a record
          whose sidecar could not be removed is left in the DB and counted as a
          failure rather than half-deleted.
        * The image file is unlinked AFTER, best-effort. An image file with no
          sidecar is inert (reconcile only re-imports images that have one), so
          it must never gate the delete.

        *delete_files* is the only difference between the two actions: True
        removes the image file too ("delete"), False leaves it ("removeFromDb").

        Returns (removed_count, fail_count).
        """
        if not image_records:
            return 0, 0

        # (record, image_path) — image_path is None when get_safe_path rejects the
        # stored filename; such a record still has a DB row to remove.
        resolved = [(rec, self._resolve_image_path(rec.filename)) for rec in image_records]

        def _unlink_sidecars() -> Dict[str, str]:
            """Unlink every sidecar; return {image_uuid: error} for the ones that failed."""
            errors: Dict[str, str] = {}
            for rec, image_path in resolved:
                if image_path is None:
                    continue
                try:
                    sidecar_path = self._get_json_sidefile(image_path)
                    if sidecar_path.exists():
                        sidecar_path.unlink()
                except Exception as e:  # noqa: BLE001 - reported per image, not fatal to the batch
                    errors[rec.uuid] = str(e)
            return errors

        sidecar_errors = await asyncio.to_thread(_unlink_sidecars)
        for image_uuid, err in sidecar_errors.items():
            logger.error("remove_image_records_bulk: sidecar delete failed for %s: %s", image_uuid, err)

        removable = [(rec, path) for rec, path in resolved if rec.uuid not in sidecar_errors]
        if not removable:
            return 0, len(sidecar_errors)

        for rec, _ in removable:
            self._remove_from_hash_index(rec)

        uuids = [rec.uuid for rec, _ in removable]
        deleted = await self._db_client.delete_images_bulk(uuids)
        if deleted != len(uuids):
            # Rows can legitimately vanish between the caller's query and this
            # delete (a concurrent single-image delete). Worth a line, not a raise.
            logger.warning(
                "remove_image_records_bulk: asked to delete %d images, DB removed %d",
                len(uuids),
                deleted,
            )

        if delete_files:

            def _unlink_images() -> None:
                for _, image_path in removable:
                    if image_path is None:
                        continue
                    try:
                        if image_path.exists():
                            image_path.unlink()
                    except Exception as e:  # noqa: BLE001 - best effort, see docstring
                        logger.warning("remove_image_records_bulk: failed to delete file %s: %s", image_path, e)

            await asyncio.to_thread(_unlink_images)

        logger.info(
            "remove_image_records_bulk: removed %d images (files %s, %d failed)",
            len(uuids),
            "deleted" if delete_files else "kept",
            len(sidecar_errors),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for rec, _ in removable:
                logger.debug("remove_image_records_bulk: removed %s (%s)", rec.filename, rec.uuid)

        return len(uuids), len(sidecar_errors)

    # ─────────────────────────────────────────────────────────────────────────
    # Perceptual hash index public API
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_images(self, phash: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images within *threshold* of *phash*."""
        return self._hash_index.find_similar_images(phash, threshold)

    def find_similar_bboxes(
        self,
        phash: str,
        label_uuids: Optional[List[str]],
        threshold: int,
    ) -> List[Tuple[str, int, str]]:
        """Return [(bbox_uuid, distance, label_uuid)] within *threshold* of *phash*."""
        return self._hash_index.find_similar_bboxes(phash, label_uuids, threshold)

    async def find_similar_bboxes_enriched(
        self,
        phash: str,
        label_uuids: Optional[List[str]],
        threshold: int,
        tag_uuids: Optional[List[str]] = None,
        exclude_tag_uuids: Optional[List[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Return [(bbox_uuid, image_uuid, distance, label_uuid)] within *threshold* of *phash*.

        Like find_similar_bboxes but enriches each result with its parent image UUID via a
        single batch query, so callers never have to deal with a missing image_uuid.
        Optional tag_uuids / exclude_tag_uuids filter the matching bboxes by their own tags.
        label_uuids restricts the BK-tree search to those label trees (None/empty = all labels).
        """
        matches = self._hash_index.find_similar_bboxes(phash, label_uuids, threshold)
        if not matches:
            return []

        # Apply tag filter if any tags are specified.
        if tag_uuids or exclude_tag_uuids:
            passing = await self._db_client.filter_bboxes_by_tags(
                [b for b, _, _ in matches],
                tag_uuids or [],
                exclude_tag_uuids or [],
                tag_match_mode,
            )
            matches = [(b, d, l) for b, d, l in matches if b in passing]
            if not matches:
                return []

        image_uuid_map = await self._db_client.get_image_uuids_for_bboxes([b for b, _, _ in matches])
        return [(b_uuid, image_uuid_map.get(b_uuid), dist, lbl) for b_uuid, dist, lbl in matches]

    async def find_similar_images_by_uuid(self, image_uuid: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images similar to the given image UUID, including the query image itself (distance 0)."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None or image.phash is None:
            return []
        return self._hash_index.find_similar_images(image.phash, threshold)

    async def find_similar_bboxes_by_uuid(
        self,
        bbox_uuid: str,
        label_uuids: Optional[List[str]],
        threshold: int,
        tag_uuids: Optional[List[str]] = None,
        exclude_tag_uuids: Optional[List[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Return [(bbox_uuid, image_uuid, distance, label_uuid)] for bboxes similar to the given bbox UUID, including the query bbox itself (distance 0)."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None:
            return []
        bbox, _ = result
        if bbox.phash is None:
            return []
        return await self.find_similar_bboxes_enriched(
            bbox.phash,
            label_uuids,
            threshold,
            tag_uuids,
            exclude_tag_uuids,
            tag_match_mode,
        )

    async def rank_images_by_diversity(
        self,
        seed_uuid: Optional[str],
        candidate_uuids: Optional[Set[str]],
        count: int,
    ) -> List[str]:
        """Return up to *count* image UUIDs ordered most-diverse first (farthest-point sampling).

        If *seed_uuid* is None, a seed phash is chosen automatically from the candidate
        set (v1 approximation: first available candidate — the diversity walk is largely
        insensitive to the exact seed choice).
        If *candidate_uuids* is None, all valid images participate.
        """
        if seed_uuid is not None:
            image = await self._db_client.get_image_by_uuid(seed_uuid)
            if image is None or image.phash is None:
                return []
            seed_phash = image.phash
        else:
            seed_phash = self._hash_index.pick_any_image_phash(candidate_uuids)
            if seed_phash is None:
                return []

        # rank_images_by_diversity is async: it materialises the candidate list on
        # the event loop (safe against concurrent index mutation) and offloads only
        # the CPU-bound ranking to a worker thread internally.
        return await self._hash_index.rank_images_by_diversity(
            seed_phash,
            candidate_uuids,
            count,
        )

    async def rank_bboxes_by_diversity(
        self,
        seed_bbox_uuid: str,
        label_uuids: Optional[List[str]],
        count: int,
    ) -> List[Tuple[str, Optional[str], str]]:
        """Return up to *count* (bbox_uuid, image_uuid, label_uuid) ordered most-diverse first."""
        result = await self._db_client.get_bbox_with_image(seed_bbox_uuid)
        if result is None:
            return []
        bbox, _ = result
        if bbox.phash is None:
            return []

        ranked = await self._hash_index.rank_bboxes_by_diversity(
            bbox.phash,
            label_uuids or None,
            count,
        )
        if not ranked:
            return []

        image_uuid_map = await self._db_client.get_image_uuids_for_bboxes([b for b, _ in ranked])
        return [(b_uuid, image_uuid_map.get(b_uuid), lbl) for b_uuid, lbl in ranked]

    async def get_most_unique_images(
        self,
        count: int,
        seed_uuid: Optional[str] = None,
        candidate_uuids: Optional[List[str]] = None,
    ) -> List[str]:
        """Return up to *count* image UUIDs ranked by visual diversity.

        Default scope: images that have at least one bbox tagged UNVERIFIED_TAG_UUID.
        Falls back to all images if no unverified images exist.

        When *candidate_uuids* is given (the caller's already-filtered set, e.g. the
        Review tab's active filters), that set is the scope instead — an explicit
        filter wins over the unverified default, otherwise filtering to, say,
        verified-only images would silently return unverified ones. An empty list
        means "nothing matched the filter" and yields no results.
        """
        if candidate_uuids is not None:
            if not candidate_uuids:
                return []
            return await self.rank_images_by_diversity(seed_uuid, self._cap_candidates(set(candidate_uuids)), count)

        unverified_uuids = self._cap_candidates(await self._db_client.get_image_uuids_by_bbox_tag(UNVERIFIED_TAG_UUID))
        return await self.rank_images_by_diversity(
            seed_uuid,
            unverified_uuids if unverified_uuids else None,
            count,
        )

    @staticmethod
    def _cap_candidates(uuids: Set[str]) -> Set[str]:
        """Bound a diversity candidate pool to the hash index's documented capacity.

        Truncating the query itself would bias which images are even eligible (e.g.
        always favoring lower-rowid images), so sample randomly instead to keep the
        pool representative.
        """
        if len(uuids) <= HashIndex.MAX_DIVERSITY_CANDIDATES:
            return uuids
        return set(random.sample(sorted(uuids), HashIndex.MAX_DIVERSITY_CANDIDATES))

    async def find_similar_bboxes_by_coords(
        self,
        image_uuid: str,
        x: float,
        y: float,
        w: float,
        h: float,
        label_uuids: Optional[List[str]],
        threshold: int,
        tag_uuids: Optional[List[str]] = None,
        exclude_tag_uuids: Optional[List[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Compute a phash from a (possibly unsaved) bbox region and return similar bboxes."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None:
            return []
        image_path = Path(image.filename)
        if not image_path.is_absolute():
            image_path = self._files_root / image_path
        phash = compute_bbox_phash(image_path, x, y, w, h)
        return await self.find_similar_bboxes_enriched(
            phash, label_uuids, threshold, tag_uuids, exclude_tag_uuids, tag_match_mode
        )

    async def get_images_by_uuids(self, image_uuids: List[str]) -> List[ImageRecordRead]:
        """Return full ImageRecordRead objects for a list of image UUIDs, in any order."""
        records = await self._db_client.get_images_by_uuids(image_uuids)
        result = []
        for img in records:
            read = ImageRecordRead.model_validate(img)
            read.filename = self._image_rel_path(img.filename)
            result.append(read)
        return result

    async def get_status(self) -> dict:
        """Return service health and diagnostic info."""
        notices: list[dict] = []

        if self._init_error:
            notices.append(
                {
                    "key": "db_init_error",
                    "level": "error",
                    "message": f"Database initialization failed: {self._init_error}",
                }
            )
        elif self._build_phase == "migrating":
            notices.append(
                {
                    "key": "db_migrating",
                    "level": "info",
                    "message": "Database migrating — other requests paused until complete",
                }
            )
        elif not self._hash_index_ready:
            if self._build_phase == "loading":
                img, bbox = self._build_img_loaded, self._build_bbox_loaded
                if img or bbox:
                    count_str = f" — {img:,} images, {bbox:,} bboxes loaded"
                else:
                    count_str = ""
                phase_detail = f" (loading hashes from DB…{count_str})"
            else:
                phase_detail = {
                    "constructing": " (constructing search trees…)",
                }.get(self._build_phase, "…")
            notices.append(
                {
                    "key": "hash_build",
                    "level": "info",
                    "message": f"Building similarity index{phase_detail} — similarity searches unavailable until complete",
                }
            )
        elif self._rebuild_in_progress:
            notices.append(
                {
                    "key": "hash_rebuild",
                    "level": "warning",
                    "message": "Hash index rebuilding — similar search may return stale results",
                }
            )
        stats = self._hash_index.stats()

        summary = f"index: {stats.get('image_valid', 0):,} imgs"
        if stats.get("needs_rebuild"):
            summary += " ⚠rebuild"

        # Bbox trees are keyed by label *uuid*, and resolving those to names means
        # a DB round-trip — which /status can't afford, since the notices loop
        # polls it every few seconds. Aggregate instead: a list of 28 raw uuids
        # tells a reader nothing anyway.
        bbox_sizes = stats.get("bbox_tree_sizes_by_label", {}) or {}
        tombstones = stats.get("bbox_tombstones_by_label", {}) or {}
        bbox_total = sum(bbox_sizes.values())
        bbox_dead = sum(tombstones.values())

        index_rows = [
            row("Images indexed", f"{stats.get('image_valid', 0):,}"),
            row("Image tree size", f"{stats.get('image_tree_size', 0):,} ({stats.get('image_tombstones', 0):,} dead)"),
            row("Bbox trees", f"{len(bbox_sizes):,} labels, {bbox_total:,} entries ({bbox_dead:,} dead)"),
            row("Largest bbox tree", f"{max(bbox_sizes.values()):,}" if bbox_sizes else "—"),
            row("Build phase", self._build_phase or "—"),
            row("Ready", "yes" if self._hash_index_ready else "no"),
            row("Needs rebuild", "yes" if stats.get("needs_rebuild") else "no"),
        ]

        return {
            "service": "db_server",
            "status": "ok",
            "hash_index": stats,
            "summary": summary,
            "notices": notices,
            "details": [section("Similarity index", index_rows)],
        }

    async def get_image_filename_by_uuid(self, image_uuid: str) -> Optional[str]:
        """Return the relative filename for an image UUID, or None if not found."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        return self._image_rel_path(image.filename) if image else None

    def check_dedup_bulk(self, image_phash: str, image_threshold: int, bboxes: list) -> dict:
        """Bulk duplicate check: one image phash + N bbox phashes → one in-memory call."""
        return self._hash_index.check_dedup_bulk(image_phash, image_threshold, bboxes)

    async def rebuild_hash_index(self, clear_first: bool = False) -> dict:
        """Rebuild the BK-tree index from the DB and return stats.

        If clear_first is True, all stored phashes are NULLed before the rebuild
        so the tree starts empty. Use this when the global hash size changes —
        old hashes are incompatible and would produce nonsense distances.
        """
        self._rebuild_in_progress = True
        try:
            if clear_first:
                await self._db_client.clear_all_phashes()
            stats = await self._hash_index.rebuild()
        finally:
            self._rebuild_in_progress = False
        return {
            "image_count": stats.image_count,
            "bbox_count": stats.bbox_count,
            "duration_ms": stats.duration_ms,
        }

    async def recompute_image_phash(
        self,
        image_uuid: str,
        image_hash_size: Optional[int] = None,
        bbox_hash_size: Optional[int] = None,
    ) -> Optional[dict]:
        """Re-read image from disk, recompute phashes, update DB + index.

        Returns {"image_phash": str, "bbox_count": int} on success, None if not found.
        """
        import cv2 as _cv2

        img_hs = image_hash_size or self._image_hash_size
        bbox_hs = bbox_hash_size or self._bbox_hash_size

        image_record = await self._db_client.get_image_by_uuid(image_uuid)
        if image_record is None:
            return None

        safe_path = get_safe_path(self._files_root, image_record.filename)
        if safe_path is None or not safe_path.exists():
            logger.warning("recompute_image_phash: file not found for %s", image_uuid)
            return None

        image_bgr = await asyncio.to_thread(_cv2.imread, str(safe_path))
        if image_bgr is None:
            logger.warning("recompute_image_phash: cv2.imread failed for %s", safe_path)
            return None

        image_phash = await asyncio.to_thread(phash_from_numpy, image_bgr, img_hs)

        bbox_phash_list = []
        for bbox in image_record.bboxes or []:
            try:
                bp = await asyncio.to_thread(
                    phash_from_numpy_crop,
                    image_bgr,
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    bbox_hs,
                )
                bbox_phash_list.append({"uuid": bbox.uuid, "phash": bp})
            except Exception:
                logger.exception("bbox phash error for %s", bbox.uuid)

        await self._db_client.update_phashes(image_uuid, image_phash, bbox_phash_list)

        # Remove old hash index entries, then re-add updated ones.
        if image_record.phash:
            self._hash_index.remove_image(image_uuid)
        for bbox in image_record.bboxes or []:
            if bbox.phash and bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)

        self._hash_index.add_image(image_uuid, image_phash)
        for bp_entry in bbox_phash_list:
            bbox_obj = next(
                (b for b in (image_record.bboxes or []) if b.uuid == bp_entry["uuid"]),
                None,
            )
            if bbox_obj and bbox_obj.label_uuid:
                self._hash_index.add_bbox(bp_entry["uuid"], bbox_obj.label_uuid, bp_entry["phash"])

        # Refresh sidecar with updated phashes.
        updated_record = await self._db_client.get_image_by_filename(safe_path)
        if updated_record:
            enriched_update = ImageRecordUpdate(
                bboxes=[
                    BBoxUpdate(
                        uuid=b.uuid,
                        label_uuid=b.label_uuid,
                        label_name=b.label.name if b.label else None,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        tag_uuids=[t.uuid for t in (b.tags or [])],
                        tag_names=[t.name for t in (b.tags or [])],
                        phash=b.phash,
                    )
                    for b in updated_record.bboxes
                ]
            )
            await asyncio.to_thread(
                self._db_client.write_image_metadata_to_json,
                self._get_json_sidefile(safe_path),
                image_uuid,
                image_record.filename,
                enriched_update,
                image_phash,
            )

        return {"image_phash": image_phash, "bbox_count": len(bbox_phash_list)}

    async def clear_orphan_generated_tags(self) -> int:
        """Delete TASK_GENERATED tags with no associated bboxes.

        Returns the number of tags deleted.
        """
        return await self._db_client.clear_orphan_generated_tags()

    ################################################################################
    # TASK CONFIGS API
    ################################################################################

    async def create_task_config(
        self,
        name: str,
        typename: str,
        params: TaskParams,
        description: Optional[str] = None,
        is_temporary: Optional[bool] = False,
    ) -> TaskConfig:
        return await self._db_client.add_task_config(
            name=name, typename=typename, params=params, description=description, is_temporary=bool(is_temporary)
        )

    async def list_task_configs(
        self,
        config_uuids: Optional[List[str]] = None,
        typename: Optional[str] = None,
        name: Optional[str] = None,
        is_temporary: Optional[bool] = None,
    ) -> Dict[str, TaskConfig]:
        return await self._db_client.get_task_configs(
            config_uuids=config_uuids, typename=typename, name=name, is_temporary=is_temporary
        )

    async def update_task_config(
        self,
        config_uuid: str,
        name: Optional[str] = None,
        typename: Optional[str] = None,
        params: Optional[TaskParams] = None,
        description: Optional[str] = None,
        marked_for_delete: Optional[bool] = None,
    ) -> None:
        await self._db_client.update_task_config(
            config_uuid=config_uuid,
            name=name,
            typename=typename,
            params=params,
            description=description,
            marked_for_delete=marked_for_delete,
        )

    async def delete_task_configs(self, config_uuids: List[str]) -> None:
        await self._db_client.delete_task_config(config_uuids)

    ################################################################################
    # TASK INSTANCES API
    ################################################################################

    async def create_task_instance(self, config_uuid: str) -> TaskInstance:
        return await self._db_client.insert_new_active_task(config_uuid)

    async def list_task_instances(
        self,
        task_uuids: Optional[List[str]] = None,
        resumable: Optional[bool] = None,
        include_resume_data: bool = False,
    ) -> List[TaskInstance]:
        return await self._db_client.get_tasks(
            task_uuid=task_uuids,
            resumable=resumable,
            include_resume_data=include_resume_data,
        )

    async def delete_task_instances(self, task_uuids: List[str]) -> None:
        await self._db_client.delete_active_tasks(task_uuids)

    async def set_task_status(
        self, instance_uuid: str, status: TaskStatus, error_message: Optional[str] = None
    ) -> None:
        await self._db_client.set_task_status(instance_uuid, status, error_message=error_message)

    async def set_task_result(self, instance_uuid: str, result: TaskResultData) -> None:
        await self._db_client.set_task_result(instance_uuid, result)

    async def set_task_resume_data(self, instance_uuid: str, resume_data: TaskResumeData) -> None:
        await self._db_client.set_task_resume_data(instance_uuid, resume_data)

    async def set_task_timestamps_and_logs(
        self,
        instance_uuid: str,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        log_json: Optional[List[str]] = None,
        status_data_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._db_client.set_task_timestamps_and_logs(
            instance_uuid,
            started_at=started_at,
            ended_at=ended_at,
            log_json=log_json,
            status_data_json=status_data_json,
        )

    async def update_task_heartbeat(self, instance_uuid: str) -> None:
        await self._db_client.update_task_heartbeat(instance_uuid)

    async def get_task_results(self, task_uuids: List[str]) -> Dict[str, TaskResultData]:
        results = await self._db_client.get_task_results(task_uuids)
        return results or {}

    ################################################################################
    # THUMBNAIL API
    ################################################################################

    def _evict_expired_counts(self, now: float) -> None:
        """Drop timed-out entries from the count cache once it grows past the ceiling.

        Entries are keyed by a hash of the filter predicates, so every distinct filter
        combination a user tries leaves one behind; without this they are only ever
        replaced by an identical query and the dict grows for the life of the process.
        """
        if len(self._count_cache) <= self._COUNT_CACHE_MAX_ENTRIES:
            return
        stale = [key for key, (_, ts, _epoch) in self._count_cache.items() if now - ts >= self._COUNT_CACHE_TTL]
        for key in stale:
            del self._count_cache[key]
        if len(self._count_cache) > self._COUNT_CACHE_MAX_ENTRIES:
            # Everything is still live (a burst of distinct filters inside one TTL
            # window). Drop the oldest to keep the ceiling meaningful.
            excess = len(self._count_cache) - self._COUNT_CACHE_MAX_ENTRIES
            for key in sorted(self._count_cache, key=lambda k: self._count_cache[k][1])[:excess]:
                del self._count_cache[key]

    def _thumb_cache_dir(self) -> Path:
        return CACHE_DIR / "thumb_cache"

    def _thumb_cache_path(self, bbox_uuid: str, width: int) -> Path:
        return self._thumb_cache_dir() / f"{bbox_uuid}_{width}{_THUMB_SUFFIX}"

    @staticmethod
    def _thumb_cache_files(cache_dir: Path, stem: str = "*"):
        """Every cached thumbnail matching *stem*, in any format the cache has used."""
        for suffix in _THUMB_SUFFIXES:
            yield from cache_dir.glob(f"{stem}{suffix}")

    @staticmethod
    def _encode_thumb(image: "PILImage.Image") -> bytes:
        """Encode a thumbnail for the cache and the wire.

        The mode conversion is load-bearing, not defensive: JPEG has no alpha channel,
        and a source PNG or a palette image reaches here as RGBA/LA/P, which Pillow
        refuses to write as JPEG.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format=_THUMB_FORMAT, quality=_THUMB_QUALITY, optimize=True)
        return buf.getvalue()

    def _trim_thumb_cache_sync(self, max_bytes: int) -> None:
        """Evict least-recently-used thumbnails until the cache is under its target.

        Note the two different bounds: the trim is *triggered* at max_bytes but *runs*
        down to _THUMB_TRIM_TARGET of it. Evicting to exactly the cap left the cache
        full, so the next write trimmed again, and the file it had just evicted was
        usually the one the next request regenerated.
        """
        cache_dir = self._thumb_cache_dir()
        if not cache_dir.exists():
            return
        entries = []
        total = 0
        for p in self._thumb_cache_files(cache_dir):
            try:
                st = p.stat()
                entries.append((st.st_mtime, st.st_size, p))
                total += st.st_size
            except FileNotFoundError:
                pass
        if total <= max_bytes:
            return
        target = int(max_bytes * _THUMB_TRIM_TARGET)
        entries.sort()  # oldest mtime first — evict these first
        for _, size, path in entries:
            if total <= target:
                break
            try:
                path.unlink(missing_ok=True)
                total -= size
                logger.info(f"Thumbnail cache evicted: {path.name}")
            except Exception as e:
                logger.warning(f"Failed to evict thumbnail {path.name}: {e}")

    def _schedule_thumb_cache_trim(self) -> None:
        """Trim the cache in a background thread if a size limit is set.

        Coalesced: each trim globs and stats the whole cache directory, so running one
        per thumbnail miss (which is how often this is called) is pure waste. If a trim
        is already in flight, this is a no-op — that trim will observe whatever the
        current on-disk state is by the time it runs.
        """
        if THUMB_CACHE_MAX_MB <= 0:
            return
        if self._thumb_trim_task is not None and not self._thumb_trim_task.done():
            return
        max_bytes = THUMB_CACHE_MAX_MB * 1024 * 1024
        # Keep a strong reference (asyncio holds only a weak one) and log failures
        # instead of letting them vanish into an un-awaited task.
        task = asyncio.create_task(asyncio.to_thread(self._trim_thumb_cache_sync, max_bytes))
        self._thumb_trim_task = task

        def _log_failure(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception() is not None:
                logger.warning(f"Thumbnail cache trim failed: {t.exception()}")

        task.add_done_callback(_log_failure)

    def _resolve_image_file_path(self, image_path: str) -> Optional[Path]:
        """Resolve an image path to a file under _files_root, or None.

        Only paths that resolve within _files_root are accepted; there is no
        raw-path fallback (that would let a user-supplied path escape the root
        via an absolute path or traversal — arbitrary file read). Trusted
        callers pass absolute paths already under _files_root, which
        get_safe_path resolves correctly.
        """
        safe = get_safe_path(self._files_root, Path(image_path))
        if safe is not None and safe.is_file():
            return safe
        # Not silent: legacy DB rows may hold absolute paths outside the root
        # (imported before FILES_ROOT was enforced) — surface why the
        # thumbnail is missing instead of returning a bare None.
        logger.warning(
            f"Image path {image_path!r} does not resolve to a file under {self._files_root}; refusing to serve it"
        )
        return None

    @staticmethod
    def _render_outlined_crop(
        img,
        box: Tuple[float, float, float, float],
        out_width: int,
        color: Optional[Tuple[int, int, int]],
        pad_frac: float = 0.2,
    ) -> bytes:
        """Crop `img` around `box`, resize to `out_width`, then stroke the box.

        `box` is (left, top, right, bottom) in source pixels; `pad_frac` is the
        context padding as a fraction of the box's own size, so the box occupies
        the same share of the thumbnail whatever its source size. `pad_frac=0`
        with `color=None` gives the bare contents of the box and nothing else,
        which is what a caller wants when its own frame *is* the box.

        The stroke is drawn AFTER the resize, at a width proportional to the
        output. Drawn before (at a fixed 3 source px) it was a third of the crop
        for a 20px box — which reads as a small box floating in context rather
        than a filled card — and less than one pixel for a 900px box, which the
        LANCZOS downscale then blurred into the background.

        Returns PNG bytes.
        """
        from PIL import Image as PILImage
        from PIL import ImageDraw

        img_w, img_h = img.size
        bl, bt, br, bb = box
        pad_x = (br - bl) * pad_frac
        pad_y = (bb - bt) * pad_frac
        left = max(0, bl - pad_x)
        top = max(0, bt - pad_y)
        right = min(img_w, br + pad_x)
        bottom = min(img_h, bb + pad_y)
        # A box smaller than a pixel (or an image only a few pixels across) would
        # otherwise crop to nothing, and a zero-size crop cannot be resized.
        right = max(right, min(img_w, left + 1))
        bottom = max(bottom, min(img_h, top + 1))

        cropped = img.crop((left, top, right, bottom))
        crop_w, crop_h = cropped.size
        aspect = crop_w / crop_h if crop_h else 1
        out_height = max(1, round(out_width / aspect))
        cropped = cropped.resize((out_width, out_height), PILImage.LANCZOS)

        if color is not None:
            sx = out_width / crop_w if crop_w else 1
            sy = out_height / crop_h if crop_h else 1
            stroke = max(_OUTLINE_MIN_PX, round(out_width * _OUTLINE_WIDTH_RATIO))
            draw = ImageDraw.Draw(cropped)
            draw.rectangle(
                [
                    (bl - left) * sx,
                    (bt - top) * sy,
                    (br - left) * sx,
                    (bb - top) * sy,
                ],
                outline=color,
                width=stroke,
            )
            del draw

        return Manager._encode_thumb(cropped)

    async def get_bbox_thumbnail(self, bbox_uuid: str, width: int = 200) -> Optional[bytes]:
        """
        Return JPEG bytes for a cropped+resized thumbnail of the given bbox.
        Serves from disk cache if available and fresh, otherwise generates and caches it.
        Cache freshness is validated against bbox.updated_at so edits made outside the
        review page (e.g. from the image-tagging canvas) are always reflected.
        """
        from PIL import Image as PILImage

        # Fetch bbox first so we can validate cache freshness against bbox.updated_at.
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None:
            return None
        bbox, image_record = result
        if image_record is None:
            return None

        cache_path = self._thumb_cache_path(bbox_uuid, width)
        if cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                if cache_mtime >= bbox.updated_at.timestamp():
                    cache_path.touch()  # update mtime for LRU tracking
                    async with aiofiles.open(cache_path, "rb") as f:
                        return await f.read()
                # Cache is older than the bbox — delete it and regenerate below.
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass

        safe_path = self._resolve_image_file_path(image_record.filename)
        if safe_path is None:
            return None

        # Extract coords and look up label color before the thread (avoids detached ORM objects).
        bx, by, bw, bh = bbox.x, bbox.y, bbox.width, bbox.height
        label_color: tuple[int, int, int] = (255, 215, 0)  # fallback gold
        if bbox.label_uuid:
            label = await self._db_client.get_label_by_uuid(bbox.label_uuid)
            if label and label.color:
                try:
                    h = label.color.lstrip("#")
                    label_color = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
                except (ValueError, IndexError):
                    pass

        def _generate() -> Optional[bytes]:
            try:
                with PILImage.open(safe_path) as img:
                    img = img.convert("RGB")
                    img_w, img_h = img.size
                    return self._render_outlined_crop(
                        img,
                        (bx * img_w, by * img_h, (bx + bw) * img_w, (by + bh) * img_h),
                        width,
                        label_color,
                    )
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for bbox {bbox_uuid}: {e}")
                return None

        async with self._thumb_decode_sem:
            jpeg_bytes = await asyncio.to_thread(_generate)
        if jpeg_bytes is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(jpeg_bytes)

        self._schedule_thumb_cache_trim()
        return jpeg_bytes

    async def invalidate_bbox_thumbnails(self, bbox_uuids: List[str]) -> None:
        """Delete all cached thumbnail sizes for the given bbox UUIDs."""
        cache_dir = self._thumb_cache_dir()

        def _remove_files() -> None:
            for uuid in bbox_uuids:
                for path in Manager._thumb_cache_files(cache_dir, stem=f"{uuid}_*"):
                    try:
                        path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete thumbnail cache for {uuid}: {e}")

        await asyncio.to_thread(_remove_files)

    @staticmethod
    def _source_version(source: Optional[Path]) -> str:
        """A cheap identity for the *contents* of an image file, for cache keys.

        Path alone is not one. A file under FILES_ROOT can be replaced in place —
        by an overwriting upload, by a rsync, by the user swapping in a better
        scan — and every path-keyed thumbnail then serves the picture that is
        gone. Keying on mtime+size instead means new bytes simply miss the cache,
        with no invalidation hook to remember to call from each of the places
        that can write over an image. The superseded entries age out through the
        same oldest-mtime trim that bounds the cache anyway.
        """
        if source is None:
            return "0"
        try:
            st = source.stat()
        except OSError:
            return "0"
        return f"{st.st_mtime_ns}-{st.st_size}"

    def _image_thumb_cache_path(self, image_path: str, width: int, source: Optional[Path] = None) -> Path:
        key = hashlib.sha1(f"{image_path}|{self._source_version(source)}".encode()).hexdigest()
        return self._thumb_cache_dir() / f"img_{key}_{width}{_THUMB_SUFFIX}"

    def _region_thumb_cache_path(
        self,
        image_path: str,
        region: Tuple[float, float, float, float],
        width: int,
        color: Optional[str] = None,
        source: Optional[Path] = None,
        pad_frac: float = 0.2,
        outline: bool = True,
    ) -> Path:
        # The region is part of the key, so moving a proposed box simply misses
        # the cache rather than needing an invalidation hook. So is the outline
        # colour: it is drawn into the PNG, and relabelling a proposed box asks
        # for the same region in a new colour -- keyed without it, the card ends
        # up showing a chip of one class beside a crop outlined in another.
        # Padding and the outline flag are keyed for the same reason as the colour:
        # both are baked into the PNG, and a tight crop asked for after a padded one of
        # the same region would otherwise be served the padded file.
        region_key = f"{region[0]:.6f},{region[1]:.6f},{region[2]:.6f},{region[3]:.6f}"
        variant = f"{pad_frac:.4f}|{'o' if outline else 'n'}"
        key = hashlib.sha1(f"{image_path}|{region_key}|{color or ''}|{variant}|{self._source_version(source)}".encode())
        return self._thumb_cache_dir() / f"rgn_{key.hexdigest()}_{width}{_THUMB_SUFFIX}"

    async def get_region_thumbnail(
        self,
        image_path: str,
        x: float,
        y: float,
        width_frac: float,
        height_frac: float,
        width: int = 200,
        color: Optional[str] = None,
        pad_frac: float = 0.2,
        outline: bool = True,
    ) -> Optional[bytes]:
        """PNG bytes for a cropped+resized thumbnail of an arbitrary normalized
        region of an image, with the region outlined.

        This is `get_bbox_thumbnail` for a box that has no bbox row: a preview
        run's proposed *new* boxes exist only as candidate geometry, so a card
        for one had to fall back to the whole frame — in which a small proposed
        box is invisible. Defaults to the same 20% context padding and stroked
        outline as the bbox crop.

        `pad_frac=0, outline=False` asks for the bare contents of the region
        instead. That is what a caller wants when its own frame already *is* the
        box: the watcher's match preview draws the crop inside a rectangle the
        user drags, and the default's padding plus stroke put a second, smaller
        box inside the first, so it was unclear which edge the overlap was
        measured from.
        """
        from PIL import Image as PILImage

        bx = min(max(x, 0.0), 1.0)
        by = min(max(y, 0.0), 1.0)
        bw = min(max(width_frac, 0.0), 1.0 - bx)
        bh = min(max(height_frac, 0.0), 1.0 - by)
        if bw <= 0 or bh <= 0:
            return None

        # Resolved before the cache lookup — see get_image_thumbnail: the source
        # file's version is part of the key.
        safe_path = self._resolve_image_file_path(image_path)
        if safe_path is None:
            return None

        cache_path = self._region_thumb_cache_path(
            image_path, (bx, by, bw, bh), width, color, source=safe_path, pad_frac=pad_frac, outline=outline
        )
        if cache_path.exists():
            try:
                cache_path.touch()  # update mtime for LRU tracking
                async with aiofiles.open(cache_path, "rb") as f:
                    return await f.read()
            except OSError:
                pass

        stroke_colour: Tuple[int, int, int] = (92, 196, 92)  # the "new box" green
        if color:
            try:
                h = color.lstrip("#")
                stroke_colour = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            except (ValueError, IndexError):
                pass

        def _generate() -> Optional[bytes]:
            try:
                with PILImage.open(safe_path) as img:
                    img = img.convert("RGB")
                    img_w, img_h = img.size
                    return self._render_outlined_crop(
                        img,
                        (bx * img_w, by * img_h, (bx + bw) * img_w, (by + bh) * img_h),
                        width,
                        stroke_colour if outline else None,
                        pad_frac=max(0.0, pad_frac),
                    )
            except Exception as e:
                logger.warning(f"Failed to generate region thumbnail for {image_path}: {e}")
                return None

        async with self._thumb_decode_sem:
            png_bytes = await asyncio.to_thread(_generate)
        if png_bytes is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(png_bytes)

        self._schedule_thumb_cache_trim()
        return png_bytes

    async def get_image_thumbnail(self, image_path: str, width: int = 640) -> Optional[bytes]:
        """Return PNG bytes for a full-image thumbnail, resized to `width` px.
        Serves from disk cache if available, otherwise generates and caches it.
        """
        from PIL import Image as PILImage

        # Resolved first: the source file's version is part of the cache key, so
        # replacing an image in place misses the cache instead of serving the
        # picture it replaced.
        safe_path = self._resolve_image_file_path(image_path)
        if safe_path is None:
            return None

        cache_path = self._image_thumb_cache_path(image_path, width, source=safe_path)
        if cache_path.exists():
            try:
                cache_path.touch()  # update mtime for LRU tracking
            except OSError:
                pass
            async with aiofiles.open(cache_path, "rb") as f:
                return await f.read()

        def _generate() -> Optional[bytes]:
            try:
                with PILImage.open(safe_path) as img:
                    img = img.convert("RGB")
                    aspect = img.width / img.height if img.height else 1
                    height = max(1, round(width / aspect))
                    resized = img.resize((width, height), PILImage.LANCZOS)
                    return self._encode_thumb(resized)
            except Exception as e:
                logger.warning(f"Failed to generate image thumbnail for {image_path}: {e}")
                return None

        async with self._thumb_decode_sem:
            png_bytes = await asyncio.to_thread(_generate)
        if png_bytes is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(png_bytes)

        self._schedule_thumb_cache_trim()
        return png_bytes

    async def open_file(self, rel_path: str) -> Optional[Tuple[aiofiles.threadpool.binary.AsyncBufferedReader, str]]:
        """
        Securely open a file under the root and return an aiofiles stream and filename.

        Args:
            rel_path (str): Relative path under the image root.

        Returns:
            Tuple[aiofiles.AsyncBufferedReader, str] or None
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory not set or invalid")
            return None

        try:
            safe_path = get_safe_path(self._files_root, rel_path)

            if safe_path is None:
                logger.warning(f"Blocked directory traversal attempt: {rel_path}")
                return None

            if not safe_path.is_file():
                logger.warning(f"File not found: {safe_path}")
                return None

            f = await aiofiles.open(safe_path, mode="rb")
            filename = safe_path.name
            return f, filename

        except Exception as e:
            logger.error(f"Failed to open file {rel_path}: {e}", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset management
    # ─────────────────────────────────────────────────────────────────────────

    async def list_datasets(self, include_views: bool = False) -> List[DatasetRead]:
        return await self._db_client.list_datasets(include_views=include_views)

    async def get_dataset(self, uuid: str) -> Optional[DatasetRead]:
        return await self._db_client.get_dataset(uuid)

    async def create_dataset(self, data: DatasetCreate) -> DatasetRead:
        return await self._db_client.create_dataset(data)

    async def update_dataset(self, uuid: str, data: DatasetUpdate) -> Optional[DatasetRead]:
        return await self._db_client.update_dataset(uuid, data)

    async def delete_dataset(self, uuid: str) -> bool:
        return await self._db_client.delete_dataset(uuid)

    async def get_dataset_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
    ) -> Optional[tuple[List[ImageRecordRead], bool]]:
        result = await self._db_client.list_dataset_images(
            uuid,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )
        if result is None:
            return None
        images, has_more = result
        records = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            records.append(record)
        return records, has_more

    async def get_dataset_labels(self, uuid: str) -> Optional[List[Label]]:
        """Return the Label objects for a dataset's relevant labels."""
        dataset = await self._db_client.get_dataset(uuid)
        if dataset is None:
            return None
        if not dataset.label_uuids:
            return []
        labels_map = await self._db_client.get_labels_by_uuids(dataset.label_uuids)
        return list(labels_map.values())

    async def stream_dataset_export(
        self,
        uuid: str,
        fmt: str,
        *,
        val_fraction: float = 0.2,
        require_verification: bool = True,
    ) -> Optional[Tuple[str, Any]]:
        """Build a standard-format export of one Dataset, streamed as a tar.

        Returns ``(download_filename, async_iterator_of_chunks)``, or None if the
        dataset doesn't exist. Returns a 0-image tar rather than an error when the
        dataset resolves to nothing — "this dataset is empty" is a legitimate answer
        and the caller can see it in the archive.

        The tar is assembled and emitted incrementally: one member at a time, each
        image read in a worker thread. Nothing larger than a single image is ever
        resident, so exporting a 100k-image dataset costs the same memory as
        exporting one, and the client starts receiving bytes immediately.
        """
        if fmt not in EXPORT_FORMAT_IDS:
            raise ValueError(f"Unknown export format: {fmt!r}")

        dataset = await self._db_client.get_dataset(uuid)
        if dataset is None:
            return None

        spec = await self._db_client.get_dataset_detection_dataset(uuid, require_verification=require_verification)
        if spec is None:
            return None

        writer = make_writer(fmt, spec["class_names"], dataset.name)
        assignments = split_images(spec["images"], val_fraction)

        safe_name = sanitize_stem(dataset.name, uuid)
        download_name = f"{safe_name}_{fmt}.tar"

        def _read(path: Path, want_size: bool):
            data = path.read_bytes()
            if not want_size:
                return data, (0, 0)
            from PIL import Image as PILImage

            with PILImage.open(io.BytesIO(data)) as img:
                return data, img.size

        async def _iter():
            tar = TarStream()
            skipped: List[str] = []
            # The splits that actually received an image, not the ones the
            # assignment *planned*: if every val image turns out to be missing
            # from disk, declaring `val: images/val` in data.yaml (or emitting an
            # empty instances_val.json / val.txt) points the trainer at a
            # directory the archive does not contain, and Ultralytics errors out.
            written_splits: set = set()
            try:
                for split, entry in assignments:
                    path = self._resolve_image_path(entry["filename"])
                    if path is None or not path.exists():
                        skipped.append(entry["filename"])
                        continue
                    try:
                        data, size = await asyncio.to_thread(_read, path, writer.needs_dimensions)
                    except Exception:
                        logger.exception("Dataset export: failed to read %s", path)
                        skipped.append(entry["filename"])
                        continue

                    base = sanitize_stem(entry["filename"], entry["image_uuid"])
                    writer.add_image(
                        tar, split, base, image_extension(entry["filename"]), data, size, entry["annotations"]
                    )
                    written_splits.add(split)
                    chunk = tar.drain()
                    if chunk:
                        yield chunk

                # "train" is always declared — a format's index files need the key
                # even when the archive turned out empty — while "val" is only
                # declared if something landed in it.
                splits = ["train"] + (["val"] if "val" in written_splits else [])
                writer.finalize(tar, splits)
                if skipped:
                    # Surfaced in the archive rather than only in the log: a
                    # streamed download has no way to report a partial failure once
                    # the first byte is out.
                    tar.add_text(
                        "SKIPPED.txt",
                        "These images are in the dataset but could not be read from disk:\n\n"
                        + "\n".join(skipped)
                        + "\n",
                    )
                tar.close()
                final = tar.drain()
                if final:
                    yield final
            except Exception:
                logger.exception("Dataset export failed for %s (%s)", uuid, fmt)
                raise

        return download_name, _iter()

    async def preview_dataset_images(
        self,
        filter_json: dict,
        label_uuids: List[str],
        *,
        limit: int = 500,
    ) -> tuple[List[ImageRecordRead], bool]:
        """Execute filter criteria in-memory without saving a dataset (unsaved preview)."""
        images, has_more = await self._db_client.preview_dataset_images(filter_json, label_uuids, limit=limit)
        records = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            records.append(record)
        return records, has_more

    # ─── Model Configurations ─────────────────────────────────────────────────

    def get_model_type_architectures(self) -> dict:
        return dict(MODEL_TYPE_ARCHITECTURES)

    async def _validate_model_config_nodes(self, nodes: List[ModelConfigNodeWrite]) -> None:
        """Validate model_type, architecture, and label_uuid for every node write object."""
        for node in nodes:
            if node.model_type not in MODEL_TYPE_ARCHITECTURES:
                raise ValueError(
                    f"Unknown model_type: {node.model_type!r}. "
                    f"Valid choices: {list(MODEL_TYPE_ARCHITECTURES.keys())}"
                )
            if node.architecture not in MODEL_TYPE_ARCHITECTURES[node.model_type]:
                raise ValueError(
                    f"Architecture {node.architecture!r} is not valid for model_type "
                    f"{node.model_type!r}. Valid choices: {MODEL_TYPE_ARCHITECTURES[node.model_type]}"
                )
            if node.label_uuid is not None:
                label = await self._db_client.get_label_by_uuid(node.label_uuid)
                if label is None:
                    raise ValueError(f"Label not found: {node.label_uuid!r}")

    async def list_model_configs(self) -> List[ModelConfigRead]:
        return await self._db_client.list_model_configs()

    async def get_model_config(self, uuid: str) -> Optional[ModelConfigRead]:
        return await self._db_client.get_model_config(uuid)

    async def get_model_config_node(self, node_uuid: str) -> Optional["ModelConfigNodeRead"]:
        return await self._db_client.get_model_config_node(node_uuid)

    async def derive_training_targets(self, model_config_uuid: str) -> Optional[list]:
        return await self._db_client.derive_training_targets(model_config_uuid)

    async def get_training_target_detection_dataset(
        self,
        model_config_uuid: str,
        node_uuid: str,
        dataset_uuids: list[str],
        label_uuids: Optional[list[str]] = None,
        class_remap: Optional[dict[str, str]] = None,
        negative_ratio: Optional[float] = None,
        negative_ratio_floor: Optional[int] = None,
    ) -> Optional[dict]:
        result = await self._db_client.get_training_target_detection_dataset(
            model_config_uuid, node_uuid, dataset_uuids, label_uuids=label_uuids, class_remap=class_remap
        )
        if result is None:
            return None
        return await self._apply_negative_ratio_cap(result, negative_ratio, negative_ratio_floor)

    async def get_training_target_crop_dataset(
        self,
        model_config_uuid: str,
        node_uuid: str,
        dataset_uuids: list[str],
        label_uuids: Optional[list[str]] = None,
        class_remap: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        # No negative-ratio cap: a crop dataset has no negative images to cap. Every
        # sample is a verified box that belongs to exactly one class.
        return await self._db_client.get_training_target_crop_dataset(
            model_config_uuid, node_uuid, dataset_uuids, label_uuids=label_uuids, class_remap=class_remap
        )

    # Default cap for a detection dataset's negative (no-annotation) images, expressed
    # as a multiple of the positive count. Bulk-verifying "absent" across a whole dataset
    # (e.g. tagging every indoor photo as dog-absent right after adding a few outdoor dog
    # photos) otherwise silently drowns a new class in negatives with no signal in the loss.
    NEGATIVE_RATIO_DEFAULT = 4.0
    NEGATIVE_RATIO_FLOOR_DEFAULT = 20

    async def _apply_negative_ratio_cap(
        self,
        result: dict,
        negative_ratio: Optional[float],
        negative_ratio_floor: Optional[int],
    ) -> dict:
        """Cap a detection dataset's negatives at ``negative_ratio`` * positive count
        (never below ``negative_ratio_floor``), keeping the perceptually most diverse
        negatives via ``HashIndex`` so a handful of near-duplicate frames from one
        static source can't be what survives the cap. No-op if already under the cap.
        """
        images = result.get("images", [])
        positives = [img for img in images if img["annotations"]]
        negatives = [img for img in images if not img["annotations"]]

        ratio = negative_ratio if negative_ratio is not None else self.NEGATIVE_RATIO_DEFAULT
        floor = negative_ratio_floor if negative_ratio_floor is not None else self.NEGATIVE_RATIO_FLOOR_DEFAULT
        max_negatives = max(int(len(positives) * ratio), floor)

        if len(negatives) <= max_negatives:
            return result

        candidate_uuids = {img["image_uuid"] for img in negatives}
        if len(candidate_uuids) > HashIndex.MAX_DIVERSITY_CANDIDATES:
            candidate_uuids = set(random.sample(sorted(candidate_uuids), HashIndex.MAX_DIVERSITY_CANDIDATES))

        kept_uuids = set(await self.rank_images_by_diversity(None, candidate_uuids, max_negatives))

        if len(kept_uuids) < max_negatives:
            # Negatives with no indexed phash can't be ranked by diversity at all; top
            # them up deterministically so the ratio/floor guarantee still holds.
            remaining = sorted(uuid for uuid in candidate_uuids if uuid not in kept_uuids)
            kept_uuids.update(remaining[: max_negatives - len(kept_uuids)])

        kept_negatives = [img for img in negatives if img["image_uuid"] in kept_uuids]
        result["images"] = positives + kept_negatives
        return result

    async def create_model_config(self, data: ModelConfigCreate) -> ModelConfigRead:
        await self._validate_model_config_nodes(data.nodes)
        return await self._db_client.create_model_config(data)

    async def update_model_config(self, uuid: str, data: ModelConfigUpdate) -> Optional[ModelConfigRead]:
        if data.nodes is not None:
            await self._validate_model_config_nodes(data.nodes)
        return await self._db_client.update_model_config(uuid, data)

    async def delete_model_config(self, uuid: str) -> bool:
        return await self._db_client.delete_model_config(uuid)

    # ── Filter chains ──────────────────────────────────────────────────────────

    async def list_filter_chains(self) -> List[FilterChainRead]:
        return await self._db_client.list_filter_chains()

    async def get_filter_chain(self, uuid: str) -> Optional[FilterChainRead]:
        return await self._db_client.get_filter_chain(uuid)

    async def create_filter_chain(self, data: FilterChainCreate) -> FilterChainRead:
        return await self._db_client.create_filter_chain(data)

    async def update_filter_chain(self, uuid: str, data: FilterChainUpdate) -> Optional[FilterChainRead]:
        return await self._db_client.update_filter_chain(uuid, data)

    async def delete_filter_chain(self, uuid: str) -> bool:
        return await self._db_client.delete_filter_chain(uuid)

    # ── Training collectors ────────────────────────────────────────────────────

    async def list_training_collectors(self) -> List[TrainingCollectorRead]:
        return await self._db_client.list_training_collectors()

    async def get_training_collector(self, uuid: str) -> Optional[TrainingCollectorRead]:
        return await self._db_client.get_training_collector(uuid)

    async def create_training_collector(self, data: TrainingCollectorCreate) -> TrainingCollectorRead:
        return await self._db_client.create_training_collector(data)

    async def update_training_collector(
        self, uuid: str, data: TrainingCollectorUpdate
    ) -> Optional[TrainingCollectorRead]:
        return await self._db_client.update_training_collector(uuid, data)

    async def delete_training_collector(self, uuid: str) -> bool:
        return await self._db_client.delete_training_collector(uuid)

    async def get_training_collector_review_actions(self, uuid: str) -> List[dict]:
        """The same "browse this collector's output" buttons a running watcher's
        Jobs card shows — available whether or not anything is currently running.

        `TrainingCollector.review_actions()` needs a live task tag (created, if
        missing, on `start()`), which a stopped collector has never had a chance
        to resolve. Reproduce that resolution read-only: compute the two
        candidate uuids the same way `_resolve_tags` would name them
        (`task_tag_candidates`), and prefer whichever one the DB says already
        exists — mirroring the live path's own "reuse the legacy tag if its name
        still matches" rule — without ever creating a tag from a page that is
        just trying to answer "what's here to review".

        A tag that doesn't exist yet (collector has never actually run) still
        gets a uuid handed back: filtering Review on a tag nothing carries yet
        just returns nothing, which is the correct answer for "not collected
        anything yet" — better than hiding the button and forcing the run to
        happen before its own review path can even be discovered.
        """
        collector = await self.get_training_collector(uuid)
        if collector is None:
            return []
        spec = CollectorSpec.from_params(collector.params)
        if not spec.generate_task_tag:
            # No task tag means nothing on disk is scoped to THIS collector
            # specifically — there's nothing to filter by that a run wouldn't
            # also fail to filter by. Matches `review_actions()`'s own guard.
            return []
        legacy_tag_uuid, task_tag_uuid, tag_name = task_tag_candidates(uuid, collector.name, spec)
        legacy_tag = await self.get_tag_by_uuid(legacy_tag_uuid)
        resolved_tag_uuid = legacy_tag_uuid if (legacy_tag and legacy_tag.name == tag_name) else task_tag_uuid
        return build_review_actions(resolved_tag_uuid, spec)

    # ── Watch profiles (StreamWatcher / NVR) ───────────────────────────────────

    async def list_watch_profiles(self) -> List[WatchProfileRead]:
        return await self._db_client.list_watch_profiles()

    async def get_watch_profile(self, uuid: str) -> Optional[WatchProfileRead]:
        return await self._db_client.get_watch_profile(uuid)

    async def create_watch_profile(self, data: WatchProfileCreate) -> WatchProfileRead:
        return await self._db_client.create_watch_profile(data)

    async def update_watch_profile(self, uuid: str, data: WatchProfileUpdate) -> Optional[WatchProfileRead]:
        return await self._db_client.update_watch_profile(uuid, data)

    async def delete_watch_profile(self, uuid: str) -> bool:
        return await self._db_client.delete_watch_profile(uuid)

    async def validate_watch_profile(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Report configuration problems that only show up at run time.

        Worth its own endpoint because every one of these fails *silently*: a rule
        with no zone never fires, a pre-roll longer than the buffer quietly produces a
        short clip, and a source with no models is analysed against nothing.
        """
        profile = await self._db_client.get_watch_profile(uuid)
        if profile is None:
            return None

        errors: List[str] = []
        warnings: List[str] = []

        if not profile.sources:
            errors.append("This watcher has no sources, so it will never analyse anything.")

        enabled_rules = [r for r in profile.rules if r.enabled]
        if not enabled_rules:
            warnings.append("No enabled rules — the watcher will track objects but never raise an event.")

        # Zones now live on the sources, so this needs a lookup rather than a walk of the
        # profile — the uuids a rule may name are not in the profile at all any more.
        regions_by_source = {}
        for src in profile.sources:
            regions_by_source[src.source_uuid] = await self._db_client.list_source_regions(src.source_uuid)
        zone_uuids = {
            r.uuid for regions in regions_by_source.values() for r in regions if r.role == RegionRole.ZONE.value
        }
        # Which source each zone is drawn on, for checking a rule's source scope against
        # the zones it names.
        zone_source = {
            r.uuid: source_uuid
            for source_uuid, regions in regions_by_source.items()
            for r in regions
            if r.role == RegionRole.ZONE.value
        }
        disabled_sources = {src.source_uuid for src in profile.sources if not src.enabled}
        for rule in profile.rules:
            for kind, refs, side in (
                (rule.subject_a_kind, rule.subject_a_refs_json, "A"),
                (rule.subject_b_kind, rule.subject_b_refs_json, "B"),
            ):
                if kind != "zone":
                    continue
                missing = [ref for ref in (refs or []) if ref not in zone_uuids]
                if missing:
                    errors.append(
                        f"Rule {rule.name!r}: subject {side} references "
                        f"{len(missing)} zone(s) that no longer exist on this watcher's sources."
                    )
            # A rule narrowed to sources it cannot run on is the same class of silent
            # failure as a missing zone: the editor shows a fully configured rule that
            # never fires. Both cases below are reachable only by editing the source list
            # after the rule, which is exactly when nobody re-reads the rule.
            scope = [str(u) for u in (rule.source_uuids_json or [])]
            if scope:
                if all(u in disabled_sources for u in scope):
                    warnings.append(
                        f"Rule {rule.name!r} is restricted to source(s) that are disabled on this "
                        "watcher, so it will never be evaluated."
                    )
                # Zones belong to one source each, so a rule scoped to camera A and
                # pointing at a zone drawn on camera B has no reachable target at all.
                if rule.subject_b_kind == "zone" and rule.subject_b_refs_json:
                    reachable = [ref for ref in rule.subject_b_refs_json if zone_source.get(ref) in set(scope)]
                    if not reachable:
                        errors.append(
                            f"Rule {rule.name!r} is restricted to source(s) that none of its zones are "
                            "drawn on, so it can never fire."
                        )

            # A slot naming nothing can never fire, and reads in the editor as though it
            # were merely unconfigured rather than broken.
            for kind, refs, side in (
                (rule.subject_a_kind, rule.subject_a_refs_json, "A"),
                (rule.subject_b_kind, rule.subject_b_refs_json, "B"),
            ):
                if kind in ("label", "zone") and not refs:
                    errors.append(f"Rule {rule.name!r}: subject {side} selects nothing, so the rule can never fire.")

        # A mask a watcher disables but which has since been deleted from the source is
        # an opt-out pointing at nothing — harmless, but it means the UI shows a checkbox
        # state derived from a mask that no longer exists.
        for src in profile.sources:
            mask_uuids = {r.uuid for r in regions_by_source.get(src.source_uuid, []) if r.role == RegionRole.MASK.value}
            stale = [u for u in (src.disabled_mask_uuids_json or []) if u not in mask_uuids]
            if stale:
                warnings.append(
                    f"Source {src.source_uuid} disables {len(stale)} mask(s) that no longer exist on it; "
                    "the setting has no effect and can be cleared."
                )

        for src in profile.sources:
            has_own = bool(src.model_group_uuid or src.model_name)
            has_default = bool(profile.default_model_group_uuid or profile.default_model_name)
            if not has_own and not has_default:
                # Motion reads pixels, not detections, so a motion-only watcher on a
                # model-less camera is a legitimate setup — a bare "did anything move at
                # the gate" recorder. Only say nothing will be detected when something
                # actually depends on a detection.
                # No rules at all still needs a model: the watcher is misconfigured in
                # two ways, and "nothing will be detected" is the one this line reports.
                needs_detections = not enabled_rules or any(
                    r.relation != RuleRelation.MOTION.value
                    for r in enabled_rules
                    if not r.source_uuids_json or src.source_uuid in [str(u) for u in r.source_uuids_json]
                )
                if needs_detections:
                    errors.append(
                        f"Source {src.source_uuid} has no models selected and the profile sets no default, "
                        "so nothing will be detected on it."
                    )
                elif enabled_rules:
                    warnings.append(
                        f"Source {src.source_uuid} has no models selected. Its motion rules will still run; "
                        "nothing else will, because nothing will be detected on it."
                    )
            fps = src.analysis_fps or profile.default_analysis_fps
            if fps > 10:
                warnings.append(
                    f"Source {src.source_uuid} analyses at {fps} fps. Per-frame inference over "
                    "base64 HTTP is the throughput ceiling; 2–5 fps is usually plenty for tracking."
                )

        # The buffer is not only the pre-roll ceiling — it is also the window the clip
        # drain works inside, which is what makes an event *longer* than the buffer
        # possible at all. That second job was invisible: the field is labelled for
        # pre-roll, so shortening it to a preferred lead-in looks harmless and silently
        # starves the drain. See `apps.helpers.recordings.clipDrain`.
        #
        # The drain reads no closer than SEGMENT_LAG_S to now and fires once the cursor
        # is `SEGMENT_LAG_S + interval` old, while the buffer drops anything older than
        # the retention. The difference is all the slack there is for a slow inference
        # pass or a paused task; at or below zero the drain can never copy out a whole
        # chunk and every long event loses its opening.
        for src in profile.sources:
            if src.buffer_mode == BufferMode.OFF.value:
                # No rewind buffer to starve — an unbuffered source uses the frame ring.
                continue
            retention = src.record_retention_s or STREAM_BUFFER_RETENTION_S
            interval = max(MIN_DRAIN_INTERVAL_S, retention * DRAIN_FRACTION)
            margin = retention - SEGMENT_LAG_S - interval
            if margin <= 0:
                warnings.append(
                    f"Source {src.source_uuid} buffers only {retention:g}s, which the clip drain "
                    f"cannot rescue video from: it reads no closer than {SEGMENT_LAG_S:g}s to live and "
                    f"runs every {interval:g}s, so material is deleted before it is copied out. Any "
                    f"event longer than {retention:g}s will lose its opening. Leave this blank unless "
                    "you have a reason not to — to change how much lead-in a clip has, set the "
                    "pre-roll on the Record clip action instead."
                )
            elif margin < interval:
                warnings.append(
                    f"Source {src.source_uuid} buffers {retention:g}s, leaving the clip drain only "
                    f"{margin:g}s of margin before video is deleted. A slow inference pass or a paused "
                    "watcher will punch holes in clips of long events. Pre-roll is set on the Record "
                    "clip action; this setting does not need to match it."
                )

        # Pre-roll longer than the retention can never be satisfied — the buffer simply
        # does not hold that much history. Catch it at save time, not at cut time.
        retentions = {src.source_uuid: (src.record_retention_s or STREAM_BUFFER_RETENTION_S) for src in profile.sources}
        for action in profile.actions:
            if action.kind != "record_clip":
                continue
            preroll = float(action.params_json.get("preroll_s", 0) or 0)
            for source_uuid, retention in retentions.items():
                if preroll > retention:
                    warnings.append(
                        f"A record_clip action asks for {preroll:g}s of pre-roll, but source "
                        f"{source_uuid} only buffers {retention:g}s. The clip will be clamped."
                    )

        # A timelapse can only keep a frame the watcher actually pulled, so an interval
        # shorter than the analysis period silently becomes the analysis period. Nothing
        # at run time says so — the frames just arrive further apart than configured.
        fps_by_source = {src.source_uuid: (src.analysis_fps or profile.default_analysis_fps) for src in profile.sources}
        for action in profile.actions:
            if action.kind != "save_timelapse":
                continue
            configured = action.params_json.get("output_dir") or ""
            if has_placeholder(configured) and not static_prefix(configured):
                # Refused at run time too, but a watcher that will not start is worth
                # catching at the moment someone types it.
                errors.append(
                    f"A save_timelapse action writes to {configured!r}, which starts with a "
                    "placeholder. Put a real folder name first, so its frames can be found "
                    "again and cleaned up — e.g. timelapse/{source}."
                )

            interval = float(action.params_json.get("interval_s", 30) or 30)
            scope = {str(u) for u in (action.source_uuids_json or [])}
            for source_uuid, fps in fps_by_source.items():
                if scope and source_uuid not in scope:
                    continue
                period = 1.0 / fps if fps else 0.0
                if period > interval:
                    warnings.append(
                        f"A save_timelapse action asks for a frame every {interval:g}s, but source "
                        f"{source_uuid} is only analysed every {period:g}s. Frames will be written "
                        "at the slower rate."
                    )

        # A rule's `max_duration_s` does not merely cap an event: reaching it closes the
        # event and, with the subject still present, opens another. So it also bounds how
        # many frames any one event can ever hold, and once that is below the action's
        # `min_frames` the timelapse compiles nothing — for every event, forever. The
        # run-time signal is the *legitimate* "below the minimum frame count" outcome, so
        # nothing looks wrong anywhere. The two numbers sit on different rows in the
        # editor and nothing else in the app relates them.
        for action in profile.actions:
            if action.kind != "record_event_timelapse" or not action.enabled:
                continue
            interval = max(1.0, float(action.params_json.get("interval_s", 10) or 10))
            min_frames = max(1, int(action.params_json.get("min_frames", 10) or 10))
            scope = {str(u) for u in (action.rule_uuids_json or [])}
            for rule in profile.rules:
                if scope and rule.uuid not in scope:
                    continue
                # An uncapped rule (0) is bounded only by how long the subject stays, and
                # a disabled one raises no events at all — neither can be judged here.
                if not rule.enabled or rule.max_duration_s <= 0:
                    continue
                # Buckets run 0..floor(span/interval), so a 300s event at 10s is 31.
                frames = int(rule.max_duration_s // interval) + 1
                if frames < min_frames:
                    warnings.append(
                        f"Rule {rule.name!r} caps its events at {rule.max_duration_s:g}s, which at one "
                        f"frame every {interval:g}s is only {frames} frame(s) — below this Event "
                        f"timelapse action's {min_frames}-frame minimum, so it will never produce a "
                        f"video. Raise the rule's max (0 removes the cap entirely), shorten the "
                        f"interval, or lower the action's minimum."
                    )

        # An event is written to the Events view only where an in-scope action writes
        # something to it — see `Action.logs_event`. A rule with no such action still
        # runs, still publishes and still shows in the task log; it just leaves no row.
        # That is usually deliberate ("a cat is present" would otherwise file hundreds of
        # empty events a day), which is why this is a warning and names the fix rather
        # than an error. Without it, a rule someone *did* want filed is indistinguishable
        # from one that never matched.
        logging_actions = [a for a in profile.actions if a.enabled and a.kind in EVENT_LOGGING_ACTION_KINDS]
        unfiled = [
            rule.name
            for rule in profile.rules
            if rule.enabled
            and not any(
                # Empty scope means every rule, as everywhere else.
                not a.rule_uuids_json or rule.uuid in {str(u) for u in a.rule_uuids_json}
                for a in logging_actions
            )
        ]
        if unfiled:
            named = ", ".join(f"{name!r}" for name in unfiled[:4])
            more = f" (and {len(unfiled) - 4} more)" if len(unfiled) > 4 else ""
            warnings.append(
                f"Nothing files events for {named}{more}: no clip, event timelapse or Save event "
                "action covers these rules, so their matches will publish and log but will not "
                "appear in Events. Add a Save event action if you want them listed."
            )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    async def list_avail_trackers(self) -> List[Dict[str, Any]]:
        """Registered trackers, with the schema the Watchers pane renders their params from.

        Returned in one call rather than the list-then-fetch-schema pair the image
        providers use: there are a handful of trackers and their schemas are a few
        fields each, so a second round trip per tracker buys nothing.
        """
        return [
            {
                "typename": typename,
                "display_name": cls.display_name(),
                "params_schema": cls.params_schema(),
            }
            for typename, cls in sorted(tracker_registry.items())
        ]

    async def list_avail_actions(self) -> List[Dict[str, Any]]:
        """Registered actions, with the schema the Watchers pane renders their params from.

        The actions registry has always carried `params_schema()` — its docstring even
        claims it "drives the UI form automatically" — but nothing ever served it, so the
        pane grew a hand-written form per action kind instead. The two lists then had no
        way to disagree out loud: a parameter added to an action simply never appeared,
        and one removed from it kept being written by the form.

        Same shape and same reasoning as `list_avail_trackers`: one call rather than
        list-then-fetch-each, because there are a handful of actions and their schemas
        are a few fields each.
        """
        return [
            {
                "kind": kind,
                "display_name": cls.display_name(),
                "params_schema": cls.params_schema(),
            }
            for kind, cls in sorted(action_registry.items())
        ]

    # ── Watch events and recordings ────────────────────────────────────────────

    async def list_watch_events(self, **kwargs) -> Tuple[List[WatchEventRead], int]:
        return await self._db_client.list_watch_events(**kwargs)

    async def get_watch_event(self, uuid: str) -> Optional[WatchEventRead]:
        return await self._db_client.get_watch_event(uuid)

    async def watch_event_facets(self, profile_uuid: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        return await self._db_client.watch_event_facets(profile_uuid=profile_uuid)

    async def watch_event_counts(self) -> List[Dict[str, Any]]:
        return await self._db_client.watch_event_counts()

    async def watch_event_groups(self, **kwargs) -> List[Dict[str, Any]]:
        return await self._db_client.watch_event_groups(**kwargs)

    async def watch_event_histogram(self, **kwargs) -> Dict[str, Any]:
        return await self._db_client.watch_event_histogram(**kwargs)

    async def watch_event_sessions(self, **kwargs) -> Tuple[List[Dict[str, Any]], int]:
        return await self._db_client.watch_event_sessions(**kwargs)

    async def create_watch_event(self, event: WatchEvent) -> WatchEventRead:
        return await self._db_client.create_watch_event(event)

    async def update_watch_event(self, uuid: str, **fields) -> Optional[WatchEventRead]:
        return await self._db_client.update_watch_event(uuid, **fields)

    async def close_stranded_watch_events(self, profile_uuid: str) -> int:
        return await self._db_client.close_stranded_watch_events(profile_uuid)

    async def list_event_blips(self, **kwargs) -> Tuple[List[EventBlipRead], int]:
        return await self._db_client.list_event_blips(**kwargs)

    async def replace_event_blips(
        self,
        event_uuid: str,
        blips: Sequence[Dict[str, Any]],
        recording_uuid: Optional[str] = None,
    ) -> List[EventBlipRead]:
        return await self._db_client.replace_event_blips(event_uuid, blips, recording_uuid=recording_uuid)

    async def set_event_blip_states(self, uuids: Sequence[str], state: str) -> int:
        return await self._db_client.set_event_blip_states(uuids, state)

    async def set_watch_event_review_states(self, uuids: Sequence[str], state: str) -> int:
        return await self._db_client.set_watch_event_review_states(uuids, state)

    async def set_watch_event_pinned(self, uuids: Sequence[str], pinned: bool) -> int:
        return await self._db_client.set_watch_event_pinned(uuids, pinned)

    async def expire_blips_for_recording(self, recording_uuid: str) -> int:
        return await self._db_client.expire_blips_for_recording(recording_uuid)

    async def watch_event_snapshot_file(self, uuid: str) -> Optional[Path]:
        """Resolve an event's card art to a file that actually exists.

        Returns None both when the event has no snapshot and when the path is stale —
        the caller cannot usefully tell those apart, and neither can be served.
        """
        path_str = await self._db_client.get_watch_event_snapshot_path(uuid)
        if not path_str:
            return None
        path = Path(path_str)
        return path if path.is_file() else None

    async def watch_event_crop_file(self, uuid: str) -> Optional[Path]:
        """Resolve an event's cropped card art to a file that actually exists.

        Same contract as `watch_event_snapshot_file`, including the deliberate blurring
        of "never had one" and "the path is stale" — the caller serves a 404 either way,
        and the card falls back to the wide shot for both.
        """
        path_str = await self._db_client.get_watch_event_crop_path(uuid)
        if not path_str:
            return None
        path = Path(path_str)
        return path if path.is_file() else None

    async def delete_watch_event(self, uuid: str) -> bool:
        """Delete an event, its card art, and every recording it owns.

        Every recording, not one: an event can own both a clip and a timelapse, and
        deleting only the first left the other's file on disk with a row pointing at a
        event that no longer exists — invisible in the UI and swept only by the orphan
        pass, an hour later at best.

        The snapshot and its crop are the event's own files (not curated training
        images), so they go with the row. A `save_snapshot` action's output lives under
        `image_datasets` and is deliberately left alone — the user curates that. So does
        a `save_timelapse` action's, which is not an event's property at all.

        Checks `pinned` before touching anything, not just before deleting the row.
        `db_client.delete_watch_event` carries the same check, but it runs *after* the
        event's recordings are already gone — checking only there would delete every
        clip and then refuse to delete the row that named them, leaving a "pinned"
        event with no video at all. Checking here first is what keeps the promise a pin
        makes atomic.
        """
        event = await self._db_client.get_watch_event(uuid)
        if event is None:
            return False
        if event.pinned:
            raise WatchEventPinnedError(f"Event {uuid} is pinned — unpin it before deleting")
        # Both paths are read before the row goes: afterwards there is nothing left to
        # ask, and a crop whose row was deleted first is an orphan the sweep has to find
        # an hour later.
        images = [
            await self._db_client.get_watch_event_snapshot_path(uuid),
            await self._db_client.get_watch_event_crop_path(uuid),
        ]
        for recording_uuid in await self._db_client.recording_uuids_for_event(uuid):
            await self.delete_recording(recording_uuid)
        deleted = await self._db_client.delete_watch_event(uuid)
        if deleted:
            for path_str in images:
                if not path_str:
                    continue
                try:
                    Path(path_str).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("Could not remove event image %s: %s", path_str, e)
        return deleted

    async def delete_watch_events(self, uuids: Sequence[str]) -> int:
        """Delete a selection of events, and report how many actually went.

        One failure does not abandon the rest: the caller selected N cards and pressed
        delete once, and leaving an arbitrary suffix of that selection alive because the
        third event's clip file was locked is worse than deleting what can be deleted
        and saying so. A uuid that no longer exists is not a failure — it is the normal
        result of deleting the same selection twice.

        A pinned event is the same story, not an error: the UI is expected to keep
        pinned events out of a bulk selection in the first place (see the events page),
        so hitting one here is normally a race — pinned from another tab after the
        selection was made — rather than something worth an exception-level log line.
        """
        deleted = 0
        for uuid in uuids:
            try:
                if await self.delete_watch_event(uuid):
                    deleted += 1
            except WatchEventPinnedError:
                logger.info("Skipping delete of pinned watch event %s", uuid)
            except Exception:
                logger.exception("Could not delete watch event %s", uuid)
        return deleted

    async def list_recordings(self, **kwargs) -> List[RecordingRead]:
        return await self._db_client.list_recordings(**kwargs)

    async def get_recording(self, uuid: str) -> Optional[RecordingRead]:
        return await self._db_client.get_recording(uuid)

    async def create_recording(self, recording: Recording) -> RecordingRead:
        return await self._db_client.create_recording(recording)

    async def update_recording(self, uuid: str, **fields) -> Optional[RecordingRead]:
        return await self._db_client.update_recording(uuid, **fields)

    async def delete_recording(self, uuid: str) -> bool:
        """Delete a recording row and the clip file it points at.

        The file is removed first-effort but a failure there must not leave an
        undeletable row: the row is the index, the file is derived from it.

        Any tracking blips found in this clip are expired here rather than deleted —
        the finding outlives the video, but it stops offering itself for review, because
        harvesting frames out of a clip that is gone cannot work and its "play the clip"
        button would open a dead player. This is the single choke point every deletion
        path funnels through (the age sweep, the size cap, the reconcile of hand-deleted
        files, and the DELETE endpoint), which is why the expiry belongs here rather
        than in each of them.

        Checks the owning event's `pinned` flag before touching the file, not just
        before deleting the row. `db_client.delete_recording` carries the same check,
        but it runs last — checking only there would unlink the clip and then refuse
        to delete the now-pointless row, which is a worse outcome than doing nothing.
        A recording with no event, or whose event has already been deleted, has
        nothing to check and is unaffected.
        """
        recording = await self._db_client.get_recording(uuid)
        if recording is None:
            return False
        if recording.event_uuid:
            owner = await self._db_client.get_watch_event(recording.event_uuid)
            if owner is not None and owner.pinned:
                raise WatchEventPinnedError(
                    f"Recording {uuid} belongs to pinned event {recording.event_uuid} — unpin it before deleting"
                )
        for path_str in (recording.file_path, recording.thumbnail_path, recording.overlay_path):
            if not path_str:
                continue
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Could not delete recording file %s: %s", path_str, e)
        # Before the row goes: `expire_blips_for_recording` is keyed on the uuid, which
        # is still valid either way, but ordering the two makes the intent legible — the
        # blips are updated while the thing they describe still exists.
        try:
            expired = await self._db_client.expire_blips_for_recording(uuid)
            if expired:
                logger.info("Expired %s tracking blip(s) whose clip %s was deleted", expired, uuid)
        except Exception:
            # A blip is a note about a video; failing to annotate it must never be the
            # reason a clip stays on a disk that is trying to free space.
            logger.exception("Could not expire tracking blips for recording %s", uuid)
        return await self._db_client.delete_recording(uuid)

    # ── Retention ──────────────────────────────────────────────────────────────

    async def get_recording_retention_days(self) -> int:
        """How long event clips and card art are kept, in days.

        Stored as a plain string in `settings` (the table holds no other type). Anything
        unparseable falls back to the default rather than raising: this runs from the
        worker loop, and a hand-edited row must not be able to stop the sweep — a stuck
        sweep is a disk that fills up silently, which is the failure this exists to
        prevent. Zero or negative means "keep everything", so there is a way to opt out
        that does not involve deleting the setting and guessing at the default.
        """
        return await self._numeric_setting(RECORDING_RETENTION_SETTING_KEY, DEFAULT_RECORDING_RETENTION_DAYS, "days")

    async def get_recording_max_total_mb(self) -> int:
        """Ceiling on total recording bytes, in MB. See `get_recording_retention_days`
        for why an unreadable value falls back rather than raising."""
        return await self._numeric_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, DEFAULT_RECORDING_MAX_TOTAL_MB, "MB")

    def _scan_recording_files(self) -> List[Tuple[Path, int, float]]:
        """Every file under the recordings roots, with size and mtime. Blocking.

        Only the managed roots are walked, and that limit is about *deletion*: this is
        what the orphan sweep deletes from, and it deletes by discovery — anything here
        with no row pointing at it goes. Walking wherever a param happens to point would
        turn that into deleting files in a directory the user chose for their own
        reasons, which is the failure mode that makes directory sweeps dangerous.

        A clip written to a hand-set absolute `output_dir` is therefore not swept. It is
        still *counted* toward the size cap, and still deleted when its row goes — see
        `_recording_disk_entries` for why following a row is a different act from
        walking a directory.
        """
        entries: List[Tuple[Path, int, float]] = []
        for root in self._recording_roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                try:
                    if not path.is_file():
                        continue
                    stat = path.stat()
                except OSError:
                    # Raced with another deleter, or unreadable. Either way it is not
                    # this sweep's business.
                    continue
                entries.append((path, stat.st_size, stat.st_mtime))
        return entries

    @staticmethod
    def _stat_external_recordings(referenced: Set[str], roots: Tuple[str, ...]) -> List[Tuple[Path, int, float]]:
        """Stat the row-referenced files that lie outside the roots. Blocking.

        Reached by following rows, never by walking. That is the whole distinction: a
        path in `referenced` names a file this application wrote and still has an index
        entry for, so its provenance is known. A file found by walking a directory the
        user chose has no provenance at all.
        """
        entries: List[Tuple[Path, int, float]] = []
        for raw in referenced:
            real = os.path.realpath(raw)
            # Already counted by the walk. Compared with a separator appended so that a
            # sibling root like `/data/recordings/clips-old` is not read as being inside
            # `/data/recordings/clips`.
            if any(real == root or real.startswith(root + os.sep) for root in roots):
                continue
            try:
                stat = os.stat(real)
            except OSError:
                # A row naming a file that is gone, unreadable, or on an unmounted share.
                # It contributes no bytes, which is exactly what the caller needs.
                continue
            entries.append((Path(real), stat.st_size, stat.st_mtime))
        return entries

    async def _recording_disk_entries(self) -> List[Tuple[Path, int, float]]:
        """Every file the size cap is accountable for, with size and mtime.

        The union of two sets that are gathered in deliberately different ways:

        * everything under the managed roots, found by walking — this is what the orphan
          sweep may also delete, so it has to include files no row names; and
        * everything a surviving row points at from *outside* those roots, found by
          following the rows.

        The second half exists because accounting scope has to match deletion scope. An
        action with an absolute `output_dir` writes outside the roots, and age pruning
        already deletes those files — it works through the row, which unlinks whatever
        `file_path` names, wherever that is. Measuring by walking alone left the size cap
        blind to a file it was perfectly willing to delete: it could never push the total
        over the cap, so a camera writing to a hand-set directory filled the disk while
        the sweep logged that it was under its bound.

        What does *not* change is that nothing walks a directory the user chose. Their
        own files, sitting beside ours in that directory, are neither counted nor
        touched — following a row to a file we wrote is a different act from discovering
        files by scanning, and only the second one can eat data that was never ours.
        """
        entries = await asyncio.to_thread(self._scan_recording_files)
        referenced = await self._db_client.referenced_recording_paths()
        roots = tuple(os.path.realpath(root) for root in self._recording_roots)
        entries += await asyncio.to_thread(self._stat_external_recordings, referenced, roots)
        return entries

    async def sweep_orphan_recording_files(self, now_ts: Optional[float] = None) -> Dict[str, int]:
        """Delete files under the recordings roots that no row points at.

        Rows are deleted through paths that unlink their own files, but a row can also
        vanish without passing through one — a watch profile deleted while events hang
        off it cascades in the database and leaves their snapshots behind, and a crashed
        ffmpeg leaves a `.part` file no row ever referenced. Nothing else ever looks at
        those files again, so without this they are permanent.

        Two rails, because a sweep that walks a directory and deletes is exactly the kind
        of thing that eats live data when it is subtly wrong:

        * every path any surviving row names is exempt, compared on the real path so a
          symlinked or non-normalised stored path still matches; and
        * anything younger than the grace period is exempt, because a clip exists on
          disk before the row referencing it is written — without this the sweep races
          clip extraction and deletes its output mid-write.
        """
        now_ts = time.time() if now_ts is None else now_ts
        referenced = await self._db_client.referenced_recording_paths()
        # Normalised on both sides: a stored path and a walked path can name the same
        # file in different spellings, and treating a live file as an orphan deletes it.
        safe = {os.path.realpath(p) for p in referenced}
        entries = await asyncio.to_thread(self._scan_recording_files)

        removed = 0
        freed = 0
        for path, size, mtime in entries:
            if os.path.realpath(path) in safe:
                continue
            if now_ts - mtime < RECORDINGS_ORPHAN_GRACE_S:
                continue
            try:
                await asyncio.to_thread(path.unlink)
            except OSError as e:
                logger.warning("Could not remove orphaned recording file %s: %s", path, e)
                continue
            removed += 1
            freed += size

        if removed:
            logger.info("Retention sweep removed %s orphaned file(s), freeing %.1f MB", removed, freed / 1048576)
        return {"files": removed, "bytes": freed}

    @staticmethod
    def _path_size(path: Optional[str], size_by_path: Dict[str, int]) -> int:
        """Bytes *path* occupies, per the walk. 0 when it lies outside the managed roots.

        Zero is the safe answer rather than a stat: the map is the same measurement the
        running total was built from, so a path missing from it contributed nothing to
        that total and crediting eviction with its bytes would make the loop stop before
        the cap was actually met. A miss now means a genuinely unmeasured file — the map
        covers row-referenced files outside the roots too.
        """
        if not path:
            return 0
        return size_by_path.get(os.path.realpath(path), 0)

    async def _event_disk_usage(self, uuid: str, size_by_path: Dict[str, int]) -> int:
        """Bytes one event will free: its two images plus every recording it owns.

        Undercounting here is not cosmetic — eviction decrements a running total by this
        figure and stops when it is back under the cap, so missing an event's second
        recording makes the sweep believe it freed less than it did and evict further
        than it needed to.
        """
        total = self._path_size(await self._db_client.get_watch_event_snapshot_path(uuid), size_by_path)
        total += self._path_size(await self._db_client.get_watch_event_crop_path(uuid), size_by_path)
        for recording_uuid in await self._db_client.recording_uuids_for_event(uuid):
            recording = await self._db_client.get_recording(recording_uuid)
            total += self._path_size(getattr(recording, "file_path", None), size_by_path)
            total += self._path_size(getattr(recording, "thumbnail_path", None), size_by_path)
            total += self._path_size(getattr(recording, "overlay_path", None), size_by_path)
        return total

    async def enforce_recording_size_cap(self) -> Dict[str, int]:
        """Delete the oldest events until the recordings roots fit under the cap.

        Oldest-first because an NVR's value decays with age — the same order the time
        bound already imposes, so the two agree rather than fighting.

        Eviction stops the moment the total is back under the cap, which means the
        running total has to be decremented *per deletion* rather than re-measured per
        batch. Deleting a whole batch and then re-walking overshoots badly: with a batch
        of fifty and a cap one clip over, it would evict fifty events to reclaim one.

        The decrement is looked up in a path→size map built from the walk, so it reflects
        the bytes actually on disk — a clip's recorded `size_bytes` can disagree with its
        file, and snapshots carry no size column at all. The reported total is re-measured
        at the end rather than accumulated, so it stays honest even if a lookup missed.
        """
        max_mb = await self.get_recording_max_total_mb()
        if max_mb <= 0:
            return {"events": 0, "recordings": 0, "bytes": 0}
        max_bytes = max_mb * 1024 * 1024

        entries = await self._recording_disk_entries()
        initial_total = sum(size for _p, size, _m in entries)
        if initial_total <= max_bytes:
            return {"events": 0, "recordings": 0, "bytes": 0}
        size_by_path = {os.path.realpath(path): size for path, size, _m in entries}
        total = initial_total

        events = 0
        recordings = 0
        for _ in range(SIZE_EVICTION_MAX_PASSES):
            if total <= max_bytes:
                break
            deleted_this_pass = 0
            now = datetime.now(timezone.utc)

            for uuid in await self._db_client.watch_event_uuids_before(now, limit=SIZE_EVICTION_BATCH):
                if total <= max_bytes:
                    break
                freed = await self._event_disk_usage(uuid, size_by_path)
                try:
                    if await self.delete_watch_event(uuid):
                        events += 1
                        deleted_this_pass += 1
                        total -= freed
                except Exception:
                    logger.exception("Size eviction could not delete watch event %s", uuid)

            if total > max_bytes and not deleted_this_pass:
                # No events left to shed, so the bytes belong to recordings with no event.
                for uuid in await self._db_client.recording_uuids_before(now, limit=SIZE_EVICTION_BATCH):
                    if total <= max_bytes:
                        break
                    recording = await self._db_client.get_recording(uuid)
                    freed = self._path_size(getattr(recording, "file_path", None), size_by_path)
                    freed += self._path_size(getattr(recording, "thumbnail_path", None), size_by_path)
                    try:
                        if await self.delete_recording(uuid):
                            recordings += 1
                            deleted_this_pass += 1
                            total -= freed
                    except Exception:
                        logger.exception("Size eviction could not delete recording %s", uuid)

            if not deleted_this_pass:
                # Over the cap with nothing left that this sweep owns: files under the
                # roots that no row names and that are still inside the orphan grace
                # period. Saying so is the useful outcome; looping is not.
                logger.warning(
                    "Recordings occupy %.1f MB, over the %s MB cap, but no deletable rows "
                    "remain — the excess is files no row points at and younger than the "
                    "orphan grace period.",
                    total / 1048576,
                    max_mb,
                )
                break

        final = sum(size for _p, size, _m in await self._recording_disk_entries())
        freed = max(0, initial_total - final)
        if events or recordings:
            logger.info(
                "Size cap (%s MB): evicted %s event(s) and %s orphan recording(s), freeing %.1f MB",
                max_mb,
                events,
                recordings,
                freed / 1048576,
            )
        return {"events": events, "recordings": recordings, "bytes": freed}

    async def prune_recordings_by_age(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """Delete events and clips past the retention window. Returns what it removed.

        Events are swept first because `delete_watch_event` already cascades to the clip
        the event owns, so sweeping recordings first would delete the file and leave the
        event pointing at nothing for the rest of the pass. The second sweep then catches
        recordings with no event — a clip whose event was deleted by hand, or one written
        before the row that should have referenced it.

        Retention is measured from when the event *started*, not from when the row was
        written: a clip is interesting because of when it was filmed.
        """
        days = await self.get_recording_retention_days()
        if days <= 0:
            return {"events": 0, "recordings": 0}

        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        events = 0
        for uuid in await self._db_client.watch_event_uuids_before(cutoff):
            try:
                if await self.delete_watch_event(uuid):
                    events += 1
            except Exception:
                # One unreadable row must not stop the sweep; the disk is the point.
                logger.exception("Retention sweep could not delete watch event %s", uuid)

        recordings = 0
        for uuid in await self._db_client.recording_uuids_before(cutoff):
            try:
                if await self.delete_recording(uuid):
                    recordings += 1
            except Exception:
                logger.exception("Retention sweep could not delete recording %s", uuid)

        if events or recordings:
            logger.info(
                "Retention sweep (%s day window): removed %s event(s) and %s orphan recording(s)",
                days,
                events,
                recordings,
            )
        return {"events": events, "recordings": recordings}

    async def prune_expired_recordings(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """Run the whole retention pass: age, then orphans, then the size cap.

        The order is what makes the size cap cheap and correct. Age first sheds whatever
        is simply too old; the orphan sweep then reclaims files nothing points at, and
        those bytes count toward the cap — so a directory full of stranded snapshots is
        fixed by deleting *them* rather than by evicting events a user still wants.
        Only what is left over gets to force an eviction.

        Both bounds are independent, matching the task-instance precedent in
        `apps/__init__.py`: either can be switched off with 0 without disabling the other,
        because "how far back do I care?" and "how much disk may this take?" are
        different questions.
        """
        aged = await self.prune_recordings_by_age(now=now)
        orphans = await self.sweep_orphan_recording_files()
        capped = await self.enforce_recording_size_cap()
        return {
            "events": aged["events"] + capped["events"],
            "recordings": aged["recordings"] + capped["recordings"],
            "orphanFiles": orphans["files"],
            "bytes": orphans["bytes"] + capped["bytes"],
        }

    async def reconcile_missing_recording_files(self, now_ts: Optional[float] = None) -> Dict[str, int]:
        """Clear rows whose media file is no longer on disk.

        The exact inverse of `sweep_orphan_recording_files`, and needed for the same
        reason in the other direction: a user tidying up in a file manager deletes clips
        and snapshots without going through the app, and what is left is an event card
        with a player that will never load and a "last clip" link that 404s.

        A recording row is deleted outright — the row exists to index a file, and an
        index entry for a file nobody can produce again is worth nothing. An *event* is
        not: it is the record that something happened, which stays true after its
        pictures are gone, so only the path column is cleared and the event survives.
        Snapshot and crop are cleared independently — an event that lost one still shows
        the other, and the db client hands them over as a row each so this can tell them
        apart.

        Three exemptions, all of which stop this from eating live data:

        * rows still `pending`/`writing`, which are *supposed* to have no file yet;
        * anything inside the same grace period the orphan sweep uses, because a clip's
          row is written around the time its file appears and the two orderings vary; and
        * paths outside the managed roots, matching the rule the rest of retention
          follows — a clip written to a custom `output_dir` on a drive that happens to be
          unplugged must not lose its row over it.

        Plus a circuit breaker: if most rows look missing at once, that is far more
        likely to be an unmounted volume than a user who deleted almost everything, so
        the pass reports and does nothing. Stale rows are recoverable; deleted history
        is not.
        """
        now_ts = time.time() if now_ts is None else now_ts
        if not any(root.is_dir() for root in self._recording_roots):
            # No managed root on disk at all: nothing to compare against, and every row
            # would look missing. This is the unmounted case in its clearest form.
            logger.warning("Skipping recording reconcile: none of the managed roots exist")
            return {"recordings": 0, "snapshots": 0}

        roots = [os.path.realpath(root) for root in self._recording_roots]
        rows = await self._db_client.rows_naming_recording_files()

        candidates: List[Tuple[Dict[str, Any], List[str]]] = []
        for row in rows:
            if str(row.get("status") or "") in _RECORDING_IN_PROGRESS_STATUSES:
                continue
            at = row.get("at")
            if isinstance(at, datetime):
                age = now_ts - at.replace(tzinfo=at.tzinfo or timezone.utc).timestamp()
                if age < RECORDINGS_ORPHAN_GRACE_S:
                    continue
            managed = [p for p in row["paths"] if any(os.path.realpath(p).startswith(root + os.sep) for root in roots)]
            if managed:
                candidates.append((row, managed))

        # One hop to the thread pool for the whole stat pass, not one per path: this runs
        # on the worker loop, and a row-at-a-time await would put a context switch between
        # every stat for no benefit.
        def _existing(paths: Sequence[str]) -> Set[str]:
            return {p for p in paths if os.path.exists(p)}

        present = await asyncio.to_thread(_existing, [p for _row, managed in candidates for p in managed])

        considered = len(candidates)
        missing = [row for row, managed in candidates if not any(p in present for p in managed)]

        if not missing:
            return {"recordings": 0, "snapshots": 0}
        if considered >= RECONCILE_GUARD_MIN_ROWS and len(missing) / considered > RECONCILE_GUARD_MAX_FRACTION:
            logger.warning(
                "Skipping recording reconcile: %s of %s rows have no file, which looks like an "
                "unavailable volume rather than deleted media. Nothing was removed.",
                len(missing),
                considered,
            )
            return {"recordings": 0, "snapshots": 0}

        recordings = 0
        snapshots = 0
        for row in missing:
            try:
                if row["kind"] == "recording":
                    if await self.delete_recording(row["uuid"]):
                        recordings += 1
                elif row["kind"] == "event_crop":
                    # Only the crop's column: an event whose crop a user deleted still
                    # has its wide shot, and clearing both would blank a card that has a
                    # perfectly good picture to fall back to.
                    await self._db_client.update_watch_event(row["uuid"], crop_path=None)
                    snapshots += 1
                else:
                    await self._db_client.update_watch_event(row["uuid"], snapshot_path=None)
                    snapshots += 1
            except Exception:
                # One unreadable row must not stop the pass; the stale index is the point.
                logger.exception("Reconcile could not clear %s %s", row["kind"], row["uuid"])

        logger.info(
            "Recording reconcile: dropped %s row(s) for deleted clips and cleared %s missing event image(s)",
            recordings,
            snapshots,
        )
        return {"recordings": recordings, "snapshots": snapshots}

    # ── Timelapse retention ────────────────────────────────────────────────────

    async def get_timelapse_retention_days(self) -> int:
        """How many days of timelapse frames to keep. 0 or less keeps everything.

        Falls back on an unreadable value for the same reason the recording bounds do:
        this runs from the worker loop, and a stuck sweep is a disk that fills silently.
        """
        return await self._numeric_setting(TIMELAPSE_RETENTION_SETTING_KEY, DEFAULT_TIMELAPSE_RETENTION_DAYS, "days")

    async def get_timelapse_max_total_mb(self) -> int:
        """Ceiling on total timelapse bytes, in MB. 0 or less means no ceiling."""
        return await self._numeric_setting(TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY, DEFAULT_TIMELAPSE_MAX_TOTAL_MB, "MB")

    async def _numeric_setting(self, key: str, fallback: int, unit: str) -> int:
        raw = await self._db_client.get_setting(key)
        if raw is None or str(raw).strip() == "":
            return fallback
        try:
            return int(float(str(raw).strip()))
        except (TypeError, ValueError):
            logger.warning("Unreadable %s setting %r; falling back to %s %s", key, raw, fallback, unit)
            return fallback

    async def timelapse_roots(self) -> List[Path]:
        """Every directory a `save_timelapse` action writes into.

        Taken from the actions themselves rather than from a single configured root,
        because the directory is per-action: two watchers can keep their timelapses
        apart, and neither should have to know about a global setting to be swept.

        A **disabled** action still contributes its directory. Disabling stops new frames
        being written; it does not make the ones already on disk somebody else's problem,
        and the alternative is a folder that grows to the size the user set and then
        silently stops being managed. Deleting the action entirely does orphan its
        frames — the same trade the recording sweep makes for a custom `output_dir`, and
        for the same reason: nothing left in the database names the directory.

        A templated directory (`timelapse/{source}`) contributes its **literal prefix**,
        and the sweep then looks for day folders at any depth beneath it. Expanding the
        template against the current sources would look more precise and be worse: a
        camera renamed last month leaves a folder the expansion no longer names, whose
        frames would then never be swept. The prefix survives renames; the per-file shape
        rails in `_scan_timelapse_frames` are what keep the wider net safe.
        """
        roots: List[Path] = []
        seen: Set[str] = set()
        for profile in await self._db_client.list_watch_profiles():
            for action in profile.actions:
                if action.kind != TIMELAPSE_ACTION_KIND:
                    continue
                configured = (action.params_json or {}).get("output_dir") or ""
                try:
                    prefix = static_prefix(configured) if has_placeholder(configured) else configured
                    if not prefix:
                        # Every segment is a placeholder, so the only directory that would
                        # bound this sweep is FILES_ROOT itself. The action refuses to be
                        # built at all in that state; this is the matching refusal for a
                        # row saved before that check existed.
                        raise ValueError(f"output_dir {configured!r} names no fixed folder to sweep")
                    root = resolve_output_dir(prefix, DEFAULT_TIMELAPSE_DIR)
                except (ValueError, OSError) as e:
                    # A path that no longer resolves is a misconfiguration to report, not
                    # a reason to abandon the sweep for every other watcher.
                    logger.warning("Skipping timelapse root for action %s: %s", action.uuid, e)
                    continue
                key = str(root)
                if key not in seen:
                    seen.add(key)
                    roots.append(root)
        return roots

    def _scan_timelapse_frames(self, roots: Sequence[Path]) -> List[Tuple[Path, int, date]]:
        """Every deletable frame under the timelapse roots, with its size and its day.

        Blocking. Three rails decide what is deletable, and all three exist because these
        roots live under FILES_ROOT alongside a user's annotated images:

        * only files with a `YYYY-MM-DD` directory *somewhere* on the path below the root,
        * only files whose name matches what the action writes, and
        * never a file with a JSON sidecar beside it — that is the app's marker for an
          image someone has curated, and a timelapse frame deliberately has none.

        The first rail used to say "an immediate subdirectory", which stopped being true
        the moment the directory could be templated: `timelapse/{source}` files the day
        folder one level down, and `timelapse/{date}/{source}` puts it one level *up*
        from the frames. Both are the same question — "which day was this filmed?" — so
        the walk carries the nearest day folder it has passed through.

        The day comes from the directory name, not from mtime. The name records when the
        frame was filmed, which is the thing retention is about; mtime would make a
        restored backup or a `cp -r` look like today's footage.

        Depth is bounded. The net is wider than it was, and an `output_dir` pointed at a
        big tree should cost a bounded walk rather than a full recursive stat of it.
        """
        entries: List[Tuple[Path, int, date]] = []

        def walk(directory: Path, day: Optional[date], depth: int) -> None:
            if depth > TIMELAPSE_SCAN_MAX_DEPTH:
                return
            try:
                children = sorted(directory.iterdir())
            except OSError:
                return
            for child in children:
                try:
                    if child.is_dir():
                        parsed = day
                        try:
                            parsed = datetime.strptime(child.name, "%Y-%m-%d").date()
                        except ValueError:
                            pass
                        walk(child, parsed, depth + 1)
                        continue
                    # A frame outside any day folder is not one this sweep wrote, whatever
                    # its name looks like.
                    if day is None or not TIMELAPSE_FRAME_RE.match(child.name):
                        continue
                    if child.with_suffix(child.suffix + ".json").exists():
                        continue
                    size = child.stat().st_size
                except OSError:
                    # Raced with another deleter, or unreadable. Not this sweep's
                    # business either way.
                    continue
                entries.append((child, size, day))

        for root in roots:
            if root.is_dir():
                walk(root, None, 0)
        return entries

    def _delete_timelapse_frames(self, paths: Sequence[Path], roots: Sequence[Path]) -> Tuple[int, int]:
        """Unlink frames and tidy up the directories left empty. Blocking.

        Returns `(files, bytes)`. The empty-directory pass is what stops a year-old
        timelapse leaving 365 empty folders behind. It walks *upward* rather than
        removing only the frame's own folder, because a templated directory nests: with
        `timelapse/{source}`, emptying every day folder under a camera leaves the camera's
        folder behind too, and a camera that has been retired would keep its empty shell
        forever.

        `rmdir` only, and never above a root: a directory holding anything the sweep chose
        not to delete stops the climb by failing, which is exactly the behaviour wanted —
        and the root itself always survives, so the action has somewhere to write when the
        camera comes back.
        """
        files = 0
        freed = 0
        touched: Set[Path] = set()
        for path in paths:
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            files += 1
            freed += size
            touched.add(path.parent)

        stop = {root.resolve() for root in roots}
        # Deepest first, so a parent is only considered after its children have had their
        # chance to go — otherwise the parent is still non-empty when it is tried.
        for directory in sorted(touched, key=lambda d: len(d.parts), reverse=True):
            current = directory
            while current.resolve() not in stop and any(root in current.parents for root in stop):
                try:
                    current.rmdir()
                except OSError:
                    # Not empty, or gone already. Both end the climb.
                    break
                current = current.parent
        return files, freed

    async def prune_timelapse_by_age(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """Delete timelapse frames filmed before the retention window opens."""
        days = await self.get_timelapse_retention_days()
        if days <= 0:
            return {"files": 0, "bytes": 0}

        # Local dates, because the day directories are named in local time — comparing
        # them against a UTC "today" would shed a day early or late for most of the world.
        today = (now or datetime.now().astimezone()).date()
        cutoff = today - timedelta(days=days)

        roots = await self.timelapse_roots()
        entries = await asyncio.to_thread(self._scan_timelapse_frames, roots)
        expired = [path for path, _size, day in entries if day < cutoff]
        if not expired:
            return {"files": 0, "bytes": 0}

        files, freed = await asyncio.to_thread(self._delete_timelapse_frames, expired, roots)
        logger.info(
            "Timelapse retention (%s day window): removed %s frame(s), freeing %.1f MB",
            days,
            files,
            freed / 1048576,
        )
        return {"files": files, "bytes": freed}

    async def enforce_timelapse_size_cap(self) -> Dict[str, int]:
        """Delete the oldest timelapse frames until the roots fit under the cap.

        Oldest-first, and frame-by-frame rather than whole days: a timelapse degrades
        gracefully under partial deletion in a way an event clip does not, and evicting
        an entire day to reclaim a few MB would put a visible hole in the playback.
        """
        max_mb = await self.get_timelapse_max_total_mb()
        if max_mb <= 0:
            return {"files": 0, "bytes": 0}
        max_bytes = max_mb * 1024 * 1024

        roots = await self.timelapse_roots()
        entries = await asyncio.to_thread(self._scan_timelapse_frames, roots)
        total = sum(size for _p, size, _d in entries)
        if total <= max_bytes:
            return {"files": 0, "bytes": 0}

        # Sorted by (day, name) rather than by mtime, so the order matches the one the
        # age bound already imposes and the two bounds cannot disagree about what "oldest"
        # means. The name carries the time within the day.
        doomed: List[Path] = []
        for path, size, day in sorted(entries, key=lambda e: (e[2], e[0].name)):
            if total <= max_bytes:
                break
            doomed.append(path)
            total -= size

        files, freed = await asyncio.to_thread(self._delete_timelapse_frames, doomed, roots)
        logger.info("Timelapse size cap (%s MB): evicted %s frame(s), freeing %.1f MB", max_mb, files, freed / 1048576)
        return {"files": files, "bytes": freed}

    async def prune_expired_timelapse(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """Run both timelapse bounds: age first, then the size cap over what is left.

        Age first for the same reason the recording pass does it: whatever is simply too
        old should go before anything still inside the window is forced out, so the cap
        only ever evicts frames a user might otherwise have kept.
        """
        aged = await self.prune_timelapse_by_age(now=now)
        capped = await self.enforce_timelapse_size_cap()
        return {"files": aged["files"] + capped["files"], "bytes": aged["bytes"] + capped["bytes"]}

    # ── Storage accounting ─────────────────────────────────────────────────────

    async def get_storage_usage(self) -> Dict[str, Any]:
        """What the recording-side of the app is holding on disk, against its bounds.

        Every number here is measured the way the sweep that deletes it measures —
        `_recording_disk_entries` and `_scan_timelapse_frames` are the same calls
        `enforce_*_size_cap` uses. That is the point: a figure produced by a
        different walk would disagree with the cap in exactly the cases that matter
        (a clip written to a custom `output_dir`, a folder full of orphans), and a
        readout that disagrees with the enforcement is worse than none.

        Deliberately three separate buckets rather than one total. They are bounded
        independently, they grow for unrelated reasons, and the useful question is
        always "which of these is the one filling the disk?".

        Walks directories, so it is a request the caller asks for — never something
        on a polling path. `/status` in particular must not grow this.
        """
        recording_entries = await self._recording_disk_entries()
        recording_bytes = sum(size for _p, size, _m in recording_entries)

        timelapse_roots = await self.timelapse_roots()
        timelapse_entries = await asyncio.to_thread(self._scan_timelapse_frames, timelapse_roots)
        timelapse_bytes = sum(size for _p, size, _d in timelapse_entries)

        thumb_files, thumb_bytes = await asyncio.to_thread(self._thumb_cache_usage)

        return {
            "recordings": {
                "bytes": recording_bytes,
                "files": len(recording_entries),
                "maxBytes": (await self.get_recording_max_total_mb()) * 1024 * 1024,
                "retentionDays": await self.get_recording_retention_days(),
                "roots": [str(root) for root in self._recording_roots],
            },
            "timelapse": {
                "bytes": timelapse_bytes,
                "files": len(timelapse_entries),
                "maxBytes": (await self.get_timelapse_max_total_mb()) * 1024 * 1024,
                "retentionDays": await self.get_timelapse_retention_days(),
                "roots": [str(root) for root in timelapse_roots],
            },
            "thumbnailCache": {
                "bytes": thumb_bytes,
                "files": thumb_files,
                "maxBytes": THUMB_CACHE_MAX_MB * 1024 * 1024,
                "dir": str(self._thumb_cache_dir()),
            },
        }

    def _thumb_cache_usage(self) -> Tuple[int, int]:
        """`(files, bytes)` in the thumbnail cache. Blocking."""
        cache_dir = self._thumb_cache_dir()
        files = 0
        total = 0
        if not cache_dir.is_dir():
            return (0, 0)
        for path in self._thumb_cache_files(cache_dir):
            try:
                total += path.stat().st_size
                files += 1
            except OSError:
                continue
        return (files, total)

    async def clear_thumbnail_cache(self) -> Dict[str, int]:
        """Delete every cached thumbnail.

        The one purge here that costs nothing but CPU: thumbnails are derived from
        images that are still on disk and are regenerated on the next request. It is
        also the honest fix for a cache that has gone wrong — a stale entry, a
        half-written png — where the alternative is hunting for the one bad file.
        """

        def _remove() -> Dict[str, int]:
            cache_dir = self._thumb_cache_dir()
            files = 0
            freed = 0
            if not cache_dir.is_dir():
                return {"files": 0, "bytes": 0}
            for path in Manager._thumb_cache_files(cache_dir):
                try:
                    size = path.stat().st_size
                    path.unlink()
                except OSError:
                    continue
                files += 1
                freed += size
            return {"files": files, "bytes": freed}

        result = await asyncio.to_thread(_remove)
        logger.info("Cleared thumbnail cache: %s file(s), %.1f MB", result["files"], result["bytes"] / 1048576)
        return result

    async def purge_all_timelapse_frames(self) -> Dict[str, int]:
        """Delete every timelapse frame under every `save_timelapse` output folder.

        Separate from the bounded sweeps rather than "set the cap to 1 byte", because
        a bound is a policy and this is an act: someone reclaiming space now should
        not have to change a setting they then have to remember to change back.

        Scope is exactly `_scan_timelapse_frames` — day folders, matching filenames,
        no JSON sidecar — which is what makes this safe to expose. It cannot touch a
        curated image even when one is sitting in the same directory tree.
        """
        roots = await self.timelapse_roots()
        entries = await asyncio.to_thread(self._scan_timelapse_frames, roots)
        if not entries:
            return {"files": 0, "bytes": 0}
        files, freed = await asyncio.to_thread(self._delete_timelapse_frames, [p for p, _s, _d in entries], roots)
        logger.info("Purged all timelapse frames: %s frame(s), %.1f MB", files, freed / 1048576)
        return {"files": files, "bytes": freed}

    # ── Model groups ───────────────────────────────────────────────────────────

    async def list_model_groups(self) -> List[ModelGroupRead]:
        return await self._db_client.list_model_groups()

    async def get_model_group(self, uuid: str) -> Optional[ModelGroupRead]:
        return await self._db_client.get_model_group(uuid)

    async def create_model_group(self, data: ModelGroupCreate) -> ModelGroupRead:
        return await self._db_client.create_model_group(data)

    async def update_model_group(self, uuid: str, data: ModelGroupUpdate) -> Optional[ModelGroupRead]:
        return await self._db_client.update_model_group(uuid, data)

    async def delete_model_group(self, uuid: str) -> bool:
        return await self._db_client.delete_model_group(uuid)

    # ── Training configs ──────────────────────────────────────────────────────

    async def create_training_config(self, data) -> "TrainingConfigRead":
        return await self._db_client.create_training_config(data)

    async def list_training_configs(self) -> "list[TrainingConfigRead]":
        return await self._db_client.list_training_configs()

    async def get_training_config(self, uuid: str) -> "Optional[TrainingConfigRead]":
        return await self._db_client.get_training_config(uuid)

    async def update_training_config(self, uuid: str, data) -> "Optional[TrainingConfigRead]":
        return await self._db_client.update_training_config(uuid, data)

    async def delete_training_config(self, uuid: str) -> bool:
        return await self._db_client.delete_training_config(uuid)

    async def get_training_config_readiness(
        self, uuid: str, restrict_to_image_uuids: "Optional[list[str]]" = None
    ) -> "Optional[dict]":
        return await self._db_client.get_training_config_readiness(
            uuid, restrict_to_image_uuids=restrict_to_image_uuids
        )

    async def get_training_config_detection_dataset(self, uuid: str) -> "Optional[dict]":
        result = await self._db_client.get_training_config_detection_dataset(uuid)
        if result is None:
            return None
        tc = await self._db_client.get_training_config(uuid)
        hyperparams = dict(tc.hyperparams_json or {}) if tc is not None else {}
        return await self._apply_negative_ratio_cap(
            result, hyperparams.get("negative_ratio"), hyperparams.get("negative_ratio_floor")
        )

    async def get_dataset_image_uuids(self, dataset_uuids: list[str]) -> list[str]:
        return await self._db_client.get_dataset_image_uuids(dataset_uuids)

    async def get_training_config_label_stats(
        self, uuid: str, restrict_to_image_uuids: "Optional[list[str]]" = None
    ) -> "Optional[dict]":
        return await self._db_client.get_training_config_label_stats(
            uuid, restrict_to_image_uuids=restrict_to_image_uuids
        )

    async def get_training_config_needs_review_images(
        self, uuid: str, payload: "NeedsReviewImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_needs_review_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
            extra_label_uuids=payload.label_uuids or None,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_ready_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_ready_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_ready_positive_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_ready_positive_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_ready_negative_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_ready_negative_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_partial_review_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_partial_review_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_not_started_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_not_started_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_image_gaps(self, uuid: str, image_uuids: "list[str]") -> "dict[str, list[dict]]":
        return await self._db_client.get_training_config_image_gaps(uuid, image_uuids)

    async def get_training_config_images_by_label_status(
        self, uuid: str, payload: "ImagesByLabelStatusPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_images_by_label_status(
            uuid,
            payload.label_uuid,
            payload.status,
            rollup_label_uuids=payload.rollup_label_uuids,
            missing_label_uuid=payload.missing_label_uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    # ── Preview runs ───────────────────────────────────────────────────────

    async def create_preview_run(self, payload: "PreviewRunCreate") -> PreviewRunRead:
        name = payload.name or f"{payload.model_name} — {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"
        run = await self._db_client.create_preview_run(
            name=name,
            model_name=payload.model_name,
            task_typename=payload.task_typename,
            params=payload.params,
            scope=payload.scope,
        )
        return PreviewRunRead.model_validate(run)

    async def list_preview_runs(self) -> List[PreviewRunRead]:
        """Sweeps expired runs first — the list is the only reliably-visited
        endpoint, so it is where the TTL gets enforced (§9 of the plan: a
        scheduler for a delete this cheap would be its own failure mode)."""
        try:
            await self._db_client.sweep_expired_preview_runs()
        except Exception:
            logger.exception("preview run TTL sweep failed; listing anyway")

        runs = await self._db_client.get_preview_runs()
        counts = await self._db_client.count_preview_decisions([r.uuid for r in runs])
        results = []
        for run in runs:
            read = PreviewRunRead.model_validate(run)
            read.decision_counts = counts.get(run.uuid, {})
            read.candidate_count = sum(read.decision_counts.values())
            results.append(read)
        return results

    async def get_preview_run(self, run_uuid: str) -> Optional[PreviewRunRead]:
        run = await self._db_client.get_preview_run(run_uuid)
        if run is None:
            return None
        read = PreviewRunRead.model_validate(run)
        read.decision_counts = (await self._db_client.count_preview_decisions([run_uuid])).get(run_uuid, {})
        read.candidate_count = sum(read.decision_counts.values())
        return read

    async def update_preview_run(self, run_uuid: str, payload: "PreviewRunUpdate") -> Optional[PreviewRunRead]:
        run = await self._db_client.update_preview_run(
            run_uuid, name=payload.name, pinned=payload.pinned, stats=payload.stats
        )
        return None if run is None else PreviewRunRead.model_validate(run)

    async def delete_preview_run(self, run_uuid: str) -> bool:
        return await self._db_client.delete_preview_runs(run_uuid) > 0

    async def add_preview_candidates(
        self,
        run_uuid: str,
        candidates: List["PreviewCandidateCreate"],
        conflict_policy: PreviewConflictPolicy = PreviewConflictPolicy.LAST,
    ) -> int:
        return await self._db_client.add_preview_candidates(run_uuid, candidates, conflict_policy)

    async def _annotate_preview_staleness(self, candidates: List[PreviewCandidate]) -> List[PreviewCandidateRead]:
        """Reconcile a run's snapshot against the live rows.

        A saved run holds UUIDs captured when the model ran. By the time the user
        comes back the bbox may have been relabeled or deleted and the image may
        be gone, so this is read-time reconciliation, not stored state:

          - bbox row gone (or image gone, for an is_new candidate) → `stale`
          - bbox still there but its label moved off the snapshot →
            `label_changed_since_run`, which is still applicable but the user
            should see it before accepting.

        `stale` is written back so the counts in the saved-runs list agree with
        what the panel shows.
        """
        bbox_uuids = {c.bbox_uuid for c in candidates if c.bbox_uuid}
        # Fetched for *every* candidate, not just the boxless ones: the panel
        # builds each card's thumbnail URL from the path, and an absent key is
        # also how a vanished image is detected.
        image_uuids = {c.image_uuid for c in candidates if c.image_uuid}

        live_labels: Dict[str, Optional[str]] = {}
        if bbox_uuids:
            live_labels = await self._db_client.get_bbox_label_uuids(list(bbox_uuids))
        live_images: Dict[str, str] = {}
        if image_uuids:
            live_images = await self._db_client.get_image_filenames(list(image_uuids))

        results: List[PreviewCandidateRead] = []
        newly_stale: List[str] = []
        for c in candidates:
            read = PreviewCandidateRead.model_validate(c)
            filename = live_images.get(c.image_uuid)
            read.image_filename = self._image_rel_path(filename) if filename else ""
            vanished = (
                c.image_uuid not in live_images if (c.is_new or not c.bbox_uuid) else c.bbox_uuid not in live_labels
            )
            if vanished:
                # Already-applied candidates are history, not stale work — an
                # applied delete legitimately removed its own bbox.
                if c.decision != PreviewDecision.APPLIED:
                    read.decision = PreviewDecision.STALE
                    if c.decision != PreviewDecision.STALE:
                        newly_stale.append(c.uuid)
            elif c.bbox_uuid and c.current_label_uuid is not None:
                read.label_changed_since_run = live_labels.get(c.bbox_uuid) != c.current_label_uuid
            results.append(read)

        if newly_stale:
            run_uuid = candidates[0].run_uuid
            await self._db_client.update_preview_candidates(run_uuid, newly_stale, decision=PreviewDecision.STALE)

        return results

    async def query_preview_candidates(
        self, run_uuid: str, query: "PreviewCandidateQuery"
    ) -> Tuple[List[PreviewCandidateRead], int]:
        candidates, total = await self._db_client.query_preview_candidates(run_uuid, query)
        return await self._annotate_preview_staleness(candidates), total

    async def update_preview_candidates(self, run_uuid: str, payload: "PreviewCandidateBulkUpdate") -> int:
        return await self._db_client.update_preview_candidates(
            run_uuid,
            payload.candidate_uuids,
            decision=payload.decision,
            decision_action=payload.decision_action,
            proposed_label_uuid=payload.proposed_label_uuid,
            geometry=payload.geometry,
        )

    async def apply_preview_run(self, run_uuid: str, mark_verified: bool = False) -> PreviewApplyResult:
        """Apply every `accepted` candidate in one server-side pass.

        Doing this server-side (rather than the browser's per-group loop) is what
        makes closing the laptop mid-apply safe on a several-thousand-candidate
        run: each candidate is marked `applied` as it lands, so a re-run picks up
        where it stopped instead of re-applying everything.

        Semantics match the client loop it replaces exactly — relabels grouped by
        proposed label with the model's geometry replacing the original, new boxes
        created UNVERIFIED (the model's suggestion is not a human verification)
        unless `mark_verified`, then bbox thumbnails invalidated.
        """
        result = PreviewApplyResult()
        accepted = await self._db_client.get_preview_candidates(run_uuid, decisions=[PreviewDecision.ACCEPTED])
        if not accepted:
            return result

        # Re-check liveness *inside* the apply, not only on read: the run may
        # have been sitting in a tab for days. Never silently skip — the stale
        # count is surfaced in the toast.
        reconciled = await self._annotate_preview_staleness(accepted)
        stale_uuids = {r.uuid for r in reconciled if r.decision == PreviewDecision.STALE}
        result.stale = len(stale_uuids)
        live = [c for c in accepted if c.uuid not in stale_uuids]

        applied_uuids: List[str] = []
        touched_bbox_uuids: List[str] = []

        # ── Deletes ────────────────────────────────────────────────────────
        to_delete = [c for c in live if c.decision_action == PreviewDecisionAction.DELETE and c.bbox_uuid]
        if to_delete:
            modified, failed = await self.bulk_apply_bboxes_by_uuids(
                [c.bbox_uuid for c in to_delete], "delete", None, []
            )
            result.deleted += modified
            result.failed += failed
            applied_uuids.extend(c.uuid for c in to_delete)
            touched_bbox_uuids.extend(c.bbox_uuid for c in to_delete)

        # ── Relabels, grouped by proposed label ────────────────────────────
        relabels = [c for c in live if c.decision_action != PreviewDecisionAction.DELETE and not c.is_new]
        groups: Dict[str, List[PreviewCandidate]] = defaultdict(list)
        for c in relabels:
            if c.bbox_uuid:
                groups[c.proposed_label_uuid].append(c)
        for label_uuid, group in groups.items():
            bbox_uuids = [c.bbox_uuid for c in group]
            geometry = {c.bbox_uuid: c.geometry_json for c in group if c.geometry_json}
            modified, failed = await self.bulk_apply_bboxes_by_uuids(
                bbox_uuids,
                "relabel",
                label_uuid,
                [],
                bbox_geometry=geometry or None,
                mark_verified=mark_verified,
            )
            result.modified += modified
            result.failed += failed
            applied_uuids.extend(c.uuid for c in group)
            touched_bbox_uuids.extend(bbox_uuids)

        # ── New boxes ──────────────────────────────────────────────────────
        new_boxes = [c for c in live if c.is_new and c.decision_action != PreviewDecisionAction.DELETE]
        for c in new_boxes:
            geo = c.geometry_json or {}
            image = await self._db_client.get_image_by_uuid(c.image_uuid)
            if image is None or not geo:
                result.failed += 1
                continue
            try:
                bbox_uuid = await self.add_single_bbox(
                    image.filename,
                    c.proposed_label_uuid,
                    geo["x"],
                    geo["y"],
                    geo["width"],
                    geo["height"],
                    [VERIFIED_TAG_UUID if mark_verified else UNVERIFIED_TAG_UUID],
                    # `confidence` is the key CollectTrainingImagesTask (the
                    # highest-volume producer) writes and the one the canvas
                    # label and review card were built around.
                    # ApplyFilterChainTask writes `source_confidence` instead —
                    # readers accept both, see CONFIDENCE_KEYS in
                    # static/js/shared/domain/image/confidence.js.
                    metadata_json={
                        # Provenance travels from the candidate onto the box, because
                        # this is where it stops being recoverable. A candidate is
                        # swept by the preview-run TTL; the bbox it created is
                        # permanent training data. Dropping the keys here means a
                        # harvested batch that turns out to be wrong can no longer be
                        # found — `source_model` alone says "some blip harvest, ever",
                        # not *which* event or run, which is the granularity a bad
                        # batch actually needs.
                        #
                        # Splatted FIRST so the candidate's own columns below always
                        # win: _provenance_from matches by `source` *prefix*, so a
                        # producer that ever wrote `source_model` into its geometry
                        # would otherwise silently overwrite c.source_model here.
                        **_provenance_from(geo),
                        "run_model_added": True,
                        "source_model": c.source_model,
                        "confidence": c.confidence,
                    },
                )
            except Exception:
                logger.exception("apply_preview_run: creating new bbox on image %s failed", c.image_uuid)
                bbox_uuid = None
            if bbox_uuid is None:
                result.failed += 1
                continue
            result.added += 1
            applied_uuids.append(c.uuid)

        if applied_uuids:
            await self._db_client.update_preview_candidates(
                run_uuid,
                applied_uuids,
                decision=PreviewDecision.APPLIED,
                applied_at=datetime.now(timezone.utc),
            )
        if touched_bbox_uuids:
            try:
                await self.invalidate_bbox_thumbnails(touched_bbox_uuids)
            except Exception:
                logger.exception("apply_preview_run: thumbnail invalidation failed")

        return result
