"""Keep the previous generations of a trained model, so a bad retrain is undoable.

Before this existed, training a model wrote straight over ``<name>.onnx`` and the
weights it replaced were gone (``_training_job_runner._save_artifact`` logged
"Overwriting existing model artifact" and did exactly that). There was no history
and nothing to roll back to.

**Layout.** The live model stays exactly where every reader already expects it —
``<models>/<name>.onnx`` and its ``.json`` sidecar — and previous generations go
into a sibling directory:

    <models>/_versions/<name>/<version_id>/{<name>.onnx, <name>.json, ...}

Nothing that enumerates models is affected: both the inference server's
``glob("*.onnx")`` listing and its status page are non-recursive, so ``_versions``
is invisible to them. That invisibility is the point — versioning must not become
a second class of model that can be pinned, assigned labels, or trained into.

**Snapshots are copies, deliberately — not hardlinks.** A hardlink archives a
generation for free and survives the install path, which replaces the *directory
entry* by atomic rename. It does not survive an in-place write, and the models
dir has several: the upload-import path rewrites a sidecar, and mapping a class
to a label rewrites ``<name>.json`` where it lies. Sharing an inode means those
edits reach back and silently rewrite history, which was measurable in about two
minutes of trying it. Disk is the cheaper thing to spend: a copy costs one model
per kept generation, bounded by ``keep`` and reported in the UI, and it cannot be
corrupted by a writer that knows nothing about versioning.

**A version is one model's whole file set**, ``.onnx`` + ``.json`` sidecar +
``.onnx.data`` external tensors, because restoring weights without the sidecar
that names their classes produces a model that loads and lies.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from apps.helpers.metadataUtils import atomic_write_bytes

logger = logging.getLogger(__name__)

#: Subdirectory of the models dir holding every archived generation. Leading
#: underscore so it sorts away from real models and reads as "not a model".
VERSIONS_DIRNAME = "_versions"

#: The files that make up one model. Order matters on the way *in*: the ``.onnx``
#: is written last because the inference server keys its reload on that file's
#: mtime alone, so its dependencies must already be on disk when it moves.
MODEL_SUFFIXES = (".json", ".onnx.data", ".onnx")

#: Settings key (db_server ``user_settings``) holding how many previous
#: generations to keep per model. Read by tasks_server on install and mirrored in
#: the Settings UI; the inference server is told the value on the push rather than
#: reading it, because a remote inference server has no route to db_server.
MODEL_VERSIONS_KEEP_SETTING_KEY = "model_versions_keep"

#: How many previous generations to keep when nothing says otherwise. Three is a
#: working default: enough to step back past a bad run and the one before it,
#: while bounding a detector's history to a few hundred MB rather than the
#: unbounded growth this module replaced.
DEFAULT_VERSIONS_KEEP = 3

_VERSION_ID_FORMAT = "%Y%m%dT%H%M%SZ"

#: Separates a model name from a version id in a *preview* model — an archived
#: generation materialised as a real, loadable model so the comparison tools can
#: run it against the live one. ``@`` because no model this app writes contains it:
#: trained names come from a slugified config name, and imports go through
#: ``import_model_upload``.
PREVIEW_SEPARATOR = "@"

#: How long a preview may sit in the models dir before it is swept. It exists for
#: the length of one comparison run; anything older is the residue of a browser
#: that closed mid-run, and leaving it is a full-size model file nobody will ever
#: look for again.
PREVIEW_MAX_AGE_S = 6 * 3600


def versions_root(models_dir: Path) -> Path:
    return Path(models_dir) / VERSIONS_DIRNAME


def model_version_dir(models_dir: Path, model_name: str) -> Path:
    return versions_root(models_dir) / model_name


def live_model_files(models_dir: Path, model_name: str) -> List[Path]:
    """Existing on-disk files belonging to ``model_name``, in write order."""
    return [p for p in (Path(models_dir) / f"{model_name}{sfx}" for sfx in MODEL_SUFFIXES) if p.exists()]


def _new_version_id(parent: Path, when: Optional[datetime] = None) -> str:
    """A timestamp id, disambiguated if this second already has one.

    The format sorts lexicographically in chronological order, which is what lets
    every other function here order versions without stat-ing them.
    """
    base = (when or datetime.now(timezone.utc)).strftime(_VERSION_ID_FORMAT)
    if not (parent / base).exists():
        return base
    for n in range(2, 100):
        candidate = f"{base}-{n}"
        if not (parent / candidate).exists():
            return candidate
    raise RuntimeError(f"Could not allocate a version id under {parent}")


def _archive_file(src: Path, dst: Path) -> None:
    """Copy one model file into a version directory, preserving its mtime.

    A copy, not a link — see the module docstring. copy2 keeps st_mtime, which is
    what dates the generation.
    """
    shutil.copy2(src, dst)


def snapshot_current(models_dir: Path, model_name: str, *, when: Optional[datetime] = None) -> Optional[str]:
    """Archive the model currently on disk as a new version.

    Returns the new version id, or None when there is nothing to archive — the
    first training run of a model has no previous generation, and that is not an
    error.
    """
    files = live_model_files(models_dir, model_name)
    if not files:
        return None

    parent = model_version_dir(models_dir, model_name)
    parent.mkdir(parents=True, exist_ok=True)
    version_id = _new_version_id(parent, when)
    dest_dir = parent / version_id
    dest_dir.mkdir()

    entries = []
    for src in files:
        _archive_file(src, dest_dir / src.name)
        entries.append({"name": src.name, "bytes": src.stat().st_size})

    meta = {
        "versionId": version_id,
        "modelName": model_name,
        # When this generation was *archived*, i.e. when it stopped being current.
        "archivedAt": datetime.now(timezone.utc).isoformat(),
        # When the weights themselves were written — the closest thing to
        # "trained at" available without trusting the sidecar's contents.
        "createdAt": datetime.fromtimestamp(files[-1].stat().st_mtime, timezone.utc).isoformat(),
        "files": entries,
    }
    (dest_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    logger.info("Archived model %s as version %s", model_name, version_id)
    return version_id


def list_versions(models_dir: Path, model_name: str) -> List[Dict]:
    """Archived generations of ``model_name``, newest first."""
    parent = model_version_dir(models_dir, model_name)
    if not parent.is_dir():
        return []

    out: List[Dict] = []
    for version_dir in sorted(parent.iterdir(), reverse=True):
        if not version_dir.is_dir():
            continue
        meta = {}
        meta_path = version_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                # A version whose metadata is unreadable is still restorable —
                # the files are what matter, so report it rather than hide it.
                logger.warning("Unreadable version metadata at %s", meta_path)
        files = [p for p in version_dir.iterdir() if p.is_file() and p.name != "meta.json"]
        out.append(
            {
                "version_id": version_dir.name,
                "archived_at": meta.get("archivedAt"),
                "created_at": meta.get("createdAt"),
                "bytes": sum(p.stat().st_size for p in files),
                "files": sorted(p.name for p in files),
            }
        )
    return out


def prune_versions(models_dir: Path, model_name: str, keep: int) -> List[str]:
    """Delete all but the ``keep`` newest versions. Returns the ids removed."""
    keep = max(0, int(keep))
    versions = list_versions(models_dir, model_name)
    doomed = versions[keep:]
    removed = []
    for entry in doomed:
        version_dir = model_version_dir(models_dir, model_name) / entry["version_id"]
        try:
            shutil.rmtree(version_dir)
            removed.append(entry["version_id"])
        except OSError:
            logger.exception("Could not remove model version %s", version_dir)
    if removed:
        logger.info("Pruned %d old version(s) of %s: %s", len(removed), model_name, removed)
    return removed


def restore_version(models_dir: Path, model_name: str, version_id: str) -> Dict:
    """Make an archived generation the live model again.

    The generation being replaced is archived first, so a rollback is itself
    undoable — restoring is never a one-way door.

    Returns ``{"restored": version_id, "archived": <new id or None>, "files": [...]}``.

    Raises FileNotFoundError when the version does not exist.
    """
    version_dir = model_version_dir(models_dir, model_name) / version_id
    if not version_dir.is_dir():
        raise FileNotFoundError(f"No version {version_id!r} for model {model_name!r}")

    sources = {p.name: p for p in version_dir.iterdir() if p.is_file() and p.name != "meta.json"}
    if not any(name.endswith(".onnx") for name in sources):
        raise FileNotFoundError(f"Version {version_id!r} of {model_name!r} has no .onnx weights")

    archived = snapshot_current(models_dir, model_name)

    written = []
    # Same ordering rule as the install path: weights last, so the reload the
    # mtime triggers never sees a new .onnx beside an old sidecar.
    for suffix in MODEL_SUFFIXES:
        name = f"{model_name}{suffix}"
        src = sources.get(name)
        if src is None:
            continue
        atomic_write_bytes(Path(models_dir) / name, src.read_bytes())
        written.append(name)

    # A generation that had no external-tensor file must not inherit the current
    # one's: onnxruntime would load the restored graph against tensors from a
    # different model and fail somewhere far away from here.
    for stale in live_model_files(models_dir, model_name):
        if stale.name not in sources:
            stale.unlink(missing_ok=True)
            logger.info("Removed %s, which the restored version does not have", stale.name)

    logger.info("Restored model %s to version %s", model_name, version_id)
    return {"restored": version_id, "archived": archived, "files": written}


def delete_versions(models_dir: Path, model_name: str) -> int:
    """Drop a model's entire history. Returns how many versions were removed.

    Called when the model itself is deleted: history that outlives its model is
    unreachable disk — nothing lists it, and a later model reusing the name would
    inherit a stranger's weights as its rollback targets.
    """
    parent = model_version_dir(models_dir, model_name)
    if not parent.is_dir():
        return 0
    count = len(list_versions(models_dir, model_name))
    shutil.rmtree(parent, ignore_errors=True)
    return count


def versions_usage(models_dir: Path) -> Dict:
    """Disk held by archived versions across every model.

    Each inode is counted once and any inode a live model file still shares is
    skipped. Versions are copies today, so that dedup changes nothing — it is
    here so the number stays honest if an archive is ever linked rather than
    copied, rather than silently doubling.
    """
    root = versions_root(models_dir)
    result = {"bytes": 0, "versions": 0, "models": 0}
    if not root.is_dir():
        return result

    live_inodes = set()
    for path in Path(models_dir).glob("*"):
        try:
            if path.is_file():
                st = path.stat()
                live_inodes.add((st.st_dev, st.st_ino))
        except OSError:
            continue

    seen = set(live_inodes)
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        result["models"] += 1
        for version_dir in model_dir.iterdir():
            if not version_dir.is_dir():
                continue
            result["versions"] += 1
            for path in version_dir.iterdir():
                try:
                    st = path.stat()
                except OSError:
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                result["bytes"] += st.st_size
    return result


def install_with_history(models_dir: Path, model_names, keep: int = DEFAULT_VERSIONS_KEEP) -> Dict[str, Optional[str]]:
    """Archive the current generation of each named model and prune to ``keep``.

    Call it *before* writing new weights: it snapshots what is on disk now. The
    caller then writes the new files over the top, and the archived hardlinks keep
    the old bytes alive.
    """
    archived: Dict[str, Optional[str]] = {}
    for name in sorted(set(model_names)):
        try:
            archived[name] = snapshot_current(models_dir, name)
            # `keep` and not `keep + 1`: the generation just archived is already one
            # of the *previous* ones (the new weights have not landed yet), and
            # `keep` counts previous generations.
            prune_versions(models_dir, name, keep)
        except OSError:
            logger.exception("Could not archive current version of model %s", name)
            archived[name] = None
    return archived


# ---------------------------------------------------------------------------
# Version previews
#
# Comparing an archived generation against the live model means *running* it, and
# every path that runs a model — pin/unpin, the keep-warm refcount, the GPU unload
# timer, the API's `{model_name}` segment — is keyed by model name. So a version is
# compared by briefly becoming a model: its files are copied out under
# `<name>@<version_id>`, which loads and scores like any other model with no change
# to any of those paths, and is deleted when the comparison is done.
#
# Previews are deliberately kept out of the *offered* model list (the endpoint
# filters them) while staying in the available-models cache, so nothing can pick one
# as a watcher's model or a training target, but inference by name still resolves.
# ---------------------------------------------------------------------------


def preview_name(model_name: str, version_id: str) -> str:
    return f"{model_name}{PREVIEW_SEPARATOR}{version_id}"


def is_version_preview(name: str) -> bool:
    """Is this model name a materialised version rather than a real model?"""
    return PREVIEW_SEPARATOR in name


def split_preview_name(name: str) -> Optional[tuple]:
    """``("cats", "20260901T101500Z")`` for a preview name, else None."""
    if not is_version_preview(name):
        return None
    model_name, _, version_id = name.rpartition(PREVIEW_SEPARATOR)
    return (model_name, version_id) if model_name and version_id else None


def materialize_version(models_dir: Path, model_name: str, version_id: str) -> str:
    """Copy an archived generation out as a runnable model; return its name.

    Idempotent: an existing preview of the same version is reused rather than
    rewritten, so two comparisons of the same pair don't fight over the files.

    Raises FileNotFoundError when the version does not exist.
    """
    version_dir = model_version_dir(models_dir, model_name) / version_id
    if not version_dir.is_dir():
        raise FileNotFoundError(f"No version {version_id!r} for model {model_name!r}")

    name = preview_name(model_name, version_id)
    target = Path(models_dir) / f"{name}.onnx"
    if target.exists():
        return name

    sources = {p.name: p for p in version_dir.iterdir() if p.is_file() and p.name != "meta.json"}
    if f"{model_name}.onnx" not in sources:
        raise FileNotFoundError(f"Version {version_id!r} of {model_name!r} has no .onnx weights")

    # Weights last, same rule as everywhere else: a reader that spots the .onnx
    # must find its sidecar and external tensors already there.
    #
    # `copyfile`, not `copy2`: an archived generation carries the mtime it was
    # *trained* at, which is days or weeks old, and `sweep_stale_previews` dates a
    # preview by that mtime. Preserving it would make every preview born already
    # older than PREVIEW_MAX_AGE_S, so the next comparison's sweep would delete a
    # preview another comparison is still running against.
    for suffix in MODEL_SUFFIXES:
        src = sources.get(f"{model_name}{suffix}")
        if src is None:
            continue
        shutil.copyfile(src, Path(models_dir) / f"{name}{suffix}")

    logger.info("Materialised %s version %s as %s", model_name, version_id, name)
    return name


def delete_preview(models_dir: Path, name: str) -> bool:
    """Remove a materialised version. Returns False for a name that isn't one.

    The guard is the point: this is called from a cleanup path with whatever name
    the caller was using, and a bug there must not be able to delete a real model.
    """
    if not is_version_preview(name):
        return False
    removed = False
    for suffix in MODEL_SUFFIXES:
        path = Path(models_dir) / f"{name}{suffix}"
        if path.exists():
            path.unlink(missing_ok=True)
            removed = True
    return removed


def sweep_stale_previews(models_dir: Path, max_age_s: float = PREVIEW_MAX_AGE_S) -> List[str]:
    """Delete previews older than ``max_age_s``. Returns the names removed.

    A preview outlives its comparison only when the browser went away mid-run, so
    age is the only signal available — there is no session to key on.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_s
    removed = []
    for path in Path(models_dir).glob(f"*{PREVIEW_SEPARATOR}*.onnx"):
        name = path.name[: -len(".onnx")]
        if not is_version_preview(name):
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        if delete_preview(models_dir, name):
            removed.append(name)
    if removed:
        logger.info("Swept %d stale model version preview(s): %s", len(removed), removed)
    return removed
