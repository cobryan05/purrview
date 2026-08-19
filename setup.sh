#!/usr/bin/env bash
#
# WhiskerWatcher deployment configurator.
#
# Interactively asks what this host should run and writes a .env that
# `docker compose` reads automatically. It picks the right image base
# (slim CPU vs CUDA), onnxruntime flavor, compose profiles, GPU overlay,
# and inter-service wiring so you don't have to remember the build-arg matrix.
#
#   ./setup.sh            # interactive, writes ./.env
#   ./setup.sh && docker compose up -d --build
#
set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE=".env"

# ---- helpers --------------------------------------------------------------
ask() {  # ask "Question" default_yn  -> returns 0 for yes, 1 for no
  local prompt="$1" def="${2:-y}" reply
  local hint="[Y/n]"; [ "$def" = "n" ] && hint="[y/N]"
  read -r -p "$prompt $hint " reply || true
  reply="${reply:-$def}"
  case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

ask_str() {  # ask_str "Question" default -> echoes answer
  local prompt="$1" def="${2:-}" reply
  if [ -n "$def" ]; then read -r -p "$prompt [$def] " reply || true; echo "${reply:-$def}"
  else read -r -p "$prompt " reply || true; echo "$reply"; fi
}

# Ask for a docker size string (e.g. 512m, 2gb). Re-prompts until valid, so a
# typo is caught here rather than becoming an opaque compose error later.
ask_size() {  # ask_size "Question" default -> echoes answer
  local prompt="$1" def="$2" reply
  while true; do
    read -r -p "$prompt [$def] " reply >&2 || true
    reply="${reply:-$def}"
    if printf '%s' "$reply" | grep -Eiq '^[0-9]+[kmg]b?$'; then echo "$reply"; return 0; fi
    echo "  !! Expected a size like 512m, 1g or 2gb — got '$reply'." >&2
  done
}

# Ask for a bare integer (MB / days / seconds). Deliberately a *different*
# validator from ask_size: these fields must not accept a '100mb'-style suffix.
ask_int() {  # ask_int "Question" default -> echoes answer
  local prompt="$1" def="$2" reply
  while true; do
    read -r -p "$prompt [$def] " reply >&2 || true
    reply="${reply:-$def}"
    if printf '%s' "$reply" | grep -Eq '^[0-9]+$'; then echo "$reply"; return 0; fi
    echo "  !! Expected a plain number with no unit suffix — got '$reply'." >&2
  done
}

menu() {  # menu "Title" "opt1" "opt2" ... -> echoes chosen index (1-based) on stderr-free stdout
  local title="$1"; shift
  local i=1
  echo "$title" >&2
  for opt in "$@"; do echo "  $i) $opt" >&2; i=$((i+1)); done
  local reply
  read -r -p "Choose [1]: " reply >&2 || true
  echo "${reply:-1}"
}

# Ask whether to use the GPU for inference. Echoes "y" or "n". Training, when
# part of the deployment, always uses the GPU regardless of this answer.
#
# Always asked, and the question says whether a GPU was actually detected. It
# used to skip the prompt entirely on an undetected host and force "n", which
# made this the only role-affecting choice in the script with no override —
# every other "no GPU found" branch offers a "continue anyway". Detection is
# `nvidia-smi` on the *host*, and a machine whose containers can reach the GPU
# perfectly well may not have it on PATH, so a failed probe is not proof of
# anything. The detection result becomes the *default* instead of the verdict.
choose_gpu_inference() {
  local prompt def
  if [ "$GPU_DEFAULT" = "y" ]; then
    prompt="  NVIDIA GPU detected. Use it for inference (onnxruntime-gpu)?"
    def="y"
  else
    prompt="  No NVIDIA GPU detected. Build for GPU inference anyway (onnxruntime-gpu)?"
    def="n"
  fi
  if ask "$prompt" "$def"; then
    if [ "$GPU_DEFAULT" != "y" ]; then
      # The GPU build reserves an NVIDIA device, and reserving one that is not
      # there stops the container from starting at all — a failure that looks
      # nothing like "wrong build flavour" when you meet it later.
      echo "  !! Building for GPU inference on a host where none was detected." >&2
      echo "     The container will reserve an NVIDIA device and will fail to start" >&2
      echo "     unless Docker can actually provide one (NVIDIA Container Toolkit)." >&2
    fi
    echo "y"
  else
    # Said out loud rather than assumed silently: CPU-only is a supported
    # configuration, not a degraded fallback. The inference path is pure ONNX at
    # runtime, so plain onnxruntime gives working inference and the slim base
    # skips the CUDA layers entirely — a much smaller image on a box that could
    # not use them anyway.
    echo "  CPU-only inference (plain onnxruntime, slim base). Fully supported —" >&2
    echo "  inference runs on CPU and the image is far smaller." >&2
    echo "n"
  fi
}

# Resolve APP_BASE/ORT_REQUIREMENTS/COMPOSE_FILE (globals) from a GPU-inference
# y/n answer, for a topology where the app container is a separate, CPU-buildable
# service (split layouts): GPU inference pulls in the CUDA base + gpu overlay,
# CPU inference stays on the slim base with no overlay.
apply_inference_choice() {  # apply_inference_choice "$GI"
  if [ "$1" = "y" ]; then
    APP_BASE="cuda-runtime"
    ORT_REQUIREMENTS="requirements-gpu.txt"
    COMPOSE_FILE="docker-compose.yaml:docker-compose.gpu.yaml"
  else
    APP_BASE="slim-runtime"
    ORT_REQUIREMENTS="requirements-cpu.txt"
    COMPOSE_FILE=""
  fi
}

# Resolve just the onnxruntime flavor from a GPU-inference y/n answer, for a
# topology that must build on the CUDA base regardless (mono/dev-mono: torch
# needs it there anyway, so only the onnxruntime flavor follows the choice).
ort_requirements_for() {  # ort_requirements_for "$GI" -> echoes filename
  if [ "$1" = "y" ]; then echo "requirements-gpu.txt"; else echo "requirements-cpu.txt"; fi
}

# ---- detect environment ---------------------------------------------------
GPU_DEFAULT="n"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_DEFAULT="y"
fi
UID_VAL="$(id -u)"
GID_VAL="$(id -g)"

echo "=========================================================="
echo " WhiskerWatcher setup"
echo "=========================================================="
if [ "$GPU_DEFAULT" = "y" ]; then
  echo "Detected an NVIDIA GPU on this host."
else
  echo "No NVIDIA GPU detected on this host."
  echo "(Detection probes 'nvidia-smi' on the host — you will still be asked, and"
  echo " can override it, wherever the choice matters.)"
fi
echo

# ---- role of this host ----------------------------------------------------
ROLE="$(menu "What should this host run?" \
  "Everything (frontend + inference + training) on one box" \
  "Frontend + inference only (train elsewhere, or not at all)" \
  "Training only (a GPU node serving a separate frontend)" \
  "Development workspace (source-mounted, live code + dev tooling)")"
echo

# defaults
APP_BASE="slim-runtime"
ORT_REQUIREMENTS="requirements-cpu.txt"
COMPOSE_PROFILES=""
COMPOSE_FILE=""
TRAINING_SERVER_HOST=""

case "$ROLE" in
  1)  # Everything on one box — training needs a GPU; inference GPU is optional
    if [ "$GPU_DEFAULT" != "y" ]; then
      echo "!! Training needs an NVIDIA GPU, but none was detected."
      ask "   Continue anyway (build will target CUDA)?" n || { echo "Aborted."; exit 1; }
    fi
    LAYOUT="$(menu "How should services be laid out?" \
      "One container (mono) — simplest single-box deploy" \
      "Separate app + training containers (split)")"
    echo "(Training always uses the GPU. The next question is about inference only.)"
    GI="$(choose_gpu_inference)"
    if [ "$LAYOUT" = "2" ]; then
      # split: app and training are separate containers, so a CPU-inference
      # frontend can use the slim base and skip CUDA entirely.
      COMPOSE_PROFILES="app,training"
      apply_inference_choice "$GI"
    else
      # mono: one container, so it must be the CUDA base (torch needs it); only
      # the onnxruntime flavor follows the inference choice.
      COMPOSE_PROFILES="mono"
      APP_BASE="cuda-runtime"
      ORT_REQUIREMENTS="$(ort_requirements_for "$GI")"
    fi
    ;;

  2)  # Frontend + inference only
    apply_inference_choice "$(choose_gpu_inference)"
    COMPOSE_PROFILES="app"
    if ask "Point at a remote training server?" n; then
      TRAINING_SERVER_HOST="$(ask_str "  Training server hostname/IP:")"
    fi
    ;;

  3)  # Training only
    if [ "$GPU_DEFAULT" != "y" ]; then
      echo "!! Training needs an NVIDIA GPU, but none was detected."
      ask "   Continue anyway?" n || { echo "Aborted."; exit 1; }
    fi
    COMPOSE_PROFILES="training"
    # APP_BASE/ORT are irrelevant to the training image (built on the PyTorch
    # base), but keep them consistent for any shared build args.
    APP_BASE="cuda-runtime"
    ORT_REQUIREMENTS="requirements-gpu.txt"
    ;;

  4)  # Development workspace (source-mounted, dev tooling)
    if [ "$GPU_DEFAULT" != "y" ]; then
      echo "!! The dev workspaces assume an NVIDIA GPU (training-dev / mono-dev reserve one)."
      ask "   Continue anyway?" n || { echo "Aborted."; exit 1; }
    fi
    LAYOUT="$(menu "Dev layout?" \
      "Single container — everything + dev tooling" \
      "Separate app-dev + training-dev containers")"
    GI="$(choose_gpu_inference)"
    if [ "$LAYOUT" = "2" ]; then
      # split: app-dev is a separate container from training-dev, so a
      # CPU-inference app-dev can use the slim base and skip the GPU
      # reservation entirely (training-dev still always reserves one).
      COMPOSE_PROFILES="dev"
      apply_inference_choice "$GI"
    else
      # dev-mono: one container, so it must be the CUDA base (torch needs it);
      # mono-dev already reserves a GPU unconditionally in the base compose.
      COMPOSE_PROFILES="dev-mono"
      APP_BASE="cuda-runtime"
      ORT_REQUIREMENTS="$(ort_requirements_for "$GI")"
    fi
    ;;
esac

# ---- sizing (RAM + disk caches) -------------------------------------------
# Every knob has a working default in docker-compose.yaml, so the whole section
# is opt-in and the common path stays one question long. Each prompt is gated on
# whether this deployment actually contains the service that consumes it:
#   mediamtx runners (stream buffer, thumb cache) -> roles 1, 2, 4
#   training containers (shm, training cache)     -> roles 1, 3, 4
STREAM_BUFFER_SIZE=""
TRAINING_SHM_SIZE=""
THUMB_CACHE_MAX_MB=""
TRAINING_CACHE_MAX_MB=""
TRAINING_CACHE_MAX_AGE_DAYS=""
TRAINING_CACHE_SWEEP_INTERVAL_SEC=""

HAS_MEDIAMTX="y"; [ "$ROLE" = "3" ] && HAS_MEDIAMTX="n"
HAS_TRAINING="y"; [ "$ROLE" = "2" ] && HAS_TRAINING="n"

echo
if ask "Tune cache and buffer sizes? (defaults are fine for most deployments)" n; then
  echo
  if [ "$HAS_MEDIAMTX" = "y" ]; then
    echo "-- Stream rewind buffer (in-RAM, tmpfs /buffer) --"
    echo "   MediaMTX records each watched camera here so events can be cut with"
    echo "   pre-roll. It's a cap, not an allocation: an empty buffer costs ~0 RAM."
    echo "   Rule of thumb: ~30 MB per camera per 60 s of rewind at 4 Mbps."
    BUF_CAMS="$(ask_int "  How many cameras will this host watch?" 4)"
    BUF_SECS="$(ask_int "  How many seconds of rewind do you want?" 60)"
    # 4 Mbps = 0.5 MB/s; double it for headroom against bitrate spikes and
    # keyframe-aligned segments that overshoot the requested retention.
    BUF_SUGGEST=$(( (BUF_CAMS * BUF_SECS / 2) * 2 ))
    [ "$BUF_SUGGEST" -lt 128 ] && BUF_SUGGEST=128
    STREAM_BUFFER_SIZE="$(ask_size "  Buffer size:" "${BUF_SUGGEST}m")"
    echo
    echo "-- Thumbnail cache --"
    echo "   Disk cap in MB. NOTE: 0 disables the limit entirely (unbounded), it"
    echo "   does not mean 'no cache'."
    THUMB_CACHE_MAX_MB="$(ask_int "  Thumbnail cache MB:" 100)"
    echo
  fi
  if [ "$HAS_TRAINING" = "y" ]; then
    echo "-- Training shared memory --"
    echo "   The PyTorch DataLoader's /dev/shm budget. Too small a value surfaces"
    echo "   as a confusing disk-full-looking crash rather than an obvious OOM."
    TRAINING_SHM_SIZE="$(ask_size "  Shared memory size:" "2gb")"
    echo
    echo "-- Training image cache --"
    TRAINING_CACHE_MAX_MB="$(ask_int "  Cache size cap (MB):" 10000)"
    TRAINING_CACHE_MAX_AGE_DAYS="$(ask_int "  Evict entries older than (days):" 30)"
    TRAINING_CACHE_SWEEP_INTERVAL_SEC="$(ask_int "  Sweep interval (seconds):" 3600)"
    echo
  fi
fi

# ---- write .env -----------------------------------------------------------
if [ -f "$ENV_FILE" ]; then
  ask "Overwrite existing $ENV_FILE?" y || { echo "Kept existing $ENV_FILE. Nothing written."; exit 0; }
fi

{
  echo "# Generated by ./setup.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Re-run ./setup.sh to reconfigure, or edit by hand (see .env.example)."
  echo
  echo "COMPOSE_PROFILES=$COMPOSE_PROFILES"
  [ -n "$COMPOSE_FILE" ] && echo "COMPOSE_FILE=$COMPOSE_FILE"
  echo
  echo "APP_BASE=$APP_BASE"
  echo "ORT_REQUIREMENTS=$ORT_REQUIREMENTS"
  echo
  [ -n "$TRAINING_SERVER_HOST" ] && echo "TRAINING_SERVER_HOST=$TRAINING_SERVER_HOST"
  echo "APPUSER_UID=$UID_VAL"
  echo "APPUSER_GID=$GID_VAL"
  # Sizing knobs are only written when the user opted into tuning them; otherwise
  # the compose defaults apply, so an untuned .env stays as short as it was.
  echo
  [ -n "$STREAM_BUFFER_SIZE" ] && echo "STREAM_BUFFER_SIZE=$STREAM_BUFFER_SIZE"
  [ -n "$THUMB_CACHE_MAX_MB" ] && echo "THUMB_CACHE_MAX_MB=$THUMB_CACHE_MAX_MB"
  [ -n "$TRAINING_SHM_SIZE" ] && echo "TRAINING_SHM_SIZE=$TRAINING_SHM_SIZE"
  [ -n "$TRAINING_CACHE_MAX_MB" ] && echo "TRAINING_CACHE_MAX_MB=$TRAINING_CACHE_MAX_MB"
  [ -n "$TRAINING_CACHE_MAX_AGE_DAYS" ] && echo "TRAINING_CACHE_MAX_AGE_DAYS=$TRAINING_CACHE_MAX_AGE_DAYS"
  [ -n "$TRAINING_CACHE_SWEEP_INTERVAL_SEC" ] && echo "TRAINING_CACHE_SWEEP_INTERVAL_SEC=$TRAINING_CACHE_SWEEP_INTERVAL_SEC"
  true
} > "$ENV_FILE"

echo
echo "Wrote $ENV_FILE:"
echo "----------------------------------------------------------"
sed 's/^/  /' "$ENV_FILE"
echo "----------------------------------------------------------"
echo
echo "Profiles: $COMPOSE_PROFILES"
[ "$APP_BASE" = "slim-runtime" ] \
  && echo "App image: SLIM (CPU-only, no CUDA)" \
  || echo "App image: CUDA (GPU inference)"
echo
echo "Note: building the app image runs a one-time OpenAPI/client build stage,"
echo "and the training/mono images build the CUDA/torch wheelhouse. Build on a"
echo "capable host and pull the images on lean/CPU-only targets if desired."
echo

if ask "Build and start now (docker compose up -d --build)?" n; then
  exec docker compose up -d --build
else
  echo "When ready:  docker compose up -d --build"
fi
