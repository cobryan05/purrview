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

menu() {  # menu "Title" "opt1" "opt2" ... -> echoes chosen index (1-based) on stderr-free stdout
  local title="$1"; shift
  local i=1
  echo "$title" >&2
  for opt in "$@"; do echo "  $i) $opt" >&2; i=$((i+1)); done
  local reply
  read -r -p "Choose [1]: " reply >&2 || true
  echo "${reply:-1}"
}

# Ask whether to use the GPU for inference. Echoes "y" or "n". Only prompts when
# a GPU exists; on a CPU-only host it silently answers "n". Training, when part
# of the deployment, always uses the GPU regardless of this answer.
choose_gpu_inference() {
  if [ "$GPU_DEFAULT" = "y" ]; then
    if ask "Use the GPU for inference (onnxruntime-gpu)?" y; then echo "y"; else echo "n"; fi
  else
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
  echo "No NVIDIA GPU detected on this host (CPU-only assumed)."
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
