#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERROR] This setup script currently supports macOS only."
    exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
    echo "[ERROR] Install Homebrew first: https://brew.sh"
    exit 1
fi

export CI=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_ENV_HINTS=1

WHISPER_CLI="${WHISPER_CLI:-whisper-cli}"
if ! command -v "$WHISPER_CLI" >/dev/null 2>&1; then
    echo "[Whisper Setup] Installing whisper.cpp with Homebrew..."
    brew install --quiet --formula whisper-cpp
fi
if ! command -v "$WHISPER_CLI" >/dev/null 2>&1; then
    echo "[ERROR] whisper-cli was not found after installing whisper-cpp."
    exit 1
fi

WHISPER_MODEL="${WHISPER_MODEL:-$HOME/Library/Caches/companion-core/whisper/ggml-small.en.bin}"
export WHISPER_CLI WHISPER_MODEL
model_directory="$(dirname -- "$WHISPER_MODEL")"
mkdir -p "$model_directory"

if [[ -f "$WHISPER_MODEL" ]] && (( $(wc -c < "$WHISPER_MODEL") >= 300000000 )); then
    echo "[Verified] Whisper model already installed at $WHISPER_MODEL"
    exit 0
fi

temporary_model="$(mktemp "$WHISPER_MODEL.tmp.XXXXXX")"
trap 'rm -f "$temporary_model"' EXIT
model_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
echo "[Whisper Setup] Downloading the more accurate English small model (about 466 MB)..."
curl --fail --location --retry 3 --silent --show-error --output "$temporary_model" "$model_url"
model_bytes="$(wc -c < "$temporary_model" | tr -d ' ')"
if (( model_bytes < 300000000 )); then
    echo "[ERROR] The Whisper model download was incomplete ($model_bytes bytes)."
    exit 1
fi
mv "$temporary_model" "$WHISPER_MODEL"
trap - EXIT
echo "[Verified] Local speech recognition is ready. Model: $WHISPER_MODEL"