#!/bin/bash
# Stop on failures, unset variables, and failed commands inside pipelines.
set -euo pipefail

# Keep runtime paths anchored to the module root, independent of the caller's working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Read simple KEY=value settings without executing the config file as shell code.
load_config() {
    local config_file line key value
    config_file="${COMPANION_CONFIG:-$PROJECT_DIR/config/config.env}"
    if [ ! -r "$config_file" ]; then
        if [ -n "${COMPANION_CONFIG:-}" ]; then
            echo "[ERROR] Cannot read selected config file: $config_file"
            return 1
        fi
        config_file="$PROJECT_DIR/config/config.example.env"
        if [ ! -r "$config_file" ]; then
            echo "[ERROR] Cannot read config file or example: $config_file"
            return 1
        fi
        echo "[Config Note] Using config/config.example.env; copy it to config/config.env for local overrides."
    fi

    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" != *=* ]]; then
            echo "[ERROR] Invalid config line; expected KEY=value: $line"
            return 1
        fi

        key="${line%%=*}"
        value="${line#*=}"
        if ! [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
            echo "[ERROR] Invalid config key: $key"
            return 1
        fi
        case "$key" in
            PORT|OLLAMA_MODEL|OLLAMA_URL|OLLAMA_KEEP_ALIVE|OLLAMA_MAX_LOADED_MODELS|OLLAMA_NUM_PARALLEL|OLLAMA_NUM_CTX|OLLAMA_NUM_PREDICT|OLLAMA_NUM_THREAD|OLLAMA_MIN_FREE_GB|MEMORY_FILE|MEMORY_MAX_BYTES|MEMORY_MAX_ITEMS|MEMORY_MAX_ENTRY_BYTES|MAX_INPUT_BYTES|MAX_CONTEXT_BYTES|MAX_CLIENTS|WHISPER_CLI|WHISPER_MODEL)
                ;;
            *)
                echo "[ERROR] Unsupported config key: $key"
                return 1
                ;;
        esac

        # Explicit environment values override the checked-in defaults.
        if ! printenv "$key" >/dev/null 2>&1; then
            export "$key=$value"
        fi
    done < "$config_file"
}

if ! load_config; then
    exit 1
fi

# The web server defaults to 8080; validate and export the selected port for Go.
PORT="${PORT:-8080}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
    echo "[ERROR] PORT must be a number from 1 to 65535."
    exit 1
fi
export PORT

# ==============================================================================
# COMPANION CORE: MAC/LINUX AUTOMATED SETUP AND RUN WRAPPER
# ==============================================================================
# This script automatically checks system pre-requisites, downloads the localized
# AI weights, installs Go dependencies, and boots the entire ecosystem.

# Color text variables for beautiful terminal log alerts
CYAN='\033[0;36m'
PINK='\033[0;35m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}[System Initializing] Checking core system dependencies...${NC}"

# 1. Verify Go Installation
if ! command -v go &> /dev/null; then
    echo -e "${RED}[ERROR] Go (Golang) is not installed or not in PATH.${NC}"
    echo "Please download and run the installer from: https://go.dev"
    exit 1
fi
echo -e "${GREEN}[Verified] Go compiler ready.${NC}"

# 2. Verify Ollama Installation
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}[ERROR] Ollama framework engine is missing.${NC}"
    echo "Please download and run the application from: https://ollama.com"
    exit 1
fi
echo -e "${GREEN}[Verified] Ollama engine service detected.${NC}"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo -e "${PINK}[Ollama Setup] Starting the local Ollama service...${NC}"
    # Limit model residency and parallel work when this script owns the Ollama process.
    OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}" \
        OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}" \
        ollama serve >/tmp/companion-ollama.log 2>&1 &
    ollama_ready=false
    for attempt in {1..30}; do
        if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            ollama_ready=true
            break
        fi
        sleep 1
    done
    if [ "$ollama_ready" != true ]; then
        echo -e "${RED}[ERROR] Ollama did not start. See /tmp/companion-ollama.log${NC}"
        exit 1
    fi
else
    echo -e "${CYAN}[Resource Note] Existing Ollama service detected; process-level OLLAMA_* limits apply only when this script starts Ollama.${NC}"
fi
echo -e "${GREEN}[Verified] Local Ollama API is reachable.${NC}"

# 3. Reuse an installed model; check free disk before downloading a missing one.
MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
if ollama show "$MODEL" >/dev/null 2>&1; then
    echo -e "${GREEN}[Verified] Model $MODEL is already installed.${NC}"
else
    minimum_free_gb="${OLLAMA_MIN_FREE_GB:-4}"
    if ! [[ "$minimum_free_gb" =~ ^[0-9]+$ ]] || [ "$minimum_free_gb" -lt 1 ]; then
        echo -e "${RED}[ERROR] OLLAMA_MIN_FREE_GB must be a positive whole number.${NC}"
        exit 1
    fi
    available_kb="$(df -Pk "$PROJECT_DIR" | awk 'END {print $4}')"
    required_kb=$((minimum_free_gb * 1024 * 1024))
    if [ "$available_kb" -lt "$required_kb" ]; then
        echo -e "${RED}[ERROR] Model download needs at least ${minimum_free_gb} GB free; only $((available_kb / 1024 / 1024)) GB is available.${NC}"
        echo "Set OLLAMA_MIN_FREE_GB to a lower positive value to override this check."
        exit 1
    fi
    echo -e "${PINK}[Model Setup] Downloading model $MODEL...${NC}"
    if ! ollama pull "$MODEL"; then
        echo -e "${RED}[ERROR] Failed to fetch model weights from Ollama registry.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}[Verified] Compact model weights warm and ready.${NC}"

# 4. Install local voice dependencies on macOS; the script is safe to rerun.
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo -e "${CYAN}[Voice Setup] Ensuring local whisper.cpp engine and model are installed...${NC}"
    if ! ./scripts/setup_voice.sh; then
        echo -e "${RED}[ERROR] Local voice setup failed; see the message above and retry setup.${NC}"
        exit 1
    fi
else
    echo -e "${CYAN}[Voice Setup] whisper.cpp bootstrap currently supports macOS; skipping on this platform.${NC}"
fi

# 5. Upgrade imported modules, tidy the module graph, and verify compatibility before launch.
echo -e "${CYAN}[Dependency Workspace] Updating Go dependencies to their latest compatible versions...${NC}"
if ! go get -u ./...; then
    echo -e "${RED}[ERROR] Failed to update Go dependencies.${NC}"
    exit 1
fi
if ! go mod tidy; then
    echo -e "${RED}[ERROR] Failed to tidy the Go module files.${NC}"
    exit 1
fi
if ! go test ./...; then
    echo -e "${RED}[ERROR] Go tests failed after dependency updates; server startup cancelled.${NC}"
    exit 1
fi
echo -e "${GREEN}[Verified] Dependencies updated and Go tests passed.${NC}"

# 6. Scan reachable code with the current Go vulnerability database before serving.
echo -e "${CYAN}[Security Check] Scanning Go code and dependencies for known vulnerabilities...${NC}"
if ! go run golang.org/x/vuln/cmd/govulncheck@latest ./...; then
    echo -e "${RED}[ERROR] Vulnerability scan failed or found an issue; server startup cancelled.${NC}"
    exit 1
fi
echo -e "${GREEN}[Verified] No reachable Go vulnerabilities found.${NC}"

# Return PIDs listening on the requested port using the platform's available utility.
port_listener_pids() {
    local target_port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$target_port" -sTCP:LISTEN 2>/dev/null || true
    elif command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "$target_port" 2>/dev/null | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) print $i }'
    else
        return 1
    fi
}

if ! command -v lsof >/dev/null 2>&1 && ! command -v fuser >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] Install lsof or fuser so the setup script can safely clear the app port.${NC}"
    exit 1
fi
stop_port_listener() {
    local target_port="$1"
    local existing_pids remaining_pids pid

    echo -e "${CYAN}[Port Check] Checking port ${target_port}...${NC}"
    existing_pids="$(port_listener_pids "$target_port")"
    if [ -n "$existing_pids" ]; then
        # Try a graceful shutdown first, then force-stop listeners that remain.
        echo -e "${PINK}[Port Check] Stopping process(es) listening on ${target_port}: $(printf '%s' "$existing_pids" | tr '\n' ' ')${NC}"
        while IFS= read -r pid; do
            [[ "$pid" =~ ^[0-9]+$ ]] && kill -TERM "$pid" 2>/dev/null || true
        done <<< "$existing_pids"

        for attempt in {1..10}; do
            remaining_pids="$(port_listener_pids "$target_port")"
            [ -z "$remaining_pids" ] && break
            sleep 1
        done

        remaining_pids="$(port_listener_pids "$target_port")"
        if [ -n "$remaining_pids" ]; then
            echo -e "${PINK}[Port Check] Force-stopping remaining listener(s): $(printf '%s' "$remaining_pids" | tr '\n' ' ')${NC}"
            while IFS= read -r pid; do
                [[ "$pid" =~ ^[0-9]+$ ]] && kill -KILL "$pid" 2>/dev/null || true
            done <<< "$remaining_pids"
        fi

        remaining_pids="$(port_listener_pids "$target_port")"
        if [ -n "$remaining_pids" ]; then
            echo -e "${RED}[ERROR] Could not free port ${target_port}; remaining listener(s): $(printf '%s' "$remaining_pids" | tr '\n' ' ')${NC}"
            return 1
        fi
    fi
    echo -e "${GREEN}[Verified] Port ${target_port} is available.${NC}"
}

# Always replace the default app listener; also clear a custom port when requested.
stop_port_listener 8080
if [ "$PORT" != "8080" ]; then
    stop_port_listener "$PORT"
fi

# 7. Start the Go server; it serves web/index.html and the WebSocket endpoint.
echo -e "${PINK}====================================================================${NC}"
echo -e "${GREEN}[Mainframe Online] Launching Companion Server Core on http://localhost:${PORT:-8080}${NC}"
echo -e "${PINK}====================================================================${NC}"

COMPANION_ROOT="$PROJECT_DIR" go run ./cmd/companion
