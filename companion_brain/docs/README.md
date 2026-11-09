# Companion Core

## Project Layout

- `cmd/companion/`: Go server, memory engine, and tests.
- `web/`: browser application.
- `data/`: production memory snapshot and its write-ahead journal.
- `config/`: runtime configuration template and local settings.
- `scripts/`: setup and voice-bootstrap scripts.
- `docs/`: project documentation and storage roadmap.
- `go.mod`, `go.sum`: Go module definition and dependency checksums.

Copy `config/config.example.env` to `config/config.env` to set the model, storage caps, and resource limits. The launcher falls back to the example if the local file is absent. It reads simple `KEY=value` lines; do not add quotes or shell commands. Environment variables override values in the file, and `COMPANION_CONFIG=/path/to/file` selects another config file. The local config and personal memory files are ignored by Git; commit only the example template.

Run `./scripts/setup_and_run.sh` from the project root, or invoke that script by absolute path from another directory. On macOS, each run also ensures the local `whisper.cpp` speech engine and English model are installed noninteractively; other platforms skip that macOS-only bootstrap. The script upgrades imported Go modules to the latest compatible versions in their current major lines, tidies the module graph, runs tests and `govulncheck`, then starts the local Ollama service/model and web app. The module requires Go 1.26.10 to include the standard-library security fixes; Go's toolchain selector downloads that version when needed. Major dependency upgrades are not automatic because they can introduce breaking API changes. Before launch, the script always stops any process listening on port `8080`; if `PORT` selects another port, it clears that port too. It tries graceful termination, then forces termination after ten seconds if necessary.

## Resource Guardrails

The defaults below are stored in `config/config.example.env`. Copy it to `config/config.env` and edit that file for persistent local changes; environment variables override individual values for one run.

| Variable | Default | Accepted range or effect |
| --- | ---: | --- |
| `MEMORY_MAX_BYTES` | `5242880` | 1 KiB to 512 MiB maximum snapshot or journal size |
| `MEMORY_MAX_ITEMS` | `1000` | 1 to 100,000 saved nodes |
| `MEMORY_MAX_ENTRY_BYTES` | `8192` | 1 byte to 1 MiB per node, including follow-ups |
| `MAX_INPUT_BYTES` | `4096` | 128 bytes to 60 KiB per message |
| `MAX_CONTEXT_BYTES` | `2048` | 128 bytes to 32 KiB of retrieved text passed to the model |
| `MAX_CLIENTS` | `8` | 1 to 32 simultaneous WebSocket sessions |
| `OLLAMA_NUM_CTX` | `2048` | 256 to 8,192 model context tokens |
| `OLLAMA_NUM_PREDICT` | `192` | 16 to 1,024 generated tokens per response |
| `OLLAMA_NUM_THREAD` | `2` | 1 to 64 CPU threads for model inference |
| `OLLAMA_KEEP_ALIVE` | `2m` | Ollama model idle retention; use `0` to unload after a request |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Ollama process setting when the script starts `ollama serve` |
| `OLLAMA_NUM_PARALLEL` | `1` | Ollama process setting when the script starts `ollama serve` |
| `OLLAMA_MIN_FREE_GB` | `4` | Required free disk space before pulling a missing model |
| `MEMORY_FILE` | `data/memory.json` | Path to the local JSON memory snapshot |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local Ollama model name |

For example, to override the memory limits for a single run:

```sh
MEMORY_MAX_BYTES=1048576 MEMORY_MAX_ITEMS=200 ./scripts/setup_and_run.sh
```

Limits are hard rejection limits: a new write or follow-up that would exceed a cap is refused, and existing memories are not silently deleted. If the configured memory snapshot exceeds the limits, startup stops and leaves the file untouched. Raise the relevant limits or choose a separate `MEMORY_FILE` to proceed.

Memory lookup includes character n-grams to tolerate small spelling errors. Prompts are not stored by default: the gatekeeper uses `RESPOND` for chat, thanks, questions, and one-time requests, and chooses `STORE` only for durable personal information that may be useful later. If Ollama fails or returns invalid JSON, the safe fallback is `RESPOND` (except clear recall/delete commands), never auto-save. The local gatekeeper also returns a conservative spelling correction before storing or responding: it may fix clear typos and missing/extra characters, using a close memory as context, but is instructed to preserve uncertain wording and never alter names, numbers, dates, places, negation, or facts.

Use the **NODES** tab to select entries, delete selected nodes, or clear the full matrix. Single deletion asks for confirmation; multi-node deletion requires typing `DELETE NODES <count>`, and clear-all requires `DELETE ALL <count>`. The server validates that count again while holding the store lock, so a changed selection or node count cancels the operation. Natural-language delete requests are classified by the local gatekeeper and use the same confirmation flow.

Memories use a small in-house append journal alongside a readable JSON snapshot. `MEMORY_FILE` names the snapshot (default `data/memory.json`); each successful mutation is appended and synced to `MEMORY_FILE.wal`, so a single update no longer rewrites the full collection. The journal stores versioned records with CRC32 checksums. Startup imports the JSON snapshot, replays complete journal records, and truncates an incomplete final record left by a crash; a complete record with a bad checksum stops startup rather than silently losing or changing data. A new atomic JSON snapshot is written every 64 journal records or 256 KiB, after which the journal is truncated. Keep the snapshot and `.wal` together when backing up the store; stop the server first for a consistent copy. Search uses an in-house inverted candidate index over exact words, adjacent words, and character n-grams, then applies the existing 512-float cosine ranker only to candidates. Very common fuzzy n-grams are skipped to avoid turning typo matching into a full scan, so exact words and rarer n-grams are the primary candidate paths. The current engine still loads all records and indexes into RAM and supports one process/writer; tenant partitioning, cross-process coordination, and replication remain future milestones. At a configured storage cap, writes are rejected rather than old memories being pruned.

The matrix is a full 3D cuboid lattice with topic columns, importance height, and relative age depth; the floor, ceiling, front/back, and left/right faces show the coordinate grid. Dashed links are only relationships returned by the model, not guesses based on nearby screen positions.

The primary memory field has **3D Field**, **2D Pages**, **Timeline**, **Memory Lab**, and **Review** views. 2D Pages group memories by category, sort entries by importance and recency, and paginate them while showing the same field scale. Timeline filters entries by their creation/update dates. Memory Lab classifies a test input and generates local-model responses with selected temporary context, optionally compared with no context; the `/api/lab/simulate` endpoint never writes to the persistent memory store. **WHY** on a node shows its saved explanation, linked entries, and any approximate retrieval match score. Its edit and forget actions require user confirmation.

In **SETTINGS → Memory behavior**, **Review new saves** optionally queues STORE and FOLLOW_UP proposals instead of writing immediately. Approving a proposal performs the write; rejecting it discards it. This review queue is session-local and pending proposals are not added to the memory snapshot until approved.

Voice transcription uses the local Whisper setup described below; it does not depend on the browser's external speech-recognition service or send audio to Ollama.

### Local Voice Input

`./scripts/setup_and_run.sh` installs `whisper.cpp` with Homebrew and downloads its more accurate English `small.en` model (about 466 MB) into `~/Library/Caches/companion-core/whisper` on macOS. You can also run `./scripts/setup_voice.sh` independently. Click **VOICE** to start local recording and click it again to stop and transcribe; recordings automatically stop after 30 seconds. A live panel shows the microphone level, elapsed time, and a local transcript preview that refreshes while you speak; the final transcript is sent through the normal local conversation flow when recording stops. Audio is normalized in the browser and posted only to this app's local Go server. The server rejects silent or steady-tone recordings, uses a temporary WAV file for inference, and deletes it afterward. Microphone access still requires browser permission. `WHISPER_CLI` and `WHISPER_MODEL` can override the default executable and model paths.

For hands-free input, open **SETTINGS → Voice input**, set a wake phrase (default `Companion`), and enable **Wake phrase listening**. The browser keeps the microphone active in that tab and sends configurable 4–10-second audio segments (default six seconds) only to the local Whisper endpoint; after it hears the phrase, words in the same segment are sent as the command, or it waits for the next segment if you pause. Shorter checks respond sooner but use more CPU. Disable the toggle to stop and release the microphone. Wake listening is intentionally off after every page reload.

Ollama process-level limits are applied only when this script starts `ollama serve`. If the Ollama app/service is already running, restart it with the desired `OLLAMA_*` environment values for those process settings to take effect. Context size, output size, and idle retention are sent with each model request regardless.

The saved-memory cap bounds the JSON snapshot and journal individually; checkpointing can temporarily require space for the snapshot and journal together. The node-count cap also bounds in-memory vectors. These are not a hard cap on total process RSS or Ollama's model-weight memory. Model choice, context size, loaded-model count, and parallelism affect those separate resource costs.