# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time VLM (Vision-Language Model) orchestrator for Alcor Labs. A streaming pipeline that watches video of a technician performing a procedure and detects step completions, errors, and idle periods by calling VLMs through the OpenRouter API.

## Commands

```bash
# Setup
make setup                    # Install dependencies (pip install -r requirements.txt)
make dry-run                  # Validate procedure JSON + video path without API calls

# Run pipeline
python src/run.py \
    --procedure data/clip_procedures/CLIP.json \
    --video data/videos_full/CLIP/Export_py/Video_pitchshift.mp4 \
    --output output/events.json \
    --speed 1.0               # 1.0 = eval speed, 10.0 = dev speed

# Evaluate
python -m src.evaluator \
    --predicted output/events.json \
    --ground-truth data/ground_truth_sample/CLIP.json \
    --tolerance 5

# Dashboard (HTML timeline visualization)
python -m src.dashboard \
    --predicted output/events.json \
    --ground-truth data/ground_truth_sample/CLIP.json \
    --output output/dashboard.html

# Code quality
make test                     # pytest tests/ -v --cov=src
make lint                     # pylint src/
make fmt                      # black src/ --line-length 100
make clean                    # Remove output/, __pycache__, .pytest_cache
```

## Architecture

The pipeline has two sides connected by a callback contract:

**Harness (input)** — `src/harness.py` (do not modify) simulates real-time video playback. It extracts audio upfront via ffmpeg, then loops through the video timeline delivering frames (as BGR numpy + base64 JPEG) and PCM audio chunks to registered callbacks at the configured `speed`. Audio is delivered before the frame at the same timestamp.

**Pipeline (output)** — `src/run.py` contains the `Pipeline` class with two callbacks (`on_frame`, `on_audio`) where all implementation work goes. When an event is detected, call `self.harness.emit_event({...})`. The harness validates the event against the schema, records wall-clock time, and computes `detection_delay_sec`.

**Data flow:**
```
StreamingHarness.run()
  → pipeline.on_audio(pcm_bytes, start_sec, end_sec)   # audio first
  → pipeline.on_frame(frame_bgr, timestamp_sec, base64) # then frame
  ← pipeline calls harness.emit_event({...})            # when event detected
  → HarnessResults (saved as JSON)
```

**Supporting modules:**
- `src/data_loader.py` — `VideoStream` class, `load_procedure_json()`, `validate_procedure_format()`, `frame_to_base64()`
- `src/evaluator.py` — Scores output against ground truth. Bipartite matching: steps matched by `step_id` + timestamp within tolerance, errors by timestamp only. Computes precision/recall/F1 per category plus latency score.
- `src/dashboard.py` — Generates interactive HTML/SVG timeline comparing predicted vs ground truth events.

## Output Conventions

When producing pipeline output files (events JSON, logs, dashboards), include a timestamp at the FRONT of the filename so runs sort chronologically and are easy to scan in a file listing. Use the pattern: `{YYYYMMDD_HHMM}_{descriptor}`. Example: `output/20260416_0132_r066_realtime_emit.json`.

`src/run.py` auto-prepends `YYYYMMDD_HHMM_` to the `--output` path if no date prefix is present, so you can pass `--output output/r066_realtime_emit.json` and the pipeline will write `output/20260416_0132_r066_realtime_emit.json` (plus `_log.json`, `_log.md`, `_audio.json` siblings). The transformation is idempotent — passing an already-timestamped path is a no-op.

## Key Data Formats

**Procedure JSON** (`data/clip_procedures/`): Has `task_name` (or `task`), `clip`, and `steps` array where each step has `step_id` (int) and `description`.

**Ground truth** (`data/ground_truth_sample/`): Has `events` array (each with `timestamp_sec`, `type`, optional `step_id`) and `idle_periods` array (each with `start_sec`, `end_sec`). Step completions are timestamped at the **end** of each step; errors at the **start** of the wrong action.

**Event schema** (`data/schema/event_log.schema.json`): Events require `timestamp_sec` and `type`. Valid types: `step_completion` (needs `step_id`), `error_detected`, `idle_detected`. Valid sources: `video`, `audio`, `both`. Confidence must be 0-1.

## Scoring Formula

```
combined = 0.40 * step_f1 + 0.40 * error_f1 + 0.20 * latency_score
latency_score = max(0, 1.0 - mean_detection_delay / 10.0)
```

Matching tolerance: +/- 5 seconds. Closest-first greedy bipartite matching.

## Environment

- Python 3.11+, FFmpeg required (audio extraction)
- `OPENROUTER_API_KEY` env var (or `--api-key` flag)
- Videos stored in `data/videos_full/` (gitignored, downloaded separately from Google Drive)
- Video path pattern: `data/videos_full/{clip_name}/Export_py/Video_pitchshift.mp4`
- Audio is pitch-shifted for privacy; instructor verbal corrections = strong error signal
- Uses `opencv-python-headless` (no GUI), `requests`, `numpy`, `Pillow`

## VLM Integration

`call_vlm()` in `src/run.py` calls OpenRouter's `/api/v1/chat/completions` endpoint. Supports `stream=True` for SSE streaming. The frame is passed as a base64 data URI in the `image_url` content part. Default model: `google/gemini-3-flash-preview` (configurable via `--model` CLI flag).

**OpenRouter model clarification:**
- `google/gemini-3-flash-preview` = reasoning/understanding model (1M context, text output, thinking levels, tool use) -- **correct for frame analysis**
- `google/gemini-3.1-flash-image-preview` = image generation model -- **wrong for our use case** (was used 2026-04-16 PM, reverted evening)
- `google/gemini-3.1-flash-lite-preview` = cheaper lite variant

## Design Constraints

- Not every frame should go to the VLM — smart sampling is critical for cost and latency
- Harness callbacks block the timeline: slow API calls at high speed increase detection delay
- `emit_event()` is thread-safe — background processing is possible
- OpenRouter supports streaming output but NOT streaming input (no persistent frame/audio connection)
- **Zero-shot only** — we deliberately use a general-purpose VLM (Gemini 2.5 Flash) via API with no task-specific training. Competitors may train custom models (e.g., ProTAS trains CAS+APP+task graph modules on labeled procedural video) and will score higher on benchmarks where training data matches test data. We accept that tradeoff for generalization, no training data requirement, and operational simplicity. See `docs/Optimization.md` for full rationale.

## Current State (as of 2026-04-16, afternoon)

**Pipeline: V5 (queue-confirm + shortened prompt)** — 8-field boolean-first VLM schema (added `current_step_happening`). Step IDs are code-determined from `self.current_step_index`, never VLM-provided. Queue-confirm step detector: 2-of-3 voter fires → step stored as "pending" → only emitted when VLM confirms NEXT step happening (breaks cascade desync). Last step emits immediately. Timeout fallback configurable (tested 8s and 15s). VLM prompt shortened to 17 lines. Audio-video synchronized error gate suppresses VLM-only errors below 0.80 confidence unless corroborated by audio correction within +/-10s.

**Models in use:**
- VLM (observer + per-frame judgment): `google/gemini-3-flash-preview` via OpenRouter, configurable via `--model` CLI flag. **Reverted from `google/gemini-3.1-flash-image-preview` on 2026-04-16 evening** — discovered 3.1-image-preview is an IMAGE GENERATION model, not a vision understanding model. It returned structured JSON despite not being designed for that, which explains erratic VLM behavior. The original A/B benchmark showing 3.1 advantage was within 0.004 (noise); "better step precision" was likely coincidental.
- Audio: real-time whisper-1 (default). `--use-audio-cache` opts into the authoritative `data/audio_cache/`.
- ~~Mother verifier: retired 2026-04-16.~~

**Latest R066 @ 1x benchmarks (V5 schema, pre-audio-sync):**
- gemini-3-flash-preview: step_f1=0.222, error_f1=0.286, idle_f1=0.625, combined=0.315
- gemini-3.1-flash-image-preview: step_f1=0.167, error_f1=0.353, idle_f1=0.667, combined=0.311
- V5 audio-video sync run pending (`v5_audiosync` tag)

**1-Frame Memory experiment (2026-04-16 evening):**
- Baseline (no memory): step_f1=0.167, error_f1=0.353, idle_f1=0.667
- Original prompt + memory: step_f1=0.133, error_f1=0.250, idle_f1=0.364 (regression)
- Test prompt ("what changed"): step_f1=0.353, error_f1=0.000, idle_f1=0.400
- Active prompt: `prompts/vlm_prompt_test.txt` — needs error instruction softened before next run

**Error FP analysis insight:** Most video-error "FPs" are not hallucinations — they are real visible mismatches that GT didn't annotate because the instructor didn't vocalize a correction. GT error = "instructor correction"; VLM error = "visible mistake". Audio-video sync gate addresses this mismatch by requiring audio corroboration for lower-confidence VLM errors.

**Load-bearing decisions (still in force):**
- Zero-shot only — no task-specific training. Rationale: `docs/Optimization.md`.
- `emit_event()` is the sole output channel — all detectors push live (no batch splice, no verifier gate).
- Step ID = code-tracked `current_step_index`, not VLM output (V5 schema change).
- VLM sees only current + next step (2-step window), not full procedure.
- Idle watcher still only runs at 1x (1s poll doesn't make sense at higher speeds).
- Output filenames use `YYYYMMDD_HHMM_` prefix (auto-prepended by `src/run.py`).
- Word-boundary regex for audio correction matching (fixes `"no"` in `"now"`).

## Reference Documents

- `docs/SYSTEM_DESIGN.md` — infrastructure + harness contract
- `docs/pipeline_design/README.md` — V4 architecture + failure-mode diagnosis
- `docs/Optimization.md` — paper-derived optimizations + zero-shot rationale
- `docs/audio/README.md` — audio research hub (4 winning models, ensemble voting, dead ends)
- `docs/mother_agent/README.md` — Mother Agent V1 benchmark + lessons
- `docs/papers/` — 12-paper literature survey
- `docs/context_history/README.md` — full session-by-session history (2026-04-14 → 2026-04-16)

## Recent Changelog

- **2026-04-16 evening** — **Model misidentification discovered and reverted.** `google/gemini-3.1-flash-image-preview` is an image GENERATION model, not a vision understanding model. It was returning structured JSON despite not being designed for that -- explains erratic VLM behavior. Reverted default to `google/gemini-3-flash-preview` (the proper reasoning/understanding model). The A/B benchmark that justified the 3.1 switch was within 0.004 (noise); "better step precision" was coincidental.
- **2026-04-16 evening** — **1-Frame Memory + temporal prompt experiment.** Added `_last_frame_description` to Pipeline: passes previous frame's VLM description into `{previous_frame}` placeholder for temporal context. Two prompt variants tested on R066 @ 1x: (1) Original prompt with memory — regressed (step_f1=0.133, error_f1=0.250, idle_f1=0.364); VLM read prior frame but didn't change boolean outputs. (2) `prompts/vlm_prompt_test.txt` — restructured around "what CHANGED between frames" — step_f1 jumped to 0.353 (+112% vs baseline 0.167), but "report ONLY if NEW error" killed all error detection (error_f1=0.000). Key insight: temporal framing dramatically helps step detection; error instruction needs softening. `src/run.py` currently points to `vlm_prompt_test.txt`. Files: `src/run.py`, `prompts/vlm_prompt.txt`, `prompts/vlm_prompt_test.txt` (new).
- **2026-04-16 late PM** — **V5 queue-confirm step detector.** Replaced decouple/timeout with queue-based confirmation: voter fires → pending → emitted only when VLM confirms next step happening. Added `current_step_happening` VLM field (8 fields now). Voter timestamp fix (was using confirmation frame's ts, now uses voter's ts). VLM prompt shortened 52 → 17 lines. Root cause identified: step 1 fires 3-8s early across all models (egocentric view ambiguity), causing 50-100s cascade desync. Files: `src/run.py`, `prompts/vlm_prompt.txt`.
- **2026-04-16 PM** — **V5 shipped: boolean-schema + audio-video sync.** VLM prompt rewritten to 7-field boolean-first schema (no step IDs in VLM output). Step completion collapsed to single 2-of-3 boolean voter. Audio-video error sync gate added (suppress low-conf VLM-only errors without audio corroboration). `--model` CLI flag added. Default VLM switched to `google/gemini-3.1-flash-image-preview` after A/B benchmark. `src/run.py` 1680 → 1625 lines. New scripts: `scripts/run_pipeline.sh`, `scripts/benchmark_vlm_models.sh`, `scripts/_compare_runs.py`, `scripts/_probe_vlm.py`.
- **2026-04-16 AM** — **Mother Verifier retired** from V4 pipeline. All 4 detectors emit directly to `harness.emit_event()`. `src/run.py` shrunk 2188 → 1677 lines; `prompts/mother_prompt.txt` deleted. R066 @ 1x: combined 0.167 → **0.265** (+58%). Full writeup: `docs/context_history/2026-04-16_mother-retired.md`.
- **2026-04-16 AM** — V4 shipped: 4 detectors + async mother verifier. R066 @ 1x combined=0.167 (baseline 0.000). *Mother retired same day.*
- **2026-04-16** — Audio cache default flipped to opt-in (`--use-audio-cache`). Output filename prefix flipped to `YYYYMMDD_HHMM_` front.
- **2026-04-15** — Mother Agent V1 (batch reasoning) shipped → retired same day in favor of real-time emit architecture.
- **2026-04-15** — Observer V2: VLM scoped to current+next step window. step_f1 0.476 → 0.600 (+26%).

See `docs/context_history/README.md` for the full per-session log.
