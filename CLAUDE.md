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

`call_vlm()` in `src/run.py` calls OpenRouter's `/api/v1/chat/completions` endpoint. Supports `stream=True` for SSE streaming. The frame is passed as a base64 data URI in the `image_url` content part. Default model: `google/gemini-2.5-flash`.

## Design Constraints

- Not every frame should go to the VLM — smart sampling is critical for cost and latency
- Harness callbacks block the timeline: slow API calls at high speed increase detection delay
- `emit_event()` is thread-safe — background processing is possible
- OpenRouter supports streaming output but NOT streaming input (no persistent frame/audio connection)

## Context History

### 2026-04-14
- [research] Created system design doc at `docs/SYSTEM_DESIGN.md` (14 sections) before implementing the Pipeline class
- [setup] Dev environment on Windows 11: conda env `vlm` (Python 3.11.15), FFmpeg 8.1 via winget, CUDA 13.0 available
- [setup] All pip deps installed, dry-run passed. API keys in `.env` (gitignored) as `PROD_OPENROUTER_API` / `MY_OPENROUTER_API`
- [resumed] Power outage interrupted session. Resumed from design phase — Pipeline class in `src/run.py` still has empty stubs (`on_frame`/`on_audio` = pass)
- [understanding] Walked through system design: harness throttles frame delivery to simulate real-time camera feed, VLM receives one frame at a time via OpenRouter API (base64 JPEG in JSON payload), model has zero memory between calls
- [decision] Audio stream exists (AAC, 48kHz, mono, pitch-shifted) but deprioritized — focus on video-only VLM analysis first, `on_audio()` stays no-op initially
- [research] OpenRouter has 153 vision models. Confirmed availability: `google/gemini-2.5-flash` ($0.30/1M in, default), `meta-llama/llama-4-scout` ($0.08/1M in), `meta-llama/llama-4-maverick` ($0.15/1M in). Llama 3.2 90B Vision NOT available — only 11B variant
- [decision] Will benchmark Llama 4 vs Gemini 2.5 Flash — model parameterized for easy swapping
- [env] conda activate vlm works via `source /c/ProgramData/Anaconda3/etc/profile.d/conda.sh && conda activate vlm` prefix. `videos.zip` already unzipped (15 videos in `data/videos_full/`)
- [understanding] Event emission contract: pipeline parses VLM natural language into structured events (step_completion, error_detected, idle_detected) via emit_event(). Evaluator compares these against human-annotated ground truth within 5s tolerance.
- [understanding] Ground truth = human-annotated answer key with exact timestamps for steps, errors, and idle periods. Harness = test harness that simulates real-time camera feed from pre-recorded video.
- [understanding] Evaluator and dashboard (`src/evaluator.py`, `src/dashboard.py`) are pre-built — only `src/run.py` Pipeline class (`on_frame`, `on_audio`) needs implementation.
- [note] Audio extraction happens automatically in harness regardless — pipeline can ignore it. Likely exists for production scenario where instructor mic = error signal.
- [baseline] Implemented simple Pipeline in `src/run.py`: threaded VLM calls (every 5th frame), JSON-structured prompts, step state tracking, timer-based idle detection, `on_audio()` no-op
- [baseline-run] First run on R066 circuit breaker (10x speed, Gemini 2.5 Flash via MY_OPENROUTER_API key): 24 events detected, 27.8s wall time
- [baseline-eval] Scores: step_f1=0.000 (2 FP, 11 FN — only steps 1-2 detected), error_f1=0.000 (4 FP, 6 FN), idle_f1=0.174 (2 TP, 16 FP, 3 FN), mean_latency=58.65s
- [baseline-issues] Three root causes identified: (1) step tracking got stuck at step 3 — VLM kept saying "error: door not closed" and never advanced, causing steps 3-11 missed; (2) massive detection delay at 10x speed because VLM calls take 3-5s wall = 30-50s video time; (3) idle detection spamming 16 false positives because last_activity_time only updates on VLM response, not on frame receipt
- [baseline-note] PROD_OPENROUTER_API key returns 401, MY_OPENROUTER_API key works. Dashboard generated at output/dashboard.html
- [analysis] Ground truth survey across all 15 videos: 4 have zero errors (R073, R090, R092, R142), some are error-heavy (R087: 20 errors/2 steps, z065: 21 errors/0 idles). Pipeline must handle both clean and error-heavy runs.
- [decision] Will use Whisper for audio transcription — instructor verbal corrections are strong error signals. Plan: try OpenRouter Whisper API first, note local `openai-whisper` (CUDA available) as fallback option.
- [insight] Audio is pitch-shifted for privacy but Whisper is robust enough to potentially still capture speech. Detecting instructor speech = error correlation signal.
- [v2-impl] Enhanced Pipeline: added `call_audio_llm()` using `openai/gpt-4o-audio-preview` via OpenRouter, threaded audio transcription in `on_audio()`, audio transcript injected into VLM prompt, audio log saved to `_audio.json` file
- [v2-fixes] Fixed step tracking (VLM can report ANY step, skips forward if later step detected), removed timer-based idle detection (now VLM-driven), improved prompt (permissive step detection, stricter error criteria)
- [v2-run] R066 at 10x speed: 11 events, step_f1=0.167 (1/11 matched, 100% precision, 9.1% recall), error_f1=0.000, idle_f1=0.133, mean_latency=50s
- [v2-audio] Audio transcription works on pitch-shifted audio. 19/36 chunks transcribed. Some garbling ("Who's the toolbox?" likely "Use the toolbox") but instructor speech captured. Cost ~$0.004/10s chunk.
- [v2-issue] Main bottleneck is 10x speed latency — VLM calls take 3-5s wall = 30-50s video time, so timestamps land far from ground truth. Need to eval at speed=1.0 for realistic scores.
- [literature] Surveyed 12 papers across 4 shelves: egocentric benchmarks (3), action detection (4), audio-visual fusion (2), robot learning bridge (3). Extracted abstract+intro+conclusion for all 12 into `docs/papers/01-12*.md`
- [literature-picks] Selected 4 papers most actionable for pipeline enhancement:
  - **06 ProTAS** (CVPR 2024): Action progress prediction (estimate % complete per step instead of binary), task graph constraints (use procedure JSON to reject implausible predictions), over-segmentation fixes (our v1 idle spam was exactly this)
  - **07 EASGs** (CVPR 2024): PRE/PNR/POST framing — decompose step detection into precondition/point-of-no-return/postcondition phases. Detect PNR to predict completion earlier → reduce latency
  - **08 EPIC-Fusion** (ICCV 2019): Temporal binding window — audio and video don't need strict alignment. Instructor corrections may lag 2-5s behind the visual event. Keep sliding buffer of last N transcripts, not just most recent
  - **09 SoundingActions** (CVPR 2024): Consensus mechanism — when audio+video agree, boost confidence; when they disagree, downgrade. Procedure text acts as bridge/anchor modality for both
- [insight] Synthesis: these 4 papers give a coherent enhancement path — progress estimation (06) + richer state representation (07) + temporal audio window (08) + multimodal consensus voting (09). All implementable as prompt-engineering + logic changes in `src/run.py`, no new models needed

### 2026-04-15
- [literature-marked] All 12 papers annotated with `chosen: true/false` + `reevaluation` or `actionable_insights` in frontmatter
- [literature-reeval] Re-evaluated the 8 non-chosen papers. Paper 02 (Ego-Exo4D) flagged as UNDERRATED — expert commentary annotation (52 coaches critiquing student technique) is directly analogous to our instructor audio signal. Proficiency ratings ≈ our error detection. Worth reading closely.
- [literature-pdfs] Downloaded full PDFs for 4 chosen papers to `docs/papers/pdfs/`: 06-ProTAS (1.6MB, CVPR Open Access), 07-EASGs (21MB, arXiv 2312.03391), 08-EPIC-Fusion (5.2MB, arXiv 1908.08498), 09-SoundingActions (7.5MB, arXiv 2404.05206)
- [literature-code] ProTAS has public code: https://github.com/Yuhan-Shen/ProTAS
