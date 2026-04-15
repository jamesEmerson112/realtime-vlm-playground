# VLM Orchestrator -- System Design

**Real-time Vision-Language Model Pipeline for Procedure Monitoring**

| | |
|---|---|
| Generated | 2026-04-14 |
| Repo | `realtime-vlm-playground` |
| Owner | Alcor Labs |

---

## 1. High-Level System Overview

The system watches a technician performing a procedure via video, detects step completions, errors, and idle periods in real time, and produces a scored event log.

```
+-----------+     +-----------+     +-----------+     +-----------+
|  INPUT    |     | STREAMING |     |  PIPELINE |     |  OUTPUT   |
|  FILES    | --> |  HARNESS  | --> | (your     | --> | events    |
|           |     |           |     |  code)    |     | .json     |
+-----------+     +-----------+     +-----------+     +-----------+
| procedure |     | simulates |     | VLM calls |     | schema-   |
| .json     |     | real-time |     | event     |     | validated |
| video.mp4 |     | playback  |     | detection |     | log       |
+-----------+     +-----------+     +-----------+     +-----------+
                                         |
                                         | OPENROUTER_API_KEY
                                         v
                                  +-------------+
                                  | OpenRouter   |
                                  | VLM API      |
                                  | (Gemini 2.5  |
                                  |  Flash etc.) |
                                  +-------------+
```

**Post-run evaluation & visualization:**

```
+-------------+     +-------------+     +-------------+
| events.json | --> | evaluator   | --> | metrics     |
| ground      |     | (bipartite  |     | (P/R/F1,    |
| truth .json |     |  matching)  |     |  latency)   |
+-------------+     +-------------+     +-------------+
      |                                       |
      |             +-------------+           |
      +-----------> | dashboard   | <---------+
                    | (HTML/SVG   |
                    |  timeline)  |
                    +-------------+
                          |
                          v
                    dashboard.html
```

---

## 2. Module Map

### `src/`

| Module | Role |
|---|---|
| `run.py` | Pipeline class + main entry point. **YOU IMPLEMENT:** `on_frame()`, `on_audio()`. **PROVIDED:** `call_vlm()`, CLI argument parsing, harness wiring. |
| `harness.py` | `StreamingHarness` (**DO NOT MODIFY**). Simulates real-time A/V delivery. Validates & timestamps emitted events. |
| `data_loader.py` | `VideoStream`, `load_procedure_json()`, `validate_procedure_format()`, `frame_to_base64()` |
| `evaluator.py` | Scoring engine: bipartite matching, precision/recall/F1, latency score |
| `dashboard.py` | HTML/SVG timeline generator. Compares predicted vs ground truth. |

### `data/`

| Directory | Contents |
|---|---|
| `clip_procedures/` | 15 procedure JSON files (`task_name`, `clip`, `steps[]`) |
| `ground_truth_sample/` | 15 matching ground truth files (`events[]`, `idle_periods[]`) |
| `schema/` | `event_log.schema.json`, `example_output.json` |
| `videos_full/` | Video files (gitignored, via Google Drive). Pattern: `{clip}/Export_py/Video_pitchshift.mp4` |

### Other

| Path | Purpose |
|---|---|
| `output/` | Pipeline output (gitignored) |
| `tests/` | pytest test suite |
| `Makefile` | Build/run/eval/lint commands |
| `requirements.txt` | Python deps |

---

## 3. Streaming Harness Internals

`StreamingHarness.run()` is the main loop. It simulates real-time playback.

### Initialization

1. Open video with `cv2.VideoCapture`
2. Extract ALL audio upfront via ffmpeg subprocess:
   ```
   ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 -f wav -
   ```
   Skip 44-byte WAV header, chunk PCM into `audio_chunk_sec` segments
3. Record start wall-clock time

### Main Loop

```python
next_frame_video_time = 0.0
frame_interval = 1.0 / frame_fps          # default: 0.5s at 2 FPS
```

```
while next_frame_video_time < video_duration:
+-------------------------------------------------------------------+
|                                                                   |
|  1. SEEK: cap.set(CAP_PROP_POS_FRAMES, int(t * video_fps))       |
|     Read frame from video                                        |
|                                                                   |
|  2. WAIT: sleep until wall-clock catches up                      |
|     target_wall = next_frame_video_time / speed                  |
|     if elapsed < target_wall: sleep(target_wall - elapsed)       |
|                                                                   |
|  3. DELIVER AUDIO: all chunks with start_sec <= current time     |
|     for each pending chunk:                                      |
|       -> cb(audio_bytes, start_sec, end_sec)                     |
|                                                                   |
|  4. ENCODE FRAME: BGR -> RGB -> JPEG (quality=80) -> base64      |
|                                                                   |
|  5. DELIVER FRAME:                                               |
|       -> cb(frame_bgr, timestamp_sec, frame_base64)              |
|                                                                   |
|  6. ADVANCE: next_frame_video_time += frame_interval             |
|                                                                   |
+-------------------------------------------------------------------+
```

### Timing Diagram (`speed=1.0`, `frame_fps=2`)

```
video time: 0.0s    0.5s    1.0s    1.5s    2.0s    2.5s ...
            |       |       |       |       |       |
audio:      A[0-5s]--.-------.-------.-------.-------.---
frames:     F0      F1      F2      F3      F4      F5
            |       |       |       |       |       |
wall time:  0.0s    0.5s    1.0s    1.5s    2.0s    2.5s
            (at speed=1.0, wall time == video time)
```

- At `speed=10.0`, a 176s video plays in ~17.6s wall time.
- Frame delivery is throttled by `sleep()` to match the speed.

### Audio Delivery Order

Audio chunks are delivered **BEFORE** the frame at the same timestamp. This ensures the pipeline has audio context when processing the frame.

```
At t=5.0s:  deliver audio[5.0-10.0s]  THEN  deliver frame at 5.0s
```

---

## 4. Pipeline Callback Contract

The Pipeline class registers two callbacks with the harness:

```
+---------------------------+      +---------------------------+
|  harness.on_frame(cb)     |      |  harness.on_audio(cb)     |
+---------------------------+      +---------------------------+
|                           |      |                           |
|  cb(                      |      |  cb(                      |
|    frame: np.ndarray,     |      |    audio_bytes: bytes,    |
|      # BGR, H x W x 3    |      |      # PCM 16kHz mono     |
|    timestamp_sec: float,  |      |      # 16-bit signed LE   |
|      # video time         |      |    start_sec: float,      |
|    frame_base64: str,     |      |    end_sec: float         |
|      # JPEG, ready for   |      |  )                        |
|      # VLM API            |      |                           |
|  )                        |      |  Called BEFORE frame at   |
|                           |      |  same timestamp.          |
+---------------------------+      +---------------------------+
```

### Event Emission

When the pipeline detects something, it calls:

```python
self.harness.emit_event({
    "timestamp_sec": float,          # REQUIRED: video time
    "type": str,                     # REQUIRED: event type
    "step_id": int,                  # for step_completion
    "confidence": float,             # 0.0 - 1.0
    "description": str,             # human-readable
    "source": str,                  # "video" | "audio" | "both"
    "vlm_observation": str,         # raw VLM output
    "error_type": str,              # for error_detected
    "severity": str,                # "info" | "warning" | "critical"
    "spoken_response": str,         # verbal correction text
})
```

The harness then:
1. Validates event against schema (raises `ValueError` if invalid)
2. Records `wall_time = monotonic() - start_time`
3. Computes `detection_delay = (wall_time * speed) - timestamp_sec`
4. Thread-safe via `threading.Lock`

### Valid Values

| Event Types | Sources | Error Types | Severities |
|---|---|---|---|
| `step_completion` | `video` | `wrong_action` | `info` |
| `error_detected` | `audio` | `wrong_sequence` | `warning` |
| `idle_detected` | `both` | `safety_violation` | `critical` |
| | | `improper_technique` | |
| | | `other` | |

---

## 5. VLM API Call Flow

`call_vlm()` in `src/run.py` calls OpenRouter `/api/v1/chat/completions`.

### Request

```
POST https://openrouter.ai/api/v1/chat/completions
Headers:
  Authorization: Bearer {OPENROUTER_API_KEY}
  HTTP-Referer:  https://github.com/alcor-labs/vlm-orchestrator-eval
  X-Title:       VLM Orchestrator Evaluation
```

```json
{
  "model": "google/gemini-2.5-flash",
  "stream": true,
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "<your prompt>"},
      {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64,{frame_base64}"
      }}
    ]
  }]
}
```

### Non-Streaming Response (`stream=false`)

```json
{
  "choices": [{
    "message": {"content": "The technician is..."}
  }]
}
```

Returns: `choices[0].message.content`

### Streaming Response (`stream=true`)

```
data: {"choices":[{"delta":{"content":"The "}}]}
data: {"choices":[{"delta":{"content":"tech"}}]}
data: {"choices":[{"delta":{"content":"nician"}}]}
...
data: [DONE]
```

Accumulated: `full_text += delta.content` for each chunk.

### Call Flow Diagram

```
Pipeline.on_frame()
      |
      | Should I analyze this frame?
      | (smart sampling decision)
      |
      +--[yes]-->  call_vlm(api_key, frame_b64, prompt)
      |                |
      |                |  POST /api/v1/chat/completions
      |                |  with base64 JPEG image
      |                v
      |           OpenRouter API
      |                |
      |                |  VLM processes image + prompt
      |                |  (1-10s depending on model)
      |                v
      |           Response text
      |                |
      |                | Parse VLM response
      |                | Detect events (step/error/idle)
      |                v
      |           self.harness.emit_event({...})
      |
      +--[no]-->  skip (save cost/latency)
```

> **WARNING:** The callback BLOCKS the harness timeline. If `call_vlm()` takes 5s at `speed=10x`, detection delay grows by 50s of video-time. Consider background threading for API calls.

---

## 6. Data Format Reference

### 6a. Procedure JSON (`data/clip_procedures/*.json`)

```json
{
  "task_name": "Change Circuit Breaker",
  "clip": "R066-15July-Circuit-Breaker-part2",
  "steps": [
    {
      "step_id": 1,
      "description": "The student grabs the circuit breaker."
    },
    { "step_id": 2, "description": "..." }
  ]
}
```

- 15 clips total, covering circuit breakers, ATVs, RAM, graphics cards
- `step_id` is 1-indexed, sequential
- Task label may be `"task"` or `"task_name"` (code handles both)

### 6b. Ground Truth (`data/ground_truth_sample/*.json`)

```json
{
  "video_name": "R066-15July-Circuit-Breaker-part2",
  "task_type": "change circuit breaker",
  "total_duration_sec": 176,
  "procedure_steps": [
    {
      "step_id": 1,
      "description": "...",
      "start_sec": 9.2,
      "end_sec": 49.653
    }
  ],
  "events": [
    {
      "timestamp_sec": 11.698,
      "type": "error_detected",
      "error_type": "wrong_action",
      "severity": "warning",
      "description": "The student grabs the wrong toolbox."
    },
    {
      "timestamp_sec": 49.653,
      "type": "step_completion",
      "step_id": 1,
      "description": "The student grabs the circuit breaker."
    }
  ],
  "idle_periods": [
    {
      "start_sec": 57.349,
      "end_sec": 64.667,
      "duration_sec": 7.318
    }
  ]
}
```

**Key conventions:**
- Step completions are timestamped at the **END** of each step
- Errors are timestamped at the **START** of the wrong action
- Idle periods are gaps between step end and next step start

### 6c. Output Event Log (`output/events.json`)

Schema: `data/schema/event_log.schema.json`

```json
{
  "task": "Change Circuit Breaker",
  "video_source": "path/to/video.mp4",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T00:03:00Z",
  "speed": 1.0,
  "video_duration_sec": 176.0,
  "wall_duration_sec": 176.5,
  "total_frames_delivered": 352,
  "total_audio_chunks_delivered": 36,
  "mean_detection_delay_sec": 2.5,
  "max_detection_delay_sec": 8.1,
  "events": [
    {
      "timestamp_sec": 49.7,
      "type": "step_completion",
      "step_id": 1,
      "confidence": 0.92,
      "description": "...",
      "source": "video",
      "vlm_observation": "...",
      "detection_delay_sec": 1.3
    }
  ]
}
```

---

## 7. Evaluation Pipeline

```
python -m src.evaluator --predicted X --ground-truth Y
```

### Input Separation

```
pred_events --> split by type --> pred_steps, pred_errors, pred_idles
gt_events   --> split by type --> gt_steps, gt_errors
gt_idles    --> from ground truth idle_periods[]
```

### Matching Algorithms

All three categories use `_min_distance_match()` -- optimal greedy bipartite matching (closest-first):

1. Generate all valid `(pred_idx, gt_idx, distance)` pairs
2. Sort by distance ascending
3. Greedily assign: skip if either side already matched
4. Return `(TP, FP, FN)`

| Category | Match Criteria | Distance |
|---|---|---|
| **Steps** | `step_id` must be equal AND timestamp within tolerance | `\|pred_t - gt_t\|` |
| **Errors** | Timestamp within tolerance (`step_id` NOT required) | `\|pred_t - gt_t\|` |
| **Idles** | Pred timestamp falls within GT idle period `[start_sec, end_sec]` | `\|pred_t - midpoint of GT period\|` |

### Scoring

For each category (steps, errors, idles):

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * P * R / (P + R)
```

Latency:

```
delays[] = detection_delay_sec from all predicted events
latency_score = max(0, 1.0 - mean(delays) / 10.0)

0s mean delay  -> 1.0 score (perfect)
10s mean delay -> 0.0 score (worst)
Linear interpolation between
```

### Combined Score

```
combined = 0.40 * step_f1
         + 0.40 * error_f1
         + 0.20 * latency_score
```

```
+------+------+---------+
| 40%  | 40%  |   20%   |
| step | err  | latency |
| F1   | F1   | score   |
+------+------+---------+
       |
       v
   combined (0.0 - 1.0)
```

---

## 8. Dashboard Generation

```
python -m src.dashboard --predicted X --ground-truth Y
```

### Pipeline

1. Load predicted events + ground truth
2. Run evaluator to get metrics
3. Run detailed matching (mirrors evaluator's greedy bipartite)
4. Generate HTML with embedded SVG timeline

### SVG Timeline Structure

```
time axis -->  0s          50s         100s        150s     176s
+--------+----+----+--------+----+--------+--------+--------+
|        |    PROCEDURE STEPS (colored bars)                 |
| Row 1  |  [Step 1       ][S2 ][S3][Step 4    ][S5]...     |
+--------+---------------------------------------------------+
|        |    GROUND TRUTH EVENTS                            |
| Row 2  |  x  x  x    *1    *2  *3      x     *4  ...     |
|        |  (errors)  (step completions)                     |
+--------+---------------------------------------------------+
|        |    PREDICTED EVENTS                               |
| Row 3  |     *1    *2  *3         *4  ...                  |
|        |  (color-coded: green=TP, red=FP)                  |
+--------+---------------------------------------------------+
|        |    IDLE PERIODS                                   |
| Row 4  |       [====]         [==]    [=]    [==]  [=]     |
|        |       (gray bars between steps)                   |
+--------+---------------------------------------------------+
```

**Legend:** `*` = step_completion, `x` = error_detected, `[=]` = idle_period. Green = matched (TP), Red = unmatched (FP/FN).

**Output:** Single self-contained HTML file with metrics summary table, interactive SVG timeline (hover for details), per-event detail table, and color-coded match status.

---

## 9. Threading Model & Latency

### Blocking Behavior

The harness main loop is **SINGLE-THREADED**. Callbacks block the timeline:

```
harness.run()  [main thread]
     |
     |-- sleep until target wall time
     |-- call on_audio(...)          <-- BLOCKS until return
     |-- call on_frame(...)          <-- BLOCKS until return
     |       |
     |       +-- call_vlm(...)       <-- 1-10s network I/O
     |       |       |
     |       |       +-- HTTP POST to OpenRouter
     |       |       +-- wait for response
     |       |       +-- parse response
     |       |
     |       +-- emit_event(...)     <-- thread-safe (locked)
     |
     |-- advance to next frame
     |-- repeat
```

### Latency Implications

```
detection_delay = (wall_elapsed * speed) - event_timestamp
```

**Example at `speed=10x`, `call_vlm()` takes 3s:**
- Wall time advances 3s
- Video time advances 3s * 10 = **30s**
- During that 3s, ~6 frames (at 2 FPS) are DELAYED
- All subsequent events accumulate this delay

### Thread-Safe `emit_event()`

`emit_event()` uses `threading.Lock`, so background threads CAN emit:

```
Pipeline.on_frame()
     |
     +-- spawn thread --> call_vlm() --> emit_event()  [safe]
     |
     +-- return immediately  [unblocks harness]
```

This is the **KEY optimization**: move VLM calls off the main thread to prevent blocking the harness timeline.

### Recommended Pattern

```python
def on_frame(self, frame, timestamp, b64):
    if should_analyze(frame):
        thread = Thread(target=self._analyze, args=(frame, timestamp, b64))
        thread.start()
    # return immediately -- don't block

def _analyze(self, frame, timestamp, b64):
    response = call_vlm(self.api_key, b64, prompt)
    events = parse_response(response)
    for event in events:
        self.harness.emit_event(event)      # thread-safe
```

---

## 10. File / Directory Map

```
realtime-vlm-playground/
|
+-- src/
|   +-- __init__.py              # empty package marker
|   +-- run.py                   # [IMPLEMENT] Pipeline + main()
|   +-- harness.py               # [READ ONLY] StreamingHarness
|   +-- data_loader.py           # VideoStream, procedure loading
|   +-- evaluator.py             # Scoring engine
|   +-- dashboard.py             # HTML timeline generator
|
+-- data/
|   +-- clip_procedures/         # 15 procedure JSONs
|   |   +-- R066-15July-Circuit-Breaker-part2.json
|   |   +-- R073-20July-GoPro.json
|   |   +-- R087-27July-GoPro.json
|   |   +-- R090-28July-ATV.json
|   |   +-- R092-28July-Circuit-Breaker.json
|   |   +-- R142-31Aug-RAM.json
|   |   +-- R190-24Oct-ATV.json
|   |   +-- R192-24Oct-CircuitBreaker.json
|   |   +-- R198-1Nov-Graphicscard.json
|   |   +-- z010-june-16-22-gopro.json
|   |   +-- z039-june-23-22-dslr.json
|   |   +-- z045-june-24-22-dslr.json
|   |   +-- z065-june-29-22-dslr.json
|   |   +-- z067-june-29-22-gopro.json
|   |   +-- z108-july-26-22-gopro.json
|   |
|   +-- ground_truth_sample/     # 15 matching GT files
|   |   +-- (same names as clip_procedures/)
|   |
|   +-- schema/
|   |   +-- event_log.schema.json
|   |   +-- example_output.json
|   |
|   +-- videos_full/             # GITIGNORED - from Google Drive
|       +-- {clip_name}/
|           +-- Export_py/
|               +-- Video_pitchshift.mp4
|
+-- output/                      # GITIGNORED - pipeline output
|   +-- events.json
|   +-- dashboard.html
|
+-- tests/                       # pytest suite
+-- docs/
|   +-- SETUP.md
|   +-- SYSTEM_DESIGN.md         # THIS FILE
|
+-- Makefile                     # make setup/run/evaluate/test/lint
+-- requirements.txt             # opencv-python-headless, numpy,
|                                # requests, Pillow, av, python-dotenv,
|                                # pytest, pytest-cov
+-- CLAUDE.md                    # AI assistant instructions
+-- README.md
```

---

## 11. Dependency Graph

### External Services

| Service | Used By |
|---|---|
| OpenRouter API | `call_vlm()` in `src/run.py` (requires `OPENROUTER_API_KEY`) |

### System Dependencies

| Dependency | Used By |
|---|---|
| `ffmpeg` | `harness.py` `_extract_audio_chunks()` |
| Python 3.11+ | All modules |

### Python Packages

| Package | Used By |
|---|---|
| `opencv-python-headless` | `harness.py`, `data_loader.py` (cv2) |
| `numpy` | Frame arrays (`np.ndarray`) |
| `requests` | `call_vlm()`, `data_loader.py` URL fetch |
| `Pillow` | `frame_to_base64()` JPEG encoding |
| `av` | Available, not currently used in core |
| `python-dotenv` | Available for `.env` loading |

### Internal Module Dependencies

```
run.py
  +-- imports harness.StreamingHarness
  +-- imports data_loader.load_procedure_json
  +-- imports data_loader.validate_procedure_format

harness.py
  +-- imports cv2, numpy, PIL.Image (frame I/O)
  +-- subprocess: ffmpeg (audio extraction)
  +-- no internal src/ imports

data_loader.py
  +-- imports cv2, numpy, PIL.Image, requests
  +-- no internal src/ imports

evaluator.py
  +-- standalone (json, statistics, dataclasses)
  +-- no internal src/ imports

dashboard.py
  +-- imports evaluator.evaluate
```

---

## 12. Configuration Knobs

### CLI Arguments (`src/run.py main()`)

| Flag | Default | Effect |
|---|---|---|
| `--procedure` | required | Path to procedure JSON |
| `--video` | required | Path to video MP4 |
| `--output` | `output/events.json` | Output JSON path |
| `--speed` | `1.0` | Playback speed (1.0=realtime, 10=dev) |
| `--frame-fps` | `2.0` | Frames/sec delivered to pipeline |
| `--audio-chunk-sec` | `5.0` | Audio chunk duration in seconds |
| `--api-key` | env var | OpenRouter API key |
| `--dry-run` | `false` | Validate inputs only, no API calls |

### Makefile Defaults

| Variable | Default |
|---|---|
| `PROCEDURE` | `data/clip_procedures/R066-15July-Circuit-Breaker-part2.json` |
| `VIDEO` | `data/videos_full/R066-.../Video_pitchshift.mp4` |
| `OUTPUT` | `output/events.json` |
| `GROUND_TRUTH` | `data/ground_truth_sample/R066-15July-Circuit-Breaker-part2.json` |

### `call_vlm()` Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | `google/gemini-2.5-flash` | OpenRouter model |
| `stream` | `False` | SSE streaming |
| `timeout` | `30s` | HTTP timeout |
| JPEG quality | 80 (harness) / 85 (loader) | Image compression |

### Audio Extraction (ffmpeg)

| Setting | Value |
|---|---|
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit signed little-endian |
| Format | Raw PCM (WAV header stripped) |

---

## 13. End-to-End Data Flow (Detailed)

### Phase 1: Setup

```
main()
  |
  +-- load_procedure_json(path)
  |     +-- json.load() -> dict
  |
  +-- validate_procedure_format(procedure)
  |     +-- check "task"/"task_name" exists
  |     +-- check "steps" is list
  |     +-- check each step has step_id + description
  |
  +-- StreamingHarness(video_path, procedure_path, speed, ...)
  |     +-- json.load(procedure_path)
  |     +-- store config
  |
  +-- Pipeline(harness, api_key, procedure)
  |     +-- store procedure, steps, task_name
  |     +-- TODO: init state (current_step, buffers, etc.)
  |
  +-- harness.on_frame(pipeline.on_frame)
  +-- harness.on_audio(pipeline.on_audio)
```

### Phase 2: Run

```
harness.run()
  |
  +-- cv2.VideoCapture(video_path) -> cap
  +-- _extract_audio_chunks() -> [(bytes, start, end), ...]
  |     +-- ffmpeg subprocess -> WAV -> strip header -> chunk PCM
  |
  +-- start_wall_time = time.monotonic()
  |
  +-- LOOP: for each frame at frame_fps intervals
  |     |
  |     +-- cap.set(frame_number) + cap.read() -> frame BGR
  |     +-- sleep until real-time target
  |     +-- deliver pending audio chunks (before frame)
  |     +-- frame_to_base64(frame) -> JPEG base64
  |     +-- call on_frame(frame, timestamp, base64)
  |     |     |
  |     |     +-- [YOUR CODE] analyze frame
  |     |     +-- [YOUR CODE] call_vlm() if needed
  |     |     +-- [YOUR CODE] emit_event() if detected
  |     |
  |     +-- progress log every 10 frames
  |
  +-- cap.release()
  +-- compute mean/max detection delays
  +-- return HarnessResults
```

### Phase 3: Save

```
harness.save_results(results, output_path)
  +-- mkdir -p output/
  +-- json.dump(asdict(results)) -> events.json
```

### Phase 4: Evaluate (separate command)

```
python -m src.evaluator --predicted events.json --ground-truth GT.json
  |
  +-- load both JSON files
  +-- split events by type
  +-- _match_steps(pred, gt, tolerance=5.0) -> (TP, FP, FN)
  +-- _match_errors(pred, gt, tolerance=5.0) -> (TP, FP, FN)
  +-- _match_idles(pred, gt_periods) -> (TP, FP, FN)
  +-- compute P/R/F1 for each
  +-- compute latency stats (mean, max, p50, p90)
  +-- print report
```

### Phase 5: Dashboard (separate command)

```
python -m src.dashboard --predicted events.json --ground-truth GT.json
  |
  +-- load files + run evaluate() for metrics
  +-- detailed matching for per-event annotation
  +-- generate HTML with:
  |     +-- metrics summary table
  |     +-- SVG timeline (procedure steps, GT, predicted, idles)
  |     +-- per-event detail table with match status
  +-- write dashboard.html
```

---

## 14. Design Constraints & Gotchas

1. **NOT EVERY FRAME SHOULD GO TO THE VLM**
   - At 2 FPS over 176s = 352 frames
   - Each VLM call costs ~$0.001-0.01 and takes 1-10s
   - Smart sampling is critical for cost AND latency

2. **CALLBACKS BLOCK THE TIMELINE**
   - Slow `on_frame()` delays ALL subsequent frames
   - At `speed=10x`, 3s API call = 30s video-time delay
   - Use background threads for API calls

3. **AUDIO IS PITCH-SHIFTED FOR PRIVACY**
   - Standard speech-to-text may not work well
   - Instructor verbal corrections are a strong ERROR signal
   - Consider energy/silence detection instead of STT

4. **OPENROUTER LIMITATIONS**
   - Supports streaming OUTPUT (SSE) for lower TTFT
   - Does NOT support streaming INPUT (no persistent connection)
   - Each call is independent -- no session/context

5. **FRAME ENCODING**
   - Harness delivers JPEG quality=80 base64
   - `data_loader.py` uses quality=85
   - Frame is BGR numpy array (OpenCV convention)
   - base64 string is ready for VLM API (`data:image/jpeg;base64,...`)

6. **EVALUATION TOLERANCE**
   - Default: +/- 5 seconds for timestamp matching
   - Step matching requires BOTH `step_id` match AND timestamp match
   - Error matching requires ONLY timestamp match
   - Idle matching requires pred timestamp inside GT period

7. **`emit_event()` IS THREAD-SAFE**
   - Uses `threading.Lock` internally
   - Safe to call from background threads
   - Validates event schema before recording
   - Raises `ValueError` on invalid events

8. **VIDEO PATH PATTERN**
   - `data/videos_full/{clip_name}/Export_py/Video_pitchshift.mp4`
   - Videos are gitignored, downloaded from Google Drive
   - Video and procedure file share the same clip_name
