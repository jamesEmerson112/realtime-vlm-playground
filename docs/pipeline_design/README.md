# Pipeline Design (V4)

**Companion to `docs/SYSTEM_DESIGN.md`** (which covers the stable harness/evaluator/dashboard infrastructure).
This doc covers the **Pipeline layer only** — everything inside `src/run.py` Pipeline class.

| | |
|---|---|
| Drafted | 2026-04-16 |
| Status | DESIGN — not yet implemented |
| Predecessors | V1 (baseline), V2 (Mother batch), V3 (real-time emit) |
| Target scoring | step_f1 ≥ 0.50, error_f1 ≥ 0.30, latency_score ≥ 0.50 at 1x |

---

## 1. Why V4 — evidence from R066 @ 10x

Latest run (`2026-04-16T17:01:06`) produced 3 events for a 176s / 11-step / 6-error / 5-idle video.

| Metric | Score | Evidence |
|---|---|---|
| step_f1 | **0.000** | Only step 1 emitted, 105s late (154.5s vs GT 49.7s). 11 FN |
| error_f1 | **0.000** | 2 FP at 64.5s and 134.5s. Nearest GT = 85.0s. 6 FN |
| idle_f1 | **0.000** | Zero idle emitted |
| latency | **0.000** | mean 86.96s, saturated |
| combined | **0.000** | |

### Four failure modes this run exposes

1. **Cascading step freeze.** The pipeline was still saying "step 1 not complete" at t=134.5s. `current_step_index` never advanced. One miss ⇒ total loss.
2. **VLM error hallucination reflects our OWN bug.** The 2 FP errors described "contradicts step 1" — the VLM was reporting back our broken state-tracking.
3. **Audio error channel is dead in first 35s.** GT has 6 verbal-correction errors clustered at t=[11.7, 13.2, 14.3, 18.2, 33.6, 85.0]. We caught zero.
4. **Idle detection does not exist.** Timer-based idle was retired with Mother V1; nothing replaced it.

---

## 2. Design principles

1. **No cascading failures.** A missed step must not block detection of any other step. Procedure order is a *prior*, not a *gate*.
2. **VLM reports what it sees, not what we tell it.** Show the full procedure; let the VLM name the step it observes. We consume; we don't force.
3. **Each output channel runs independently.** Steps, video-errors, audio-errors, idle → four separate detectors, each emitting on its own evidence. No shared write state that can deadlock.
4. **Early-stream is audio-heavy.** Before step 1 completes, the instructor is typically coaching. Weight audio higher, widen correction keywords.
5. **Idle is a timer, not a judgment.** "No VLM-reported hands_active for >N seconds" is simple and unkillable.
6. **Respect single-frame epistemics.** Don't ask the VLM questions it structurally cannot answer (e.g., "is this the end state?" from one frame). Ask about disengagement or next-step onset instead.

---

## 3. Architecture

```
                   ┌─────────────────────────────────┐
                   │   StreamingHarness (unchanged)  │
                   └──┬───────────────┬──────────────┘
                      │ on_frame      │ on_audio
        ┌─────────────▼───┐   ┌───────▼────────────┐
        │  FrameWorker    │   │  AudioWorker       │
        │  (throttled)    │   │  (pre-computed     │
        │  VLM call →     │   │   transcripts,     │
        │  Observation    │   │   verified)        │
        └────┬────────────┘   └───────┬────────────┘
             │                        │
             ▼                        ▼
   ┌─────────────────────────────────────────────┐
   │   Shared Streams  (thread-safe deques)       │
   │   observation_stream  |  transcript_stream   │
   └──────┬──────────┬──────────┬─────────┬──────┘
          │          │          │         │
          ▼          ▼          ▼         ▼
   ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
   │ Step     ││ Video    ││ Audio    ││ Idle     │
   │ Tracker  ││ Error    ││ Error    ││ Detector │
   │          ││ Detector ││ Detector ││ (timer)  │
   └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
        │           │           │           │
        └───────────┴─────┬─────┴───────────┘
                          ▼
                   candidate_queue
                          │
                          ▼
                 ┌─────────────────┐
                 │ Mother Verifier │  (async worker thread)
                 │   gpt-5.4-mini  │
                 └────────┬────────┘
                          │ confirm / modify
                          ▼
                   harness.emit_event()
```

**1x pipeline** — candidates go through the mother verifier before `emit_event`.
**≥2x pipeline** — mother is disabled; detectors emit directly (V3 dev mode).

### Component count summary

| Component | Input | Output | Thread |
|---|---|---|---|
| FrameWorker | frame + base64 + transcript snapshot | Observation dict → `observation_stream` | on_frame callback thread |
| AudioWorker | audio bytes + timestamps | Transcript dict → `transcript_stream` | on_audio callback thread |
| StepTracker | observation_stream | `step_completion` events | same as FrameWorker (synchronous) |
| VideoErrorDetector | observation_stream | `error_detected` (source=video) | same as FrameWorker |
| AudioErrorDetector | transcript_stream | `error_detected` (source=audio) | same as AudioWorker |
| IdleDetector | observation_stream + wall clock | `idle_detected` | background `threading.Timer` |
| Emitter | all above | `harness.emit_event()` | whichever detector called |

---

## 4. Component specs

### 4.1 FrameWorker (VLM observation)

**Input:** frame, timestamp_sec, frame_base64, recent transcripts (10s window)

**Output:** Observation dict with fields:

| Field | Type | Purpose |
|---|---|---|
| `timestamp_sec` | float | video time of frame |
| `hands` | str | lightweight context, logged |
| `objects` | str | lightweight context, logged |
| `hands_active` | bool | **drives IdleDetector** — true if any hand motion visible |
| `step_id_seen` | int \| null | **drives StepTracker** — which step does the VLM think is happening NOW |
| `step_id_just_completed` | int \| null | **drives StepTracker** — step the VLM saw finishing in this frame |
| `next_step_starting` | "yes" \| "no" \| "unclear" | secondary completion signal for current step |
| `current_step_progress_pct` | int 0-100 | progress smoother input |
| `error_visible` | str \| null | drives VideoErrorDetector |
| `error_type` | enum \| null | matches harness vocab |
| `audio_supports_current_step` | bool | corroboration signal |
| `confidence` | float 0-1 | overall confidence |

**Design notes:**
- VLM sees **full procedure** (not narrow current+next window). Lets it identify out-of-order activity.
- Prompt orders RULES before SCHEMA.
- Includes one few-shot example in prompt.
- Throttle: max 2 concurrent VLM calls (already exists in V3).

### 4.2 AudioWorker (transcript)

**Input:** audio_bytes, start_sec, end_sec
**Output:** Transcript dict: `{start_sec, end_sec, text, source_model}` → `transcript_stream`

**Design notes:**
- Pre-computed audio pipeline (whisper-1 + hallucination filter + 4-model ensemble verify) stays the default per `--use-audio-cache`.
- Real-time fallback stays available for new videos.
- No Gemini audio passthrough in V4 (flagged as a V5 possibility).

**Known noise patterns — drop before downstream processing:**

These are boot-up / system phrases that recur across videos and carry no procedural signal. Strip them before the transcript reaches StepTracker / AudioErrorDetector.

| Pattern (case-insensitive substring) | Reason |
|---|---|
| `"Please wait. Looking for a server heartbeat"` | System boot phrase, appears at t=0 in multiple clips |
| `"Capturing."` (when paired with boot phrase) | Same setup noise |
| `"NO_SPEECH"` | Placeholder the ensemble emits for silent chunks — treat as silence, not text |

Implementation: a `_is_noise(transcript)` filter applied once when reading transcript_stream. Extensible as we discover more boot-up / device-status phrases across R087, z065, etc.

### 4.3 StepTracker

**Input:** observation_stream
**Output:** `step_completion` events

**Decision logic (any of these fires, whichever first):**

| Rule | Trigger | Confidence |
|---|---|---|
| A. Explicit completion | `step_id_just_completed == N` for 2 of last 3 observations | 0.85 |
| B. Next-step onset | `next_step_starting == "yes"` AND `step_id_seen == N+1` for 2 of last 3 | 0.75 |
| C. Progress smoother | Any 2 of last 5 `current_step_progress_pct` ≥ 85 for step N | 0.65 |
| D. Out-of-order | `step_id_seen == M > current_expected` consistently — emit ALL steps current..M with lower confidence | 0.40 |

**Per-step state:**
- `_progress_history[step_id]: deque(maxlen=5)`
- `_completion_votes[step_id]: deque(maxlen=3)` for Rules A/B
- `_emitted_steps: set`

**Critical change vs V3:** Rules run **for every step in the procedure**, not just `current_step_index`. A step emitted early (out of order) is valid and does NOT block later steps.

**Procedure order as prior, not gate:** If the VLM says step 7 while we haven't emitted steps 2-6, we emit step 7 with confidence penalty (0.40), but we DO emit it. We also emit backfill events for skipped steps if there's corroborating evidence in the observation history.

### 4.4 VideoErrorDetector

**Input:** observation_stream
**Output:** `error_detected` (source=video)

**Decision logic:**
- Emit when `error_visible` is non-null AND `error_type` is set AND `confidence ≥ 0.6`.
- Passthrough `error_type` to the event.
- 5s dedup window (same as V3).

**Change vs V3:** Requires `error_type` to be non-null. Forces the VLM to commit, filters hallucinations.

### 4.5 AudioErrorDetector

**Input:** transcript_stream
**Output:** `error_detected` (source=audio)

**Decision logic:**
- Regex word-boundary match on correction keywords (V3 fix stays).
- **Two keyword tiers:**
  - Tier 1 (always active): `no|stop|wrong|don't|not that|hold on|wait`
  - Tier 2 (only before first step_completion emitted): `try again|pick up|put down|hold on|that's not|other one|different`
- 5s dedup window.
- Timestamp = `start_sec` of the transcript chunk (closer to correction than current video time).

**Change vs V3:** Two-tier keyword vocab addresses failure mode 3 (early-stream error cluster).

### 4.6 IdleDetector

**Input:** observation_stream + wall clock
**Output:** `idle_detected` events

**Decision logic (background thread):**
- Poll every 1s.
- Track `last_active_video_time` = timestamp of last observation with `hands_active == true`.
- If `current_video_time - last_active_video_time ≥ 3s`:
  - Start idle period. Mark `idle_start = last_active_video_time`.
  - Continue while `hands_active` stays false.
  - On next `hands_active == true`:
    - Emit `idle_detected` with `timestamp_sec = (idle_start + idle_end) / 2` (midpoint of period).

**Threshold justification:** GT idles range 2.1s–7.3s. 3s threshold catches 4/5 GT idles (misses the 2.1s one). 2s threshold would catch all but risk more false positives during brief VLM throttling gaps.

**Critical:** This runs on a `threading.Timer`, NOT on the frame callback. It must not be blocked by VLM throttling.

### 4.7 Mother Verifier (async per-candidate)  *— added in V4 implementation*

**Input:** one candidate dict from any detector + evidence snapshot (last 5 observations, audio window, already-emitted events, current step index).

**Output:** a `{decision, reason, event}` verdict. `decision ∈ {"confirm", "reject", "modify"}`.

**Placement:** the mother runs on a **background worker thread**. Detectors produce candidates into a `queue.Queue`; the worker drains one at a time, calls `gpt-5.4-mini` with `reasoning_effort="low"`, parses strict JSON, and on `confirm`/`modify` calls `harness.emit_event()`. On `reject` the candidate is logged and dropped.

```
Detector produces candidate → _queue_candidate → queue.Queue
                                                    │
                              mother_worker thread ◀┘  (async, non-blocking)
                                       │
                              _verify_with_mother(cand, evidence)
                                       │
                      ┌────────────────┼────────────────┐
                      ▼                ▼                ▼
                  confirm          modify            reject
                   │                 │                 │
                   └──▶ harness.emit_event(event) ◀─┘ │
                                                      ▼
                                              logger.log(mother_reject)
```

**Why async:**
- Main pipeline never blocks on mother. Detectors fire, queue, return immediately.
- Mother calls take 1-3s on `gpt-5.4-mini` — blocking would saturate detection_delay_sec.
- If mini returns malformed JSON, one fallback attempt to `gpt-5.4` (still `reasoning_effort="low"`).

**When the mother is skipped:**
- `speed ≥ 2.0` disables the mother entirely — at 10x dev speed, detectors emit directly via `harness.emit_event`. This keeps 10x iteration fast (no $-per-run mother cost) while 1x stays accurate.
- `OPENAI_API` must be set when `speed < 2.0`, else main() errors out early.

**Drain on shutdown:**
- `main()` calls `pipeline.stop_background_workers()` after `harness.run()` returns.
- The helper polls the queue until empty (60s timeout), then joins the worker thread.
- After stop returns, `main()` rebuilds `results.events` from `harness._emitted_events` so post-return mother confirmations are included.

**Candidate schema** (what detectors emit → queue → mother input):

| Field | Type | Purpose |
|---|---|---|
| `timestamp_sec` | float | Detector's best estimate of event time |
| `type` | str | `step_completion`, `error_detected`, or `idle_detected` |
| `detector_source` | str | e.g. `step_tracker:A_just_completed`, `audio_error_detector:tier2`, `idle_detector` |
| `evidence_text` | str | One-line human-readable summary for the mother |
| `event` | dict | Proposed event the mother can confirm verbatim |

**Mother prompt structure:** full procedure, current step index, already-emitted events, last-5 observations, audio window, candidate. See `prompts/mother_prompt.txt`.

---

## 5. Prompt contract (field → consumer map)

Every field in the VLM JSON must have a declared consumer. Decorative fields are removed.

| Field | Consumer | If missing | Action if missing |
|---|---|---|---|
| `hands` | logger | - | log "unknown" |
| `objects` | logger | - | log "unknown" |
| `hands_active` | IdleDetector | high | default to true (fail-safe against spurious idle) |
| `step_id_seen` | StepTracker | medium | skip step tracking for this frame |
| `step_id_just_completed` | StepTracker Rule A | medium | skip Rule A |
| `next_step_starting` | StepTracker Rule B | low | treat as "unclear" |
| `current_step_progress_pct` | StepTracker Rule C | low | treat as 0 |
| `error_visible` | VideoErrorDetector | low | treat as null |
| `error_type` | VideoErrorDetector | high | REQUIRED when `error_visible` set |
| `audio_supports_current_step` | VideoErrorDetector (tiebreaker) | low | treat as false |
| `confidence` | VideoErrorDetector gate | low | treat as 0.5 |

---

## 6. Scoring targets (expected impact per component)

| Component | Fixes failure | Expected impact on R066 |
|---|---|---|
| StepTracker parallel rules | Cascading step freeze | step_f1: 0.00 → 0.50+ (5/11 at worst) |
| VideoErrorDetector error_type gate | VLM hallucination FPs | error_f1 FPs: -70%, TPs unchanged |
| AudioErrorDetector two-tier keywords | Early-stream error cluster | error_f1: 0.00 → 0.30+ (catches 3/6 GT errors) |
| IdleDetector timer | Missing idle channel | idle_f1: 0.00 → 0.30+ (catches 3/5 GT idles) |
| All channels real-time via emit_event | Latency saturation | latency_score: 0.00 → 0.60+ at 1x |

**Target combined (1x):** `0.4 × 0.50 + 0.4 × 0.30 + 0.2 × 0.60 = 0.44`

That would be a ~2.5× improvement over the current 17% high-water mark.

---

## 7. Open design questions

### Q1: Should VLM be forced to pick `step_id_seen` from the procedure, or allow null?

- **Forced:** VLM always commits; cleaner downstream parsing; risk of hallucinated picks.
- **Nullable:** VLM can say "I don't know"; more truthful; more null-handling downstream.
- **Proposal:** Nullable, with confidence weighting. VLM picks null when `confidence < 0.4`.

### Q2: What's the out-of-order backfill policy for StepTracker Rule D?

- If VLM reports step 7 while current is at step 2, do we:
  - (a) Emit only step 7 with confidence 0.4, accept 3/4/5/6 as FN?
  - (b) Emit 3/4/5/6 at interpolated timestamps with confidence 0.2?
  - (c) Scan observation history for best timestamp per skipped step?
- **Proposal:** (c), scan backward through observation_stream for the latest observation where each skipped step had `step_id_just_completed == that_step` OR `progress >= 80`. If none found, treat as FN.

### Q3: Idle detection threshold — 2s, 3s, or adaptive?

- GT has 5 idles: durations [7.3, 4.9, 2.1, 3.6, 2.7]. Median 3.6s.
- 3s threshold: catches 4/5.
- 2s threshold: catches 5/5 but risks FPs during VLM throttle gaps.
- Adaptive: threshold = median of past idle durations, starting at 3s.
- **Proposal:** Start at 3s fixed. Make threshold a CLI flag.

### Q4: Should `audio_supports_current_step == false` suppress step_completion emission?

- Yes (SoundingActions consensus): ensures audio and video agree.
- No: audio is often silent during steps; many false negatives.
- **Proposal:** Use only as confidence multiplier on Rule C emissions (progress-based, the weakest rule). Do NOT use on Rules A or B.

### Q5: What keyword additions for Tier 2 (early-stream)?

- Need to grep verified transcripts for R066, R087, z065 early-stream sections.
- Do this empirically, not speculatively.

---

## 8. Migration from current (V3-realtime)

### KEEP
- Pre-computed audio pipeline (`precompute_audio` + `verify_audio` + `--use-audio-cache`)
- `harness.emit_event()` as the sole output channel (no more bypass)
- `_find_correction_hit` word-boundary regex
- Throttling: `pending_calls ≥ 2` VLM concurrency cap
- PipelineLogger (all log formats)
- Filename timestamp prefix convention

### CHANGE
- `_build_prompt`: show full procedure, add `step_id_seen`/`step_id_just_completed`/`hands_active`/`error_type` fields, few-shot example, move RULES before SCHEMA.
- `_decide_and_emit` → split into `StepTracker`, `VideoErrorDetector` classes.
- Rule 2-of-5 ≥ 85 stays as Rule C, but is now only one of four completion paths.
- Error emission now requires `error_type`.
- Audio error detection gets two-tier keywords.

### ADD
- `IdleDetector` class running on `threading.Timer`.
- Out-of-order step emission with backfill (Rule D).
- `observation_stream` / `transcript_stream` as explicit thread-safe deques (replace ad-hoc `observation_buffer` + `audio_history`).

### DELETE
- `run_mother_batch()` method body (already dead code per `[realtime-pivot-impl]`).
- `mother_events` attribute.
- `prompts/mother_prompt.txt` (retain only if we pursue V5 retrospective catch-up).
- `_consecutive_complete_votes` state (replaced by per-step `_completion_votes`).

---

## 9. What this doc is NOT

- Not the plan for implementing V4. That comes next, in a separate file.
- Not a prompt file. The actual `prompts/vlm_prompt.txt` revision will live there.
- Not a replacement for `docs/SYSTEM_DESIGN.md` — that covers infrastructure (harness, evaluator, dashboard, schemas). Keep it stable.
- Not a scoring report. Run reports go under `docs/{topic}/README.md` per the established pattern (see `docs/mother_agent/README.md`).

---

## 10. Next steps (explicitly NOT started)

1. Answer Q1–Q5 in § 7.
2. Write the revised `prompts/vlm_prompt.txt`.
3. Write a V4 implementation plan.
4. Execute plan with benchmarks on R066, R087 (error-heavy), z065 (no-idle).

Each of those needs its own explicit green-light.
