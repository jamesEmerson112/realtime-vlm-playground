# Mother Agent V1 — Final Report

Date: 2026-04-15

## Overview

Mother Agent V1 is the first shipped implementation of the **Observer + Judge** orchestration pattern described in [`docs/Optimization.md`](../Optimization.md) (Optimization #7). The VLM is demoted to a pure *observer* (describes hands, objects, and actions per frame) and a stronger reasoning LLM (the *mother*) fires **once at the end of the stream** to produce the final event list from the full observation buffer + audio transcripts + procedure JSON.

V1 deliberately trades live latency (`latency_score ≈ 0`) for accuracy. The scoring formula weights F1 accuracy at 0.80 and latency at 0.20, so a 3–5× F1 gain dominates even with latency pinned at zero.

## Run Configuration

| Component          | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| Observer VLM       | `google/gemini-3-flash-preview` (OpenRouter)                 |
| Mother LLM         | `gpt-5.4`, `reasoning_effort="low"` (OpenAI direct)          |
| Video              | R066-15July-Circuit-Breaker-part2 (176s, 11 steps, 6 errors) |
| Speeds tested      | 10x and 1x                                                   |
| Frame sampling     | every 5th frame, max 2 concurrent VLM calls                  |
| Audio              | pre-computed (whisper-1 + ensemble verify + hallucination filter) |
| Event stamping     | Option G1 — `detection_delay_sec = video_duration - timestamp_sec` (honest, per-event) |

## Scoring Results

Tolerance for bipartite matching: ±5 seconds.

| Run             | step_f1 | error_f1 | idle_f1 | latency_score | combined | × baseline |
| --------------- | ------- | -------- | ------- | ------------- | -------- | ---------- |
| Baseline (10x)  | 0.167   | 0.000    | 0.133   | 0             | 0.067    | 1.00×      |
| Mother V1 @ 10x | 0.444   | 0.250    | 0.000   | 0             | **0.278**| **4.15×**  |
| Mother V1 @ 1x  | 0.476   | 0.000    | 0.000   | 0             | **0.190**| **2.84×**  |

Per-category counts at 1x: step_completion 5/11 matched (5 FP, 6 FN). error_detected 0/6 matched (1 FP, 6 FN). idle_detected 0/5 matched (0 FP, 5 FN).

## Observation Buffer

Mother input scales with observation count, which depends on playback speed (higher speed = more throttling).

| Speed | Frames delivered | Observations buffered | % throttled | Audio chunks (speech) | Mother prompt size |
| ----- | ---------------- | --------------------- | ----------- | --------------------- | ------------------ |
| 10x   | 353              | 20                    | ~80%        | 19                    | ~19K chars         |
| 1x    | 353              | 69                    | ~0%         | 17                    | ~55K chars         |

Mother wall-clock latency: 21s (10x run) and 41s (1x run).

## Event Comparison (V1 @ 1x vs Ground Truth)

Abbreviated for clarity — see [full events JSON](../../output/r066_mother_v1_1x_20260415_225353.json).

| GT time | GT type         | Step | Mother time | Mother type     | Step | Match? | Note                                           |
| ------- | --------------- | ---- | ----------- | --------------- | ---- | ------ | ---------------------------------------------- |
| 11.7s   | error_detected  | -    | —           | (none)          | —    | miss   | Wrong toolbox — mother didn't flag             |
| 13.2s   | error_detected  | -    | —           | (none)          | —    | miss   | Wrong toolbox (continuing)                     |
| 33.6s   | error_detected  | -    | —           | (none)          | —    | miss   | Wrong part grabbed                             |
| 49.7s   | step_completion | 1    | 44.5s       | step_completion | 1    | miss   | 5.2s off — just outside ±5s tolerance          |
| 57.3s   | step_completion | 2    | 54.5s       | step_completion | 9    | **miss**| **Step-number confusion** (reported 9 for 2)  |
| 68.7s   | step_completion | 3    | 67.0s       | step_completion | 10   | **miss**| **Step-number confusion** (reported 10 for 3) |
| 85.0s   | error_detected  | -    | 97.0s       | error_detected  | -    | miss   | 12s off — closest error but out of tolerance   |
| 98.7s   | step_completion | 4    | 74.5s       | step_completion | 4    | miss   | 24s early — mother jumped the gun              |
| 110.3s  | step_completion | 5    | 109.5s      | step_completion | 5    | ✓      |                                                |
| 115.0s  | step_completion | 6    | 114.5s      | step_completion | 6    | ✓      |                                                |
| 123.8s  | step_completion | 7    | 122.0s      | step_completion | 7    | ✓      |                                                |
| 136.8s  | step_completion | 8    | 132.0s      | step_completion | 8    | ✓      |                                                |
| 158.0s  | step_completion | 9    | 157.0s      | step_completion | 9    | ✓      |                                                |
| 163.4s  | step_completion | 10   | 164.5s      | step_completion | 11   | **miss**| **Step-number confusion** (reported 11 for 10)|
| 171.7s  | step_completion | 11   | —           | (none)          | —    | miss   | Missed final step                              |

**Pattern:** mother nails the mid-video span (steps 5–9) where each step is physically distinct and takes ~10s. It fails at transitions that pack multiple short actions into a few seconds, and at the boundary (steps 10–11 overlap by 8s in wall time).

## Failure Modes

### 1. Step-number confusion in dense regions

Mother reports `step_id=9` at 54.5s and `step_id=10` at 67s, when GT has `step_id=2` at 57.3s and `step_id=3` at 68.7s. The *content* descriptions are consistent with steps 2 and 3 (opening door, closing door). The mother's `reasoning` fields show it is anchoring on specific visible objects and skipping to a later step number that matches those objects' final state, without enforcing procedure order.

**Hypothesis:** the mother prompt asks it to "use procedure order" but doesn't enforce it hard. When observations show an object already in its end state, the mother picks whichever step matches that end-state, even if intermediate steps haven't been recorded.

**V1 fix candidate:** tighten `mother_prompt.txt` — require step_id assignments to be monotonically non-decreasing, with at most one step skipped between adjacent completions.

### 2. Zero `idle_detected`

Mother emitted 0 idle events across both runs. GT has 5 idle periods. The prompt threshold of "5+ seconds of continuous stillness" may be too strict given the observation density — at ~1 observation per 2.5s, 5 seconds is only 2 adjacent observations, which the mother interprets as transitional rather than idle.

**V1 fix candidate:** relax threshold wording ("3+ seconds" or "at least 2 consecutive observations with hands still") and explicitly list example idle signals (put-down tool, waiting, looking around).

### 3. 10x vs 1x error_f1 inversion

Counterintuitive: at 10x (20 observations), mother caught 1 error (F1 0.250). At 1x (69 observations), it caught 0 errors (F1 0.000).

**Hypothesis:** at 1x, the 3.5× larger observation buffer dilutes error signals. With more "normal" observations around each mistake, the mother's confidence in any single error observation drops below its emission threshold. At 10x the sparse buffer amplifies outliers.

**Needs regression** on R087 (error-heavy: 20 errors / 2 steps) and z065 (error-heavy: 21 errors / 0 idles) before concluding whether this is structural or R066-specific.

### 4. First-error window missed entirely

GT has 5 errors clustered in the first 35 seconds (wrong toolbox repeated, wrong part grabbed). Mother emitted zero events in that window. Causes: observations in 0–35s didn't include strong audio correction signals (precomputed audio for that range had weak transcripts like "Take out"), and visual signals ("grabs the wrong toolbox") require comparing what's grabbed against the procedure's expected toolbox — a reasoning step the observer VLM isn't asked to make.

**V1 fix candidate:** augment observer prompt to explicitly call out *mismatches* with procedure expectations (e.g., "if the visible tool doesn't match the current step's expected tool, say so in `visual_cues`").

## Speed Analysis

V1 is **latency-score-independent** from playback speed. Option G1 stamps each event with `detection_delay_sec = video_duration - event_timestamp`, which is a function of the video clock, not the wall clock. So `latency_score = 0` regardless of whether you run at 1x, 10x, or 100x.

However, **observation count is speed-dependent**. The `on_frame` callback samples every 5th frame, and threading caps at 2 concurrent VLM calls. Above ~1.5x playback, frame delivery outpaces VLM response time, so frames get `reason=throttled` skipped. At 10x, ~80% of sampled frames are dropped — leaving the mother with 20 observations instead of 69.

**Correct default: always run V1 at 1x.** Faster speeds were useful under the old stream-emit architecture (VLM called `emit_event` live, so faster playback meant faster iteration), but under V1 they just degrade mother input without speed benefit.

## Cost (Estimate per R066 Run)

| Component              | Calls | Tokens in | Tokens out | $ per 1M in | $ per 1M out | Cost    |
| ---------------------- | ----- | --------- | ---------- | ----------- | ------------ | ------- |
| Observer VLM (Gemini 3 Flash) | 69  | ~200K     | ~70K       | $0.30       | $2.50        | ~$0.24  |
| Mother LLM (gpt-5.4, low)     | 1   | ~14K      | ~1.5K      | ~$2.00      | ~$8.00       | ~$0.04  |
| Audio precompute (whisper-1 + ensemble) | 144  | ~5 min audio | —          | $0.006/min  | —            | ~$0.03  |
| **Total**              |       |           |            |             |              | **~$0.31** |

Comparable to baseline VLM-only pipeline (~$0.24 — same observer cost, no mother). Mother adds ~$0.04 per run — negligible given the accuracy gain.

## Artifacts

Run output files (relative to repo root):

| Path | Description |
| ---- | ----------- |
| [`output/r066_mother_v1_1x_20260415_225353.json`](../../output/r066_mother_v1_1x_20260415_225353.json) | 1x run events (evaluator-compatible) |
| [`output/r066_mother_v1_1x_20260415_225353.html`](../../output/r066_mother_v1_1x_20260415_225353.html) | 1x dashboard (timeline visualization) |
| [`output/r066_mother_v1_1x_20260415_225353_log.md`](../../output/r066_mother_v1_1x_20260415_225353_log.md) | 1x pipeline log (human-readable) |
| [`output/r066_mother_v1_1x_20260415_225353_log.json`](../../output/r066_mother_v1_1x_20260415_225353_log.json) | 1x pipeline log (structured JSON) |
| [`output/r066_mother_v1_1x_20260415_225353_audio.json`](../../output/r066_mother_v1_1x_20260415_225353_audio.json) | 1x audio transcripts |
| [`output/r066_mother_v1_10x_20260415_224716.json`](../../output/r066_mother_v1_10x_20260415_224716.json) | 10x run events |
| [`output/r066_mother_v1_10x_20260415_224716_log.json`](../../output/r066_mother_v1_10x_20260415_224716_log.json) | 10x pipeline log |

Source files:

| Path | Description |
| ---- | ----------- |
| [`prompts/vlm_prompt.txt`](../../prompts/vlm_prompt.txt) | Observer-only VLM prompt |
| [`prompts/mother_prompt.txt`](../../prompts/mother_prompt.txt) | Mother batch reasoning prompt |
| [`src/run.py`](../../src/run.py) | `call_reasoning_llm`, `Pipeline.run_mother_batch`, Option G1 splice |

## Next Steps

1. **Regression** on R087 (error-heavy) and z065 (no-idle) to confirm error_f1 variance is not R066-specific.
2. **Prompt tuning:**
   - Strengthen procedure-order prior in `mother_prompt.txt` (monotonic step IDs, bounded skip distance).
   - Relax idle threshold wording.
   - Augment observer prompt to call out *expected vs. visible tool mismatches* → lift first-35s error recall.
3. **V2 (real-time mother)** — periodic batch every ~10s of video, or streaming judge. Recovers `latency_score` at the cost of fresh F1 drop. Worth pursuing only after V1 F1 stabilizes across videos.
4. **V2 scope decision** needs a baseline on 3+ videos. Do not start V2 until regression is complete.

## Related Documents

- [`docs/Optimization.md`](../Optimization.md) — Optimization #7 (design rationale)
- [`docs/audio/README.md`](../audio/README.md) — Audio research report (report format precedent)
- [`CLAUDE.md`](../../CLAUDE.md) — Context history entries `[mother-v1-*]`
