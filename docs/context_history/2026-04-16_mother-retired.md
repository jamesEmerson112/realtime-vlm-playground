# 2026-04-16 — Mother Verifier Retired (V4 → V4.1)

Session time: ~12:11 PM local (based on output filename `output/20260416_1211_r066_postmother_1x.json`).

## Summary

Retired the Mother Verifier entirely from the V4 pipeline. All four detectors (step / video-error / audio-error / idle) now emit directly to `harness.emit_event()` — no async verification, no queue, no second LLM call in the hot path. At `speed >= 2.0` nothing changed (the direct-emit path already existed as the dev-mode bypass); at `speed < 2.0` the queue+mother path was replaced by the same direct emit. Net result on R066 @ 1x: combined score **0.167 → ~0.265** (+58%), mean latency **~10s → 5.09s** (−50%).

## Motivation

The V4 mother wasn't filtering noise — it was *rewriting* the signals it saw. Three consecutive 1x R066 runs with three different mother prompts all produced `error_f1=0.000` while the detectors themselves were producing live matches. Specifically:

- The mother was **synthesizing its own timestamps** at positions that missed ground truth by >5s (outside matching tolerance).
- The mother was **swallowing the audio-correction detector output** — pre-retirement morning run had 4/6 error matches live; with mother wired in, 0/6.
- The mother's "too ambiguous / generic coaching phrase" rejections killed legitimate early-stream Tier-2 audio candidates at 5–25s where GT has 5 clustered errors at 11–33s.

Rolling back to direct-emit (the path that already worked at speed ≥ 2.0) immediately restored the signals.

## Changes to `src/run.py`

Line count: **2188 → 1677** (−511 lines, matches the plan estimate of ~480).

**Removed imports / constants:**
- `import queue`
- `MOTHER_MODEL`, `MOTHER_REASONING_EFFORT`, `MOTHER_FALLBACK_MODEL`, `MOTHER_TRIGGER_IDLE_SCAN_INTERVAL_SEC`

**Removed functions / methods:**
- `call_reasoning_llm()` (OpenAI direct-API helper, only used by mother)
- `Pipeline._build_mother_prompt`
- `Pipeline._call_mother_once`
- `Pipeline._verify_with_mother`
- `Pipeline._mother_worker`
- `Pipeline._handle_verdict`

**Collapsed:**
- `_queue_candidate()` kept its name and signature (call-site stability) but the body is now unconditional direct emit. Previously branched on `speed`: `>= 2.0` bypass = GOOD, `< 2.0` queue = BAD. Now always the bypass path.

**Pruned from `Pipeline.__init__`:**
- `_candidate_queue`, `_mother_thread`, `_emitted_events_log`, `_mother_model`, `_last_silence_scan_time`
- Load of `prompts/mother_prompt.txt` (template no longer needed)
- `mother_model` / `mother_enabled` fields in the `run_start` log payload

**Renamed (idle watcher still needed):**
- `_mother_stop` → `_idle_stop`
- `start_background_workers` → `start_idle_worker`
- `stop_background_workers` → `stop_idle_worker`

**Simplified `main()`:**
- Removed the mother-gating check at `speed < 2`.
- Removed the queue-drain block in the `finally` clause.

**Trimmed `PipelineLogger`:**
- Removed branches for `mother_verify_request` / `mother_verify_response` / `mother_verify_error`, `mother_reject`, `mother_error`, `candidate_queued`.
- Added `emit_rejected` branch.
- Removed the "Mother Verifier — per-candidate input & output" section from the markdown output.

## Files deleted

- `prompts/mother_prompt.txt` (84 lines, no longer loaded anywhere).

## Files intentionally unchanged

- `src/harness.py`, `src/evaluator.py`, `src/dashboard.py`
- `scripts/cache_audio.py`, `scripts/audio_enhance.py`, `scripts/benchmark_audio.py`
- `prompts/vlm_prompt.txt`, `prompts/audio_prompt.txt`

## Verification

- **Syntax:** `ast.parse(src/run.py)` passes.
- **Grep sweep** for `mother|MOTHER_|_candidate_queue|call_reasoning_llm`: one intentional match (a docstring explaining the removal); no live references.
- **Grep sweep** for `start_background_workers|stop_background_workers`: zero matches.
- **File deletion:** `prompts/mother_prompt.txt` confirmed gone.

## Benchmark: R066 @ 1x

Output: `output/20260416_1211_r066_postmother_1x.json`

Run configuration:
- Fresh real-time `whisper-1` transcription (no `--use-audio-cache` flag).
- 22 events emitted.
- 36 audio chunks transcribed live.
- 67 VLM calls.

| Metric | V4 (with mother, morning) | Post-retirement | Δ |
|---|---|---|---|
| step_f1 | 0.267 | **0.286** | +0.019 |
| error_f1 | 0.000 | **0.133** | +0.133 |
| idle_f1 | 0.000 | **0.533** | +0.533 |
| Mean detection latency | ~10s | **5.09s** | −50% |
| Combined score | 0.167 | **~0.265** | +58% |

Per-category event tallies:
- `step_completion` — 2/11 matched, 1 FP, 9 FN
- `error_detected` — 1/6 matched, 8 FP, 5 FN (video-error detector over-fires once mother isn't suppressing — next tuning target)
- `idle_detected` — 4/5 matched, 6 FP, 1 FN

## Key insight / lesson learned

The mother verifier's failure mode was not "too conservative filter" — it was "pretends to be the detector". It was rewriting timestamps into positions that missed GT tolerance, synthesizing its own events, and dropping detector output it judged "ambiguous". The live detectors were already producing the right signals; the mother was the thing masking them. The old `speed >= 2.0` bypass path (dev-mode direct emit) was inadvertently the correct production architecture all along.

Three consecutive 1x R066 runs with different mother prompts all produced `error_f1 = 0.000`. That should have been a louder alarm than it was — the mother's reject rate was the real bug, not the specific prompt wording.

## Load-bearing decisions captured

- **No backward-compat flag for the mother.** Explicitly rejected in the plan. If it comes back, revert this commit.
- **Idle watcher still runs at 1x only.** Its 1s poll doesn't make sense at higher speeds.
- **All four detectors feed the same `_queue_candidate` path.** The name is preserved for call-site stability even though the body is now a direct emit.

## Next tuning target (out of scope for this change)

Video-error detector over-fires — 11% precision, 8 FPs. Two options:

1. Tighten the confidence gate from 0.6 → 0.75.
2. Require 2-frame agreement before emitting.

Either should push `error_f1` from 0.133 toward 0.4+.
