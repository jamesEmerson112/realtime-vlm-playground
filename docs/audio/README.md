# Audio Transcription Research — Final Report

Date: 2026-04-15

## Overview

The VLM pipeline receives pitch-shifted audio from a technician performing procedures while an instructor observes. Instructor verbal corrections ("no", "wrong", "stop") are a strong error signal. This research evaluated how to transcribe that audio reliably despite pitch-shifting, silence hallucinations, and no speaker separation.

## Model Benchmark

Tested 11 audio-to-text models across 3 videos (846 total chunks, 5s each).

### Speech Detection Rates

| Model | Speech % | Verdict |
|-------|----------|---------|
| gpt-4o-transcribe | 100% | Hallucinator — claims speech on every chunk, echoes prompt back |
| whisper-1 | 93% | Best on real speech, but hallucinates on silence ("Audio may be pitch-shifted.", "fema.gov") |
| gemini-3-flash-preview | 77% | Moderate — some hallucination but less aggressive |
| gpt-audio | 74% | Eliminated — no advantage over top 4 |
| gpt-audio-mini | 71% | Eliminated |
| mimo-v2-omni | 64% | Eliminated |
| gemini-2.5-pro | 57% | Eliminated — incomplete coverage, slow |
| gpt-4o-audio-preview | 42% | Conservative — best for avoiding false positives |
| gemini-2.5-flash | 40% | Eliminated — similar to gpt-4o-audio-preview but less accurate |
| gemini-3.1-pro-preview | 75% | Eliminated — only 1 video tested |
| voxtral-small-24b | 12% | Eliminated — too conservative, misses real speech |

### Winners (4 models)

| Model | API | Role | Cost (per 10s) |
|-------|-----|------|----------------|
| whisper-1 | OpenAI direct | Best transcriber, needs hallucination filter | ~$0.001 |
| gpt-4o-transcribe | OpenAI direct | Fast, aggressive — useful in ensemble | ~$0.001 |
| gemini-3-flash-preview | OpenRouter | Balanced middle ground | ~$0.001 |
| gpt-4o-audio-preview | OpenRouter | Conservative anchor — tiebreaker for silence | ~$0.002 |

All other models eliminated. See [AUDIO_BENCHMARK_FINDINGS.md](AUDIO_BENCHMARK_FINDINGS.md) for detailed benchmark data.

## Enhancement Methods

Tested 4 methods to improve transcription accuracy beyond single-model calls.

| Method | API Calls/Chunk | Result | Verdict |
|--------|----------------|--------|---------|
| **Passes** (whisper x3, majority vote) | 3 | Whisper hallucinates deterministically — all 3 passes agree on the same hallucination | Dead end |
| **Rechunk** (overlapping 2.5s stride) | 1 per window | More chunks = more hallucinations. Merge picks longest = hallucination wins | Dead end |
| **Ensemble** (4 models, voting) | 4 | Conservative models outvote hallucinations. GARBAGE category flags truly distorted audio | Winner |
| **Filtered Ensemble** (hallucination filter + voting) | 4 | Pre-filtering adds near-zero uplift — ensemble voting already handles hallucinations | Redundant |

### Ensemble vs Filtered Ensemble

Side-by-side comparison across 73 chunks (R066 + z065):
- **5/73 chunks differ** (6.8%)
- 3 cosmetic (verbose NO_SPEECH → clean NO_SPEECH)
- 1 false positive ("Your class is over" flagged as hallucination — it's real speech)
- 1 neutral

**Conclusion:** The voting mechanism *is* the hallucination filter. Pre-filtering is redundant.

Full comparison: [ensemble_vs_filtered_comparison.md](ensemble_vs_filtered_comparison.md)

### Ensemble Voting Logic

```
Priority: whisper-1 > gpt-4o-transcribe > gemini-3-flash > gpt-4o-audio-preview

1. Run all 4 models on the 5s chunk
2. Exclude hallucination patterns (is_hallucination() check)
3. If 2+ models produce similar transcripts (>60% word overlap):
   → Use highest-priority agreeing model's transcript
4. If all disagree → check gpt-4o-audio-preview:
   - NO_SPEECH → final = NO_SPEECH
   - Otherwise → final = GARBAGE (unreliable/distorted audio)
```

## Dead Ends

### Speaker Diarization

pyannote/speaker-diarization-3.1 tested on R066 and z065. Pitch-shifting destroys voice embeddings — diarization can't distinguish speakers. Auto-detect clusters everything as 1 speaker; forced 2-speaker mode splits arbitrarily.

This is a **dataset artifact**. In production, separate microphones per speaker would eliminate the need for diarization.

### Multipass Whisper

Whisper-1 is deterministically wrong on silence. Running it 3 times produces 3 identical hallucinations. No variance to exploit.

### Re-chunking

Overlapping windows produce more chunks but whisper hallucinates on all of them. The merge logic picks the longest transcript, which is typically the hallucination.

## Recommended Pipeline Config

| Scenario | Model/Method | Cost | Rationale |
|----------|-------------|------|-----------|
| **Cost-sensitive** | whisper-1 + hallucination filter | 1 call/chunk | Best transcriber, filter catches silence hallucinations |
| **Accuracy-sensitive** | 4-model ensemble | 4 calls/chunk | Voting catches hallucinations + flags distorted audio as GARBAGE |
| **Current pipeline** | gpt-4o-audio-preview (single) | 1 call/chunk | Conservative, avoids false positives, but misses 58% of real speech |

**Recommendation:** Start with single whisper-1 + hallucination filter. Upgrade to ensemble only if false-positive error detections are a problem.

## Cost Analysis

| Method | Calls/Chunk | Cost/Chunk | Cost/Min | Cost/10min Video |
|--------|------------|-----------|---------|-----------------|
| Single whisper-1 | 1 | ~$0.001 | ~$0.012 | ~$0.12 |
| Single gpt-4o-audio-preview | 1 | ~$0.002 | ~$0.024 | ~$0.24 |
| 4-model ensemble | 4 | ~$0.005 | ~$0.060 | ~$0.60 |

All costs approximate based on OpenAI/OpenRouter pricing at time of benchmark.

## File Index

### Scripts

| File | Description |
|------|-------------|
| `scripts/benchmark_audio.py` | 11-model benchmark runner (846 results, HTML report) |
| `scripts/audio_enhance.py` | Enhancement methods: passes, rechunk, ensemble, filtered_ensemble |
| `scripts/compare_ensemble.py` | Offline comparison: ensemble vs filtered_ensemble (no API calls) |
| `scripts/list_audio_models.py` | OpenRouter audio model lister with pricing |
| `scripts/test_diarization.py` | Speaker diarization feasibility test (dead end) |

### Docs

| File | Description |
|------|-------------|
| `docs/audio/README.md` | This report |
| `docs/audio/AUDIO_BENCHMARK_FINDINGS.md` | Detailed benchmark data, diarization results, enhancement experiments |
| `docs/audio/ensemble_vs_filtered_comparison.md` | Side-by-side ensemble vs filtered ensemble (73 chunks, 5 diffs) |

### Raw Data (gitignored, in `output/audio_optimization/`)

| File | Description |
|------|-------------|
| `audio_benchmark_raw.json` | Raw benchmark: 846 model results |
| `audio_benchmark_report.html` | Interactive HTML benchmark report |
| `audio_enhance_ensemble_{R066,z065}.json` | Ensemble voting results per chunk |
| `audio_enhance_passes_{R066,z065}.json` | Multipass whisper results per chunk |
| `audio_enhance_rechunk_{R066,z065}.json` | Re-chunking results per chunk |
| `audio_enhance_*.md` | Per-method markdown summaries |

## Hallucination Patterns

Known whisper/gpt-4o-transcribe hallucinations on silence (used in `HALLUCINATION_PATTERNS` list in `scripts/audio_enhance.py`):

```
"audio may be pitch-shifted"        # Most common — 19+ occurrences
"for more information, visit www"   # URL boilerplate (fema.gov, ncbi, aclu, etc.)
"thank you for watching"            # Video ending boilerplate
"transcribe instructor speech"      # Prompt echo
"video produced by"                 # Credits boilerplate
"audio is muted"                    # Silence description
```

These patterns are deterministic — whisper produces the exact same hallucination every time on the same silent chunk.
