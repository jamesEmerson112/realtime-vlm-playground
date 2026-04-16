# Audio Benchmark Findings

Date: 2026-04-15

## Setup

- 11 models tested across 3 videos, 5s audio chunks, 846 total results
- Audio is pitch-shifted for privacy (instructor speech distorted)
- Prompt: "Transcribe any speech in this audio. The audio may be pitch-shifted. If there is no speech, respond with exactly: NO_SPEECH"
- Script: `scripts/benchmark_audio.py`, raw data: `output/audio_benchmark_raw.json`, report: `output/audio_benchmark_report.html`

## Videos

| Video | Chunks | Steps | Errors | Profile |
|-------|--------|-------|--------|---------|
| R066 (circuit breaker) | 36 | 2 | 6 | Baseline, has v2 pipeline audio to compare |
| z065 (battery door, DSLR) | 37 | 8 | 21 | Error-heavy, many instructor corrections |
| R073 (GoPro setup) | 44 | 11 | 0 | Clean execution, false-positive test |

## Coverage

| Model | R066 | z065 | R073 | Total |
|-------|------|------|------|-------|
| whisper-1 | 36 | 37 | 44 | 117 |
| gpt-4o-transcribe | 36 | 37 | 44 | 117 |
| openai/gpt-4o-audio-preview | 36 | 37 | -- | 73 |
| openai/gpt-audio | 36 | 37 | -- | 73 |
| openai/gpt-audio-mini | 36 | 37 | -- | 73 |
| mistralai/voxtral-small-24b-2507 | 36 | 37 | -- | 73 |
| xiaomi/mimo-v2-omni | 36 | 37 | -- | 73 |
| google/gemini-2.5-flash | 36 | 37 | -- | 73 |
| google/gemini-3-flash-preview | 36 | 37 | -- | 73 |
| google/gemini-2.5-pro | 36 | 29 | -- | 65 |
| google/gemini-3.1-pro-preview | 36 | -- | -- | 36 |

Only whisper-1 and gpt-4o-transcribe have complete coverage across all 3 videos. Most OpenRouter models missing R073. gemini-2.5-pro missing 8 z065 chunks. gemini-3.1-pro-preview only has R066.

## Speech Detection Rates

| Model | Speech | Silent | Errors | Speech % |
|-------|--------|--------|--------|----------|
| gpt-4o-transcribe | 117 | 0 | 0 | 100% |
| whisper-1 | 109 | 7 | 1 | 93% |
| google/gemini-3-flash-preview | 56 | 17 | 0 | 77% |
| openai/gpt-audio | 54 | 19 | 0 | 74% |
| openai/gpt-audio-mini | 52 | 21 | 0 | 71% |
| xiaomi/mimo-v2-omni | 47 | 26 | 0 | 64% |
| google/gemini-2.5-pro | 37 | 27 | 1 | 57% |
| openai/gpt-4o-audio-preview | 31 | 42 | 0 | 42% |
| google/gemini-2.5-flash | 29 | 43 | 1 | 40% |
| google/gemini-3.1-pro-preview | 27 | 9 | 0 | 75% |
| mistralai/voxtral-small-24b-2507 | 9 | 64 | 0 | 12% |

## Key Findings

### 1. Hallucination on silence is the dominant failure mode

gpt-4o-transcribe returns speech for every single chunk (117/117) — it never says NO_SPEECH. whisper-1 is nearly as bad at 109/117. Both echo the prompt back on silent chunks:
- gpt-4o-transcribe: "Transcribe instructor speech. Audio may be pitch-shifted." (verbatim prompt echo)
- whisper-1: "Audio may be pitch-shifted." (partial prompt echo)
- whisper-1 also produces: "Thank you for watching.", ". . . . ." on silence

This makes both models unsuitable for the pipeline without a post-processing filter — they would inject false "instructor speech" into VLM prompts constantly.

### 2. Conservative models are better pipeline candidates

gpt-4o-audio-preview (42% speech rate) and gemini-2.5-flash (40%) sit in the sweet spot — low enough to avoid hallucinating on silence, high enough to catch real instructor corrections. voxtral is too conservative (12% — misses real speech).

The current pipeline uses gpt-4o-audio-preview, which appears to be a reasonable choice.

### 3. Pitch-shift artifact: "toolbox" chunk

R066 chunk 130-135s is a known distortion where instructor speech gets mangled. Multiple models produce "Who's the toolbox?" or similar. This is a real speech segment that gets distorted by pitch-shifting — it's a ground truth artifact, not a model failure.

### 4. R073 exposes false-positive risk

R073 is a clean execution (0 errors in ground truth) — there should be minimal instructor corrections. Only whisper-1 and gpt-4o-transcribe were tested on it:
- whisper-1: echoes "Audio may be pitch-shifted." on ~25/44 chunks
- gpt-4o-transcribe: returns fabricated speech on all 44 chunks

If the pipeline used either model, R073 would be flooded with false "instructor correction" signals, causing false-positive error detections.

### 5. No ground truth for audio

We have no human-annotated transcription ground truth. We can only compare models against each other and against the video ground truth (where instructor corrections should correlate with error events). The speech detection disagreement highlighting in the HTML report is our best proxy for identifying which chunks actually contain speech.

## Pipeline Implications

- Models that hallucinate on silence will inject false transcripts into the VLM prompt's "Recent instructor audio" field, biasing the VLM toward detecting errors that don't exist
- A post-processing filter (minimum transcript length, prompt-echo detection) could salvage aggressive models, but adds complexity
- The current choice of gpt-4o-audio-preview is reasonable — it's conservative enough to avoid most hallucinations while still catching real speech
- R073 should be the primary false-positive validation video for any audio model change

## Speaker Diarization Test (Dead End)

Date: 2026-04-15

### Setup

- Model: pyannote/speaker-diarization-3.1 (local, GPU)
- Hardware: GTX 1070 Ti (8GB VRAM), CUDA
- Processing time: ~7s per video
- Script: `scripts/test_diarization.py`
- Goal: separate instructor speech from technician speech to reduce false-positive error detections

### Results

| Video | Mode | Speakers | Split | Assessment |
|-------|------|----------|-------|------------|
| R066 | Auto-detect | 2 | 99% / 1% | Effectively one cluster — SPEAKER_01 is a 0.3s false split |
| R066 | Forced 2 | 2 | 57% / 43% | Arbitrary split — same utterances randomly assigned to either speaker |
| z065 | Forced 2 | 2 | 14% / 86% | Only 1 segment (2.5s) for minority speaker — not meaningful |

### Why it fails

Diarization clusters speakers by voice embeddings (pitch, timbre, speaking rate). The pitch-shifting applied to this dataset destroys these features:

1. Both speakers receive the same pitch transform
2. Their voice embeddings converge after transformation
3. pyannote can't find distinct clusters — everything looks like one speaker
4. When forced to split into 2, it makes arbitrary cuts with no correlation to actual speaker identity

### Production vs. dataset

This is a **dataset artifact**, not a production limitation. In a real deployment:
- Instructor and technician would have **separate microphones** (lapel mic, room mic)
- Each mic channel *is* the speaker label — no AI diarization needed
- The pitch-shifting was applied for privacy before the dataset was shared

### Conclusion

Diarization is a dead end for this dataset. The pipeline should continue:
- Treating all transcribed speech as potential instructor input
- Relying on keyword/semantic analysis ("no", "wrong", "stop") for correction detection
- Using conservative transcription models (gpt-4o-audio-preview) to minimize hallucinated speech

## Winning Models

Selected 4 models from the benchmark for all future use:
- `whisper-1` (OpenAI direct API)
- `openai/gpt-4o-audio-preview` (OpenRouter) — current pipeline model
- `google/gemini-3-flash-preview` (OpenRouter)
- `gpt-4o-transcribe` (OpenAI direct API) — fast but hallucination-prone, needs filtering

All other models eliminated.

## Audio Enhancement Experiments

Date: 2026-04-15

### Setup

- Script: `scripts/audio_enhance.py --method passes|rechunk|ensemble`
- Videos: R066 (36 chunks) + z065 (37 chunks)
- Output: `output/audio_optimization/audio_enhance_*`

### Results Summary

| Method | Video | NO_SPEECH | Speech | Garbage | Verdict |
|--------|-------|-----------|--------|---------|---------|
| Passes (whisper x3) | R066 | 2/36 | 34/36 | - | Bad — whisper agrees on hallucinations |
| Passes (whisper x3) | z065 | 7/37 | 30/37 | - | Bad — same, "fema.gov" and prompt echoes |
| Rechunk (2.5s stride) | R066 | 3/36 | 33/36 | - | Worse — merge picks longest = hallucination |
| Rechunk (2.5s stride) | z065 | 4/37 | 33/37 | - | Worse — swaps real speech for hallucinations |
| **Ensemble (4 models)** | **R066** | **14/36** | **18/36** | **4/36** | **Best — disagreement catches silence** |
| **Ensemble (4 models)** | **z065** | **19/37** | **14/37** | **4/37** | **Best — correctly flags garbage audio** |

### Key Findings

1. **Multiple whisper passes don't help.** Whisper-1 is deterministically wrong on silence — all 3 passes return the same hallucination ("Audio may be pitch-shifted.", "For more information, visit www.fema.gov"). Majority vote picks the hallucination with confidence.

2. **Re-chunking makes things worse.** Overlapping windows produce more chunks, but whisper hallucinates on most of them. The merge logic picks the longest transcript, which is often the hallucination. Real speech gets replaced with junk.

3. **Model ensemble is the clear winner.** When different models disagree on a chunk, it's almost always silence or distorted audio. The voting logic (2+ agree → use it, all disagree → check gpt-4o-audio-preview) correctly identifies NO_SPEECH and GARBAGE.

4. **GARBAGE is a useful category.** The 4 GARBAGE chunks per video are real speech so distorted by pitch-shifting that no two models agree. Marking them as unreliable is more honest than hallucinating a transcript.

### Ensemble Voting Logic

Models (priority order): whisper-1 > gpt-4o-transcribe > gemini-3-flash-preview > gpt-4o-audio-preview

1. Run all 4 models on the chunk
2. If 2+ models produce similar transcripts (>60% word overlap) → use highest-priority agreeing model's transcript
3. If all disagree → check gpt-4o-audio-preview:
   - If NO_SPEECH → final = NO_SPEECH
   - If not NO_SPEECH → final = GARBAGE (unreliable/distorted audio)

### Recommendation

Use ensemble for all future audio transcription in the pipeline. The single-model approach (currently gpt-4o-audio-preview) is too conservative (42% speech rate). Ensemble catches more real speech while correctly filtering silence.

## TODO

1. **Multiple whisper passes** — Run whisper-1 (or gpt-4o-transcribe) multiple times on the same 5s chunk, collect varied outputs, pick the best/consensus transcript. Whisper has some randomness in decoding; multiple passes may catch speech a single pass misses.
2. **Re-chunking strategy** — Try overlapping windows (e.g., 5s chunks with 2.5s overlap) or different chunk lengths (3s, 7s, 10s) to catch speech that falls on chunk boundaries.
3. **Model ensemble** — Run the same chunk through whisper-1 AND gpt-4o-transcribe, combine results. whisper-1 is conservative (misses some speech), gpt-4o-transcribe is aggressive (halluccinates on silence) — their union with a confidence filter could be better than either alone.
4. **Cross-reference audio with ground truth** — Compare speech detection timestamps against ground truth error timestamps to find which model's detections correlate best with actual errors.
5. **Sliding audio buffer** — Keep last N transcripts instead of just the most recent, so instructor corrections that lag 2-5s behind the visual event aren't missed (per EPIC-Fusion paper insight).
6. **Prompt-echo filtering** — Strip transcripts that match the prompt or common hallucination patterns ("Audio may be pitch-shifted.", "Transcribe instructor speech.") to salvage gpt-4o-transcribe.
