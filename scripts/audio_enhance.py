"""
Audio Enhancement Experiments

Tests 4 methods to improve audio transcription accuracy:
  1. passes             — Run whisper-1 three times per chunk, majority vote
  2. rechunk            — Overlapping 5s windows with 2.5s stride
  3. ensemble           — Run all 4 winning models, voting logic
  4. filtered_ensemble  — Filter hallucinations BEFORE voting, then ensemble

Usage:
    python scripts/audio_enhance.py --method passes --video R066
    python scripts/audio_enhance.py --method rechunk --video z065
    python scripts/audio_enhance.py --method ensemble --video all
    python scripts/audio_enhance.py --method filtered_ensemble --video R066
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Import shared functions from benchmark script
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_audio import (
    VIDEO_MAP,
    REPO_ROOT,
    SAMPLE_RATE,
    CHANNELS,
    BYTES_PER_SAMPLE,
    CHUNK_SEC,
    extract_audio_chunks,
    call_whisper,
    call_openrouter_audio,
)

# Hallucination patterns to filter out (substring match, case-insensitive)
HALLUCINATION_PATTERNS = [
    # Boilerplate disclaimers (most prevalent — 19+ occurrences across multipass data)
    "audio may be pitch-shifted",
    "audio is muted",
    "silence",
    # URL/reference boilerplate
    "for more information, visit www",
    "for more information, please visit",
    "video produced by",
    "video provided by",
    # Video ending boilerplate
    "thank you for watching",
    "thanks for watching",
    "your class is over",
    "the end",
    # Prompt-echo / self-referential
    "transcribe instructor speech",
    "transcribe any speech",
    "if there is no speech",
    "now pitch-shift",
    "please attend the conference",
    # Page/document artifacts
    "page 2 of",
    "page 3 of",
]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


def normalize_transcript(text: str) -> str:
    """Normalize transcript for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_hallucination(text: str) -> bool:
    """Check if transcript is a known hallucination / prompt echo."""
    norm = normalize_transcript(text)
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in norm:
            return True
    # Very short non-word outputs
    if len(norm) <= 3 and norm not in ("no", "stop", "wait"):
        return True
    return False


def is_no_speech(text: str) -> bool:
    """Check if transcript indicates no speech."""
    return "NO_SPEECH" in text.upper()


def transcripts_similar(a: str, b: str) -> bool:
    """Check if two transcripts are similar (>60% word overlap)."""
    if is_no_speech(a) and is_no_speech(b):
        return True
    if is_no_speech(a) or is_no_speech(b):
        return False

    words_a = set(normalize_transcript(a).split())
    words_b = set(normalize_transcript(b).split())

    if not words_a or not words_b:
        return False

    overlap = len(words_a & words_b)
    min_len = min(len(words_a), len(words_b))
    return (overlap / min_len) > 0.6 if min_len > 0 else False


def safe_print(text: str):
    """Print with ASCII-safe encoding for Windows console."""
    print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# METHOD 1: MULTIPLE WHISPER PASSES
# ---------------------------------------------------------------------------


def run_multipass(chunks, video_key, openai_key):
    """Run whisper-1 three times per chunk, pick best via majority vote."""
    results = []
    total = len(chunks)

    for i, (pcm, start, end) in enumerate(chunks):
        passes = []
        for p in range(3):
            t0 = time.time()
            try:
                transcript = call_whisper(openai_key, pcm, "whisper-1")
                latency = int((time.time() - t0) * 1000)
                passes.append({"transcript": transcript, "latency_ms": latency, "error": None})
            except Exception as e:
                latency = int((time.time() - t0) * 1000)
                passes.append({"transcript": None, "latency_ms": latency, "error": str(e)})
            time.sleep(0.3)

        # Voting logic
        transcripts = [p["transcript"] for p in passes if p["transcript"]]

        if not transcripts:
            final = "NO_SPEECH"
            method = "all_errors"
        elif all(is_no_speech(t) for t in transcripts):
            final = "NO_SPEECH"
            method = "no_speech"
        else:
            # Filter out hallucinations
            clean = [t for t in transcripts if not is_hallucination(t) and not is_no_speech(t)]

            if not clean:
                final = "NO_SPEECH"
                method = "all_hallucinations"
            else:
                # Check for majority (2+ same normalized transcript)
                norms = [normalize_transcript(t) for t in clean]
                for j, n in enumerate(norms):
                    count = norms.count(n)
                    if count >= 2:
                        final = clean[j]
                        method = "majority_vote"
                        break
                else:
                    # All differ — pick longest
                    final = max(clean, key=len)
                    method = "longest"

        agreement = sum(1 for t in transcripts if normalize_transcript(t) == normalize_transcript(final)) if transcripts else 0

        result = {
            "chunk_start": start,
            "chunk_end": end,
            "pass_1": passes[0]["transcript"],
            "pass_2": passes[1]["transcript"],
            "pass_3": passes[2]["transcript"],
            "final_transcript": final,
            "method": method,
            "agreement": agreement,
        }
        results.append(result)

        display = final[:60] if final else "ERROR"
        safe_print(f"  [{i+1}/{total}] {start:.0f}-{end:.0f}s: [{method}] {display}")

    return results


# ---------------------------------------------------------------------------
# METHOD 2: RE-CHUNKING (OVERLAPPING WINDOWS)
# ---------------------------------------------------------------------------


def extract_overlapping_chunks(video_path: str, stride_sec: float = 2.5):
    """Extract audio with overlapping 5s windows and given stride."""
    import subprocess
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()

    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
            "-f", "wav", "-",
        ],
        capture_output=True, timeout=60,
    )

    if result.returncode != 0:
        print(f"  ERROR: ffmpeg failed")
        return [], 0.0

    pcm_data = result.stdout[44:]  # skip WAV header
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    chunk_bytes = chunk_samples * BYTES_PER_SAMPLE
    stride_bytes = int(stride_sec * SAMPLE_RATE) * BYTES_PER_SAMPLE

    chunks = []
    offset = 0
    while offset < len(pcm_data):
        end_offset = min(offset + chunk_bytes, len(pcm_data))
        chunk = pcm_data[offset:end_offset]
        start_sec = offset / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        end_sec = min(start_sec + CHUNK_SEC, duration)
        chunks.append((chunk, start_sec, end_sec))
        offset += stride_bytes

    return chunks, duration


def run_rechunk(video_path, video_key, openai_key):
    """Run whisper-1 on overlapping 5s chunks with 2.5s stride, merge results."""
    # Get overlapping chunks
    overlapping_chunks, duration = extract_overlapping_chunks(video_path, stride_sec=2.5)
    print(f"  Overlapping chunks: {len(overlapping_chunks)} (2.5s stride)")

    # Also get original non-overlapping chunks for comparison
    original_chunks = extract_audio_chunks(video_path)
    print(f"  Original chunks: {len(original_chunks)} (5s non-overlapping)")

    # Transcribe all overlapping chunks
    overlap_transcripts = []
    for i, (pcm, start, end) in enumerate(overlapping_chunks):
        t0 = time.time()
        try:
            transcript = call_whisper(openai_key, pcm, "whisper-1")
            latency = int((time.time() - t0) * 1000)
        except Exception as e:
            transcript = "NO_SPEECH"
            latency = int((time.time() - t0) * 1000)
            safe_print(f"    [{i+1}/{len(overlapping_chunks)}] {start:.1f}-{end:.1f}s: ERROR {e}")
            continue

        overlap_transcripts.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "transcript": transcript,
            "latency_ms": latency,
        })

        display = transcript[:50] if transcript else ""
        safe_print(f"    [{i+1}/{len(overlapping_chunks)}] {start:.1f}-{end:.1f}s ({latency}ms): {display}")
        time.sleep(0.3)

    # Also transcribe original chunks for comparison
    print(f"\n  Transcribing original chunks for comparison...")
    original_transcripts = {}
    for i, (pcm, start, end) in enumerate(original_chunks):
        t0 = time.time()
        try:
            transcript = call_whisper(openai_key, pcm, "whisper-1")
        except Exception:
            transcript = "NO_SPEECH"
        original_transcripts[start] = transcript
        time.sleep(0.3)

    # Merge: for each original chunk window, collect overlapping transcripts
    results = []
    for orig_start in sorted(original_transcripts.keys()):
        orig_end = orig_start + CHUNK_SEC

        # Find all overlapping chunks that intersect this window
        overlapping = []
        for ot in overlap_transcripts:
            if ot["start"] < orig_end and ot["end"] > orig_start:
                overlapping.append(ot)

        # Merge: collect unique non-hallucination, non-NO_SPEECH transcripts
        speech_texts = []
        for ot in overlapping:
            t = ot["transcript"]
            if not is_no_speech(t) and not is_hallucination(t):
                norm = normalize_transcript(t)
                if norm and norm not in [normalize_transcript(s) for s in speech_texts]:
                    speech_texts.append(t)

        if speech_texts:
            # Pick the longest unique transcript as merged result
            merged = max(speech_texts, key=len)
        else:
            merged = "NO_SPEECH"

        orig_transcript = original_transcripts.get(orig_start, "NO_SPEECH")
        changed = normalize_transcript(merged) != normalize_transcript(orig_transcript)

        result = {
            "original_start": orig_start,
            "original_end": round(orig_end, 2),
            "overlapping_chunks": overlapping,
            "merged_transcript": merged,
            "original_transcript": orig_transcript,
            "changed": changed,
        }
        results.append(result)

        if changed:
            safe_print(f"  CHANGED {orig_start:.0f}-{orig_end:.0f}s: \"{orig_transcript[:40]}\" -> \"{merged[:40]}\"")

    return results


# ---------------------------------------------------------------------------
# HALLUCINATION FILTER
# ---------------------------------------------------------------------------


def filter_transcript(text: str) -> tuple[str, bool]:
    """Filter known hallucination patterns to NO_SPEECH.

    Returns (filtered_text, was_filtered).
    """
    if not text or is_no_speech(text):
        return text or "NO_SPEECH", False
    if is_hallucination(text):
        return "NO_SPEECH", True
    return text, False


# ---------------------------------------------------------------------------
# METHOD 3: MODEL ENSEMBLE
# ---------------------------------------------------------------------------

ENSEMBLE_MODELS = [
    ("whisper-1", "whisper"),
    ("gpt-4o-transcribe", "whisper"),
    ("google/gemini-3-flash-preview", "openrouter"),
    ("openai/gpt-4o-audio-preview", "openrouter"),
]


def run_ensemble(chunks, video_key, openai_key, openrouter_key):
    """Run all 4 winning models on each chunk, apply voting logic."""
    results = []
    total = len(chunks)

    for i, (pcm, start, end) in enumerate(chunks):
        model_results = {}

        for model_id, api_type in ENSEMBLE_MODELS:
            t0 = time.time()
            try:
                if api_type == "whisper":
                    transcript = call_whisper(openai_key, pcm, model_id)
                else:
                    transcript = call_openrouter_audio(openrouter_key, model_id, pcm)
                latency = int((time.time() - t0) * 1000)
                model_results[model_id] = {"transcript": transcript, "latency_ms": latency, "error": None}
            except Exception as e:
                latency = int((time.time() - t0) * 1000)
                model_results[model_id] = {"transcript": None, "latency_ms": latency, "error": str(e)}
            time.sleep(0.3)

        # Collect valid transcripts (non-error, non-hallucination)
        valid = {}
        for model_id, r in model_results.items():
            t = r["transcript"]
            if t and not is_hallucination(t):
                valid[model_id] = t

        # Voting logic
        agreeing_models = []
        final = "NO_SPEECH"
        method = "no_agreement"

        if valid:
            # Check for 2+ models agreeing
            model_ids = list(valid.keys())
            best_agreement = []
            for j in range(len(model_ids)):
                group = [model_ids[j]]
                for k in range(j + 1, len(model_ids)):
                    if transcripts_similar(valid[model_ids[j]], valid[model_ids[k]]):
                        group.append(model_ids[k])
                if len(group) > len(best_agreement):
                    best_agreement = group

            if len(best_agreement) >= 2:
                # Use transcript from highest-priority agreeing model
                priority = [m for m, _ in ENSEMBLE_MODELS]
                best_model = min(best_agreement, key=lambda m: priority.index(m))
                final = valid[best_model]
                agreeing_models = best_agreement
                method = "agreement"
            else:
                # All disagree — check gpt-4o-audio-preview
                preview_result = model_results.get("openai/gpt-4o-audio-preview", {})
                preview_transcript = preview_result.get("transcript", "")

                if preview_transcript and is_no_speech(preview_transcript):
                    final = "NO_SPEECH"
                    method = "no_speech"
                elif preview_transcript:
                    final = "GARBAGE"
                    method = "garbage"
                else:
                    final = "NO_SPEECH"
                    method = "no_speech"

        result = {
            "chunk_start": start,
            "chunk_end": end,
            "whisper_1": model_results.get("whisper-1", {}).get("transcript"),
            "gpt_4o_transcribe": model_results.get("gpt-4o-transcribe", {}).get("transcript"),
            "gemini_3_flash": model_results.get("google/gemini-3-flash-preview", {}).get("transcript"),
            "gpt_4o_audio_preview": model_results.get("openai/gpt-4o-audio-preview", {}).get("transcript"),
            "final_transcript": final,
            "method": method,
            "agreeing_models": agreeing_models,
            "agreement_count": len(agreeing_models),
        }
        results.append(result)

        display = final[:60] if final else "ERROR"
        safe_print(f"  [{i+1}/{total}] {start:.0f}-{end:.0f}s: [{method}] ({len(agreeing_models)} agree) {display}")

    return results


# ---------------------------------------------------------------------------
# METHOD 4: FILTERED ENSEMBLE
# ---------------------------------------------------------------------------


def run_filtered_ensemble(chunks, video_key, openai_key, openrouter_key):
    """Run all 4 models, filter hallucinations BEFORE voting, then apply ensemble logic."""
    results = []
    total = len(chunks)

    for i, (pcm, start, end) in enumerate(chunks):
        model_results = {}

        for model_id, api_type in ENSEMBLE_MODELS:
            t0 = time.time()
            try:
                if api_type == "whisper":
                    transcript = call_whisper(openai_key, pcm, model_id)
                else:
                    transcript = call_openrouter_audio(openrouter_key, model_id, pcm)
                latency = int((time.time() - t0) * 1000)
                model_results[model_id] = {"transcript": transcript, "latency_ms": latency, "error": None}
            except Exception as e:
                latency = int((time.time() - t0) * 1000)
                model_results[model_id] = {"transcript": None, "latency_ms": latency, "error": str(e)}
            time.sleep(0.3)

        # Pre-filter: apply hallucination filter to each model's output
        filtered = {}
        filters_applied = []
        for model_id, r in model_results.items():
            raw = r["transcript"]
            if raw:
                filt, was_filtered = filter_transcript(raw)
                filtered[model_id] = filt
                if was_filtered:
                    filters_applied.append(model_id)
            else:
                filtered[model_id] = "NO_SPEECH"

        # Voting on FILTERED outputs (same logic as run_ensemble but on filtered texts)
        valid = {}
        for model_id, t in filtered.items():
            if not is_no_speech(t):
                valid[model_id] = t

        agreeing_models = []
        final = "NO_SPEECH"
        method = "no_agreement"

        if valid:
            model_ids = list(valid.keys())
            best_agreement = []
            for j in range(len(model_ids)):
                group = [model_ids[j]]
                for k in range(j + 1, len(model_ids)):
                    if transcripts_similar(valid[model_ids[j]], valid[model_ids[k]]):
                        group.append(model_ids[k])
                if len(group) > len(best_agreement):
                    best_agreement = group

            if len(best_agreement) >= 2:
                priority = [m for m, _ in ENSEMBLE_MODELS]
                best_model = min(best_agreement, key=lambda m: priority.index(m))
                final = valid[best_model]
                agreeing_models = best_agreement
                method = "agreement"
            else:
                # All disagree — check gpt-4o-audio-preview (filtered)
                preview_filtered = filtered.get("openai/gpt-4o-audio-preview", "NO_SPEECH")

                if is_no_speech(preview_filtered):
                    final = "NO_SPEECH"
                    method = "no_speech"
                else:
                    final = "GARBAGE"
                    method = "garbage"
        else:
            method = "no_speech"

        # Key names for output
        key_map = {
            "whisper-1": "whisper_1",
            "gpt-4o-transcribe": "gpt_4o_transcribe",
            "google/gemini-3-flash-preview": "gemini_3_flash",
            "openai/gpt-4o-audio-preview": "gpt_4o_audio_preview",
        }

        result = {
            "chunk_start": start,
            "chunk_end": end,
        }
        for model_id, key in key_map.items():
            raw = model_results.get(model_id, {}).get("transcript")
            result[f"{key}_raw"] = raw
            result[f"{key}_filtered"] = filtered.get(model_id, "NO_SPEECH")

        result.update({
            "final_transcript": final,
            "method": method,
            "agreeing_models": agreeing_models,
            "agreement_count": len(agreeing_models),
            "filters_applied": filters_applied,
        })
        results.append(result)

        filt_tag = f" [filtered: {','.join(filters_applied)}]" if filters_applied else ""
        display = final[:60] if final else "ERROR"
        safe_print(f"  [{i+1}/{total}] {start:.0f}-{end:.0f}s: [{method}] ({len(agreeing_models)} agree) {display}{filt_tag}")

    return results


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------


def save_results(results, method_name, video_key):
    """Save results as JSON + markdown summary."""
    output_dir = REPO_ROOT / "output" / "audio_optimization"
    os.makedirs(output_dir, exist_ok=True)

    base = f"audio_enhance_{method_name}_{video_key}"

    # JSON
    json_path = output_dir / f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON saved: {json_path}")

    # Markdown
    md_path = output_dir / f"{base}.md"
    lines = []
    lines.append(f"# Audio Enhancement: {method_name} — {video_key}\n")

    # Stats
    total = len(results)
    if method_name == "passes":
        no_speech = sum(1 for r in results if is_no_speech(r["final_transcript"]))
        speech = total - no_speech
        methods = {}
        for r in results:
            m = r["method"]
            methods[m] = methods.get(m, 0) + 1

        lines.append("## Stats\n")
        lines.append(f"- Chunks processed: {total}")
        lines.append(f"- NO_SPEECH: {no_speech} | Speech: {speech}")
        lines.append(f"- Decision methods: {methods}")
        lines.append("")

        lines.append("## Results\n")
        lines.append("| Time | Pass 1 | Pass 2 | Pass 3 | Final | Method | Agreement |")
        lines.append("|------|--------|--------|--------|-------|--------|-----------|")
        for r in results:
            t = f"{r['chunk_start']:.0f}-{r['chunk_end']:.0f}s"
            p1 = (r["pass_1"] or "ERR")[:30]
            p2 = (r["pass_2"] or "ERR")[:30]
            p3 = (r["pass_3"] or "ERR")[:30]
            final = (r["final_transcript"] or "ERR")[:30]
            lines.append(f"| {t} | {p1} | {p2} | {p3} | **{final}** | {r['method']} | {r['agreement']}/3 |")

    elif method_name == "rechunk":
        changed = sum(1 for r in results if r["changed"])
        no_speech = sum(1 for r in results if is_no_speech(r["merged_transcript"]))

        lines.append("## Stats\n")
        lines.append(f"- Original chunks: {total}")
        lines.append(f"- Changed from original: {changed}/{total} ({changed*100//total}%)")
        lines.append(f"- NO_SPEECH: {no_speech} | Speech: {total - no_speech}")
        lines.append("")

        lines.append("## Results\n")
        lines.append("| Time | Original | Merged | Changed |")
        lines.append("|------|----------|--------|---------|")
        for r in results:
            t = f"{r['original_start']:.0f}-{r['original_end']:.0f}s"
            orig = (r["original_transcript"] or "")[:40]
            merged = (r["merged_transcript"] or "")[:40]
            ch = "YES" if r["changed"] else ""
            lines.append(f"| {t} | {orig} | **{merged}** | {ch} |")

    elif method_name == "ensemble":
        no_speech = sum(1 for r in results if is_no_speech(r["final_transcript"]))
        garbage = sum(1 for r in results if r["final_transcript"] == "GARBAGE")
        speech = total - no_speech - garbage
        methods = {}
        for r in results:
            m = r["method"]
            methods[m] = methods.get(m, 0) + 1

        lines.append("## Stats\n")
        lines.append(f"- Chunks processed: {total}")
        lines.append(f"- NO_SPEECH: {no_speech} | Speech: {speech} | Garbage: {garbage}")
        lines.append(f"- Decision methods: {methods}")
        lines.append("")

        lines.append("## Results\n")
        lines.append("| Time | whisper-1 | gpt-4o-transcribe | gemini-3-flash | gpt-4o-audio | Final | Method | Agree |")
        lines.append("|------|-----------|-------------------|----------------|--------------|-------|--------|-------|")
        for r in results:
            t = f"{r['chunk_start']:.0f}-{r['chunk_end']:.0f}s"
            w = (r["whisper_1"] or "ERR")[:20]
            g = (r["gpt_4o_transcribe"] or "ERR")[:20]
            gf = (r["gemini_3_flash"] or "ERR")[:20]
            ap = (r["gpt_4o_audio_preview"] or "ERR")[:20]
            final = (r["final_transcript"] or "ERR")[:20]
            lines.append(f"| {t} | {w} | {g} | {gf} | {ap} | **{final}** | {r['method']} | {r['agreement_count']} |")

    elif method_name == "filtered_ensemble":
        no_speech = sum(1 for r in results if is_no_speech(r["final_transcript"]))
        garbage = sum(1 for r in results if r["final_transcript"] == "GARBAGE")
        speech = total - no_speech - garbage
        total_filtered = sum(len(r.get("filters_applied", [])) for r in results)
        chunks_filtered = sum(1 for r in results if r.get("filters_applied"))
        methods = {}
        for r in results:
            m = r["method"]
            methods[m] = methods.get(m, 0) + 1

        lines.append("## Stats\n")
        lines.append(f"- Chunks processed: {total}")
        lines.append(f"- NO_SPEECH: {no_speech} | Speech: {speech} | Garbage: {garbage}")
        lines.append(f"- Hallucinations filtered: {total_filtered} (across {chunks_filtered} chunks)")
        lines.append(f"- Decision methods: {methods}")
        lines.append("")

        lines.append("## Results\n")
        lines.append("| Time | whisper-1 raw | w1 filt | gpt-4o-tr raw | g4t filt | gemini raw | gem filt | gpt-4o-au raw | g4a filt | Final | Method | Agree | Filtered |")
        lines.append("|------|---------------|---------|---------------|---------|------------|---------|---------------|---------|-------|--------|-------|----------|")
        for r in results:
            t = f"{r['chunk_start']:.0f}-{r['chunk_end']:.0f}s"
            w_r = (r.get("whisper_1_raw") or "ERR")[:15]
            w_f = (r.get("whisper_1_filtered") or "ERR")[:15]
            g_r = (r.get("gpt_4o_transcribe_raw") or "ERR")[:15]
            g_f = (r.get("gpt_4o_transcribe_filtered") or "ERR")[:15]
            gf_r = (r.get("gemini_3_flash_raw") or "ERR")[:15]
            gf_f = (r.get("gemini_3_flash_filtered") or "ERR")[:15]
            ap_r = (r.get("gpt_4o_audio_preview_raw") or "ERR")[:15]
            ap_f = (r.get("gpt_4o_audio_preview_filtered") or "ERR")[:15]
            final = (r["final_transcript"] or "ERR")[:20]
            filt = ", ".join(r.get("filters_applied", [])) or "-"
            lines.append(f"| {t} | {w_r} | {w_f} | {g_r} | {g_f} | {gf_r} | {gf_f} | {ap_r} | {ap_f} | **{final}** | {r['method']} | {r['agreement_count']} | {filt} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Markdown saved: {md_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Audio enhancement experiments")
    parser.add_argument("--method", required=True, choices=["passes", "rechunk", "ensemble", "filtered_ensemble"],
                        help="Enhancement method to run")
    parser.add_argument("--video", default="all", choices=["R066", "z065", "all"],
                        help="Video to process (default: all)")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API", "")
    openrouter_key = os.environ.get("MY_OPENROUTER_API", "")

    if not openai_key:
        print("ERROR: OPENAI_API not set in .env")
        sys.exit(1)
    if args.method in ("ensemble", "filtered_ensemble") and not openrouter_key:
        print("ERROR: MY_OPENROUTER_API not set in .env (needed for ensemble)")
        sys.exit(1)

    videos = ["R066", "z065"] if args.video == "all" else [args.video]

    print(f"\n{'='*60}")
    print(f"  AUDIO ENHANCEMENT: {args.method.upper()}")
    print(f"{'='*60}")
    print(f"  Videos: {videos}")
    print(f"{'='*60}")

    for video_key in videos:
        clip_name = VIDEO_MAP[video_key]
        video_path = str(
            REPO_ROOT / "data" / "videos_full" / clip_name / "Export_py" / "Video_pitchshift.mp4"
        )

        if not os.path.exists(video_path):
            print(f"\n  SKIP {video_key}: video not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {video_key} ({clip_name})")
        print(f"{'='*60}")

        if args.method == "rechunk":
            results = run_rechunk(video_path, video_key, openai_key)
        else:
            chunks = extract_audio_chunks(video_path)
            print(f"  {len(chunks)} chunks ({CHUNK_SEC}s each)\n")

            if args.method == "passes":
                results = run_multipass(chunks, video_key, openai_key)
            elif args.method == "ensemble":
                results = run_ensemble(chunks, video_key, openai_key, openrouter_key)
            elif args.method == "filtered_ensemble":
                results = run_filtered_ensemble(chunks, video_key, openai_key, openrouter_key)

        save_results(results, args.method, video_key)

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
