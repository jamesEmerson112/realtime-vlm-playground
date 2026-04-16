"""
Audio-to-Text Model Benchmark

Tests all available audio models (OpenAI Whisper + OpenRouter chat-based)
on pitch-shifted instructor audio from the VLM pipeline videos.

Usage:
    python scripts/benchmark_audio.py                              # Full benchmark
    python scripts/benchmark_audio.py --video R066                 # Single video
    python scripts/benchmark_audio.py --model whisper-1 --video R066  # Single model + video
    python scripts/benchmark_audio.py --resume                     # Resume from saved progress
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

VIDEO_MAP = {
    "R066": "R066-15July-Circuit-Breaker-part2",
    "z065": "z065-june-29-22-dslr",
    "R073": "R073-20July-GoPro",
}

PROMPT = (
    "Transcribe any speech in this audio. The audio may be pitch-shifted. "
    "If there is no speech, respond with exactly: NO_SPEECH"
)

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
CHUNK_SEC = 5.0

# Models: (model_id, api_type)
# api_type: "whisper" | "openrouter" | "openrouter_gemini"
MODELS = [
    ("whisper-1", "whisper"),
    ("gpt-4o-transcribe", "whisper"),
    ("openai/gpt-4o-audio-preview", "openrouter"),
    ("openai/gpt-audio", "openrouter"),
    ("openai/gpt-audio-mini", "openrouter"),
    ("mistralai/voxtral-small-24b-2507", "openrouter"),
    ("xiaomi/mimo-v2-omni", "openrouter"),
    ("google/gemini-2.5-flash", "openrouter"),
    ("google/gemini-3-flash-preview", "openrouter"),
    ("google/gemini-2.5-pro", "openrouter"),
    ("google/gemini-3.1-pro-preview", "openrouter"),
]

# ---------------------------------------------------------------------------
# AUDIO EXTRACTION
# ---------------------------------------------------------------------------


def extract_audio_chunks(video_path: str) -> List[Tuple[bytes, float, float]]:
    """Extract audio from video as 5s PCM chunks, matching harness behavior."""
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
        print(f"  ERROR: ffmpeg failed for {video_path}")
        print(f"  stderr: {result.stderr.decode()[:200]}")
        return []

    audio_data = result.stdout
    pcm_data = audio_data[44:]  # skip WAV header

    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    chunk_bytes = chunk_samples * BYTES_PER_SAMPLE

    chunks = []
    t = 0.0
    offset = 0
    while offset < len(pcm_data):
        end_offset = min(offset + chunk_bytes, len(pcm_data))
        chunk = pcm_data[offset:end_offset]
        end_t = min(t + CHUNK_SEC, duration)
        chunks.append((chunk, t, end_t))
        t = end_t
        offset = end_offset

    return chunks


def pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API CALLERS
# ---------------------------------------------------------------------------


def call_whisper(openai_key: str, pcm_bytes: bytes, model: str = "whisper-1") -> str:
    """Call OpenAI Whisper API (direct, file upload)."""
    wav_bytes = pcm_to_wav_bytes(pcm_bytes)

    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {openai_key}"},
        files={"file": ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={
            "model": model,
            "prompt": "Transcribe instructor speech. Audio may be pitch-shifted.",
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json().get("text", "").strip()
    return text if text else "NO_SPEECH"


def call_openrouter_audio(openrouter_key: str, model: str, pcm_bytes: bytes) -> str:
    """Call an OpenRouter chat model with audio input."""
    wav_bytes = pcm_to_wav_bytes(pcm_bytes)
    audio_b64 = base64.b64encode(wav_bytes).decode()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ],
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
            "X-Title": "VLM Orchestrator Audio Benchmark",
        },
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------


def run_benchmark(
    videos: List[str],
    models: List[Tuple[str, str]],
    openai_key: str,
    openrouter_key: str,
    existing_results: Optional[List[dict]] = None,
) -> List[dict]:
    """Run the full benchmark, returning list of result dicts."""
    results = existing_results or []

    # Build set of already-completed (model, video, chunk_start) for resume
    done = set()
    for r in results:
        done.add((r["model"], r["video"], r["chunk_start"]))

    output_path = REPO_ROOT / "output" / "audio_optimization" / "audio_benchmark_raw.json"

    for video_key in videos:
        clip_name = VIDEO_MAP[video_key]
        video_path = str(
            REPO_ROOT / "data" / "videos_full" / clip_name / "Export_py" / "Video_pitchshift.mp4"
        )

        if not os.path.exists(video_path):
            print(f"\n  SKIP {video_key}: video not found at {video_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Extracting audio: {video_key} ({clip_name})")
        print(f"{'='*60}")
        chunks = extract_audio_chunks(video_path)
        print(f"  Got {len(chunks)} chunks ({CHUNK_SEC}s each)")

        for model_id, api_type in models:
            print(f"\n  --- Model: {model_id} ---")
            success_count = 0
            error_count = 0

            for i, (pcm, start, end) in enumerate(chunks):
                # Skip if already done (resume mode)
                if (model_id, video_key, start) in done:
                    print(f"    [{i+1}/{len(chunks)}] {start:.0f}-{end:.0f}s: SKIP (already done)")
                    continue

                t0 = time.time()
                transcript = None
                error = None

                try:
                    if api_type == "whisper":
                        transcript = call_whisper(openai_key, pcm, model_id)
                    else:
                        transcript = call_openrouter_audio(openrouter_key, model_id, pcm)
                    success_count += 1
                except Exception as e:
                    error = str(e)
                    error_count += 1

                latency_ms = int((time.time() - t0) * 1000)

                result = {
                    "video": video_key,
                    "chunk_start": start,
                    "chunk_end": end,
                    "model": model_id,
                    "transcript": transcript,
                    "latency_ms": latency_ms,
                    "error": error,
                }
                results.append(result)

                # Print progress (ascii-safe for Windows cp1252)
                display = transcript[:60] if transcript else f"ERROR: {error[:60]}"
                display = display.encode("ascii", errors="replace").decode("ascii")
                print(f"    [{i+1}/{len(chunks)}] {start:.0f}-{end:.0f}s ({latency_ms}ms): {display}")

                # Save incrementally after each call
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

                # Rate limit: small delay between calls
                time.sleep(0.5)

            print(f"  {model_id}: {success_count} OK, {error_count} errors")

    return results


# ---------------------------------------------------------------------------
# HTML REPORT GENERATOR
# ---------------------------------------------------------------------------


def load_v2_baseline() -> Dict[Tuple[float, float], str]:
    """Load the existing v2 audio baseline for R066."""
    path = REPO_ROOT / "output" / "events_v2_audio.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {(e["start_sec"], e["end_sec"]): e["transcript"] for e in data}


def is_speech(transcript: Optional[str]) -> bool:
    """Check if transcript contains actual speech (not NO_SPEECH or error)."""
    if not transcript:
        return False
    return "NO_SPEECH" not in transcript.upper()


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_report(results: List[dict], output_path: str):
    """Generate an HTML side-by-side comparison report."""
    v2_baseline = load_v2_baseline()

    # Organize: video -> chunk_key -> model -> result
    by_video: Dict[str, Dict[float, Dict[str, dict]]] = {}
    all_models = []
    for r in results:
        v = r["video"]
        if v not in by_video:
            by_video[v] = {}
        cs = r["chunk_start"]
        if cs not in by_video[v]:
            by_video[v][cs] = {}
        by_video[v][cs][r["model"]] = r
        if r["model"] not in all_models:
            all_models.append(r["model"])

    # Compute per-model stats
    model_stats: Dict[str, dict] = {}
    for m in all_models:
        m_results = [r for r in results if r["model"] == m]
        total = len(m_results)
        errors = sum(1 for r in m_results if r["error"])
        speech = sum(1 for r in m_results if is_speech(r.get("transcript")))
        latencies = [r["latency_ms"] for r in m_results if not r["error"]]
        avg_lat = int(sum(latencies) / len(latencies)) if latencies else 0
        model_stats[m] = {
            "total": total,
            "errors": errors,
            "speech_detected": speech,
            "no_speech": total - errors - speech,
            "avg_latency_ms": avg_lat,
        }

    # Build HTML
    html_parts = [
        """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Audio Model Benchmark Report</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }
  h1, h2, h3 { color: #f0f6fc; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
  th, td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; vertical-align: top; }
  th { background: #161b22; color: #f0f6fc; position: sticky; top: 0; z-index: 1; }
  .speech { background: #0d2818; }
  .no-speech { background: #1c1c1c; color: #6e7681; }
  .error { background: #3d1117; color: #f85149; }
  .disagree { border-left: 3px solid #d29922 !important; }
  .chunk-time { font-weight: bold; white-space: nowrap; color: #79c0ff; }
  .stats-table td { text-align: center; }
  .stats-table th { text-align: center; }
  .transcript { max-width: 250px; word-wrap: break-word; font-size: 12px; }
  .latency { font-size: 11px; color: #6e7681; }
  .model-header { writing-mode: vertical-rl; text-orientation: mixed; white-space: nowrap; }
  .summary { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin: 16px 0; }
  .highlight { background: #2d1a00; border-left: 3px solid #d29922; }
  .toggle-bar { display: flex; flex-wrap: wrap; gap: 8px 16px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; margin: 16px 0; position: sticky; top: 0; z-index: 10; }
  .toggle-bar label { display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 13px; color: #c9d1d9; white-space: nowrap; }
  .toggle-bar input[type="checkbox"] { accent-color: #58a6ff; cursor: pointer; }
  .toggle-bar .toggle-title { font-weight: bold; color: #f0f6fc; margin-right: 8px; }
</style>
<script>
function toggleModel(modelId, visible) {
  document.querySelectorAll('[data-model="' + modelId + '"]').forEach(function(el) {
    el.style.display = visible ? '' : 'none';
  });
  // Persist
  var state = JSON.parse(localStorage.getItem('audioBenchHidden') || '{}');
  if (visible) { delete state[modelId]; } else { state[modelId] = true; }
  localStorage.setItem('audioBenchHidden', JSON.stringify(state));
}
function initToggles() {
  var hidden = JSON.parse(localStorage.getItem('audioBenchHidden') || '{}');
  document.querySelectorAll('.toggle-bar input[type="checkbox"]').forEach(function(cb) {
    var mid = cb.getAttribute('data-toggle');
    if (hidden[mid]) {
      cb.checked = false;
      toggleModel(mid, false);
    }
  });
}
window.addEventListener('DOMContentLoaded', initToggles);
</script>
</head><body>
<h1>Audio-to-Text Model Benchmark</h1>
<p>Generated: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
<p>Prompt: <code>""" + escape_html(PROMPT) + """</code></p>
"""
    ]

    # Toggle bar
    html_parts.append('<div class="toggle-bar"><span class="toggle-title">Models:</span>')
    for m in all_models:
        short = m.split("/")[-1] if "/" in m else m
        mid_safe = escape_html(m)
        html_parts.append(
            f'<label><input type="checkbox" checked data-toggle="{mid_safe}" '
            f'onchange="toggleModel(\'{m}\', this.checked)">{escape_html(short)}</label>'
        )
    html_parts.append("</div>")

    # Summary stats table
    html_parts.append("<h2>Model Summary</h2>")
    html_parts.append('<table class="stats-table"><tr><th>Model</th><th>Total Chunks</th>')
    html_parts.append("<th>Speech Detected</th><th>No Speech</th><th>Errors</th><th>Avg Latency</th></tr>")
    for m in all_models:
        s = model_stats[m]
        html_parts.append(
            f'<tr data-model="{escape_html(m)}"><td><b>{escape_html(m)}</b></td>'
            f"<td>{s['total']}</td>"
            f"<td>{s['speech_detected']}</td>"
            f"<td>{s['no_speech']}</td>"
            f'<td class="{"error" if s["errors"] else ""}">{s["errors"]}</td>'
            f"<td>{s['avg_latency_ms']}ms</td></tr>"
        )
    html_parts.append("</table>")

    # Per-video comparison tables
    for video_key in sorted(by_video.keys()):
        chunks = by_video[video_key]
        html_parts.append(f"<h2>Video: {video_key} ({VIDEO_MAP.get(video_key, video_key)})</h2>")

        # Header row
        html_parts.append("<div style='overflow-x: auto;'><table><tr><th>Time</th>")
        for m in all_models:
            short_name = m.split("/")[-1] if "/" in m else m
            html_parts.append(f'<th data-model="{escape_html(m)}">{escape_html(short_name)}</th>')
        html_parts.append("</tr>")

        # Data rows
        for cs in sorted(chunks.keys()):
            model_results = chunks[cs]
            # Check if models disagree on speech presence
            speech_votes = []
            for m in all_models:
                if m in model_results and model_results[m].get("transcript"):
                    speech_votes.append(is_speech(model_results[m]["transcript"]))

            has_disagreement = len(set(speech_votes)) > 1 if speech_votes else False

            html_parts.append("<tr>")
            ce = model_results[all_models[0]]["chunk_end"] if all_models[0] in model_results else cs + CHUNK_SEC
            html_parts.append(f'<td class="chunk-time">{cs:.0f}-{ce:.0f}s</td>')


            for m in all_models:
                dm = f' data-model="{escape_html(m)}"'
                if m not in model_results:
                    html_parts.append(f'<td{dm} class="no-speech">—</td>')
                    continue
                r = model_results[m]
                if r["error"]:
                    cls = "error"
                    text = f"ERROR: {r['error'][:80]}"
                elif is_speech(r["transcript"]):
                    cls = "speech"
                    text = r["transcript"][:120]
                else:
                    cls = "no-speech"
                    text = "NO_SPEECH"

                if has_disagreement:
                    cls += " disagree"

                lat = f'<br><span class="latency">{r["latency_ms"]}ms</span>'
                html_parts.append(f'<td{dm} class="{cls} transcript">{escape_html(text)}{lat}</td>')

            html_parts.append("</tr>")

        html_parts.append("</table></div>")

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\n  Report saved: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark audio-to-text models")
    parser.add_argument(
        "--video",
        choices=list(VIDEO_MAP.keys()),
        help="Test single video (default: all 3)",
    )
    parser.add_argument(
        "--model",
        help="Test single model ID (e.g., whisper-1, google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output/audio_benchmark_raw.json",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip API calls, just regenerate the HTML report from existing raw data",
    )
    args = parser.parse_args()

    # Keys
    openai_key = os.environ.get("OPENAI_API", "")
    openrouter_key = os.environ.get("MY_OPENROUTER_API", "")

    if not openai_key and not args.report_only:
        print("WARNING: OPENAI_API not set, Whisper tests will fail")
    if not openrouter_key and not args.report_only:
        print("WARNING: MY_OPENROUTER_API not set, OpenRouter tests will fail")

    # Select videos
    videos = [args.video] if args.video else list(VIDEO_MAP.keys())

    # Select models
    if args.model:
        matched = [(mid, atype) for mid, atype in MODELS if mid == args.model]
        if not matched:
            print(f"ERROR: Unknown model '{args.model}'. Available:")
            for mid, _ in MODELS:
                print(f"  {mid}")
            sys.exit(1)
        models = matched
    else:
        models = MODELS

    # Output paths
    raw_path = REPO_ROOT / "output" / "audio_optimization" / "audio_benchmark_raw.json"
    report_path = REPO_ROOT / "output" / "audio_optimization" / "audio_benchmark_report.html"
    os.makedirs(REPO_ROOT / "output" / "audio_optimization", exist_ok=True)

    # Load existing results for resume
    existing = []
    if (args.resume or args.report_only) and raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing results from {raw_path}")

    if args.report_only:
        if not existing:
            print("ERROR: No raw data found. Run benchmark first.")
            sys.exit(1)
        generate_report(existing, str(report_path))
        return

    # Print plan
    print(f"\n{'='*60}")
    print(f"  AUDIO BENCHMARK")
    print(f"{'='*60}")
    print(f"  Videos: {videos}")
    print(f"  Models: {[m[0] for m in models]}")
    print(f"  Chunk duration: {CHUNK_SEC}s")
    print(f"  Resume: {args.resume} ({len(existing)} existing results)")
    print(f"{'='*60}")

    results = run_benchmark(videos, models, openai_key, openrouter_key, existing)

    # Save final raw results
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw results saved: {raw_path} ({len(results)} entries)")

    # Generate report
    generate_report(results, str(report_path))

    print(f"\n  Done! Open {report_path} in a browser to compare.")


if __name__ == "__main__":
    main()
