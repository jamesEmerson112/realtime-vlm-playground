"""
Vision Model Benchmark — Raw Response Comparison

Sends the same frames + same prompt to multiple VLMs via OpenRouter
and generates a side-by-side comparison report (HTML + markdown).

No event parsing — compares raw model outputs to judge visual comprehension.

Usage:
    python scripts/benchmark_vision.py                          # Full benchmark
    python scripts/benchmark_vision.py --model meta-llama/llama-4-scout  # Single model
    python scripts/benchmark_vision.py --resume                 # Resume interrupted run
    python scripts/benchmark_vision.py --report-only            # Regenerate reports only
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = str(
    REPO_ROOT
    / "data"
    / "videos_full"
    / "R066-15July-Circuit-Breaker-part2"
    / "Export_py"
    / "Video_pitchshift.mp4"
)
PROCEDURE_PATH = str(
    REPO_ROOT / "data" / "clip_procedures" / "R066-15July-Circuit-Breaker-part2.json"
)
OUTPUT_DIR = REPO_ROOT / "output" / "vision_benchmark"

# Models: (openrouter_id, display_name)
MODELS = [
    ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
    ("meta-llama/llama-4-scout", "Llama 4 Scout"),
    ("meta-llama/llama-4-maverick", "Llama 4 Maverick"),
]

# Benchmark frames: (timestamp_sec, category, description)
# Selected from R066 ground truth to cover steps, errors, mid-step, idle
BENCHMARK_FRAMES = [
    # Errors
    (11.698, "error", "Grabs wrong toolbox"),
    (14.306, "error", "Slides wrong toolbox"),
    (33.639, "error", "Grabs wrong part"),
    (85.004, "error", "Presses button slightly"),
    # Mid-step
    (29.4, "mid_step", "Mid step 1 (grabbing breaker)"),
    (84.5, "mid_step", "Mid step 4 (turning on power)"),
    (120.5, "mid_step", "Mid step 7 (disassemble)"),
    (149.1, "mid_step", "Mid step 9 (insert breaker)"),
    # Step completions
    (49.653, "step_complete", "Step 1: grabs circuit breaker"),
    (57.349, "step_complete", "Step 2: turns on breaker box"),
    (68.657, "step_complete", "Step 3: closes panel door"),
    (98.716, "step_complete", "Step 4: turns on main power"),
    (136.765, "step_complete", "Step 8: returns breaker to toolbox"),
    (157.951, "step_complete", "Step 9: inserts second breaker"),
    (171.716, "step_complete", "Step 11: turns on breaker"),
    # Idle
    (61.0, "idle", "Idle between steps 2-3"),
    (101.2, "idle", "Idle between steps 4-5"),
    (138.5, "idle", "Idle between steps 8-9"),
]

# Sort by timestamp for display
BENCHMARK_FRAMES.sort(key=lambda x: x[0])


# ---------------------------------------------------------------------------
# FRAME EXTRACTION
# ---------------------------------------------------------------------------


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convert BGR frame to base64 JPEG."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frame_to_thumbnail_base64(frame: np.ndarray, width: int = 160) -> str:
    """Resize frame and encode as small JPEG base64 for HTML embedding."""
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    resized = cv2.resize(frame, (width, new_h))
    return frame_to_base64(resized, quality=60)


def extract_frames(
    video_path: str, timestamps: List[float]
) -> Dict[float, Tuple[np.ndarray, str, str]]:
    """
    Extract frames at specific timestamps.

    Returns: {timestamp: (bgr_frame, base64_full, base64_thumbnail)}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {fps:.1f} fps, {total_frames} frames")

    frames = {}
    for ts in sorted(timestamps):
        frame_num = int(ts * fps)
        if frame_num >= total_frames:
            print(f"  WARN: timestamp {ts:.1f}s exceeds video length, skipping")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARN: failed to read frame at {ts:.1f}s, skipping")
            continue
        b64_full = frame_to_base64(frame)
        b64_thumb = frame_to_thumbnail_base64(frame)
        frames[ts] = (frame, b64_full, b64_thumb)

    cap.release()
    print(f"  Extracted {len(frames)}/{len(timestamps)} frames")
    return frames


# ---------------------------------------------------------------------------
# VLM API CALLER
# ---------------------------------------------------------------------------


def call_vlm_benchmark(
    api_key: str, frame_base64: str, prompt: str, model: str
) -> Tuple[str, int]:
    """
    Call a VLM via OpenRouter (non-streaming).

    Returns: (response_text, latency_ms)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Vision Benchmark",
    }
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}"
                        },
                    },
                ],
            }
        ],
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return text, latency_ms


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------


def build_benchmark_prompt(procedure: dict, timestamp_sec: float) -> str:
    """Build a VLM prompt from the template for a specific frame timestamp."""
    template_path = REPO_ROOT / "prompts" / "vlm_prompt.txt"
    template = template_path.read_text(encoding="utf-8")

    task_name = procedure.get("task_name") or procedure.get("task", "unknown")
    steps = procedure.get("steps", [])
    steps_text = "\n".join(
        f"  {s['step_id']}. {s['description']}" for s in steps
    )

    first_step = steps[0] if steps else {"step_id": 1, "description": "unknown"}
    current_step_line = (
        f'Expected current step: {first_step["step_id"]} — "{first_step["description"]}"'
    )

    prompt = template.format(
        task_name=task_name,
        steps_text=steps_text,
        completed_steps="none yet",
        current_step_line=current_step_line,
        audio_line="",
        timestamp_sec=f"{timestamp_sec:.1f}",
    )
    return prompt


# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------


def run_benchmark(
    models: List[Tuple[str, str]],
    frames: Dict[float, Tuple[np.ndarray, str, str]],
    frame_metadata: List[Tuple[float, str, str]],
    procedure: dict,
    api_key: str,
    existing_results: Optional[List[dict]] = None,
) -> List[dict]:
    """Run the full benchmark with resume logic and incremental saves."""
    results = existing_results or []

    # Resume set: skip (model, timestamp) already done
    done = {(r["model"], r["timestamp"]) for r in results}

    raw_path = OUTPUT_DIR / "vision_benchmark_raw.json"

    for model_id, display_name in models:
        print(f"\n  --- Model: {model_id} ({display_name}) ---")
        success_count = 0
        error_count = 0

        for i, (ts, category, desc) in enumerate(frame_metadata):
            if ts not in frames:
                continue

            if (model_id, ts) in done:
                print(f"    [{i+1}/{len(frame_metadata)}] {ts:.1f}s: SKIP (already done)")
                continue

            _, b64_full, _ = frames[ts]
            prompt = build_benchmark_prompt(procedure, ts)
            response_text = None
            error = None
            latency_ms = 0

            try:
                response_text, latency_ms = call_vlm_benchmark(
                    api_key, b64_full, prompt, model_id
                )
                success_count += 1
            except Exception as e:
                error = str(e)
                latency_ms = 0
                error_count += 1

            result = {
                "model": model_id,
                "model_display": display_name,
                "timestamp": ts,
                "category": category,
                "frame_description": desc,
                "response_text": response_text,
                "latency_ms": latency_ms,
                "error": error,
            }
            results.append(result)

            # Print progress (ascii-safe)
            display = response_text[:60] if response_text else f"ERROR: {error[:60]}"
            display = display.encode("ascii", errors="replace").decode("ascii")
            print(f"    [{i+1}/{len(frame_metadata)}] {ts:.1f}s [{category}] ({latency_ms}ms): {display}")

            # Save incrementally
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Rate limit
            time.sleep(0.5)

        print(f"  {model_id}: {success_count} OK, {error_count} errors")

    return results


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


CATEGORY_COLORS = {
    "step_complete": ("#0d2818", "#3fb950", "step"),
    "error": ("#3d1117", "#f85149", "error"),
    "mid_step": ("#0c2d48", "#58a6ff", "mid-step"),
    "idle": ("#1c1c1c", "#6e7681", "idle"),
}


def category_badge(cat: str) -> str:
    """Return an HTML badge span for a category."""
    bg, fg, label = CATEGORY_COLORS.get(cat, ("#1c1c1c", "#c9d1d9", cat))
    return (
        f'<span style="background:{bg}; color:{fg}; padding:2px 6px; '
        f'border-radius:3px; font-size:11px; font-weight:bold;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------


def generate_html_report(
    results: List[dict],
    frames_thumbs: Dict[float, str],
    frame_metadata: List[Tuple[float, str, str]],
    output_path: str,
):
    """Generate an HTML side-by-side comparison report with thumbnails."""

    # Organize: timestamp -> model -> result
    by_ts: Dict[float, Dict[str, dict]] = {}
    all_models = []
    for r in results:
        ts = r["timestamp"]
        if ts not in by_ts:
            by_ts[ts] = {}
        by_ts[ts][r["model"]] = r
        if r["model"] not in all_models:
            all_models.append(r["model"])

    # Model display names
    model_names = {}
    for r in results:
        model_names[r["model"]] = r.get("model_display", r["model"])

    # Per-model stats
    model_stats = {}
    for m in all_models:
        m_results = [r for r in results if r["model"] == m]
        total = len(m_results)
        errors = sum(1 for r in m_results if r["error"])
        latencies = [r["latency_ms"] for r in m_results if not r["error"]]
        avg_lat = int(sum(latencies) / len(latencies)) if latencies else 0
        resp_lens = [len(r["response_text"]) for r in m_results if r["response_text"]]
        avg_len = int(sum(resp_lens) / len(resp_lens)) if resp_lens else 0
        model_stats[m] = {
            "total": total,
            "errors": errors,
            "avg_latency_ms": avg_lat,
            "avg_response_len": avg_len,
        }

    html = [
        """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Vision Model Benchmark Report</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }
  h1, h2, h3 { color: #f0f6fc; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
  th, td { border: 1px solid #30363d; padding: 8px 10px; text-align: left; vertical-align: top; }
  th { background: #161b22; color: #f0f6fc; position: sticky; top: 42px; z-index: 1; }
  .response { max-width: 350px; word-wrap: break-word; font-size: 12px; white-space: pre-wrap; }
  .error-cell { background: #3d1117; color: #f85149; }
  .latency { font-size: 11px; color: #6e7681; margin-top: 4px; }
  .frame-info { white-space: nowrap; }
  .frame-ts { font-weight: bold; color: #79c0ff; font-size: 14px; }
  .frame-desc { font-size: 11px; color: #8b949e; margin-top: 2px; }
  .thumb { border-radius: 4px; margin-bottom: 4px; }
  .stats-table td, .stats-table th { text-align: center; }
  .toggle-bar { display: flex; flex-wrap: wrap; gap: 8px 16px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; margin: 16px 0; position: sticky; top: 0; z-index: 10; }
  .toggle-bar label { display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 13px; color: #c9d1d9; }
  .toggle-bar .toggle-title { font-weight: bold; color: #f0f6fc; margin-right: 8px; }
</style>
<script>
function toggleModel(modelId, visible) {
  document.querySelectorAll('[data-model="' + modelId + '"]').forEach(function(el) {
    el.style.display = visible ? '' : 'none';
  });
}
</script>
</head><body>
<h1>Vision Model Benchmark — Raw Response Comparison</h1>
<p>Generated: """
        + time.strftime("%Y-%m-%d %H:%M:%S")
        + """</p>
<p>Video: R066-15July-Circuit-Breaker-part2 | Frames: """
        + str(len(frame_metadata))
        + """ | Models: """
        + str(len(all_models))
        + """</p>
"""
    ]

    # Toggle bar
    html.append('<div class="toggle-bar"><span class="toggle-title">Models:</span>')
    for m in all_models:
        short = model_names.get(m, m)
        mid_safe = escape_html(m)
        html.append(
            f'<label><input type="checkbox" checked '
            f'onchange="toggleModel(\'{mid_safe}\', this.checked)">'
            f"{escape_html(short)}</label>"
        )
    html.append("</div>")

    # Summary stats
    html.append("<h2>Model Summary</h2>")
    html.append(
        '<table class="stats-table"><tr><th>Model</th><th>Frames</th>'
        "<th>Errors</th><th>Avg Latency</th><th>Avg Response Length</th></tr>"
    )
    for m in all_models:
        s = model_stats[m]
        err_cls = ' class="error-cell"' if s["errors"] else ""
        html.append(
            f'<tr data-model="{escape_html(m)}">'
            f"<td><b>{escape_html(model_names.get(m, m))}</b><br>"
            f'<span style="font-size:11px;color:#6e7681">{escape_html(m)}</span></td>'
            f"<td>{s['total']}</td>"
            f"<td{err_cls}>{s['errors']}</td>"
            f"<td>{s['avg_latency_ms']}ms</td>"
            f"<td>{s['avg_response_len']} chars</td></tr>"
        )
    html.append("</table>")

    # Main comparison table
    html.append("<h2>Frame-by-Frame Comparison</h2>")
    html.append("<div style='overflow-x: auto;'><table><tr><th>Frame</th>")
    for m in all_models:
        short = model_names.get(m, m)
        html.append(f'<th data-model="{escape_html(m)}">{escape_html(short)}</th>')
    html.append("</tr>")

    for ts, category, desc in frame_metadata:
        if ts not in by_ts:
            continue

        # Frame info cell with thumbnail
        thumb_b64 = frames_thumbs.get(ts, "")
        thumb_html = ""
        if thumb_b64:
            thumb_html = (
                f'<img class="thumb" src="data:image/jpeg;base64,{thumb_b64}" '
                f'width="160"><br>'
            )

        badge = category_badge(category)
        html.append(
            f'<tr><td class="frame-info">{thumb_html}'
            f'<span class="frame-ts">{ts:.1f}s</span> {badge}<br>'
            f'<span class="frame-desc">{escape_html(desc)}</span></td>'
        )

        for m in all_models:
            dm = f' data-model="{escape_html(m)}"'
            if m not in by_ts[ts]:
                html.append(f"<td{dm}>—</td>")
                continue

            r = by_ts[ts][m]
            if r["error"]:
                html.append(
                    f'<td{dm} class="error-cell response">'
                    f'ERROR: {escape_html(r["error"][:200])}</td>'
                )
            else:
                text = r["response_text"] or ""
                lat = r["latency_ms"]
                html.append(
                    f'<td{dm} class="response">{escape_html(text)}'
                    f'<div class="latency">{lat}ms</div></td>'
                )

        html.append("</tr>")

    html.append("</table></div>")
    html.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"\n  HTML report: {output_path}")


# ---------------------------------------------------------------------------
# MARKDOWN SUMMARY
# ---------------------------------------------------------------------------


def generate_markdown_summary(results: List[dict], output_path: str):
    """Generate a markdown summary comparing model performance."""
    all_models = []
    for r in results:
        if r["model"] not in all_models:
            all_models.append(r["model"])

    model_names = {}
    for r in results:
        model_names[r["model"]] = r.get("model_display", r["model"])

    lines = []
    lines.append("# Vision Model Benchmark Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("Video: R066-15July-Circuit-Breaker-part2\n")

    # Per-model stats
    lines.append("## Model Stats\n")
    lines.append("| Model | Avg Latency | Avg Response Length | Errors |")
    lines.append("|-------|------------|-------------------|--------|")
    for m in all_models:
        m_results = [r for r in results if r["model"] == m]
        errors = sum(1 for r in m_results if r["error"])
        latencies = [r["latency_ms"] for r in m_results if not r["error"]]
        avg_lat = int(sum(latencies) / len(latencies)) if latencies else 0
        resp_lens = [len(r["response_text"]) for r in m_results if r["response_text"]]
        avg_len = int(sum(resp_lens) / len(resp_lens)) if resp_lens else 0
        lines.append(f"| {model_names.get(m, m)} | {avg_lat}ms | {avg_len} chars | {errors} |")

    # Per-category breakdown
    categories = ["step_complete", "error", "mid_step", "idle"]
    cat_labels = {
        "step_complete": "Step Completions",
        "error": "Errors",
        "mid_step": "Mid-Step",
        "idle": "Idle",
    }

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        # Get unique timestamps for this category
        timestamps = sorted(set(r["timestamp"] for r in cat_results))

        lines.append(f"\n## {cat_labels[cat]} ({len(timestamps)} frames)\n")

        header = "| Timestamp | Description |"
        sep = "|-----------|-------------|"
        for m in all_models:
            header += f" {model_names.get(m, m)} |"
            sep += "-------------|"
        lines.append(header)
        lines.append(sep)

        for ts in timestamps:
            desc = ""
            row = f"| {ts:.1f}s |"
            for r in cat_results:
                if r["timestamp"] == ts:
                    desc = r["frame_description"]
                    break
            row += f" {desc} |"

            for m in all_models:
                match = [r for r in cat_results if r["model"] == m and r["timestamp"] == ts]
                if match:
                    r = match[0]
                    if r["error"]:
                        cell = "ERROR"
                    else:
                        text = (r["response_text"] or "")[:100]
                        text = text.replace("\n", " ").replace("|", "/")
                        cell = text
                else:
                    cell = "—"
                row += f" {cell} |"
            lines.append(row)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Markdown summary: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark vision models on R066 frames")
    parser.add_argument("--model", help="Test single model ID")
    parser.add_argument("--resume", action="store_true", help="Resume from existing raw JSON")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from existing raw data (no API calls)",
    )
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("MY_OPENROUTER_API", "")
    if not api_key and not args.report_only:
        print("ERROR: MY_OPENROUTER_API not set. Add it to .env or export it.")
        sys.exit(1)

    # Select models
    if args.model:
        matched = [(mid, name) for mid, name in MODELS if mid == args.model]
        if not matched:
            print(f"ERROR: Unknown model '{args.model}'. Available:")
            for mid, name in MODELS:
                print(f"  {mid} ({name})")
            sys.exit(1)
        models = matched
    else:
        models = MODELS

    # Output paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = OUTPUT_DIR / "vision_benchmark_raw.json"
    html_path = OUTPUT_DIR / "vision_benchmark_report.html"
    md_path = OUTPUT_DIR / "vision_benchmark_summary.md"

    # Load existing results
    existing = []
    if (args.resume or args.report_only) and raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing results")

    # Load procedure
    with open(PROCEDURE_PATH, encoding="utf-8") as f:
        procedure = json.load(f)

    if args.report_only:
        if not existing:
            print("ERROR: No raw data found. Run benchmark first.")
            sys.exit(1)
        # Still need thumbnails for HTML — extract frames
        print("\n  Extracting frames for thumbnails...")
        timestamps = [ts for ts, _, _ in BENCHMARK_FRAMES]
        frames = extract_frames(VIDEO_PATH, timestamps)
        thumbs = {ts: data[2] for ts, data in frames.items()}
        generate_html_report(existing, thumbs, BENCHMARK_FRAMES, str(html_path))
        generate_markdown_summary(existing, str(md_path))
        return

    # Extract frames
    print(f"\n{'='*60}")
    print(f"  VISION MODEL BENCHMARK")
    print(f"{'='*60}")
    print(f"  Models: {[m[0] for m in models]}")
    print(f"  Frames: {len(BENCHMARK_FRAMES)}")
    print(f"  Total calls: {len(BENCHMARK_FRAMES) * len(models)}")
    print(f"\n  Extracting frames...")
    timestamps = [ts for ts, _, _ in BENCHMARK_FRAMES]
    frames = extract_frames(VIDEO_PATH, timestamps)

    # Run benchmark
    results = run_benchmark(models, frames, BENCHMARK_FRAMES, procedure, api_key, existing)

    # Save final
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Raw results: {raw_path} ({len(results)} entries)")

    # Generate reports
    thumbs = {ts: data[2] for ts, data in frames.items()}
    generate_html_report(results, thumbs, BENCHMARK_FRAMES, str(html_path))
    generate_markdown_summary(results, str(md_path))

    print(f"\n{'='*60}")
    print(f"  DONE — {len(results)} results across {len(models)} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
