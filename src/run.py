"""
VLM Orchestrator — Starter Template

This is where you implement your pipeline. The harness feeds you frames
and audio in real-time. You call VLMs, detect events, and emit them back.

Usage:
    python src/run.py \\
        --procedure data/clip_procedures/CLIP.json \\
        --video path/to/Video_pitchshift.mp4 \\
        --output output/events.json \\
        --speed 1.0
"""

import json
import os
import re
import sys
import io
import time
import wave
import base64
import argparse
import threading

# Force UTF-8 stdout/stderr on Windows — whisper outputs Unicode (em-dashes,
# smart quotes, accents) that PowerShell's default cp1252 encoding can't print,
# which crashes precompute_audio() mid-chunk before the cache gets saved.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

import requests
import numpy as np

# Add scripts/ to path for audio utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from benchmark_audio import extract_audio_chunks, call_whisper, call_openrouter_audio
from audio_enhance import (
    filter_transcript as _filter_transcript,
    is_no_speech,
    normalize_transcript,
    transcripts_similar,
    ENSEMBLE_MODELS,
)


# ==========================================================================
# PIPELINE LOGGER
# ==========================================================================

class PipelineLogger:
    """Logs every pipeline decision to JSON, console, and markdown."""

    def __init__(self, output_path: str):
        self.entries = []
        self.seq = 0
        self.start_time = time.time()
        # Base path without extension, e.g. "output/events"
        self.base_path = str(Path(output_path).with_suffix(""))

    def log(self, time_video: float, event_type: str, data: dict):
        entry = {
            "seq": self.seq,
            "time_wall": round(time.time() - self.start_time, 3),
            "time_video": round(time_video, 2),
            "event_type": event_type,
            "data": data,
        }
        self.seq += 1
        self.entries.append(entry)
        self._print_console(entry)

    def _print_console(self, entry):
        w = f"{entry['time_wall']:7.1f}s"
        v = f"{entry['time_video']:6.1f}s"
        t = entry["event_type"]
        d = entry["data"]

        if t == "run_start":
            print(f"  [{w} | {v}] RUN_START    {d['task_name']} ({d['step_count']} steps) speed={d['speed']}x")
        elif t == "frame_received":
            print(f"  [{w} | {v}] FRAME_RECV   #{d['frame_count']} sampled={d['sampled']} pending={d['pending_calls']}")
        elif t == "frame_skipped":
            print(f"  [{w} | {v}] FRAME_SKIP   #{d['frame_count']} reason={d['reason']}")
        elif t == "vlm_request":
            step_note = f"step={d['expected_step']} " if d.get("expected_step") is not None else ""
            print(f"  [{w} | {v}] VLM_REQ      {step_note}prompt={d['prompt_len']}chars")
        elif t == "vlm_response":
            p = d["parsed"]
            # V5 schema: description is the free-text summary. Fall back to
            # legacy "action" for back-compat with older logs.
            action_brief = (p.get("description") or p.get("action") or "")[:50]
            conf = p.get("confidence", "-")
            print(f"  [{w} | {v}] VLM_RESP     {d['latency_ms']}ms conf={conf} desc=\"{action_brief}\"")
        elif t == "vlm_error":
            print(f"  [{w} | {v}] VLM_ERR      {d['error']}")
        elif t == "emit_rejected":
            reason = str(d.get("reason", ""))[:80]
            print(f"  [{w} | {v}] EMIT_REJECT  {reason}")
        elif t == "state_change":
            print(f"  [{w} | {v}] STATE        {d['field']}: {d['old']} -> {d['new']}")
        elif t == "event_emitted":
            evt = d["event"]
            print(f"  [{w} | {v}] EVENT        {evt['type']} step_id={evt.get('step_id')} conf={evt.get('confidence')}")
        elif t == "idle_check":
            print(f"  [{w} | {v}] IDLE_CHECK   emitted={d['emitted']} reason={d['reason']}")
        elif t == "audio_received":
            source = d.get("source", "live")
            txt = d.get("transcript", "")[:40]
            print(f"  [{w} | {v}] AUDIO_RECV   {d['start_sec']:.1f}-{d['end_sec']:.1f}s [{source}] \"{txt}\"")
        elif t == "audio_skipped":
            print(f"  [{w} | {v}] AUDIO_SKIP   {d['start_sec']:.1f}-{d['end_sec']:.1f}s reason={d['reason']}")
        elif t == "audio_request":
            print(f"  [{w} | {v}] AUDIO_REQ    {d['start_sec']:.1f}-{d['end_sec']:.1f}s model={d['model']}")
        elif t == "audio_response":
            txt = d["transcript"][:60] if d["transcript"] else ""
            print(f"  [{w} | {v}] AUDIO_RESP   {d['latency_ms']}ms speech={d['is_speech']} \"{txt}\"")
        elif t == "audio_error":
            print(f"  [{w} | {v}] AUDIO_ERR    {d['start_sec']:.1f}-{d['end_sec']:.1f}s {d['error']}")
        elif t == "run_end":
            print(f"  [{w} | {v}] RUN_END      vlm_calls={d['total_vlm_calls']} audio_calls={d['total_audio_calls']} events={d['total_events']}")

    def save_json(self):
        path = self.base_path + "_log.json"
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)
        print(f"  Pipeline log (JSON): {path} ({len(self.entries)} entries)")

    def save_markdown(self):
        path = self.base_path + "_log.md"
        lines = []

        # --- Run config ---
        start_entry = next((e for e in self.entries if e["event_type"] == "run_start"), None)
        end_entry = next((e for e in self.entries if e["event_type"] == "run_end"), None)
        lines.append("# Pipeline Run Log\n")
        if start_entry:
            d = start_entry["data"]
            lines.append("## Run Configuration\n")
            lines.append(f"- **Task:** {d['task_name']} ({d['step_count']} steps)")
            lines.append(f"- **Video:** {d.get('video_path', 'N/A')}")
            lines.append(f"- **Speed:** {d['speed']}x")
            lines.append(f"- **VLM Model:** {d.get('model', 'N/A')}")
            lines.append(f"- **Audio Model:** {d.get('audio_model', 'N/A')}")
            lines.append("")

        # --- Summary stats ---
        if end_entry:
            d = end_entry["data"]
            lines.append("## Summary\n")
            lines.append(f"- **VLM calls:** {d['total_vlm_calls']} | **Audio calls:** {d['total_audio_calls']} | **Events emitted:** {d['total_events']}")
            lines.append(f"- **Steps detected:** {d.get('steps_detected', 'N/A')}/{d.get('total_steps', 'N/A')}")
            lines.append(f"- **Errors detected:** {d.get('errors_detected', 0)} | **Idles detected:** {d.get('idles_detected', 0)}")
            if d.get("mean_vlm_latency_ms"):
                lines.append(f"- **Mean VLM latency:** {d['mean_vlm_latency_ms']}ms | **Mean audio latency:** {d.get('mean_audio_latency_ms', 'N/A')}ms")
            lines.append(f"- **Wall duration:** {d['wall_duration']:.1f}s")
            lines.append("")

        # --- Timeline table ---
        timeline_types = {"vlm_request", "vlm_response", "vlm_error", "state_change",
                          "event_emitted", "idle_check", "audio_request", "audio_response", "audio_error",
                          "emit_rejected"}
        timeline = [e for e in self.entries if e["event_type"] in timeline_types]

        if timeline:
            lines.append("## Timeline\n")
            lines.append("| Video Time | Wall Time | Event | Details |")
            lines.append("|------------|-----------|-------|---------|")
            for e in timeline:
                vt = f"{e['time_video']:.1f}s"
                wt = f"{e['time_wall']:.1f}s"
                t = e["event_type"]
                d = e["data"]
                if t == "vlm_request":
                    step_note = f"Step {d['expected_step']} expected, " if d.get("expected_step") is not None else ""
                    detail = f"{step_note}prompt {d['prompt_len']} chars"
                elif t == "vlm_response":
                    p = d.get("parsed", {}) or {}
                    action_brief = (p.get("description") or p.get("action") or "")[:60]
                    detail = f"conf={p.get('confidence', '-')} ({d.get('latency_ms', '-')}ms) \"{action_brief}\""
                elif t == "vlm_error":
                    detail = f"Error: {d['error']}"
                elif t == "state_change":
                    detail = f"`{d['field']}` {d['old']} -> {d['new']}"
                elif t == "event_emitted":
                    evt = d["event"]
                    detail = f"`{evt['type']}` step_id={evt.get('step_id')} conf={evt.get('confidence')}"
                elif t == "idle_check":
                    detail = f"emitted={d['emitted']} ({d['reason']})"
                elif t == "audio_request":
                    detail = f"{d['start_sec']:.1f}-{d['end_sec']:.1f}s via {d['model']}"
                elif t == "audio_response":
                    txt = d["transcript"][:80] if d["transcript"] else "NO_SPEECH"
                    detail = f"speech={d['is_speech']} ({d['latency_ms']}ms) \"{txt}\""
                elif t == "audio_error":
                    detail = f"{d['start_sec']:.1f}-{d['end_sec']:.1f}s Error: {d['error']}"
                elif t == "emit_rejected":
                    cand = d.get("candidate", {}) or {}
                    detail = f"{cand.get('type')} — {str(d.get('reason', ''))[:120]}"
                else:
                    detail = str(d)[:100]
                lines.append(f"| {vt} | {wt} | {t} | {detail} |")
            lines.append("")

        # --- Full VLM prompts & responses ---
        vlm_pairs = []
        for i, e in enumerate(self.entries):
            if e["event_type"] == "vlm_request":
                resp = next((r for r in self.entries[i:] if r["event_type"] == "vlm_response"
                             and r["time_video"] == e["time_video"]), None)
                vlm_pairs.append((e, resp))

        if vlm_pairs:
            lines.append("## Full VLM Prompts & Responses\n")
            for idx, (req, resp) in enumerate(vlm_pairs, 1):
                lines.append(f"### VLM Call #{idx} (video: {req['time_video']:.1f}s)\n")
                lines.append("**Prompt:**")
                lines.append("```")
                lines.append(req["data"]["prompt"])
                lines.append("```\n")
                if resp:
                    lines.append("**Response:**")
                    lines.append("```")
                    lines.append(resp["data"]["response"])
                    lines.append("```\n")
                    p = resp["data"]["parsed"]
                    # V5 schema: 7 boolean/enum/text fields.
                    # Legacy schemas (pre-V5) had hands/action/status/step_id — fall through to else.
                    if "current_step_just_completed" in p or "description" in p:
                        description = (p.get("description") or p.get("action") or "")[:120]
                        lines.append(
                            f"**Parsed (V5):** desc=\"{description}\", "
                            f"current_completed={p.get('current_step_just_completed')}, "
                            f"next_starting={p.get('next_step_starting')}, "
                            f"hands_active={p.get('hands_active')}, "
                            f"error_visible={p.get('error_visible')}, "
                            f"confidence={p.get('confidence')}\n"
                        )
                    elif "hands" in p or "action" in p:
                        action = (p.get("action") or p.get("description") or "")[:120]
                        hands = (p.get("hands") or "")[:80]
                        lines.append(
                            f"**Parsed (observer):** action=\"{action}\", "
                            f"confidence={p.get('confidence')}, "
                            f"hands=\"{hands}\"\n"
                        )
                    else:
                        lines.append(
                            f"**Parsed:** {p.get('status', '-')}, "
                            f"step_id={p.get('step_id')}, "
                            f"confidence={p.get('confidence')}\n"
                        )
                else:
                    lines.append("**Response:** (error or missing)\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Pipeline log (MD):   {path}")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format


# ==========================================================================
# VLM API HELPER (provided — feel free to modify)
# ==========================================================================

def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemini-3.1-flash-image-preview",
    stream: bool = False,
) -> str:
    """
    Call a VLM via OpenRouter.

    Args:
        api_key: OpenRouter API key
        frame_base64: Base64-encoded JPEG frame
        prompt: Text prompt
        model: OpenRouter model string
        stream: If True, use streaming (SSE) responses for lower time-to-first-token

    Returns:
        Model response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    if stream:
        # Streaming: read SSE chunks as they arrive
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_text += delta["content"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return full_text
    else:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ==========================================================================
# AUDIO API HELPER
# ==========================================================================

def call_audio_llm(api_key: str, pcm_bytes: bytes, prompt: str) -> str:
    """
    Transcribe audio via OpenRouter using gpt-4o-audio-preview.

    Args:
        api_key: OpenRouter API key
        pcm_bytes: Raw PCM audio (16kHz, mono, 16-bit signed LE)
        prompt: Text prompt for the audio model

    Returns:
        Model response text
    """
    # Wrap raw PCM in a WAV container
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": "openai/gpt-4o-audio-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ==========================================================================
# PIPELINE CONSTANTS
# ==========================================================================

IDLE_THRESHOLD_SEC = 3.0


# ==========================================================================
# YOUR PIPELINE — IMPLEMENT THESE CALLBACKS
# ==========================================================================

# Audio correction keywords — any of these appearing in a transcript within the
# 10s sliding audio window triggers a live error_detected event (source="audio").
# Matched with word boundaries so "no" doesn't hit "now", "not" doesn't hit
# "note", etc. Multi-word phrases are matched by simple substring.
_SINGLE_WORD_CORRECTIONS = {
    "no", "stop", "wrong", "don't", "dont", "incorrect", "mistake", "careful",
}
_PHRASE_CORRECTIONS = {
    "not that", "not like", "other one", "hold on", "wait no", "oh no",
}


def _find_correction_hit(text: str) -> Optional[str]:
    """Return the first matching keyword/phrase, or None. Word-boundary safe."""
    lower = text.lower()
    for phrase in _PHRASE_CORRECTIONS:
        if phrase in lower:
            return phrase
    for word in _SINGLE_WORD_CORRECTIONS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return word
    return None


# V4: Tier-2 audio correction vocab — only active BEFORE the first step_completion
# emits, since early-stream audio is coaching-heavy and uses softer correction
# language than the Tier-1 sharp "no / stop / wrong" vocab.
_TIER2_PHRASE_CORRECTIONS = {
    "try again", "once more", "let's try", "that's not", "other one",
    "click harder", "press harder",
    "pick up the", "put down the",
    # Removed "go to the" / "move to the" — too generic, fired on forward
    # instructions like "go to the red toolbox and open the top compartment"
    # (R066 t=5.0s FP, 2026-04-16).
}


def _find_tier2_hit(text: str) -> Optional[str]:
    """Return the first Tier-2 phrase match, or None."""
    lower = text.lower()
    for phrase in _TIER2_PHRASE_CORRECTIONS:
        if phrase in lower:
            return phrase
    return None


# V4: Noise transcripts — boot-up / system phrases to skip before routing into
# detectors. Applied once at ingest (on_audio and the cache path).
_AUDIO_NOISE_PATTERNS = (
    "please wait. looking for a server heartbeat",
    "capturing.",
    "no_speech",
)


def _is_noise_transcript(text: str) -> bool:
    """True if the transcript is empty, whitespace, NO_SPEECH, or a known boot phrase."""
    if not text:
        return True
    t = text.lower().strip()
    if not t:
        return True
    return any(p in t for p in _AUDIO_NOISE_PATTERNS)


# ==========================================================================
# AUDIO TRANSCRIPT CACHE — skip setup+verify on repeat runs of the same video
# ==========================================================================

_AUDIO_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "audio_cache"


def _audio_cache_key(video_path: str) -> str:
    """Derive a stable per-video stem. Grandparent dir name — all videos
    are named Video_pitchshift.mp4 so the parent's parent is the discriminator."""
    p = Path(video_path)
    return p.parent.parent.name


def audio_cache_path(video_path: str) -> Path:
    return _AUDIO_CACHE_DIR / f"{_audio_cache_key(video_path)}.json"


def load_audio_cache(video_path: str):
    """Returns (transcripts_dict, audio_log) on valid cache hit; (None, None) otherwise.

    Validates: file exists, size matches, mtime matches (+/- 1ms).
    """
    cache_file = audio_cache_path(video_path)
    if not cache_file.exists():
        return None, None
    video = Path(video_path)
    if not video.exists():
        return None, None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    try:
        stat = video.stat()
        if data.get("video_size") != stat.st_size:
            return None, None
        if abs(float(data.get("video_mtime", 0.0)) - stat.st_mtime) > 1e-3:
            return None, None
        transcripts = {float(k): v for k, v in data["transcripts"].items()}
        audio_log = [(float(s), float(e), t) for s, e, t in data["audio_log"]]
    except (KeyError, TypeError, ValueError):
        return None, None
    return transcripts, audio_log


def _write_audio_cache_file(path: Path, video_path: str, transcripts: dict,
                            audio_log: list, verify_ran: bool) -> Path:
    """Internal writer — JSON-header construction only. Caller provides the target path.

    Used by save_audio_cache() for the authoritative data/audio_cache/ location,
    and by main()'s fresh-run path for the debug sister-file beside --output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)
    stat = video.stat()
    try:
        rel = video.relative_to(Path.cwd())
    except ValueError:
        rel = video
    data = {
        "video_name": _audio_cache_key(video_path),
        "video_relpath": str(rel).replace("\\", "/"),
        "video_size": stat.st_size,
        "video_mtime": stat.st_mtime,
        "audio_chunk_sec": 5.0,
        "cached_at": datetime.utcnow().isoformat() + "Z",
        "verify_ran": verify_ran,
        "transcripts": {f"{k}": v for k, v in sorted(transcripts.items())},
        "audio_log": [[float(s), float(e), t] for s, e, t in audio_log],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_audio_cache(video_path: str, transcripts: dict, audio_log: list, verify_ran: bool = True):
    """Write the authoritative cache file to data/audio_cache/<name>.json."""
    return _write_audio_cache_file(
        audio_cache_path(video_path), video_path, transcripts, audio_log, verify_ran
    )


class Pipeline:
    """
    VLM orchestration pipeline with pre-computed audio transcription.

    Audio flow: Setup (whisper-1 + filter) → Verify (4-model ensemble) → Pipeline (lookup).
    Uses Gemini 3.0 Flash for video frame analysis. Audio corrections signal errors.
    """

    def __init__(self, harness: StreamingHarness, api_key: str, procedure: Dict[str, Any],
                 output_path: str = "output/events.json", speed: float = 1.0,
                 openai_key: str = "", use_audio_cache: bool = False,
                 vlm_model: str = "google/gemini-3.1-flash-image-preview"):
        self.harness = harness
        self.api_key = api_key
        self.vlm_model = vlm_model
        # OpenAI key — used for real-time whisper-1 transcription in on_audio
        # when use_audio_cache=False.
        self.openai_key = openai_key
        # True = on_audio does dict lookup against _precomputed_audio.
        # False = on_audio spawns a whisper-1 worker per chunk.
        self.use_audio_cache = use_audio_cache
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]

        # Step tracking — detectors emit step_completion candidates directly.
        self.current_step_index = 0
        self.completed_steps: set = set()

        # V5 step detection — single vote deque shared across the "current" step.
        # Holds booleans: True when VLM asserts current_step_just_completed OR
        # next_step_starting on that frame. Rule fires on 2+ True in last 3.
        # Cleared when current_step_index advances (in _detect_step_completion).
        self._step_vote_history: deque = deque(maxlen=3)
        self._emitted_steps: set = set()
        # V5 queue-confirm: when voter fires, we DON'T emit immediately.
        # Instead we advance the queue and wait for the VLM to confirm the
        # NEW current step is happening (current_step_happening=yes).
        # Only then do we emit step_completion for the previous step.
        # This prevents premature emission and cascade desync.
        self._pending_step_id: Optional[int] = None
        self._pending_step_ts: float = 0.0
        self._pending_step_conf: float = 0.0
        self._pending_step_rule: str = ""
        self._pending_step_timeout: float = 15.0  # seconds before force-emit
        self._emitted_error_timestamps: list = []  # 5s dedup on error_detected
        self._last_frame_description: str = ""

        # Observation buffer — unbounded, used for debugging.
        self.observation_buffer: list = []

        # Frame sampling
        self.frame_count = 0
        self.pending_calls = 0
        self.lock = threading.Lock()

        # Audio state
        self.recent_transcript = ""
        self.recent_transcript_time = 0.0
        self.audio_history: deque = deque(maxlen=3)  # last 3 chunks, filtered to 10s window
        self.pending_audio_calls = 0
        self.audio_log: list = []  # [(start_sec, end_sec, transcript)]
        self._precomputed_audio: dict = {}  # {start_sec: transcript}
        self._audio_chunks: list = []  # [(pcm, start, end)]

        # V4 idle + current-video-time tracking
        # Populated in _analyze_frame; idle worker thread polls these under lock.
        self._current_video_time_obs = 0.0
        self._last_hands_active_time = 0.0
        self._first_frame_seen = False
        self._idle_active = False
        self._idle_start = 0.0
        self._last_emit_time = 0.0

        # Idle worker thread (1x only)
        self._idle_thread: Optional[threading.Thread] = None
        self._idle_stop = threading.Event()

        # Load prompt templates from files
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        with open(prompts_dir / "vlm_prompt_test.txt", "r", encoding="utf-8") as f:
            self._vlm_template = f.read()
        with open(prompts_dir / "audio_prompt.txt", "r", encoding="utf-8") as f:
            self._audio_prompt = f.read().strip()

        # Logger
        self.logger = PipelineLogger(output_path)
        self._speed = speed
        self._vlm_latencies = []
        self._audio_latencies = []
        self.logger.log(0.0, "run_start", {
            "task_name": self.task_name,
            "step_count": len(self.steps),
            "steps": self.steps,
            "speed": speed,
            "video_path": harness.video_path,
            "model": self.vlm_model,
            "audio_model": "openai/gpt-4o-audio-preview",
        })

    def _build_prompt(self, timestamp_sec: float) -> str:
        with self.lock:
            idx = self.current_step_index
            audio_entries = [
                (s, e, t) for s, e, t in self.audio_history
                if (timestamp_sec - s) < 10.0 and not _is_noise_transcript(t)
            ]

        # Current + next step windows — VLM only sees these two, not the full procedure.
        if idx < len(self.steps):
            cs = self.steps[idx]
            current_step_block = f"CURRENT STEP ({cs['step_id']}): {cs['description']}"
        else:
            current_step_block = "CURRENT STEP: all steps completed"

        if idx + 1 < len(self.steps):
            ns = self.steps[idx + 1]
            next_step_block = f"NEXT STEP ({ns['step_id']}): {ns['description']}"
        else:
            next_step_block = "NEXT STEP: none (last step of procedure)"

        if audio_entries:
            lines = []
            for s, e, t in audio_entries:
                lines.append(f'  [{s:.0f}-{e:.0f}s] "{t}"')
            audio_line = (
                "Recent audio from the scene:\n"
                + "\n".join(lines) + "\n"
                "Negative language, corrections, or warnings (e.g., \"no\", \"stop\", \"wrong\", frustration) suggest an error is happening.\n"
            )
        else:
            audio_line = ""

        if self._last_frame_description:
            previous_frame = f'Previous frame: "{self._last_frame_description}"'
        else:
            previous_frame = ""

        return self._vlm_template.format(
            task_name=self.task_name,
            current_step_block=current_step_block,
            next_step_block=next_step_block,
            audio_line=audio_line,
            timestamp_sec=f"{timestamp_sec:.1f}",
            previous_frame=previous_frame,
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: keyword matching
        lower = response.lower()
        if "step_complete" in lower or "step complete" in lower:
            step_match = re.search(r"step[_\s]*(\d+)", lower)
            step_id = int(step_match.group(1)) if step_match else None
            return {"status": "step_complete", "step_id": step_id, "description": response[:200], "confidence": 0.5}
        if "error" in lower:
            return {"status": "error", "step_id": None, "description": response[:200], "error_description": response[:200], "confidence": 0.5}
        if "idle" in lower:
            return {"status": "idle", "step_id": None, "description": response[:200], "confidence": 0.5}
        return {"status": "in_progress", "step_id": None, "description": response[:200], "confidence": 0.3}

    # ======================================================================
    # V4 DETECTORS — each produces a candidate event dict, emitted directly.
    # ======================================================================

    def _on_new_observation(self, observation: dict, audio_window: list):
        """Called after each VLM observation. Runs all video-side detectors."""
        # Step completion (4 parallel rules)
        step_cand = self._detect_step_completion(observation)
        if step_cand is not None:
            self._queue_candidate(step_cand, observation, audio_window)

        # Video-side error
        video_err_cand = self._detect_video_error(observation)
        if video_err_cand is not None:
            self._queue_candidate(video_err_cand, observation, audio_window)

    def _make_step_candidate(self, ts: float, step_id: int, conf: float,
                             rule: str, observation: dict) -> dict:
        return {
            "timestamp_sec": float(ts),
            "type": "step_completion",
            "detector_source": f"step_tracker:{rule}",
            "evidence_text": (
                f"Step {step_id} implicated by rule {rule}; "
                f"description='{observation.get('description', '')[:120]}'"
            ),
            "event": {
                "timestamp_sec": float(ts),
                "type": "step_completion",
                "step_id": int(step_id),
                "confidence": round(float(conf), 3),
                "source": "video",
                "description": (
                    f"Step {step_id} end-state inferred via {rule}"
                ),
                "vlm_description": observation.get("description", ""),
            },
        }

    def _make_video_error_candidate(self, ts: float, err_visible: str,
                                    err_type: str, conf: float,
                                    observation: dict,
                                    source: str = "video") -> dict:
        return {
            "timestamp_sec": float(ts),
            "type": "error_detected",
            "detector_source": "video_error_detector",
            "evidence_text": (
                f"VLM reports visible error ('{err_visible[:80]}', type={err_type}); "
                f"description='{observation.get('description', '')[:80]}'"
            ),
            "event": {
                "timestamp_sec": float(ts),
                "type": "error_detected",
                "error_type": err_type,
                "severity": "warning",
                "confidence": round(float(conf), 3),
                "source": source,
                "description": f"VLM: {err_visible[:200]}",
            },
        }

    def _make_audio_error_candidate(self, ts: float, transcript: str,
                                    hit: str, tier: int) -> dict:
        return {
            "timestamp_sec": float(ts),
            "type": "error_detected",
            "detector_source": f"audio_error_detector:tier{tier}",
            "evidence_text": (
                f"Audio tier-{tier} correction '{hit}' in transcript: "
                f"\"{transcript.strip()[:120]}\""
            ),
            "event": {
                "timestamp_sec": float(ts),
                "type": "error_detected",
                "error_type": "wrong_action",
                "severity": "warning",
                "confidence": 0.65 if tier == 1 else 0.55,
                "source": "audio",
                "description": f"Audio correction ('{hit}'): {transcript.strip()[:200]}",
            },
        }

    def _make_idle_candidate(self, ts_mid: float, idle_start: float,
                             idle_end: float) -> dict:
        duration = max(0.0, idle_end - idle_start)
        return {
            "timestamp_sec": float(ts_mid),
            "type": "idle_detected",
            "detector_source": "idle_detector",
            "evidence_text": (
                f"No hands_active observations between {idle_start:.1f}s and "
                f"{idle_end:.1f}s ({duration:.1f}s span)"
            ),
            "event": {
                "timestamp_sec": float(ts_mid),
                "type": "idle_detected",
                "confidence": 0.7,
                "source": "video",
                "description": (
                    f"Technician idle from {idle_start:.1f}-{idle_end:.1f}s "
                    f"({duration:.1f}s)"
                ),
            },
        }

    def _detect_step_completion(self, observation: dict) -> Optional[dict]:
        """Queue-confirm step detector.

        When the voter fires (2-of-3 on completed/next_starting), we DON'T
        emit immediately. Instead we store the step as "pending", advance
        the queue so the VLM sees the NEXT step, and wait for the VLM to
        confirm the new step is happening (current_step_happening=yes).
        Only then do we emit step_completion for the previous step.

        This prevents cascade desync: step N can't be emitted too early
        because we need proof that step N+1 has actually started.
        """
        ts = float(observation["timestamp_sec"])
        happening = bool(observation.get("current_step_happening", False))
        completed = bool(observation.get("current_step_just_completed", False))
        next_starting = bool(observation.get("next_step_starting", False))
        signal = completed or next_starting

        with self.lock:
            # --- Normal voting for current step ---
            idx = self.current_step_index
            if idx >= len(self.steps):
                return None
            current_step_id = self.steps[idx]["step_id"]
            if current_step_id in self._emitted_steps:
                return None

            self._step_vote_history.append(signal)

            if sum(1 for v in self._step_vote_history if v) >= 2:
                try:
                    conf = float(observation.get("confidence", 0.85))
                except (TypeError, ValueError):
                    conf = 0.85
                conf = max(0.5, min(1.0, conf))
                if completed and next_starting:
                    rule_tag = "both_signals"
                elif completed:
                    rule_tag = "current_completed"
                else:
                    rule_tag = "next_starting"

                # Emit immediately — queue-confirm disabled.
                # Advance step index and clear voter for next step.
                self._emitted_steps.add(current_step_id)
                self.completed_steps.add(current_step_id)
                while (self.current_step_index < len(self.steps)
                       and self.steps[self.current_step_index]["step_id"]
                       in self._emitted_steps):
                    self.current_step_index += 1
                self._step_vote_history.clear()
                return self._make_step_candidate(
                    ts, current_step_id, conf, rule_tag, observation
                )

        return None

    def _detect_video_error(self, observation: dict) -> Optional[dict]:
        """
        Emit a video-error candidate only when:
          - error_visible is a non-empty string AND
          - error_type is a valid VALID_ERROR_TYPES value AND
          - confidence >= 0.6 AND
          - audio corroboration exists within ±10s OR VLM confidence >= 0.80.

        The audio gate aligns VLM detections with the GT's implicit definition
        (error = instructor correction) while preserving high-confidence safety
        violations the instructor doesn't vocalize.
        """
        ts = float(observation["timestamp_sec"])
        err_vis = observation.get("error_visible")
        err_type = observation.get("error_type")
        try:
            conf = float(observation.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5

        if not err_vis or not isinstance(err_vis, str):
            return None
        if err_vis.strip().lower() in ("", "null", "none", "unclear"):
            return None
        if err_type not in StreamingHarness.VALID_ERROR_TYPES:
            return None
        if conf < 0.6:
            return None

        with self.lock:
            # 5s dedup against already-emitted error timestamps.
            if any(abs(ts - t) < 5.0 for t in self._emitted_error_timestamps):
                return None
            # Audio cross-reference: check if any recent transcript contains
            # correction vocabulary (Tier-1 or Tier-2) within ±10s.
            audio_corroborated = any(
                _find_correction_hit(t) or _find_tier2_hit(t)
                for s, e, t in self.audio_history
                if abs(ts - s) < 10.0 and not _is_noise_transcript(t)
            )

        if audio_corroborated:
            # Both modalities agree — boost confidence, tag source as "both".
            conf = min(1.0, conf + 0.15)
            return self._make_video_error_candidate(
                ts, err_vis.strip(), err_type, conf, observation, source="both"
            )
        elif conf >= 0.80:
            # VLM-only but very confident — likely a real safety violation
            # the instructor didn't vocalize. Let it through.
            return self._make_video_error_candidate(
                ts, err_vis.strip(), err_type, conf, observation
            )
        else:
            # VLM-only at moderate confidence — suppress to align with GT.
            return None

    def _on_new_transcript(self, transcript_text: str, start_sec: float,
                          end_sec: float):
        """Audio-error detector runs here. Called from transcribe workers + cache path."""
        if _is_noise_transcript(transcript_text):
            return

        with self.lock:
            first_step_emitted = bool(self._emitted_steps)

        # Tier 1 — always active.
        hit = _find_correction_hit(transcript_text)
        tier = 1
        # Tier 2 — only before the first step_completion is confirmed.
        if not hit and not first_step_emitted:
            hit = _find_tier2_hit(transcript_text)
            tier = 2 if hit else tier

        if not hit:
            return

        # 5s dedup in video-time.
        with self.lock:
            if any(abs(start_sec - t) < 5.0 for t in self._emitted_error_timestamps):
                return

        cand = self._make_audio_error_candidate(start_sec, transcript_text, hit, tier)
        audio_window = [(start_sec, end_sec, transcript_text)]
        self._queue_candidate(cand, observation=None, audio_window=audio_window)

    # ======================================================================
    # DIRECT EMIT
    # ======================================================================

    def _queue_candidate(self, candidate: dict, observation: Optional[dict],
                         audio_window: list):
        """Emit a detector candidate directly to the harness.

        Name retained for call-site stability. There is no queue anymore —
        every candidate becomes a live harness.emit_event() call. Mother
        verifier was retired after three 1x R066 runs showed error_f1 dropped
        from 0.500 (live emit) to 0.000 (post-mother). See shipping notes.
        """
        event = candidate["event"]
        ts = float(event.get("timestamp_sec", 0.0))
        try:
            self.harness.emit_event(event)
            with self.lock:
                if event.get("type") == "step_completion":
                    sid = int(event.get("step_id"))
                    self._emitted_steps.add(sid)
                    self.completed_steps.add(sid)
                elif event.get("type") == "error_detected":
                    self._emitted_error_timestamps.append(ts)
            self.logger.log(ts, "event_emitted", {
                "event": event, "candidate": candidate,
            })
        except ValueError as e:
            self.logger.log(ts, "emit_rejected", {
                "reason": f"harness rejected: {e}",
                "candidate": candidate,
            })

    def _idle_worker(self):
        """Background idle detector. Polls once per wall-second.

        Uses video-time deltas (not wall-time) so it stays correct at non-1x speeds.
        A span is closed when hands_active returns AND has been absent >= IDLE_THRESHOLD_SEC.
        """
        while not self._idle_stop.is_set():
            # Poll at wall-second cadence; at 1x this ≈ 1 video-second.
            if self._idle_stop.wait(1.0):
                break

            with self.lock:
                seen = self._first_frame_seen
                current = self._current_video_time_obs
                last_active = self._last_hands_active_time
                idle_active = self._idle_active
                idle_start = self._idle_start

            if not seen:
                continue

            gap = current - last_active
            if not idle_active and gap >= IDLE_THRESHOLD_SEC:
                with self.lock:
                    self._idle_active = True
                    self._idle_start = last_active
            elif idle_active and gap < IDLE_THRESHOLD_SEC:
                idle_end = current
                ts_mid = (idle_start + idle_end) / 2.0
                with self.lock:
                    self._idle_active = False
                    audio_snap = [
                        (s, e, t) for s, e, t in self.audio_history
                        if abs(ts_mid - s) < 10.0 and not _is_noise_transcript(t)
                    ]
                cand = self._make_idle_candidate(ts_mid, idle_start, idle_end)
                self._queue_candidate(cand, observation=None, audio_window=audio_snap)

        # On shutdown, close any open idle span so we don't lose the final one.
        with self.lock:
            idle_active = self._idle_active
            idle_start = self._idle_start
            current = self._current_video_time_obs
            self._idle_active = False
        if idle_active and current > idle_start + IDLE_THRESHOLD_SEC:
            ts_mid = (idle_start + current) / 2.0
            cand = self._make_idle_candidate(ts_mid, idle_start, current)
            self._queue_candidate(cand, observation=None, audio_window=[])

    def start_idle_worker(self):
        """Spawn the idle watcher thread (1x only — the watcher's 1s polling
        doesn't scale to higher speeds). No-op at 2x+."""
        if self._speed >= 2.0:
            return
        if self._idle_thread is not None:
            return
        self._idle_stop.clear()
        self._idle_thread = threading.Thread(
            target=self._idle_worker, name="idle_worker", daemon=True,
        )
        self._idle_thread.start()

    def stop_idle_worker(self):
        """Signal the idle watcher and join. No queue to drain anymore."""
        if self._idle_thread is None:
            return
        self._idle_stop.set()
        self._idle_thread.join(timeout=5.0)
        self._idle_thread = None

    def _analyze_frame(self, timestamp_sec: float, frame_base64: str):
        """
        Real-time pipeline: VLM observes, detectors decide, harness.emit_event() fires live.
        """
        try:
            prompt = self._build_prompt(timestamp_sec)
            self.logger.log(timestamp_sec, "vlm_request", {
                "prompt_len": len(prompt),
                "prompt": prompt,
            })

            t0 = time.time()
            response = call_vlm(self.api_key, frame_base64, prompt, model=self.vlm_model)
            latency_ms = round((time.time() - t0) * 1000)
            self._vlm_latencies.append(latency_ms)

            parsed = self._parse_response(response)

            self.logger.log(timestamp_sec, "vlm_response", {
                "response": response,
                "latency_ms": latency_ms,
                "parsed": parsed,
            })

            # V5 observer fields — 7 fields + timestamp. VLM produces only
            # booleans, enums, and free text — no step ids, no progress %.
            # hands_active: fail-safe to True when VLM omits it — spurious
            # idle is worse than spurious activity.
            raw_hands_active = parsed.get("hands_active")
            if raw_hands_active is None:
                hands_active = True
            else:
                hands_active = bool(raw_hands_active)

            err_type = parsed.get("error_type")
            if err_type not in StreamingHarness.VALID_ERROR_TYPES:
                err_type = None

            try:
                confidence = float(parsed.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            observation = {
                "timestamp_sec": timestamp_sec,
                "current_step_happening": str(
                    parsed.get("current_step_happening", "no")
                ).lower().strip() == "yes",
                "current_step_just_completed": bool(
                    parsed.get("current_step_just_completed", False)
                ),
                "next_step_starting": bool(
                    parsed.get("next_step_starting", False)
                ),
                "hands_active": hands_active,
                "error_visible": parsed.get("error_visible"),
                "error_type": err_type,
                "confidence": confidence,
                "description": str(parsed.get("description", "")),
                "raw_response": response[:1000],
            }

            self._last_frame_description = observation.get("description", "")

            with self.lock:
                self.observation_buffer.append(observation)
                audio_snapshot = [
                    (s, e, t) for s, e, t in self.audio_history
                    if not _is_noise_transcript(t)
                ]
                self._current_video_time_obs = max(
                    self._current_video_time_obs, float(timestamp_sec)
                )
                if not self._first_frame_seen:
                    self._first_frame_seen = True
                    # Seed last-active at first frame so early frames don't auto-idle.
                    self._last_hands_active_time = float(timestamp_sec)
                if hands_active:
                    self._last_hands_active_time = float(timestamp_sec)

            self._on_new_observation(observation, audio_snapshot)

        except Exception as e:
            self.logger.log(timestamp_sec, "vlm_error", {"error": str(e)})
        finally:
            with self.lock:
                self.pending_calls -= 1

    def _transcribe_chunk(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """Thread body: whisper-1 + hallucination filter on a single 5s audio chunk.

        Mutates _precomputed_audio / audio_log / audio_history / recent_transcript
        under self.lock so on_frame() / _build_prompt() see consistent state.
        Called by on_audio() when real-time mode is active (self.openai_key set).
        """
        try:
            self.logger.log(start_sec, "audio_request", {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "model": "whisper-1",
                "source": "realtime",
            })

            t0 = time.time()
            try:
                raw = call_whisper(self.openai_key, audio_bytes, "whisper-1")
            except Exception as e:
                self.logger.log(start_sec, "audio_error", {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "error": str(e),
                })
                with self.lock:
                    self.audio_log.append((start_sec, end_sec, "NO_SPEECH"))
                return
            latency_ms = round((time.time() - t0) * 1000)
            self._audio_latencies.append(latency_ms)

            filtered, was_filtered = _filter_transcript(raw)
            is_speech = not is_no_speech(filtered)
            noise = _is_noise_transcript(filtered)

            with self.lock:
                self.audio_log.append((start_sec, end_sec, filtered))
                if is_speech and not noise:
                    self._precomputed_audio[start_sec] = filtered
                    self.recent_transcript = filtered
                    self.recent_transcript_time = start_sec
                    self.audio_history.append((start_sec, end_sec, filtered))

            self.logger.log(start_sec, "audio_response", {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "transcript": filtered,
                "raw": raw[:200],
                "latency_ms": latency_ms,
                "is_speech": is_speech,
                "was_filtered": was_filtered,
                "source": "realtime",
            })

            # V4 audio-error detector — fires on Tier-1 keywords always,
            # Tier-2 only before the first step_completion is confirmed.
            if is_speech and not noise:
                self._on_new_transcript(filtered, start_sec, end_sec)
        finally:
            with self.lock:
                self.pending_audio_calls -= 1

    def _transcribe_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        try:
            prompt = self._audio_prompt
            self.logger.log(start_sec, "audio_request", {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "model": "openai/gpt-4o-audio-preview",
                "prompt": prompt,
            })

            t0 = time.time()
            transcript = call_audio_llm(self.api_key, audio_bytes, prompt)
            latency_ms = round((time.time() - t0) * 1000)
            self._audio_latencies.append(latency_ms)

            clean = transcript.strip()
            is_speech = "NO_SPEECH" not in transcript.upper()

            with self.lock:
                self.audio_log.append((start_sec, end_sec, clean))

            self.logger.log(start_sec, "audio_response", {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "transcript": clean,
                "latency_ms": latency_ms,
                "is_speech": is_speech,
            })

            if not is_speech:
                return

            with self.lock:
                self.recent_transcript = clean
                self.recent_transcript_time = start_sec
                self.audio_history.append((start_sec, end_sec, clean))

        except Exception as e:
            self.logger.log(start_sec, "audio_error", {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "error": str(e),
            })
        finally:
            with self.lock:
                self.pending_audio_calls -= 1

    # ------------------------------------------------------------------
    # PRE-COMPUTED AUDIO: SETUP + VERIFY
    # ------------------------------------------------------------------

    def precompute_audio(self, video_path: str, openai_key: str):
        """Step 1: Extract all audio chunks and transcribe with whisper-1 + hallucination filter."""
        print("  [SETUP] Extracting audio chunks...")
        self._audio_chunks = extract_audio_chunks(video_path)
        total = len(self._audio_chunks)
        print(f"  [SETUP] {total} chunks ({5.0}s each). Transcribing with whisper-1...")

        for i, (pcm, start, end) in enumerate(self._audio_chunks):
            try:
                transcript = call_whisper(openai_key, pcm, "whisper-1")
                filtered, was_filtered = _filter_transcript(transcript)
            except Exception as e:
                filtered = "NO_SPEECH"
                was_filtered = False
                print(f"    [{i+1}/{total}] {start:.0f}-{end:.0f}s: ERROR {e}")
                continue

            self.audio_log.append((start, end, filtered))

            if not is_no_speech(filtered):
                self._precomputed_audio[start] = filtered

            tag = " [filtered]" if was_filtered else ""
            display = filtered[:50] if not is_no_speech(filtered) else "NO_SPEECH"
            print(f"    [{i+1}/{total}] {start:.0f}-{end:.0f}s: {display}{tag}")
            time.sleep(0.3)

        speech_count = len(self._precomputed_audio)
        print(f"  [SETUP] Done. {speech_count} speech / {total - speech_count} silence\n")

    def verify_audio(self, openai_key: str, openrouter_key: str):
        """Step 2: Run ensemble on all chunks, auto-correct whisper transcripts."""
        total = len(self._audio_chunks)
        print(f"  [VERIFY] Running ensemble (3 models) on {total} chunks...")

        corrections = 0
        for i, (pcm, start, end) in enumerate(self._audio_chunks):
            whisper_transcript = self._precomputed_audio.get(start, "NO_SPEECH")

            # Run the other 3 models
            model_outputs = {"whisper-1": whisper_transcript}
            for model_id, api_type in ENSEMBLE_MODELS:
                if model_id == "whisper-1":
                    continue
                try:
                    if api_type == "whisper":
                        model_outputs[model_id] = call_whisper(openai_key, pcm, model_id)
                    else:
                        model_outputs[model_id] = call_openrouter_audio(openrouter_key, model_id, pcm)
                except Exception as e:
                    model_outputs[model_id] = "NO_SPEECH"
                time.sleep(0.3)

            # Filter all outputs
            filtered = {}
            for model_id, text in model_outputs.items():
                filt, _ = _filter_transcript(text)
                filtered[model_id] = filt

            # Ensemble voting on filtered outputs
            valid = {m: t for m, t in filtered.items() if not is_no_speech(t)}
            ensemble_result = "NO_SPEECH"

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
                    ensemble_result = valid[best_model]
                else:
                    # All disagree — check gpt-4o-audio-preview
                    g4a = filtered.get("openai/gpt-4o-audio-preview", "NO_SPEECH")
                    ensemble_result = "NO_SPEECH" if is_no_speech(g4a) else "GARBAGE"

            # Auto-correct
            old = self._precomputed_audio.get(start, "NO_SPEECH")
            if ensemble_result == "GARBAGE":
                ensemble_result = "NO_SPEECH"  # Don't inject garbage into VLM prompt

            if normalize_transcript(old) != normalize_transcript(ensemble_result):
                corrections += 1
                if is_no_speech(ensemble_result):
                    self._precomputed_audio.pop(start, None)
                else:
                    self._precomputed_audio[start] = ensemble_result

            # Update audio log
            for idx, (s, e, _) in enumerate(self.audio_log):
                if s == start:
                    self.audio_log[idx] = (s, e, ensemble_result if not is_no_speech(ensemble_result) else "NO_SPEECH")
                    break

            display = ensemble_result[:50] if not is_no_speech(ensemble_result) else "NO_SPEECH"
            changed = " [corrected]" if normalize_transcript(old) != normalize_transcript(ensemble_result) else ""
            print(f"    [{i+1}/{total}] {start:.0f}-{end:.0f}s: {display}{changed}")

        speech_count = len(self._precomputed_audio)
        print(f"  [VERIFY] Done. {corrections} corrections. Final: {speech_count} speech / {total - speech_count} silence\n")

    # ------------------------------------------------------------------
    # HARNESS CALLBACKS
    # ------------------------------------------------------------------

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        """Called by the harness for each video frame."""
        self.frame_count += 1

        # Sample every 5th frame (~1 call per 2.5s at 2 FPS)
        if self.frame_count % 5 != 0:
            return

        # Limit concurrent VLM calls
        with self.lock:
            if self.pending_calls >= 2:
                self.logger.log(timestamp_sec, "frame_skipped", {
                    "frame_count": self.frame_count, "reason": "throttled", "pending_calls": self.pending_calls,
                })
                return
            self.pending_calls += 1

        self.logger.log(timestamp_sec, "frame_received", {
            "frame_count": self.frame_count, "sampled": True, "pending_calls": self.pending_calls,
        })

        thread = threading.Thread(target=self._analyze_frame, args=(timestamp_sec, frame_base64), daemon=True)
        thread.start()

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """Called by the harness for each 5s audio chunk.

        Real-time mode (use_audio_cache=False — default):
            Spawn a thread that calls whisper-1 + hallucination filter, then
            appends the result to audio_log / audio_history / _precomputed_audio.
            Does NOT block the harness timeline.
        Cache mode (use_audio_cache=True — --use-audio-cache path):
            Dict lookup into the pre-populated _precomputed_audio; update the
            sliding audio_history window for the VLM prompt.
        """
        if not self.use_audio_cache and self.openai_key:
            # Real-time: spawn a thread so we don't block the harness timeline.
            with self.lock:
                self.pending_audio_calls += 1
            threading.Thread(
                target=self._transcribe_chunk,
                args=(audio_bytes, start_sec, end_sec),
                daemon=True,
            ).start()
            return

        # Cache mode — existing dict-lookup behavior (--use-audio-cache).
        transcript = self._precomputed_audio.get(start_sec, "")
        if transcript and not _is_noise_transcript(transcript):
            with self.lock:
                self.recent_transcript = transcript
                self.recent_transcript_time = start_sec
                self.audio_history.append((start_sec, end_sec, transcript))
            self.logger.log(start_sec, "audio_received", {
                "start_sec": start_sec, "end_sec": end_sec,
                "transcript": transcript, "source": "precomputed",
            })
            # V4: feed transcript into audio-error detector.
            self._on_new_transcript(transcript, start_sec, end_sec)
        else:
            self.logger.log(start_sec, "audio_received", {
                "start_sec": start_sec, "end_sec": end_sec,
                "transcript": transcript or "NO_SPEECH", "source": "precomputed",
            })


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

def _apply_timestamp_prefix(output_path: str) -> str:
    """Prepend YYYYMMDD_HHMM_ to the filename stem so runs sort chronologically.

    Idempotent: if the stem already starts with an 8-digit date + 4-digit time,
    the path is returned unchanged.

    Example:
        output/r066_realtime.json       -> output/20260416_0132_r066_realtime.json
        output/20260416_0132_r066.json  -> output/20260416_0132_r066.json (unchanged)
    """
    p = Path(output_path)
    if re.match(r"^\d{8}_\d{4}_", p.stem):
        return output_path
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return str(p.parent / f"{ts}_{p.stem}{p.suffix}")


def main():
    parser = argparse.ArgumentParser(description="VLM Orchestrator Pipeline")
    parser.add_argument("--procedure", required=True, help="Path to procedure JSON")
    parser.add_argument("--video", required=True, help="Path to video MP4 (with audio)")
    parser.add_argument("--output", default="output/events.json", help="Output JSON path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time, 2.0 = 2x, etc.)")
    parser.add_argument("--frame-fps", type=float, default=2.0,
                        help="Frames per second delivered to pipeline (default: 2)")
    parser.add_argument("--audio-chunk-sec", type=float, default=5.0,
                        help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument(
        "--model",
        default="google/gemini-3.1-flash-image-preview",
        help="VLM model string passed to OpenRouter (default: google/gemini-3.1-flash-image-preview).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    parser.add_argument(
        "--use-audio-cache",
        action="store_true",
        help="Opt-in: load pre-built transcripts from data/audio_cache/<video>.json "
             "instead of re-transcribing. Errors out if no valid cache exists. "
             "(Default: always re-transcribe with OPENAI_API and save a debug copy "
             "next to --output.)",
    )
    args = parser.parse_args()
    args.output = _apply_timestamp_prefix(args.output)

    # Load procedure
    print("=" * 60)
    print("  VLM ORCHESTRATOR")
    print("=" * 60)
    print()

    procedure = load_procedure_json(args.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    print(f"  Procedure: {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:     {args.video}")
    print(f"  Speed:     {args.speed}x")
    print()

    if args.dry_run:
        if not Path(args.video).exists():
            print(f"  WARNING: Video not found: {args.video}")
            print("  [DRY RUN] Procedure validated. Video not checked (file missing).")
        else:
            print("  [DRY RUN] Inputs validated. Skipping pipeline.")
        return

    if not Path(args.video).exists():
        print(f"  ERROR: Video not found: {args.video}")
        sys.exit(1)

    # Load .env FIRST so subsequent os.getenv() calls can see every key,
    # including OPENROUTER_API_KEY / MY_OPENROUTER_API / OPENAI_API.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # VLM/OpenRouter key — accept any of (--api-key, OPENROUTER_API_KEY, MY_OPENROUTER_API).
    # The .env in this repo historically uses MY_OPENROUTER_API; keep that working
    # without forcing the user to rename or export manually.
    api_key = (
        args.api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("MY_OPENROUTER_API")
    )
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY (or MY_OPENROUTER_API in .env) or pass --api-key")
        sys.exit(1)

    openai_key = os.getenv("OPENAI_API", "")
    openrouter_key = os.getenv("MY_OPENROUTER_API", "") or api_key

    # Create harness and pipeline
    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(
        harness, api_key, procedure,
        output_path=args.output, speed=args.speed,
        # openai_key drives real-time whisper-1 transcription in on_audio.
        # use_audio_cache controls whether on_audio uses the dict lookup
        # path vs the real-time whisper-1 path.
        openai_key=openai_key,
        use_audio_cache=args.use_audio_cache,
        vlm_model=args.model,
    )

    # STEP 1: Audio — default is real-time (whisper-1 + filter inline in on_audio).
    #         --use-audio-cache loads the authoritative cache at data/audio_cache/<video>.json.
    if args.use_audio_cache:
        cached_transcripts, cached_audio_log = load_audio_cache(args.video)
        if cached_transcripts is None:
            parser.error(
                f"--use-audio-cache requested, but no valid cache at "
                f"{audio_cache_path(args.video)}. Build one with "
                f"'python scripts/cache_audio.py', or drop --use-audio-cache to "
                f"transcribe in real-time."
            )
        pipeline._precomputed_audio = cached_transcripts
        pipeline.audio_log = list(cached_audio_log)
        print(
            f"  [CACHE] HIT {audio_cache_path(args.video).name} "
            f"— {len(cached_transcripts)} speech chunks, "
            f"{len(cached_audio_log)} total. on_audio() will lookup, not transcribe.\n"
        )
    elif openai_key:
        print(
            f"  [AUDIO] Real-time transcription enabled (whisper-1 + hallucination filter).\n"
            f"          Chunks stream in via on_audio() during playback; no upfront batch.\n"
            f"          For ensemble-verified transcripts, pre-build a cache with\n"
            f"          'python scripts/cache_audio.py' and pass --use-audio-cache.\n"
        )
    else:
        parser.error(
            "OPENAI_API not set in environment. Either set it to enable real-time "
            "transcription (default), or pass --use-audio-cache to load a pre-built "
            "cache from data/audio_cache/."
        )

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    # Start the idle watcher (no-op at speed >= 2.0).
    pipeline.start_idle_worker()
    if pipeline._speed < 2.0:
        print(f"  [LIVE EMIT] Detectors emit directly to harness "
              f"(idle_threshold={IDLE_THRESHOLD_SEC}s)\n")
    else:
        print(f"  [LIVE EMIT] speed={pipeline._speed}x ≥ 2 — idle watcher "
              f"disabled, detectors emit directly\n")

    # STEP 3: Run pipeline
    print("  [PIPELINE] Starting harness...")
    try:
        results = harness.run()
    finally:
        # Stop the idle watcher.
        pipeline.stop_idle_worker()

    # Rebuild results.events + delays from the harness's authoritative list.
    rebuilt_events = []
    rebuilt_delays = []
    for ee in harness._emitted_events:
        ev = dict(ee.event)
        ev["detection_delay_sec"] = round(ee.detection_delay_sec, 3)
        rebuilt_events.append(ev)
        rebuilt_delays.append(ee.detection_delay_sec)
    results.events = rebuilt_events
    results.mean_detection_delay_sec = round(
        sum(rebuilt_delays) / len(rebuilt_delays), 3,
    ) if rebuilt_delays else 0.0
    results.max_detection_delay_sec = round(
        max(rebuilt_delays), 3
    ) if rebuilt_delays else 0.0

    # Log run end
    events_by_type = {}
    for e in results.events:
        t = e.get("type", "unknown")
        events_by_type[t] = events_by_type.get(t, 0) + 1

    pipeline.logger.log(
        results.video_duration_sec,
        "run_end",
        {
            "total_vlm_calls": len(pipeline._vlm_latencies),
            "total_audio_calls": len(pipeline._audio_latencies),
            "total_events": len(results.events),
            "steps_detected": events_by_type.get("step_completion", 0),
            "total_steps": len(pipeline.steps),
            "errors_detected": events_by_type.get("error_detected", 0),
            "idles_detected": events_by_type.get("idle_detected", 0),
            "mean_vlm_latency_ms": round(sum(pipeline._vlm_latencies) / len(pipeline._vlm_latencies)) if pipeline._vlm_latencies else None,
            "mean_audio_latency_ms": round(sum(pipeline._audio_latencies) / len(pipeline._audio_latencies)) if pipeline._audio_latencies else None,
            "wall_duration": results.wall_duration_sec,
        },
    )

    # Save outputs
    harness.save_results(results, args.output)
    pipeline.logger.save_json()
    pipeline.logger.save_markdown()

    # Save audio transcript log
    if pipeline.audio_log:
        audio_log_path = str(Path(args.output).with_suffix("")) + "_audio.json"
        with open(audio_log_path, "w") as f:
            json.dump([{"start_sec": s, "end_sec": e, "transcript": t} for s, e, t in pipeline.audio_log], f, indent=2)
        print(f"  Audio log: {audio_log_path} ({len(pipeline.audio_log)} chunks)")

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(results.events)}")
    print()

    if not results.events:
        print("  WARNING: No events detected. Implement Pipeline.on_frame() and Pipeline.on_audio().")


if __name__ == "__main__":
    main()
