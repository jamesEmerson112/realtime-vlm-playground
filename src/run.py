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
            # Observer-only schema: no status/step_id, just hands/objects/action
            action_brief = (p.get("action") or p.get("description") or "")[:50]
            conf = p.get("confidence", "-")
            print(f"  [{w} | {v}] VLM_RESP     {d['latency_ms']}ms conf={conf} action=\"{action_brief}\"")
        elif t == "vlm_error":
            print(f"  [{w} | {v}] VLM_ERR      {d['error']}")
        elif t == "mother_batch_request":
            print(f"  [{w} | {v}] MOTHER_REQ   model={d.get('model')} obs={d.get('observation_count')} audio={d.get('audio_count')} prompt={d.get('prompt_len')}chars")
        elif t == "mother_batch_response":
            print(f"  [{w} | {v}] MOTHER_RESP  {d.get('elapsed_sec')}s events={d.get('events_count')}")
        elif t == "mother_batch_error":
            print(f"  [{w} | {v}] MOTHER_ERR   {d.get('error')}")
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
                          "mother_batch_request", "mother_batch_response", "mother_batch_error"}
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
                    action_brief = (p.get("action") or p.get("description") or "")[:60]
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
                elif t == "mother_batch_request":
                    detail = f"model={d.get('model')} obs={d.get('observation_count')} audio={d.get('audio_count')} prompt={d.get('prompt_len')}chars"
                elif t == "mother_batch_response":
                    detail = f"{d.get('elapsed_sec')}s, {d.get('events_count')} events"
                elif t == "mother_batch_error":
                    detail = f"Error: {d.get('error')}"
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
                    # Observer schema (Mother V1): hands/objects/action/visual_cues/confidence
                    # Older schema (pre-V1): status/step_id/confidence — handle both
                    if "hands" in p or "action" in p:
                        action = p.get("action", p.get("description", ""))[:120]
                        lines.append(
                            f"**Parsed (observer):** action=\"{action}\", "
                            f"confidence={p.get('confidence')}, "
                            f"hands=\"{p.get('hands', '')[:80]}\"\n"
                        )
                    else:
                        lines.append(
                            f"**Parsed:** {p.get('status', '-')}, "
                            f"step_id={p.get('step_id')}, "
                            f"confidence={p.get('confidence')}\n"
                        )
                else:
                    lines.append("**Response:** (error or missing)\n")

        # --- Mother Agent input & output ---
        mother_pairs = []
        for i, e in enumerate(self.entries):
            if e["event_type"] == "mother_batch_request":
                # Pair with the next response OR error entry after this request.
                outcome = next(
                    (r for r in self.entries[i + 1:]
                     if r["event_type"] in ("mother_batch_response", "mother_batch_error")),
                    None,
                )
                mother_pairs.append((e, outcome))

        if mother_pairs:
            lines.append("## Mother Agent Input & Output\n")
            for idx, (req, outcome) in enumerate(mother_pairs, 1):
                rd = req["data"]
                prompt_text = rd.get("prompt", "(prompt not captured)")
                prompt_len = rd.get("prompt_len", len(prompt_text) if isinstance(prompt_text, str) else 0)
                obs_count = rd.get("observation_count", "?")
                audio_count = rd.get("audio_count", "?")

                header_suffix = f" #{idx}" if len(mother_pairs) > 1 else ""
                lines.append(
                    f"### Input{header_suffix} "
                    f"(system prompt, {prompt_len} chars, "
                    f"{obs_count} observations + {audio_count} audio lines)\n"
                )
                lines.append("```")
                lines.append(prompt_text)
                lines.append("```\n")

                if outcome is None:
                    lines.append("**Output:** (no response or error logged)\n")
                    continue

                od = outcome["data"]
                if outcome["event_type"] == "mother_batch_response":
                    elapsed = od.get("elapsed_sec", "?")
                    events_count = od.get("events_count", "?")
                    response_text = od.get("response", "")
                    lines.append(
                        f"### Output{header_suffix} ({elapsed}s, {events_count} events)\n"
                    )
                    lines.append("```")
                    lines.append(response_text)
                    lines.append("```\n")
                else:  # mother_batch_error
                    elapsed = od.get("elapsed_sec", "?")
                    err_msg = od.get("error", "(no error message)")
                    response_text = od.get("response", "")
                    lines.append(f"### Error{header_suffix} ({elapsed}s)\n")
                    lines.append(f"**Error:** {err_msg}\n")
                    if response_text:
                        lines.append("**Raw response:**")
                        lines.append("```")
                        lines.append(response_text)
                        lines.append("```\n")

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
    model: str = "google/gemini-3-flash-preview",
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
# MOTHER AGENT — REASONING LLM HELPER (OpenAI direct, text-only)
# ==========================================================================

MOTHER_MODEL = "gpt-5.4"
MOTHER_REASONING_EFFORT = "high"


def call_reasoning_llm(
    api_key: str,
    system: str,
    user: str,
    model: str = MOTHER_MODEL,
    reasoning_effort: str = MOTHER_REASONING_EFFORT,
    max_completion_tokens: int = 8000,
) -> str:
    """
    Call a reasoning LLM directly via OpenAI Chat Completions (no OpenRouter).
    Used by the Mother Agent for post-stream event decisions.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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

class Pipeline:
    """
    VLM orchestration pipeline with pre-computed audio transcription.

    Audio flow: Setup (whisper-1 + filter) → Verify (4-model ensemble) → Pipeline (lookup).
    Uses Gemini 3.0 Flash for video frame analysis. Audio corrections signal errors.
    """

    def __init__(self, harness: StreamingHarness, api_key: str, procedure: Dict[str, Any],
                 output_path: str = "output/events.json", speed: float = 1.0):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]

        # Step tracking — progress-pct smoother emits step_completion live via
        # harness.emit_event() when smoothed progress crosses 90.
        self.current_step_index = 0
        self.completed_steps: set = set()

        # Real-time emission state — smoother + dedup guards
        # maxlen=5 gives us a ~12s rolling window at ~2.5s sampling; we emit
        # on "any 2 of last 5 at >= 85" which tolerates sporadic VLM hallucinations
        # while still requiring corroborating evidence.
        self._progress_history: dict = {
            s["step_id"]: deque(maxlen=5) for s in self.steps
        }
        self._emitted_steps: set = set()          # step_ids that already fired live
        self._emitted_error_timestamps: list = [] # for 5s dedup window on error_detected

        # Observation buffer — kept unbounded for post-stream reduced Mother
        # (idle_detected + missed step_completion catch-up).
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

        # Build steps text once for the prompt
        self._steps_text = "\n".join(
            f"  Step {s['step_id']}: {s['description']}" for s in self.steps
        )

        # Load prompt templates from files
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        with open(prompts_dir / "vlm_prompt.txt", "r", encoding="utf-8") as f:
            self._vlm_template = f.read()
        with open(prompts_dir / "audio_prompt.txt", "r", encoding="utf-8") as f:
            self._audio_prompt = f.read().strip()
        with open(prompts_dir / "mother_prompt.txt", "r", encoding="utf-8") as f:
            self._mother_template = f.read()

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
            "model": "google/gemini-3-flash-preview",
            "audio_model": "openai/gpt-4o-audio-preview",
        })

    def _build_prompt(self, timestamp_sec: float) -> str:
        with self.lock:
            idx = self.current_step_index
            completed = sorted(self.completed_steps)
            # Snapshot audio history — only chunks within 10s
            audio_entries = [
                (s, e, t) for s, e, t in self.audio_history
                if (timestamp_sec - s) < 10.0
            ]

        # 2-step window: only show current + next to reduce hallucination surface.
        if idx >= len(self.steps):
            current_step_block = (
                "All procedure steps appear complete. "
                "Watch for idle/post-procedure activity."
            )
            next_step_block = ""
        else:
            cur = self.steps[idx]
            current_step_block = (
                f'CURRENT STEP (focus here): Step {cur["step_id"]} — "{cur["description"]}"\n'
                f'  Estimate current_step_progress_pct relative to visible completion of THIS step.'
            )
            if idx + 1 < len(self.steps):
                nxt = self.steps[idx + 1]
                next_step_block = (
                    f'NEXT STEP (in case you see it starting): '
                    f'Step {nxt["step_id"]} — "{nxt["description"]}"'
                )
            else:
                next_step_block = ""

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

        return self._vlm_template.format(
            task_name=self.task_name,
            completed_steps=completed if completed else "none yet",
            current_step_block=current_step_block,
            next_step_block=next_step_block,
            audio_line=audio_line,
            timestamp_sec=f"{timestamp_sec:.1f}",
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

    def _decide_and_emit(self, observation: dict, audio_window: list):
        """
        Real-time decision layer. Runs on VLM worker thread after each observation.
        Emits events directly via self.harness.emit_event():
          - step_completion: when the 3-frame moving-average of current_step_progress_pct
            crosses 90 (and the step hasn't been emitted yet).
          - error_detected (source=video): when VLM reports error_visible.
          - error_detected (source=audio): when the 10s audio window contains any
            CORRECTION_KEYWORDS. Dedup'd within a 5s window so we don't spam errors.
        """
        ts = observation["timestamp_sec"]

        # ------------------------------------------------------------------
        # (1) STEP_COMPLETION via progress-pct smoothing
        # ------------------------------------------------------------------
        should_emit_step = False
        emit_step_id = None
        emit_confidence = 0.0
        emit_smoothed = 0.0
        old_idx = None

        with self.lock:
            idx = self.current_step_index
            if idx < len(self.steps):
                step_id = self.steps[idx]["step_id"]
                raw_pct = observation.get("current_step_progress_pct", 0)
                try:
                    raw_pct = int(raw_pct)
                except (TypeError, ValueError):
                    raw_pct = 0
                raw_pct = max(0, min(100, raw_pct))

                hist = self._progress_history[step_id]
                hist.append(raw_pct)
                # "Any 2 of last 5 at >= 85" — tolerant of sporadic VLM 100% hits
                # while still requiring corroborating evidence within ~12s.
                high_hits = sum(1 for p in hist if p >= 85)
                if high_hits >= 2 and step_id not in self._emitted_steps:
                    smoothed = sum(hist) / len(hist)
                    should_emit_step = True
                    emit_step_id = step_id
                    emit_confidence = min(1.0, max(0.5, smoothed / 100.0))
                    emit_smoothed = smoothed
                    old_idx = idx
                    self._emitted_steps.add(step_id)
                    self.completed_steps.add(step_id)
                    self.current_step_index += 1

        if should_emit_step:
            event = {
                "timestamp_sec": ts,
                "type": "step_completion",
                "step_id": int(emit_step_id),
                "confidence": round(emit_confidence, 3),
                "source": "video",
                "description": f"Step {emit_step_id} end-state visible (smoothed progress {emit_smoothed:.0f}%)",
            }
            try:
                self.harness.emit_event(event)
                self.logger.log(ts, "event_emitted", {"event": event})
                self.logger.log(ts, "state_change", {
                    "field": "current_step_index",
                    "old": old_idx,
                    "new": old_idx + 1,
                    "advanced_step_id": emit_step_id,
                    "trigger": {"progress_smoothed": round(emit_smoothed, 1)},
                })
            except ValueError as e:
                self.logger.log(ts, "vlm_error", {"error": f"emit_event rejected step: {e}"})

        # ------------------------------------------------------------------
        # (2) ERROR_DETECTED from VLM error_visible
        # ------------------------------------------------------------------
        err_vis = observation.get("error_visible")
        if err_vis and isinstance(err_vis, str) and err_vis.strip().lower() not in ("", "null", "none"):
            self._maybe_emit_error(ts, f"VLM: {err_vis.strip()}", "video")

        # ------------------------------------------------------------------
        # (3) ERROR_DETECTED from audio correction keywords in 10s window
        # ------------------------------------------------------------------
        for (a_start, _a_end, transcript) in audio_window:
            if not transcript:
                continue
            hit = _find_correction_hit(transcript)
            if hit:
                # Anchor the error timestamp at the audio chunk start — closer
                # to when the correction was spoken than the current video time.
                self._maybe_emit_error(
                    float(a_start),
                    f"Audio correction ('{hit}'): {transcript.strip()[:120]}",
                    "audio",
                )
                break

    def _maybe_emit_error(self, ts: float, description: str, source: str):
        """5-second dedup on error_detected emissions, then harness.emit_event()."""
        with self.lock:
            if any(abs(ts - t) < 5.0 for t in self._emitted_error_timestamps):
                return
            self._emitted_error_timestamps.append(ts)

        event = {
            "timestamp_sec": float(ts),
            "type": "error_detected",
            "confidence": 0.7,
            "source": source,
            "description": description,
        }
        try:
            self.harness.emit_event(event)
            self.logger.log(ts, "event_emitted", {"event": event})
        except ValueError as e:
            self.logger.log(ts, "vlm_error", {"error": f"emit_event rejected error: {e}"})

    def _analyze_frame(self, timestamp_sec: float, frame_base64: str):
        """
        Real-time pipeline: VLM observes, sub-mother decides, harness.emit_event() fires live.
        """
        try:
            prompt = self._build_prompt(timestamp_sec)
            self.logger.log(timestamp_sec, "vlm_request", {
                "prompt_len": len(prompt),
                "prompt": prompt,
            })

            t0 = time.time()
            response = call_vlm(self.api_key, frame_base64, prompt)
            latency_ms = round((time.time() - t0) * 1000)
            self._vlm_latencies.append(latency_ms)

            parsed = self._parse_response(response)

            self.logger.log(timestamp_sec, "vlm_response", {
                "response": response,
                "latency_ms": latency_ms,
                "parsed": parsed,
            })

            # Extract observer fields with safe fallbacks (missing fields → empty)
            visual_cues = parsed.get("visual_cues", [])
            if not isinstance(visual_cues, list):
                visual_cues = [str(visual_cues)]

            # current_step_progress_pct: coerce to int in [0, 100]; default 0 on missing/bad
            raw_pct = parsed.get("current_step_progress_pct", 0)
            try:
                progress_pct = max(0, min(100, int(raw_pct)))
            except (TypeError, ValueError):
                progress_pct = 0

            observation = {
                "timestamp_sec": timestamp_sec,
                "hands": str(parsed.get("hands", "")),
                "objects": str(parsed.get("objects", "")),
                "action": str(parsed.get("action", parsed.get("description", ""))),
                "visual_cues": visual_cues,
                "end_state_visible": str(parsed.get("end_state_visible", "unclear")).lower(),
                "heard_in_audio_matches_current_step": bool(
                    parsed.get("heard_in_audio_matches_current_step", False)
                ),
                "current_step_progress_pct": progress_pct,
                "error_visible": parsed.get("error_visible"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "raw_response": response[:1000],
            }

            with self.lock:
                self.observation_buffer.append(observation)
                audio_snapshot = list(self.audio_history)
            self._decide_and_emit(observation, audio_snapshot)

        except Exception as e:
            self.logger.log(timestamp_sec, "vlm_error", {"error": str(e)})
        finally:
            with self.lock:
                self.pending_calls -= 1

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
    # MOTHER AGENT — BATCH-AT-END REASONING
    # ------------------------------------------------------------------

    def run_mother_batch(self, openai_key: str, video_duration_sec: float):
        """
        Reduced Mother Agent V2 (post-stream catch-up): runs once after harness.run()
        completes. Live emission already handled step_completion + error_detected during
        the stream. This pass only finds RETROSPECTIVE events:
          - idle_detected (needs hindsight — "nothing happened for 5+s")
          - step_completion for step_ids NOT in self._emitted_steps (live missed them)

        Events are emitted via self.harness.emit_event() so detection_delay_sec is
        computed honestly by the harness. Retrospective delays will be large but GT
        tolerates this class of latency.
        """
        obs_count = len(self.observation_buffer)
        speech_audio = [(s, e, t) for s, e, t in self.audio_log
                        if t and "NO_SPEECH" not in t.upper() and t != "GARBAGE"]
        audio_count = len(speech_audio)

        already_emitted = sorted(self._emitted_steps)
        print(f"  [MOTHER] Catch-up pass: {obs_count} obs + {audio_count} audio, "
              f"already-emitted steps={already_emitted} ({video_duration_sec:.1f}s video)...")

        if obs_count == 0:
            print("  [MOTHER] No observations — skipping.")
            self.logger.log(video_duration_sec, "mother_batch_error",
                            {"error": "no observations"})
            return

        # Interleave observations + audio into a single chronological stream.
        def _format_obs_block(obs: dict) -> str:
            ts = obs.get("timestamp_sec", 0.0)
            hands = obs.get("hands", "") or "-"
            objects = obs.get("objects", "") or "-"
            action = obs.get("action", "") or "-"
            cues = obs.get("visual_cues", []) or []
            cues_str = "; ".join(str(c) for c in cues) if isinstance(cues, list) else str(cues)
            end_state = obs.get("end_state_visible", "unclear") or "unclear"
            progress = obs.get("current_step_progress_pct")
            error_visible = obs.get("error_visible")
            conf = obs.get("confidence", 0.0)
            lines = [
                f"  [t={ts:6.1f}s | obs | conf={conf:.2f}]",
                f"    hands: {hands}",
                f"    objects: {objects}",
                f"    action: {action}",
            ]
            if cues_str:
                lines.append(f"    visual_cues: {cues_str}")
            lines.append(f"    end_state_visible: {end_state}")
            if progress is not None:
                lines.append(f"    current_step_progress_pct: {progress}")
            if error_visible and str(error_visible).lower() not in ("none", "null", ""):
                lines.append(f"    error_visible: {error_visible}")
            return "\n".join(lines)

        stream_items = []
        for obs in self.observation_buffer:
            stream_items.append((float(obs.get("timestamp_sec", 0.0)), "obs", obs))
        for s, e, t in speech_audio:
            stream_items.append((float(s), "audio", (s, e, t)))
        stream_items.sort(key=lambda x: x[0])

        stream_lines = []
        for _, kind, payload in stream_items:
            if kind == "obs":
                stream_lines.append(_format_obs_block(payload))
            else:
                s, e, t = payload
                stream_lines.append(f'  [t={s:6.1f}-{e:.1f}s | audio]\n    "{t}"')
        stream_text = "\n".join(stream_lines) if stream_lines else "  (no stream entries)"

        # Build slim catch-up prompt
        system_prompt = self._mother_template.format(
            task_name=self.task_name,
            steps_text=self._steps_text,
            stream_text=stream_text,
            video_duration=f"{video_duration_sec:.1f}",
            already_emitted_step_ids=already_emitted,
        )

        self.logger.log(video_duration_sec, "mother_batch_request", {
            "model": MOTHER_MODEL,
            "reasoning_effort": MOTHER_REASONING_EFFORT,
            "observation_count": obs_count,
            "audio_count": audio_count,
            "already_emitted_step_ids": already_emitted,
            "prompt_len": len(system_prompt),
            "prompt": system_prompt,
        })

        t0 = time.time()
        try:
            response = call_reasoning_llm(
                openai_key,
                system=system_prompt,
                user="Analyze the stream above and produce the retrospective events JSON.",
                model=MOTHER_MODEL,
                reasoning_effort=MOTHER_REASONING_EFFORT,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            self.logger.log(video_duration_sec, "mother_batch_error", {
                "error": f"API call failed: {exc}",
                "elapsed_sec": round(elapsed, 2),
            })
            print(f"  [MOTHER] ERROR after {elapsed:.1f}s: {exc}")
            return

        elapsed = time.time() - t0

        # Parse response (strip markdown fences same as _parse_response)
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            raw_events = parsed.get("events", [])
            if not isinstance(raw_events, list):
                raise ValueError(f"'events' is not a list: {type(raw_events).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.log(video_duration_sec, "mother_batch_error", {
                "error": f"JSON parse: {exc}",
                "elapsed_sec": round(elapsed, 2),
                "response": response,
            })
            print(f"  [MOTHER] JSON parse failed after {elapsed:.1f}s: {exc}")
            return

        # Filter: only accept idle_detected OR step_completion with a NEW step_id.
        # Emit each live via harness.emit_event() so detection_delay_sec is computed.
        allowed_types = {"idle_detected", "step_completion"}
        emitted_count = 0
        skipped_count = 0

        for ev in raw_events:
            if not isinstance(ev, dict):
                skipped_count += 1
                continue
            ev_type = ev.get("type", "")
            if ev_type not in allowed_types:
                skipped_count += 1
                continue
            try:
                ts = float(ev.get("timestamp_sec", 0.0))
            except (TypeError, ValueError):
                skipped_count += 1
                continue

            out = {
                "timestamp_sec": ts,
                "type": ev_type,
                "confidence": float(ev.get("confidence", 0.5)),
                "description": str(ev.get("description", "")),
                "source": "both",
            }

            if ev_type == "step_completion":
                step_id = ev.get("step_id")
                try:
                    step_id_int = int(step_id)
                except (TypeError, ValueError):
                    skipped_count += 1
                    continue
                if step_id_int in self._emitted_steps:
                    skipped_count += 1
                    continue
                out["step_id"] = step_id_int
                with self.lock:
                    self._emitted_steps.add(step_id_int)
                    self.completed_steps.add(step_id_int)

            try:
                self.harness.emit_event(out)
                self.logger.log(ts, "event_emitted", {"event": out, "source_mother": True})
                emitted_count += 1
            except ValueError as e:
                self.logger.log(ts, "mother_batch_error", {
                    "error": f"emit_event rejected: {e}", "event": out,
                })
                skipped_count += 1

        self.logger.log(video_duration_sec, "mother_batch_response", {
            "elapsed_sec": round(elapsed, 2),
            "response_len": len(response),
            "events_count": emitted_count,
            "skipped_count": skipped_count,
            "response": response,
        })
        print(f"  [MOTHER] Done in {elapsed:.1f}s — emitted {emitted_count} retrospective "
              f"events ({skipped_count} skipped)")

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
        """Called by the harness for each audio chunk. Looks up pre-computed transcript."""
        transcript = self._precomputed_audio.get(start_sec, "")
        if transcript:
            with self.lock:
                self.recent_transcript = transcript
                self.recent_transcript_time = start_sec
                self.audio_history.append((start_sec, end_sec, transcript))
            self.logger.log(start_sec, "audio_received", {
                "start_sec": start_sec, "end_sec": end_sec,
                "transcript": transcript, "source": "precomputed",
            })
        else:
            self.logger.log(start_sec, "audio_received", {
                "start_sec": start_sec, "end_sec": end_sec,
                "transcript": "NO_SPEECH", "source": "precomputed",
            })


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

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
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

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

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    # Load .env for audio API keys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
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

    pipeline = Pipeline(harness, api_key, procedure, output_path=args.output, speed=args.speed)

    # STEP 1: Setup — pre-compute audio transcripts with whisper-1
    if openai_key:
        pipeline.precompute_audio(args.video, openai_key)

        # STEP 2: Verify — ensemble voting on all chunks
        if openrouter_key:
            pipeline.verify_audio(openai_key, openrouter_key)
        else:
            print("  [VERIFY] Skipped — MY_OPENROUTER_API not set\n")
    else:
        print("  [SETUP] Skipped — OPENAI_API not set. Audio will use real-time fallback.\n")

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    # STEP 3: Run pipeline
    print("  [PIPELINE] Starting harness...")
    results = harness.run()

    # No post-stream Mother. All events were emitted live via harness.emit_event()
    # during the stream — the harness already populated results.events and
    # mean/max_detection_delay_sec in harness.run() before returning.

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
