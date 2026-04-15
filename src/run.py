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
import wave
import base64
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

import requests
import numpy as np

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
    model: str = "google/gemini-2.5-flash",
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
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    },
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
# YOUR PIPELINE — IMPLEMENT THESE CALLBACKS
# ==========================================================================

CORRECTION_KEYWORDS = {"stop", "no", "wrong", "don't", "wait", "not", "incorrect", "mistake"}

class Pipeline:
    """
    VLM orchestration pipeline with audio transcription.

    Uses Gemini 2.5 Flash for video frame analysis and GPT-4o Audio Preview
    for instructor speech transcription. Audio corrections signal errors.
    """

    def __init__(self, harness: StreamingHarness, api_key: str, procedure: Dict[str, Any]):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]

        # Step tracking
        self.current_step_index = 0
        self.completed_steps: set = set()

        # Idle detection — driven by VLM "idle" responses, not timer
        self.last_vlm_activity_time = 0.0
        self.idle_emitted_times: set = set()

        # Frame sampling
        self.frame_count = 0
        self.pending_calls = 0
        self.lock = threading.Lock()

        # Audio state
        self.recent_transcript = ""
        self.recent_transcript_time = 0.0
        self.pending_audio_calls = 0
        self.audio_log: list = []  # [(start_sec, end_sec, transcript)]

        # Build steps text once for the prompt
        self._steps_text = "\n".join(
            f"  Step {s['step_id']}: {s['description']}" for s in self.steps
        )

    def _build_prompt(self, timestamp_sec: float) -> str:
        with self.lock:
            idx = self.current_step_index
            transcript = self.recent_transcript
            transcript_time = self.recent_transcript_time
            completed = sorted(self.completed_steps)
        current_step = self.steps[idx] if idx < len(self.steps) else None

        prompt = f"""You are monitoring a technician performing: "{self.task_name}"

The procedure steps are:
{self._steps_text}

Steps already completed: {completed if completed else "none yet"}
"""
        if current_step:
            prompt += f"Expected current step: {current_step['step_id']} — \"{current_step['description']}\"\n\n"
        else:
            prompt += "All steps may be completed.\n\n"

        # Include recent audio transcript if available and recent
        if transcript and (timestamp_sec - transcript_time) < 15.0:
            prompt += f"Recent instructor audio: \"{transcript}\"\n\n"

        prompt += """Analyze this video frame. Respond ONLY with a JSON object (no markdown fences):
{
  "status": "step_complete" or "error" or "in_progress" or "idle",
  "step_id": <integer step number completed or being worked on, or null>,
  "description": "<what the person is doing>",
  "error_description": "<if error: what went wrong. otherwise null>",
  "confidence": <0.0 to 1.0>
}

IMPORTANT RULES:
- You may report ANY step as complete, not just the expected current step. Steps can be completed out of order or you may have missed earlier ones.
- Report "step_complete" ONLY when you can clearly see the step has been finished.
- Report "error" ONLY if the technician is clearly doing something WRONG (wrong tool, wrong part, wrong sequence). NOT just because they haven't started the next step yet.
- Report "idle" if the technician is standing still, waiting, or not actively working.
- Default to "in_progress" if unsure."""
        return prompt

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

    def _analyze_frame(self, timestamp_sec: float, frame_base64: str):
        try:
            prompt = self._build_prompt(timestamp_sec)
            response = call_vlm(self.api_key, frame_base64, prompt)
            parsed = self._parse_response(response)

            status = parsed.get("status", "in_progress")
            step_id = parsed.get("step_id")
            confidence = parsed.get("confidence", 0.5)
            description = parsed.get("description", "")

            with self.lock:
                if status == "step_complete" and step_id is not None and step_id not in self.completed_steps:
                    self.completed_steps.add(step_id)
                    self.last_vlm_activity_time = timestamp_sec

                    # Advance step pointer — skip ahead if VLM reports a later step
                    while (self.current_step_index < len(self.steps)
                           and self.steps[self.current_step_index]["step_id"] <= step_id):
                        self.completed_steps.add(self.steps[self.current_step_index]["step_id"])
                        self.current_step_index += 1

                    self.harness.emit_event({
                        "timestamp_sec": timestamp_sec,
                        "type": "step_completion",
                        "step_id": step_id,
                        "confidence": confidence,
                        "description": description,
                        "source": "video",
                        "vlm_observation": response[:500],
                    })

                elif status == "error":
                    error_desc = parsed.get("error_description", description)
                    self.last_vlm_activity_time = timestamp_sec

                    self.harness.emit_event({
                        "timestamp_sec": timestamp_sec,
                        "type": "error_detected",
                        "error_type": "wrong_action",
                        "severity": "warning",
                        "confidence": confidence,
                        "description": error_desc,
                        "source": "video",
                        "vlm_observation": response[:500],
                        "spoken_response": f"Stop — {error_desc}",
                    })

                elif status == "idle":
                    rounded = round(timestamp_sec, 0)
                    if rounded not in self.idle_emitted_times:
                        self.idle_emitted_times.add(rounded)
                        self.harness.emit_event({
                            "timestamp_sec": timestamp_sec,
                            "type": "idle_detected",
                            "confidence": confidence,
                            "description": description,
                            "source": "video",
                            "vlm_observation": response[:500],
                        })

                else:
                    # in_progress — update activity time
                    self.last_vlm_activity_time = timestamp_sec

        except Exception as e:
            print(f"  [pipeline] VLM call failed at {timestamp_sec:.1f}s: {e}")
        finally:
            with self.lock:
                self.pending_calls -= 1

    def _transcribe_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        try:
            prompt = "Transcribe any speech in this audio. The audio may be pitch-shifted. If there is no speech, respond with exactly: NO_SPEECH"
            transcript = call_audio_llm(self.api_key, audio_bytes, prompt)

            clean = transcript.strip()
            with self.lock:
                self.audio_log.append((start_sec, end_sec, clean))

            if "NO_SPEECH" in transcript.upper():
                return

            with self.lock:
                self.recent_transcript = clean
                self.recent_transcript_time = start_sec

            print(f"  [audio] {start_sec:.1f}-{end_sec:.1f}s: {clean[:80]}")

        except Exception as e:
            print(f"  [audio] Transcription failed at {start_sec:.1f}s: {e}")
        finally:
            with self.lock:
                self.pending_audio_calls -= 1

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        """Called by the harness for each video frame."""
        self.frame_count += 1

        # Sample every 5th frame (~1 call per 2.5s at 2 FPS)
        if self.frame_count % 5 != 0:
            return

        # Limit concurrent VLM calls
        with self.lock:
            if self.pending_calls >= 2:
                return
            self.pending_calls += 1

        thread = threading.Thread(target=self._analyze_frame, args=(timestamp_sec, frame_base64), daemon=True)
        thread.start()

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """Called by the harness for each audio chunk. Transcribes in background."""
        with self.lock:
            if self.pending_audio_calls >= 1:
                return
            self.pending_audio_calls += 1

        thread = threading.Thread(target=self._transcribe_audio, args=(audio_bytes, start_sec, end_sec), daemon=True)
        thread.start()


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

    # Create harness and pipeline
    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(harness, api_key, procedure)

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    # Run
    results = harness.run()

    # Save
    harness.save_results(results, args.output)

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
