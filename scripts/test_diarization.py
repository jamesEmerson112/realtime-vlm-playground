"""
Speaker Diarization Feasibility Test

Tests whether pyannote.audio can distinguish speakers in pitch-shifted audio.
If it can separate instructor from technician, we can reduce false-positive
error detections by only feeding instructor speech to the VLM.

Usage:
    python scripts/test_diarization.py --video R066
    python scripts/test_diarization.py --video R066 --speakers 2
    python scripts/test_diarization.py                          # all 3 videos
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

VIDEO_MAP = {
    "R066": "R066-15July-Circuit-Breaker-part2",
    "z065": "z065-june-29-22-dslr",
    "R073": "R073-20July-GoPro",
}

CHUNK_SEC = 5.0


def extract_wav(video_path: str, output_wav: str):
    """Extract full audio as 16kHz mono WAV file."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            output_wav,
        ],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  ERROR: ffmpeg failed: {result.stderr.decode()[:200]}")
        sys.exit(1)


def load_benchmark_transcripts(video_key: str):
    """Load transcripts from the audio benchmark for cross-referencing."""
    raw_path = REPO_ROOT / "output" / "audio_benchmark_raw.json"
    if not raw_path.exists():
        return {}
    with open(raw_path) as f:
        data = json.load(f)
    # Use gpt-4o-audio-preview transcripts (moderate, less hallucination)
    transcripts = {}
    for r in data:
        if r["video"] == video_key and r["model"] == "openai/gpt-4o-audio-preview":
            if r["transcript"] and "NO_SPEECH" not in r["transcript"].upper():
                transcripts[(r["chunk_start"], r["chunk_end"])] = r["transcript"]
    return transcripts


def find_overlapping_transcript(seg_start, seg_end, transcripts):
    """Find benchmark transcript that overlaps with a diarization segment."""
    matches = []
    for (cs, ce), text in transcripts.items():
        # Check overlap
        if seg_start < ce and seg_end > cs:
            matches.append(text)
    return " | ".join(matches) if matches else None


def run_diarization(video_key: str, hf_token: str, num_speakers: int = None):
    """Run pyannote diarization on a video's audio."""
    clip_name = VIDEO_MAP[video_key]
    video_path = str(
        REPO_ROOT / "data" / "videos_full" / clip_name / "Export_py" / "Video_pitchshift.mp4"
    )

    if not os.path.exists(video_path):
        print(f"  SKIP {video_key}: video not found at {video_path}")
        return None

    # Extract audio to temp WAV
    wav_path = str(REPO_ROOT / "output" / f"_diarize_{video_key}.wav")
    print(f"\n{'='*60}")
    print(f"  Diarization: {video_key} ({clip_name})")
    print(f"{'='*60}")
    print(f"  Extracting audio...")
    extract_wav(video_path, wav_path)

    # Load pyannote pipeline
    print(f"  Loading pyannote/speaker-diarization-3.1...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(device)

    # Load audio as waveform (scipy avoids torchcodec/ffmpeg DLL issues on Windows)
    from scipy.io import wavfile
    sample_rate, audio_np = wavfile.read(wav_path)
    # Convert int16 to float32 normalized [-1, 1], shape (1, num_samples)
    waveform = torch.from_numpy(audio_np.astype("float32") / 32768.0).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # Run diarization
    print(f"  Running diarization...")
    t0 = time.time()

    kwargs = {}
    if num_speakers is not None:
        kwargs["min_speakers"] = num_speakers
        kwargs["max_speakers"] = num_speakers
        print(f"  Forcing {num_speakers} speakers")

    diarize_output = pipeline(audio_input, **kwargs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Load transcripts for cross-referencing
    transcripts = load_benchmark_transcripts(video_key)

    # Parse segments (pyannote 4.x returns DiarizeOutput with .speaker_diarization Annotation)
    annotation = diarize_output.speaker_diarization
    segments = []
    speaker_times = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        seg = {
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "duration": round(turn.end - turn.start, 2),
            "speaker": speaker,
        }

        # Cross-reference with transcripts
        transcript = find_overlapping_transcript(turn.start, turn.end, transcripts)
        if transcript:
            seg["overlapping_transcript"] = transcript

        segments.append(seg)

        if speaker not in speaker_times:
            speaker_times[speaker] = 0.0
        speaker_times[speaker] += turn.end - turn.start

    # Print results
    num_detected = len(speaker_times)
    total_speech = sum(speaker_times.values())

    print(f"\n  Speakers detected: {num_detected}")
    print(f"  Total speech: {total_speech:.1f}s")
    for spk, dur in sorted(speaker_times.items()):
        pct = (dur / total_speech * 100) if total_speech > 0 else 0
        print(f"    {spk}: {dur:.1f}s ({pct:.0f}%)")

    print(f"\n  Timeline:")
    for seg in segments:
        transcript_str = f'  "{seg["overlapping_transcript"][:60]}"' if "overlapping_transcript" in seg else ""
        # ASCII-safe output
        transcript_str = transcript_str.encode("ascii", errors="replace").decode("ascii")
        print(f"    {seg['start']:6.1f}s - {seg['end']:6.1f}s  [{seg['speaker']}]  ({seg['duration']:.1f}s){transcript_str}")

    # Clean up temp WAV
    try:
        os.remove(wav_path)
    except OSError:
        pass

    return {
        "video": video_key,
        "model": "pyannote/speaker-diarization-3.1",
        "num_speakers_forced": num_speakers,
        "num_speakers_detected": num_detected,
        "total_speech_sec": round(total_speech, 2),
        "per_speaker_sec": {k: round(v, 2) for k, v in speaker_times.items()},
        "elapsed_sec": round(elapsed, 1),
        "device": str(device),
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description="Test speaker diarization on pitch-shifted audio")
    parser.add_argument("--video", choices=list(VIDEO_MAP.keys()), help="Single video (default: all 3)")
    parser.add_argument("--speakers", type=int, help="Force exact number of speakers")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set in .env")
        sys.exit(1)

    videos = [args.video] if args.video else list(VIDEO_MAP.keys())
    os.makedirs(REPO_ROOT / "output", exist_ok=True)

    all_results = []
    for video_key in videos:
        result = run_diarization(video_key, hf_token, args.speakers)
        if result:
            all_results.append(result)

    # Save results
    output_path = REPO_ROOT / "output" / "diarization_test.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
