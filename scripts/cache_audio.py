"""Bulk pre-cache audio transcripts for every video in data/videos_full/.

Usage:
    python scripts/cache_audio.py          # skip already-cached videos
    python scripts/cache_audio.py --force  # re-cache all videos

Requires OPENAI_API and MY_OPENROUTER_API in .env (same as src/run.py).
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_procedure_json
from src.harness import StreamingHarness
from src.run import Pipeline, load_audio_cache, save_audio_cache, audio_cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-cache even if valid cache exists")
    ap.add_argument("--root", default="data/videos_full",
                    help="Directory containing <video_stem>/Export_py/Video_pitchshift.mp4")
    args = ap.parse_args()

    # Load .env for API keys (same pattern as src/run.py).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    openai_key = os.getenv("OPENAI_API", "")
    openrouter_key = os.getenv("MY_OPENROUTER_API", "")
    if not openai_key:
        sys.exit("ERROR: OPENAI_API not set in environment")
    if not openrouter_key:
        print("WARN: MY_OPENROUTER_API not set — will save unverified cache")

    videos = sorted(Path(args.root).glob("*/Export_py/Video_pitchshift.mp4"))
    print(f"Found {len(videos)} videos under {args.root}\n")

    # Use any clip_procedures JSON as a dummy procedure for the Pipeline ctor —
    # audio methods don't touch self.procedure or self.steps.
    any_procedure = next(Path("data/clip_procedures").glob("*.json"))
    procedure = load_procedure_json(str(any_procedure))

    tmp_dir = Path(tempfile.gettempdir())

    results = []
    for i, video_path in enumerate(videos, 1):
        stem = video_path.parent.parent.name
        if not args.force:
            cached_t, _ = load_audio_cache(str(video_path))
            if cached_t is not None:
                print(f"[{i}/{len(videos)}] {stem}: CACHED ({len(cached_t)} chunks) — skip")
                results.append((stem, "skipped", len(cached_t)))
                continue

        print(f"\n[{i}/{len(videos)}] {stem}: processing...")
        t0 = time.time()

        # Dummy harness on THIS video just so Pipeline ctor has something valid.
        # audio_chunk_sec must match the cache format (5.0).
        harness = StreamingHarness(
            video_path=str(video_path),
            procedure_path=str(any_procedure),
            speed=1.0,
            audio_chunk_sec=5.0,
        )
        pipeline = Pipeline(
            harness, openrouter_key or "", procedure,
            output_path=str(tmp_dir / f"cache_{stem}.json"), speed=1.0,
        )

        try:
            pipeline.precompute_audio(str(video_path), openai_key)
            # Guard: if every whisper call errored (network dropout, auth failure,
            # etc.) we end up with an empty audio_log. Saving that as a "cache" is
            # a silent poison — it'd be indistinguishable from a real silent video.
            # Bail before verify/save so the retry picks this video back up.
            if not pipeline.audio_log:
                raise RuntimeError(
                    "empty audio_log after precompute — likely network/API failure"
                )
            if openrouter_key:
                pipeline.verify_audio(openai_key, openrouter_key)
                verify_ran = True
            else:
                verify_ran = False
            save_audio_cache(
                str(video_path),
                pipeline._precomputed_audio,
                pipeline.audio_log,
                verify_ran=verify_ran,
            )
            dt = time.time() - t0
            print(f"[{i}/{len(videos)}] {stem}: CACHED {len(pipeline._precomputed_audio)} chunks in {dt:.0f}s")
            results.append((stem, "cached", len(pipeline._precomputed_audio)))
        except Exception as e:
            print(f"[{i}/{len(videos)}] {stem}: ERROR {e}")
            results.append((stem, f"error: {e}", 0))

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for stem, status, n in results:
        print(f"  {stem:50s} {status:12s} {n} chunks")


if __name__ == "__main__":
    main()
