"""Retroactively apply the hallucination filter to every existing audio cache.

Usage:
    python scripts/clean_cache.py           # clean all caches in-place
    python scripts/clean_cache.py --dry-run # show what WOULD change, write nothing

Applies:
  - NO_SPEECH overwrite for entries matched by is_hallucination()
  - Removes the corresponding key from `transcripts` dict
  - Rewrites cache_file with a new `cleaned_at` marker

Idempotent — re-running on an already-cleaned cache is a no-op, because
NO_SPEECH entries are skipped before is_hallucination() is called.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audio_enhance import is_hallucination, is_no_speech


def clean_cache(cache_path: Path, dry_run: bool = False) -> dict:
    """Returns {"path": str, "changed": N, "kept": M, "hits": [(start, before, after)]}."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    new_audio_log = []
    new_transcripts = {}
    hits = []

    for entry in data["audio_log"]:
        s, e, t = entry[0], entry[1], entry[2]
        if t and not is_no_speech(t) and t != "GARBAGE" and is_hallucination(t):
            hits.append((float(s), t, "NO_SPEECH"))
            new_audio_log.append([float(s), float(e), "NO_SPEECH"])
            # Drop from transcripts dict (keyed by start time as string)
            continue
        new_audio_log.append([float(s), float(e), t])
        if t and not is_no_speech(t) and t != "GARBAGE":
            new_transcripts[f"{float(s)}"] = t

    changed = len(hits)
    if changed and not dry_run:
        data["audio_log"] = new_audio_log
        data["transcripts"] = new_transcripts
        data["cleaned_at"] = datetime.utcnow().isoformat() + "Z"
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {
        "path": str(cache_path),
        "changed": changed,
        "kept": len(new_transcripts),
        "hits": hits,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Show changes but don't write cache files")
    ap.add_argument("--cache-dir", default="data/audio_cache",
                    help="Directory containing per-video cache JSONs")
    args = ap.parse_args()

    caches = sorted(Path(args.cache_dir).glob("*.json"))
    print(f"Scanning {len(caches)} caches...\n")

    total_changed = 0
    for cache in caches:
        result = clean_cache(cache, dry_run=args.dry_run)
        tag = "DRY " if args.dry_run else ""
        print(f"{tag}{cache.stem:50s} {result['changed']:3d} filtered, {result['kept']:3d} kept")
        for s, before, after in result["hits"]:
            snippet = before[:70].replace("\n", " ")
            print(f"    [{s:6.1f}s] \"{snippet}\" -> {after}")
        total_changed += result["changed"]

    print(f"\n{'DRY ' if args.dry_run else ''}Total filtered across all caches: {total_changed}")


if __name__ == "__main__":
    main()
