"""
Offline comparison: Ensemble vs Filtered Ensemble.

Reads existing ensemble JSON results, applies hallucination filter,
re-runs voting, and generates a side-by-side markdown report.

No API calls — purely offline analysis.

Usage:
    python scripts/compare_ensemble.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio_enhance import (
    ENSEMBLE_MODELS,
    filter_transcript,
    is_no_speech,
    normalize_transcript,
    transcripts_similar,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output" / "audio_optimization"

KEY_TO_MODEL = {
    "whisper_1": "whisper-1",
    "gpt_4o_transcribe": "gpt-4o-transcribe",
    "gemini_3_flash": "google/gemini-3-flash-preview",
    "gpt_4o_audio_preview": "openai/gpt-4o-audio-preview",
}

SHORT_NAMES = {
    "whisper-1": "w1",
    "gpt-4o-transcribe": "g4t",
    "google/gemini-3-flash-preview": "gem",
    "openai/gpt-4o-audio-preview": "g4a",
}


def rerun_filtered(chunk):
    """Apply filter to raw transcripts and re-vote."""
    filtered = {}
    filters_applied = []

    for short_key, model_id in KEY_TO_MODEL.items():
        raw = chunk.get(short_key)
        if raw:
            filt, was_filtered = filter_transcript(raw)
            filtered[model_id] = filt
            if was_filtered:
                filters_applied.append(model_id)
        else:
            filtered[model_id] = "NO_SPEECH"

    # Vote on filtered outputs
    valid = {m: t for m, t in filtered.items() if not is_no_speech(t)}
    agreeing_models = []
    final = "NO_SPEECH"
    method = "no_speech"

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
            pf = filtered.get("openai/gpt-4o-audio-preview", "NO_SPEECH")
            if is_no_speech(pf):
                final = "NO_SPEECH"
                method = "no_speech"
            else:
                final = "GARBAGE"
                method = "garbage"

    return final, method, agreeing_models, filters_applied, filtered


def classify(text):
    if not text:
        return "NO_SPEECH"
    if is_no_speech(text):
        return "NO_SPEECH"
    if text == "GARBAGE":
        return "GARBAGE"
    return "speech"


def trunc(text, n=35):
    if not text:
        return "-"
    t = text.replace("\n", " ")
    return t[:n] + "..." if len(t) > n else t


def generate_report(videos_data):
    lines = []
    lines.append("# Ensemble vs Filtered Ensemble: Comparison Report\n")
    lines.append("> Generated offline from existing ensemble data. No API calls.\n")

    # --- Summary table ---
    lines.append("## Summary\n")
    lines.append("| Video | Method | NO_SPEECH | Speech | GARBAGE | Total |")
    lines.append("|-------|--------|-----------|--------|---------|-------|")

    all_changes = {}

    for video, data in videos_data.items():
        ens = {"NO_SPEECH": 0, "speech": 0, "GARBAGE": 0}
        filt = {"NO_SPEECH": 0, "speech": 0, "GARBAGE": 0}
        changes = []

        for chunk in data:
            ens_final = chunk["final_transcript"]
            filt_final, filt_method, filt_agree, filt_applied, filt_map = rerun_filtered(chunk)

            ens[classify(ens_final)] += 1
            filt[classify(filt_final)] += 1

            if normalize_transcript(ens_final or "") != normalize_transcript(filt_final or ""):
                changes.append({
                    "chunk": chunk,
                    "ens_final": ens_final,
                    "filt_final": filt_final,
                    "ens_method": chunk["method"],
                    "filt_method": filt_method,
                    "filters_applied": filt_applied,
                    "filt_map": filt_map,
                })

        total = len(data)
        lines.append(f"| {video} | Ensemble | {ens['NO_SPEECH']} | {ens['speech']} | {ens['GARBAGE']} | {total} |")
        lines.append(f"| {video} | Filtered | {filt['NO_SPEECH']} | {filt['speech']} | {filt['GARBAGE']} | {total} |")
        lines.append(f"| {video} | **Delta** | **{filt['NO_SPEECH']-ens['NO_SPEECH']:+d}** | **{filt['speech']-ens['speech']:+d}** | **{filt['GARBAGE']-ens['GARBAGE']:+d}** | |")
        all_changes[video] = changes

    lines.append("")

    # --- Per-video side-by-side ---
    for video, data in videos_data.items():
        lines.append(f"## {video}: Side-by-Side\n")
        lines.append("| Time | whisper-1 | gpt-4o-transcribe | gemini-3-flash | gpt-4o-audio | Ensemble | Filtered | Changed |")
        lines.append("|------|-----------|-------------------|----------------|--------------|----------|----------|---------|")

        for chunk in data:
            ens_final = chunk["final_transcript"]
            filt_final, _, _, filt_applied, _ = rerun_filtered(chunk)

            t = f"{chunk['chunk_start']:.0f}-{chunk['chunk_end']:.0f}s"
            w1 = trunc(chunk.get("whisper_1"), 20)
            g4t = trunc(chunk.get("gpt_4o_transcribe"), 20)
            gem = trunc(chunk.get("gemini_3_flash"), 20)
            g4a = trunc(chunk.get("gpt_4o_audio_preview"), 20)
            e = trunc(ens_final, 20)
            f_ = trunc(filt_final, 20)

            changed = normalize_transcript(ens_final or "") != normalize_transcript(filt_final or "")
            ch_mark = "**YES**" if changed else ""

            lines.append(f"| {t} | {w1} | {g4t} | {gem} | {g4a} | **{e}** | **{f_}** | {ch_mark} |")

        lines.append("")

    # --- Diff highlights ---
    lines.append("## Changed Chunks: Detail\n")

    total_changes = sum(len(v) for v in all_changes.values())
    if total_changes == 0:
        lines.append("No chunks changed between methods.\n")
    else:
        lines.append(f"**{total_changes} chunks differ** across all videos.\n")

        for video, changes in all_changes.items():
            if not changes:
                continue

            lines.append(f"### {video} ({len(changes)} changes)\n")

            for c in changes:
                chunk = c["chunk"]
                t = f"{chunk['chunk_start']:.0f}-{chunk['chunk_end']:.0f}s"
                filt_names = ", ".join(SHORT_NAMES.get(m, m) for m in c["filters_applied"]) or "none"

                lines.append(f"**{t}** | `{c['ens_method']}` -> `{c['filt_method']}` | filtered: {filt_names}")
                lines.append("")
                lines.append("| Model | Raw Transcript |")
                lines.append("|-------|----------------|")
                for short_key in KEY_TO_MODEL:
                    model_id = KEY_TO_MODEL[short_key]
                    raw = chunk.get(short_key) or "-"
                    filt_tag = " **[FILTERED]**" if model_id in c["filters_applied"] else ""
                    lines.append(f"| {SHORT_NAMES[model_id]} | {trunc(raw, 60)}{filt_tag} |")
                lines.append(f"| | **Ensemble final:** {trunc(c['ens_final'], 60)} |")
                lines.append(f"| | **Filtered final:** {trunc(c['filt_final'], 60)} |")
                lines.append("")

    # --- Analysis ---
    lines.append("## Analysis\n")
    lines.append("### Why the delta is small\n")
    lines.append("The existing `run_ensemble()` already calls `is_hallucination()` to exclude hallucinated")
    lines.append("transcripts from the valid pool before voting. The filtered ensemble converts them to")
    lines.append("NO_SPEECH *votes* instead of simply excluding them, but since conservative models")
    lines.append("(gemini-3-flash, gpt-4o-audio-preview) already vote NO_SPEECH on silence, the outcome")
    lines.append("rarely changes.\n")
    lines.append("### False positive risk\n")
    lines.append('The pattern `"your class is over"` matches real instructor speech in R066 170-175s')
    lines.append("(instructor ending the procedure). This converts a correct 2-model agreement into GARBAGE.")
    lines.append('Recommend removing `"your class is over"` and `"the end"` from HALLUCINATION_PATTERNS —')
    lines.append("these are plausible classroom/procedure phrases.\n")
    lines.append("### Conclusion\n")
    lines.append("Filtered ensemble provides negligible uplift over plain ensemble on this dataset.")
    lines.append("The main value of the expanded pattern list is for the `passes` method (single-model"),
    lines.append("hallucination filtering). For the ensemble, the voting mechanism already handles")
    lines.append("hallucinations effectively.\n")

    return "\n".join(lines)


def main():
    videos_data = {}
    for video in ["R066", "z065"]:
        path = OUTPUT_DIR / f"audio_enhance_ensemble_{video}.json"
        with open(path, encoding="utf-8") as f:
            videos_data[video] = json.load(f)

    report = generate_report(videos_data)

    out_path = OUTPUT_DIR / "ensemble_vs_filtered_comparison.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
