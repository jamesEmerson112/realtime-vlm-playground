"""Side-by-side comparison of two pipeline runs against the same ground truth.

Usage:
    python scripts/_compare_runs.py <predicted_a.json> <predicted_b.json> \
        <ground_truth.json> <label_a> <label_b>

Called by scripts/benchmark_vlm_models.sh. Prints an F1 / latency / combined-
score comparison and declares a winner per the CLAUDE.md scoring formula:
    combined = 0.40 * step_f1 + 0.40 * error_f1 + 0.20 * latency_score
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluator import evaluate as eval_run  # noqa: E402


def _score(predicted: str, gt: str) -> dict:
    if not Path(gt).exists():
        with open(predicted) as f:
            pred = json.load(f)
        return {
            "no_gt": True,
            "events_total": len(pred.get("events", [])),
            "mean_delay": pred.get("mean_detection_delay_sec"),
            "max_delay": pred.get("max_detection_delay_sec"),
        }
    m = eval_run(predicted, gt, time_tolerance_sec=5.0, verbose=False)
    d = asdict(m)
    latency_score = max(0.0, 1.0 - d["mean_detection_delay_sec"] / 10.0)
    combined = (
        0.40 * d["step_f1"]
        + 0.40 * d["error_f1"]
        + 0.20 * latency_score
    )
    d["latency_score"] = round(latency_score, 3)
    d["combined"] = round(combined, 3)
    d["no_gt"] = False
    return d


def _fmt(x, width: int = 10) -> str:
    if x is None:
        return f"{'-':<{width}s}"
    if isinstance(x, float):
        return f"{x:<{width}.3f}"
    return f"{str(x):<{width}s}"


def _delta(a, b) -> str:
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return "-"
    return f"{b - a:+.3f}"


def main():
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    out_a, out_b, gt, label_a, label_b = sys.argv[1:6]
    a = _score(out_a, gt)
    b = _score(out_b, gt)

    if a.get("no_gt") or b.get("no_gt"):
        print()
        print("Events / latency only (no GT to score against):")
        print(f"  A  {label_a:<48s}  events={a.get('events_total'):<4}  mean_delay={_fmt(a.get('mean_delay'))}")
        print(f"  B  {label_b:<48s}  events={b.get('events_total'):<4}  mean_delay={_fmt(b.get('mean_delay'))}")
        return

    rows = [
        ("step_f1",           "step_f1"),
        ("step_precision",    "step_precision"),
        ("step_recall",       "step_recall"),
        ("step_tp",           "step_tp"),
        ("step_fp",           "step_fp"),
        ("step_fn",           "step_fn"),
        ("error_f1",          "error_f1"),
        ("error_precision",   "error_precision"),
        ("error_recall",      "error_recall"),
        ("error_tp",          "error_tp"),
        ("error_fp",          "error_fp"),
        ("error_fn",          "error_fn"),
        ("idle_f1",           "idle_f1"),
        ("idle_precision",    "idle_precision"),
        ("idle_recall",       "idle_recall"),
        ("idle_tp",           "idle_tp"),
        ("idle_fp",           "idle_fp"),
        ("idle_fn",           "idle_fn"),
        ("mean_delay_sec",    "mean_detection_delay_sec"),
        ("max_delay_sec",     "max_detection_delay_sec"),
        ("latency_score",     "latency_score"),
        ("COMBINED",          "combined"),
    ]

    col_a_head = f"A ({label_a[-28:]})"
    col_b_head = f"B ({label_b[-28:]})"

    print()
    print(f"{'metric':<18s}  {col_a_head:<34s}  {col_b_head:<34s}  {'B - A':>10s}")
    print("-" * 100)
    for display_name, key in rows:
        va = a.get(key)
        vb = b.get(key)
        print(f"{display_name:<18s}  {_fmt(va, 34)}  {_fmt(vb, 34)}  {_delta(va, vb):>10s}")

    print()
    winner = None
    if a["combined"] > b["combined"]:
        winner = f"Model A  ({label_a})"
    elif b["combined"] > a["combined"]:
        winner = f"Model B  ({label_b})"
    if winner:
        print(f"Winner by combined score: {winner}")
        print(f"  A.combined = {a['combined']:.3f}   B.combined = {b['combined']:.3f}")
    else:
        print("Tie on combined score.")


if __name__ == "__main__":
    main()
