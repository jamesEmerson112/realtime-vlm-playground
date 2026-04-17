#!/usr/bin/env bash
# scripts/benchmark_vlm_models.sh — Compare two VLM models on the same clip.
#
# Runs the full pipeline twice (once per model) on the same clip at the same
# speed, then prints a side-by-side comparison of step/error/idle F1 and
# detection latency pulled from each run's predicted events vs. ground truth.
#
# Usage:
#   ./scripts/benchmark_vlm_models.sh [clip_name] [speed]
#
# Arguments:
#   clip_name   procedure JSON basename (default: R066-15July-Circuit-Breaker-part2)
#   speed       playback speed multiplier (default: 1.0)
#
# Environment overrides:
#   MODEL_A     first VLM model   (default: google/gemini-3.1-flash-image-preview — current default)
#   MODEL_B     second VLM model  (default: google/gemini-3-flash-preview — former default)
#   TAG_A       filename tag for model A (default: m-gemini31image)
#   TAG_B       filename tag for model B (default: m-gemini3flash)
#
# Output:
#   Two independent pipeline runs under output/<timestamp>_<clip>_<tag>_<speed>x.*
#   A side-by-side F1 / latency report printed to stdout.

set -euo pipefail

CLIP="${1:-R066-15July-Circuit-Breaker-part2}"
SPEED="${2:-1.0}"

MODEL_A="${MODEL_A:-google/gemini-3.1-flash-image-preview}"
MODEL_B="${MODEL_B:-google/gemini-3-flash-preview}"
TAG_A="${TAG_A:-m-gemini31image}"
TAG_B="${TAG_B:-m-gemini3flash}"

GT="data/ground_truth_sample/${CLIP}.json"
RUNNER="./scripts/run_pipeline.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: $RUNNER not found or not executable." >&2
  exit 1
fi

# ---------- python interpreter (for post-run summary) ----------
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import json" 2>/dev/null; then
  for CAND in \
      "/c/Users/voan2/.conda/envs/vlm/python.exe" \
      "$HOME/.conda/envs/vlm/python.exe"; do
    [[ -x "$CAND" ]] && PYTHON="$CAND" && break
  done
fi

SPEED_TAG="$(echo "$SPEED" | tr . _)x"
CLIP_SHORT="$(echo "$CLIP" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')"

run_and_capture() {
  local model="$1"
  local tag="$2"
  local outvar="$3"
  local basename_pattern="${CLIP_SHORT}_${tag}_${SPEED_TAG}"

  echo
  echo "#############################################################"
  echo "#  BENCHMARK RUN — $model"
  echo "#############################################################"

  "$RUNNER" "$CLIP" "$SPEED" "$tag" "$model"

  # newest matching events file (not dashboard/log sidecars)
  local found
  found="$(ls -t output/*_"${basename_pattern}".json 2>/dev/null \
           | grep -v '_log.json$' \
           | grep -v '_audio.json$' \
           | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "ERROR: could not locate events file for $tag ($basename_pattern)" >&2
    exit 1
  fi
  printf -v "$outvar" '%s' "$found"
}

run_and_capture "$MODEL_A" "$TAG_A" OUT_A
run_and_capture "$MODEL_B" "$TAG_B" OUT_B

echo
echo "============================================================"
echo "  BENCHMARK SUMMARY — ${CLIP} @ ${SPEED}x"
echo "============================================================"
echo "  Model A:  ${MODEL_A}  ->  $OUT_A"
echo "  Model B:  ${MODEL_B}  ->  $OUT_B"
echo "  Ground:   $GT"
echo "============================================================"

if [[ ! -f "$GT" ]]; then
  echo "(no ground truth — only raw event counts will be reported)"
fi

"$PYTHON" scripts/_compare_runs.py "$OUT_A" "$OUT_B" "$GT" "$MODEL_A" "$MODEL_B"
