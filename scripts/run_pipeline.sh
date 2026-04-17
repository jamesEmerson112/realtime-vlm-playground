#!/usr/bin/env bash
# scripts/run_pipeline.sh — Run VLM pipeline + evaluator + dashboard on a clip.
#
# Usage:
#   ./scripts/run_pipeline.sh [clip_name] [speed] [tag] [model]
#
# Arguments (all optional — positional):
#   clip_name   procedure JSON basename (default: R066-15July-Circuit-Breaker-part2)
#   speed       playback speed multiplier, 1.0 = eval speed (default: 1.0)
#   tag         short descriptor that appears in the output filename (default: v5)
#   model       OpenRouter model string (default: google/gemini-3.1-flash-image-preview)
#
# Examples:
#   ./scripts/run_pipeline.sh                                      # R066 @ 1x, tag=v5
#   ./scripts/run_pipeline.sh R066-15July-Circuit-Breaker-part2 2.0 fast
#   ./scripts/run_pipeline.sh R066-15July-Circuit-Breaker-part2 1.0 v5 google/gemini-3.1-flash-image-preview
#
# Output files (all auto-prefixed YYYYMMDD_HHMM_):
#   output/<timestamp>_<clip_short>_<tag>_<speed>x.json            # events
#   output/<timestamp>_<clip_short>_<tag>_<speed>x_log.json        # pipeline log JSON
#   output/<timestamp>_<clip_short>_<tag>_<speed>x_log.md          # pipeline log markdown
#   output/<timestamp>_<clip_short>_<tag>_<speed>x_audio.json      # audio transcript
#   output/<timestamp>_<clip_short>_<tag>_<speed>x_dashboard.html  # timeline dashboard

set -euo pipefail

# ---------- python interpreter ----------
# Prefer $PYTHON if the caller sets it. Otherwise use `python`, but fall back
# to the conda `vlm` env if `python` doesn't have cv2 (common on Windows where
# the shell's `python` is the base env).
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import cv2" 2>/dev/null; then
  for CAND in \
      "/c/Users/voan2/.conda/envs/vlm/python.exe" \
      "$HOME/.conda/envs/vlm/python.exe" \
      "$HOME/miniconda3/envs/vlm/python.exe" \
      "$HOME/anaconda3/envs/vlm/python.exe"; do
    if [[ -x "$CAND" ]] && "$CAND" -c "import cv2" 2>/dev/null; then
      PYTHON="$CAND"
      break
    fi
  done
fi

if ! "$PYTHON" -c "import cv2" 2>/dev/null; then
  echo "ERROR: no python interpreter with cv2 available." >&2
  echo "       Activate the 'vlm' conda env, or set PYTHON=/path/to/python." >&2
  exit 1
fi

# ---------- args ----------
CLIP="${1:-R066-15July-Circuit-Breaker-part2}"
SPEED="${2:-1.0}"
TAG="${3:-v5}"
MODEL="${4:-google/gemini-3.1-flash-image-preview}"

PROCEDURE="data/clip_procedures/${CLIP}.json"
VIDEO="data/videos_full/${CLIP}/Export_py/Video_pitchshift.mp4"
GT="data/ground_truth_sample/${CLIP}.json"

# ---------- preflight ----------
for f in "$PROCEDURE" "$VIDEO"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing input file: $f" >&2
    echo "       make sure the clip name matches files under data/" >&2
    exit 1
  fi
done

# short clip id for filename: first dash-separated token, lowercased (e.g. R066 -> r066)
CLIP_SHORT="$(echo "$CLIP" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')"
# speed tag: 1.0 -> 1_0x, 2.0 -> 2_0x
SPEED_TAG="$(echo "$SPEED" | tr . _)x"

# src/run.py auto-prepends YYYYMMDD_HHMM_ (see CLAUDE.md) — we just supply the descriptor
OUTPUT_BASENAME="${CLIP_SHORT}_${TAG}_${SPEED_TAG}"
OUTPUT="output/${OUTPUT_BASENAME}.json"

# ---------- run pipeline ----------
echo "==============================================="
echo "  VLM Pipeline"
echo "==============================================="
echo "  Clip:      $CLIP"
echo "  Procedure: $PROCEDURE"
echo "  Video:     $VIDEO"
echo "  Speed:     ${SPEED}x"
echo "  Model:     $MODEL"
echo "  Output:    $OUTPUT  (auto-prefixed with timestamp)"
echo "==============================================="
echo

"$PYTHON" src/run.py \
  --procedure "$PROCEDURE" \
  --video "$VIDEO" \
  --output "$OUTPUT" \
  --speed "$SPEED" \
  --model "$MODEL"

# ---------- locate the actual (timestamped) output ----------
# run.py prepends YYYYMMDD_HHMM_ so the file ends up as output/<ts>_<basename>.json.
# Grab the newest file that matches.
ACTUAL_OUTPUT="$(ls -t output/*_"${OUTPUT_BASENAME}".json 2>/dev/null | head -n 1 || true)"

if [[ -z "${ACTUAL_OUTPUT:-}" ]]; then
  echo "ERROR: pipeline did not produce output/*_${OUTPUT_BASENAME}.json" >&2
  exit 1
fi

BASE="${ACTUAL_OUTPUT%.json}"

# ---------- evaluate ----------
if [[ -f "$GT" ]]; then
  echo
  echo "==============================================="
  echo "  Evaluator"
  echo "==============================================="
  echo "  Predicted:    $ACTUAL_OUTPUT"
  echo "  Ground truth: $GT"
  echo "  Tolerance:    5s"
  echo "==============================================="
  echo

  "$PYTHON" -m src.evaluator \
    --predicted "$ACTUAL_OUTPUT" \
    --ground-truth "$GT" \
    --tolerance 5
else
  echo
  echo "(no ground truth at $GT — skipping evaluation)"
fi

# ---------- dashboard ----------
if [[ -f "$GT" ]]; then
  DASHBOARD="${BASE}_dashboard.html"
  echo
  echo "==============================================="
  echo "  Dashboard"
  echo "==============================================="
  echo "  Output: $DASHBOARD"
  echo "==============================================="
  echo

  "$PYTHON" -m src.dashboard \
    --predicted "$ACTUAL_OUTPUT" \
    --ground-truth "$GT" \
    --output "$DASHBOARD"
fi

# ---------- summary ----------
echo
echo "==============================================="
echo "  Done — output files"
echo "==============================================="
echo "  events:    $ACTUAL_OUTPUT"
for suffix in _log.json _log.md _audio.json _dashboard.html; do
  [[ -f "${BASE}${suffix}" ]] && echo "  $(printf '%-10s' "${suffix#_}:") ${BASE}${suffix}"
done
echo "==============================================="
