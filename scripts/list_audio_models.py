"""
List all audio-input-capable models on OpenRouter.

Usage:
    python scripts/list_audio_models.py
    python scripts/list_audio_models.py --modality audio_input
    python scripts/list_audio_models.py --modality audio_output
"""

import argparse
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def fetch_models():
    """Fetch all models from OpenRouter /api/v1/models."""
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def filter_audio_models(models, modality="audio_input"):
    """
    Filter models that support audio input or output.

    OpenRouter model schema has 'architecture.modality' like:
        "text->text", "text+image->text", "text+image+audio->text", etc.

    We check if 'audio' appears on the input side (left of ->) or output side.
    """
    results = []
    for m in models:
        arch = m.get("architecture", {})
        modality_str = arch.get("modality", "")

        # Split on '->' to get input and output sides
        parts = modality_str.split("->")
        if len(parts) != 2:
            continue

        input_side, output_side = parts

        if modality == "audio_input" and "audio" in input_side.lower():
            results.append(m)
        elif modality == "audio_output" and "audio" in output_side.lower():
            results.append(m)
        elif modality == "any_audio" and "audio" in modality_str.lower():
            results.append(m)

    return results


def format_price(price_str):
    """Convert price string to $/1M tokens."""
    if not price_str:
        return "N/A"
    try:
        per_token = float(price_str)
        per_million = per_token * 1_000_000
        return f"${per_million:.2f}/M"
    except (ValueError, TypeError):
        return price_str


def main():
    parser = argparse.ArgumentParser(description="List audio models on OpenRouter")
    parser.add_argument(
        "--modality",
        choices=["audio_input", "audio_output", "any_audio"],
        default="audio_input",
        help="Filter by audio capability (default: audio_input)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    print(f"Fetching models from OpenRouter...")
    all_models = fetch_models()
    print(f"Total models: {len(all_models)}")

    audio_models = filter_audio_models(all_models, args.modality)
    print(f"Models with {args.modality}: {len(audio_models)}\n")

    if args.json:
        print(json.dumps(audio_models, indent=2))
        return

    # Table output
    print(f"{'Model ID':<50} {'Modality':<30} {'Input Price':<15} {'Output Price':<15}")
    print("-" * 110)

    for m in sorted(audio_models, key=lambda x: x.get("id", "")):
        model_id = m.get("id", "?")
        arch = m.get("architecture", {})
        modality_str = arch.get("modality", "?")
        pricing = m.get("pricing", {})
        in_price = format_price(pricing.get("prompt"))
        out_price = format_price(pricing.get("completion"))

        print(f"{model_id:<50} {modality_str:<30} {in_price:<15} {out_price:<15}")

    # Show context length for top picks
    print(f"\n--- Details for top audio-input models ---\n")
    for m in sorted(audio_models, key=lambda x: x.get("id", "")):
        model_id = m.get("id", "?")
        ctx = m.get("context_length", "?")
        desc = m.get("description", "")[:120]
        print(f"  {model_id}")
        print(f"    Context: {ctx} tokens")
        print(f"    {desc}...")
        print()


if __name__ == "__main__":
    main()
