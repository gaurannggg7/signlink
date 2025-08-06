#!/usr/bin/env python3
import subprocess
import json
import sys
import argparse

MODEL_NAME = None  # will be auto-detected via ollama list


def pick_model():
    """
    Auto-detect the first Ollama model you have pulled via `ollama list`.
    """
    res = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print("🛑 Failed to list Ollama models:", res.stderr.strip(), file=sys.stderr)
        sys.exit(1)

    # Skip header, take first model entry
    lines = res.stdout.strip().splitlines()[1:]
    if not lines:
        print(
            "🛑 No Ollama models found. Please pull or register a model first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Model name is the first column of the first line
    return lines[0].split()[0]


def generate_gloss(model: str, transcript: str) -> list[str]:
    """
    Run on-device inference via Ollama and parse JSON output for ASL gloss tokens.
    Supports multiple JSON output schemas:
      - {"completion": "..."}
      - {"gloss": {"text": "..."}}
      - {"gloss": {"gloss": "..."}}
      - {"gloss": {"word": "..."}}
      - {"gloss": "..."}
    """
    prompt = f"English: {transcript}\nASL Gloss:"
    cmd = ["ollama", "run", model, "--format", "json", prompt]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("🛑 Ollama inference error:", proc.stderr.strip(), file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("🛑 Failed to parse JSON. Got:", repr(proc.stdout), file=sys.stderr)
        sys.exit(1)

    # Extract text from various possible fields
    if "completion" in data:
        text = data["completion"]
    elif "gloss" in data:
        gloss_obj = data["gloss"]
        # If gloss is directly a string
        if isinstance(gloss_obj, str):
            text = gloss_obj
        # If gloss is a dict, check for common keys
        elif isinstance(gloss_obj, dict):
            if "text" in gloss_obj:
                text = gloss_obj["text"]
            elif "gloss" in gloss_obj:
                text = gloss_obj["gloss"]
            elif "word" in gloss_obj:
                text = gloss_obj["word"]
            else:
                print("🛑 Unexpected 'gloss' dict structure:", gloss_obj, file=sys.stderr)
                sys.exit(1)
        else:
            print("🛑 Unexpected 'gloss' type:", type(gloss_obj), file=sys.stderr)
            sys.exit(1)
    else:
        print("🛑 Unexpected JSON structure:", data, file=sys.stderr)
        sys.exit(1)

    # Tokenize to uppercase words
    return text.upper().split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on-device Gemma3n via Ollama (auto-detects model)"
    )
    parser.add_argument(
        "text",
        help="English text to translate to ASL gloss",
    )
    args = parser.parse_args()

    model = pick_model()
    print(f"Using Ollama model: {model}")
    tokens = generate_gloss(model, args.text)
    print("→", tokens)
