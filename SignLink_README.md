# SignLink — Voice-to-Sign Language AI Pipeline

> Real-time English speech/text → American Sign Language video, deployed on Hugging Face Spaces.

[**▶ Try the live demo**](https://huggingface.co/spaces/gaurannggg7/Signlink) · [Landing page](https://huggingface.co/spaces/gaurannggg7/asl-translator-website) · [Dataset](https://huggingface.co/datasets/gaurannggg7/asl-dictionary)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Spaces-yellow) ![Whisper](https://img.shields.io/badge/ASR-Whisper-green) ![License](https://img.shields.io/badge/assets-CC0-lightgrey)

---

## What it does

SignLink takes spoken or typed English and renders a continuous American Sign Language video. The pipeline:

1. **Transcribe** — OpenAI Whisper converts speech to text.
2. **Gloss** — a constrained-prompt LLM stage rewrites English into ASL gloss (drops articles/linking verbs, present-tense root forms).
3. **Resolve** — a 3-stage out-of-vocabulary system maps every token to a motion-capture clip with zero silent drops.
4. **Render** — an FFmpeg pipeline normalizes and stitches the matched clips into one playable video.

## Why it's interesting

**3-stage OOV resolution → 100% token coverage.** No input ever fails silently:

| Stage | Method | Example |
|-------|--------|---------|
| 1. Exact match | `os.walk` runtime indexer over 1,566 nested assets, greedy multi-word phrase matching | `CLEAN` → `CLEAN.mp4` |
| 2. Lemmatization | Regex suffix stripping (-ING, -ED, -LY, -S …) | `CLEANING` → `CLEAN` |
| 3. Fingerspelling | A–Z letter clips spell anything unknown | `DELTA` → `D·E·L·T·A` |

**Cloud-native architecture.** The original monolithic script OOM-crashed on CPU containers (4GB+ model weights). Refactored into a decoupled design: a Hugging Face Dataset acts as the storage layer (1,566 clips), Whisper runs CPU-optimized, LLM inference is offloaded, and the Streamlit UI runs in a ~100MB container.

**FFmpeg synthesis pipeline.** Two-pass subprocess approach — normalize each clip to 1280×720 individually, then concat-demux with a lossless copy. Replaced an `ffmpeg-python` filter-graph approach that broke on reused filter nodes ("multiple outgoing edges"). PID-namespaced temp files keep concurrent requests from colliding.

## Architecture

```
English speech/text
      │
      ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌──────────────┐
│   Whisper    │──▶│  Gloss stage │──▶│  3-stage OOV resolve │──▶│   FFmpeg     │
│  (ASR, CPU)  │   │ (constrained │   │  exact→lemma→spell   │   │  normalize + │
│              │   │   prompt)    │   │  1,566 mocap assets  │   │  concat      │
└──────────────┘   └──────────────┘   └─────────────────────┘   └──────────────┘
                                                                        │
                                                                        ▼
                                                                  ASL video out
```

## Tech stack

**ML/AI:** Hugging Face Transformers · Whisper (ASR) · Gemma-2B (gloss) · PyTorch
**Video:** FFmpeg (multi-pass normalization + concat demuxer)
**App:** Streamlit (custom dark theme)
**Data:** Hugging Face Datasets (1,540 signs + A–Z fingerspelling)
**Deploy:** Hugging Face Spaces · Docker · headless-container optimized

## Run locally

```bash
git clone https://github.com/gaurannggg7/signlink.git
cd signlink
pip install -r requirements.txt
streamlit run app.py
```

The app pulls the ASL asset dataset from Hugging Face at startup. To run the LLM gloss stage you'll need an `HF_TOKEN` set in your environment; without it the app falls back to a stopword-based gloss builder.

## Assets & attribution

Motion-capture sign clips are from the [StudioGalt Sign-Language Mocap Archive](https://github.com/StudioGalt/Sign-Language-Mocap-Archive) (CC0). Avatar: "Galtis."

## License

Code released under MIT. Sign-language assets are CC0 (StudioGalt).
