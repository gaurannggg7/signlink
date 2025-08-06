Offline ASL Translator 🗣️ 🎥👐

An end‑to‑end Streamlit app that converts English speech or text into American Sign Language (ASL) gloss tokens and stitches corresponding sign‑language video clips into a playable video.

Features

Multi‑Modal Input: Text, audio file (WAV/MP3), or live microphone recording

Gemma 3n Integration: Switch between models (gemma3n_E2B, gemma3n-E4B) via dropdown

Device Fallback: Automatically chooses CUDA → MPS → CPU; falls back to CPU on OOM

Auto‑Download: Missing Gemma models are downloaded from Hugging Face at runtime

Live Captions: Toggle real‑time transcript display

ASL Gloss Builder: Extracts or generates uppercase gloss tokens

Dictionary Lookup: Maps tokens to local video clips via CSV index

Video Rendering: Concatenates clips into a single output stream

Error Handling: Friendly pop‑ups for missing mic, model load failures, or missing clips

Prerequisites

Python 3.8+

Git (for cloning)

Streamlit

PyTorch or Apple Silicon build for MPS

Optional: Coral Edge TPU & compiled gemma3n_edgetpu.tflite

Usage

Select a Gemma model (gemma3n_E2B for local, gemma3n-E4B to auto‑download large model).

Choose Input mode: Text, Audio File, or Record Mic.

Click Translate to process.

View Live Transcript (if enabled) and Gloss Tokens.

The video player displays the concatenated ASL clip.

Customization

Model Aliases: In gemma_loader.py’s hf_alias dict, map any dropdown key to a HF repo or local folder.

Device Selection: Override get_best_device() in utils.py to change device priority.

Styling: Tweak CSS in app.py to adjust theme, fonts, or layout.

Contributing

Fork this repository

Create a feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m "Add feature")

Push (git push origin feature/YourFeature)

Open a Pull Request

Please ensure tests pass and add new ones for any new functionality.

