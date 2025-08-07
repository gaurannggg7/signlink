# Offline ASL Translator 🗣️ 🎥👐

An end‑to‑end Streamlit app that converts English speech or text into American Sign Language (ASL) gloss tokens and stitches corresponding sign‑language video clips into a playable video.

## ⚠️ Important: Video Dataset File Structure

Due to GitHub's limitation of 1000 files per directory, the ASL video dataset is organized across multiple folders in `content/asl_app_data/dictionary/`. **You will need to ensure the file paths in the code correctly reference this multi-folder structure.**

### Directory Structure:
```
signlink/
├── content/
│   └── asl_app_data/
│       └── dictionary/
│           ├── folder1/    # First batch of ASL video files
│           ├── folder2/    # Second batch of ASL video files
│           ├── folder3/    # Additional video files
│           └── ...         # More folders as needed
└── ...
```

### Important Setup Notes:
1. The ASL video clips are stored in `content/asl_app_data/dictionary/`
2. Due to GitHub's 1000-file limit per directory, videos are distributed across multiple subfolders
3. The CSV index file should map gloss tokens to the correct video paths including subfolder names
4. You may need to update the base path variable in your code to point to `content/asl_app_data/dictionary/`

Example path configuration:
```python
# Base directory for ASL videos
BASE_VIDEO_PATH = "content/asl_app_data/dictionary/"

# Example video path with subfolder
video_path = os.path.join(BASE_VIDEO_PATH, "folder1", "HELLO.mp4")
```

## Features

- **Multi‑Modal Input**: Text, audio file (WAV/MP3), or live microphone recording
- **Gemma 3n Integration**: Switch between models (gemma3n_E2B, gemma3n-E4B) via dropdown
- **Device Fallback**: Automatically chooses CUDA → MPS → CPU; falls back to CPU on OOM
- **Auto‑Download**: Missing Gemma models are downloaded from Hugging Face at runtime
- **Live Captions**: Toggle real‑time transcript display
- **ASL Gloss Builder**: Extracts or generates uppercase gloss tokens
- **Dictionary Lookup**: Maps tokens to local video clips via CSV index
- **Video Rendering**: Concatenates clips into a single output stream
- **Error Handling**: Friendly pop‑ups for missing mic, model load failures, or missing clips

## Prerequisites

- Python 3.8+
- Git (for cloning)
- Streamlit
- PyTorch or Apple Silicon build for MPS
- Optional: Coral Edge TPU & compiled gemma3n_edgetpu.tflite

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/gaurannggg7/signlink.git
   cd signlink
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify video dataset structure**: Ensure all video files are properly organized in `content/asl_app_data/dictionary/` with its subfolders.

4. **Update configuration**: Check that the video lookup paths in your code point to the correct directory structure.

## Usage

1. Select a Gemma model (gemma3n_E2B for local, gemma3n-E4B to auto‑download large model).
2. Choose Input mode: Text, Audio File, or Record Mic.
3. Click Translate to process.
4. View Live Transcript (if enabled) and Gloss Tokens.
5. The video player displays the concatenated ASL clip.

## Customization

- **Model Aliases**: In `gemma_loader.py`'s `hf_alias` dict, map any dropdown key to a HF repo or local folder.
- **Device Selection**: Override `get_best_device()` in `utils.py` to change device priority.
- **Video Path Configuration**: Update the base path to `content/asl_app_data/dictionary/` and ensure the video lookup logic handles the multi-folder structure.
- **Styling**: Tweak CSS in `app.py` to adjust theme, fonts, or layout.

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure tests pass and add new ones for any new functionality.

## Example Usage for Ollama.

```bash
python demo_ollama.py "English: Hello, can you please clean the counter? ASL Gloss:"
```

## Troubleshooting

- **Missing video files**: Verify all video folders are present in `content/asl_app_data/dictionary/`
- **Path errors**: Ensure the CSV index file contains paths relative to the base directory including subfolder names
- **File not found errors**: Check that your code's base path variable points to `content/asl_app_data/dictionary/`
- **Performance issues**: Consider loading videos on-demand rather than all at once if dealing with large datasets
