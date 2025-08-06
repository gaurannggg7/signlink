import os
import sounddevice as sd
import soundfile as sf
import tempfile
import whisper
from utils import get_best_device

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration: model name via environment
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")


def load_whisper_model(model_name: str = WHISPER_MODEL_NAME):
    """
    Load and cache the Whisper model on the best available device,
    with fallback to CPU if the preferred device is unsupported.
    """
    if not hasattr(load_whisper_model, "_model"):
        # Determine best device (e.g., cuda, mps, or cpu)
        best = get_best_device().type
        try:
            print(f"Loading Whisper model '{model_name}' on {best}…")
            load_whisper_model._model = whisper.load_model(model_name, device=best)
        except NotImplementedError as e:
            print(f"Warning: failed to load model on {best}: {e}. Falling back to CPU.")
            load_whisper_model._model = whisper.load_model(model_name, device="cpu")
    return load_whisper_model._model


def transcribe_file(path: str, model_name: str = WHISPER_MODEL_NAME) -> str:
    """
    Transcribe an audio file to text using the loaded Whisper model.
    Raises a RuntimeError on failure.
    """
    model = load_whisper_model(model_name)
    try:
        result = model.transcribe(path, fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")


def list_input_devices():
    """
    Return a list of (index, name) for all input-capable audio devices.
    """
    devices = sd.query_devices()
    return [(i, d.get('name', '')) for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]


def record_audio(duration: int, sr: int = 16000, device_index: int = None) -> tuple[int, list[float]]:
    """
    Record `duration` seconds of mono audio at sample rate `sr`.
    Automatically selects a valid input device if none is provided.
    Raises RuntimeError on failure.
    """
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No input audio devices found.")

    # Choose device: user-specified or default, else first available
    if device_index is None:
        default = sd.default.device
        if isinstance(default, (tuple, list)) and default[0] is not None:
            device_index = default[0]
        else:
            device_index = devices[0][0]

    # Validate device
    if device_index not in [i for i, _ in devices]:
        device_index = devices[0][0]

    # Configure recorder
    sd.default.device = (device_index, None)
    sd.default.samplerate = sr
    sd.default.channels = 1

    print(f"Recording {duration}s at {sr}Hz from device #{device_index}…")
    try:
        data = sd.rec(int(duration * sr), dtype='float32', channels=1)
        sd.wait()
    except Exception as e:
        raise RuntimeError(f"Audio recording failed: {e}")

    # Convert to 1D float32 list
    wav = data.flatten().tolist()
    return sr, wav


def transcribe_from_mic(duration: int) -> str:
    """
    Record `duration` seconds from the microphone, save to a temp WAV, transcribe, then clean up.
    Raises RuntimeError on recording or transcription errors.
    """
    try:
        sr, wav = record_audio(duration)
    except Exception as e:
        raise RuntimeError(f"Microphone recording error: {e}")

    # Write to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, wav, sr)

    try:
        text = transcribe_file(tmp_path)
    finally:
        # Ensure cleanup
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return text
