import streamlit as st
from whisper_transcribe import transcribe_file, transcribe_from_mic
from gloss_builder import build_gloss_sequence
from mapping import ASLDictionary
from renderer import stitch_clips
import re

# Page configuration
st.set_page_config("Offline ASL Translator", layout="centered")
st.title("Offline ASL Translator 🎥👐")

# Dropdown to switch Gemma models
model_option = st.selectbox(
     "Select Gemma Model:",
     ["gemma3n_E2B", "gemma3n-E4B"]
)

# Input mode selection
mode = st.radio("Input mode:", ["Text", "Audio File", "Record Mic"])
transcript = ""

# Capture transcript
if mode == "Text":
    txt = st.text_input("Enter text:")
    if st.button("Translate") and txt:
        transcript = txt

elif mode == "Audio File":
    upload = st.file_uploader("Upload WAV/MP3:", type=["wav","mp3"])
    if upload and st.button("Translate"):
        tmp = "temp_input.wav"
        with open(tmp, "wb") as f:
            f.write(upload.getbuffer())
        transcript = transcribe_file(tmp)

else:
    dur = st.slider("Duration (sec)", 1, 30, 5)
    if st.button("Translate"):
        transcript = transcribe_from_mic(dur)

# Process and display results
if transcript:
    st.subheader("Transcript")
    st.write(transcript)

    # Build gloss tokens
    if mode == "Text":
        # Split alphabetic words and uppercase
        raw = re.findall(r"[A-Za-z]+", transcript)
        gloss = [w.upper() for w in raw if w.upper() not in {"A", "AN", "THE"}]
    else:
        # Use Gemma model specified by dropdown
        gloss = build_gloss_sequence(transcript, model_option)

    st.subheader("Gloss Tokens")
    st.write(gloss)

    # Lookup and render video clips
    dictionary = ASLDictionary()
    paths = dictionary.get_paths(gloss)
    if paths:
        out_video = stitch_clips(paths)
        st.video(out_video)
    else:
        st.error("No ASL clips found for the given input.")
