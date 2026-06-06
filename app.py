import streamlit as st
from whisper_transcribe import transcribe_file
from gloss_builder import build_gloss_sequence
from mapping import ASLDictionary
from renderer import stitch_clips
import re
import os
from huggingface_hub import snapshot_download

st.set_page_config(
    page_title="SignLink | ASL Translator",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>
/* ── RESET & BASE ── */
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
body, .stApp {
    background-color: #0F0F1A !important;
    color: #E2E8F0 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
/* ── HERO BANNER ── */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3em 0.9em;
    border-radius: 999px;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    border: 1px solid rgba(255,255,255,0.2);
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.75rem;
    line-height: 1.15;
}
.hero p {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.85);
    max-width: 560px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}
.hero-links {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.hero-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.55em 1.4em;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    text-decoration: none;
    transition: all 0.2s;
    cursor: pointer;
}
.hero-link.primary {
    background: white;
    color: #4F46E5;
}
.hero-link.secondary {
    background: transparent;
    color: white;
    border: 2px solid rgba(255,255,255,0.6);
}
.hero-link:hover { opacity: 0.88; transform: translateY(-1px); }
/* ── STATS BAR ── */
.stats-bar {
    background: #13131F;
    border-bottom: 1px solid #1E1E35;
    display: flex;
    justify-content: center;
    gap: 0;
    padding: 0;
}
.stat-item {
    padding: 1.1rem 2.5rem;
    text-align: center;
    border-right: 1px solid #1E1E35;
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-size: 1.6rem;
    font-weight: 800;
    color: #818CF8;
    display: block;
}
.stat-label {
    font-size: 0.7rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
/* ── MAIN CONTENT AREA ── */
.main-content {
    max-width: 900px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem;
}
/* ── SECTION HEADERS ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #818CF8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 1.5rem;
}
/* ── DEMO SENTENCE BUTTONS ── */
.demo-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}
.demo-pill {
    background: #1A1A2E;
    border: 1px solid #2D2D4E;
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
    text-align: center;
    cursor: pointer;
    font-size: 0.82rem;
    color: #A5B4FC;
    font-weight: 500;
    transition: all 0.2s;
    line-height: 1.3;
}
.demo-pill:hover {
    background: #252545;
    border-color: #4F46E5;
    color: white;
    transform: translateY(-1px);
}
/* ── INPUT AREA ── */
.input-card {
    background: #13131F;
    border: 1px solid #1E1E35;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
/* Override Streamlit textarea */
.stTextArea textarea {
    background-color: #0F0F1A !important;
    color: #E2E8F0 !important;
    border: 1px solid #2D2D4E !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.25) !important;
}
.stTextArea label { color: #94A3B8 !important; font-size: 0.85rem !important; }
/* ── TRANSLATE BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.4) !important;
}
/* ── PIPELINE STATUS ── */
.stStatus {
    background: #13131F !important;
    border: 1px solid #1E1E35 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
/* ── VIDEO PLAYER ── */
.stVideo {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #1E1E35 !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}
video {
    border-radius: 10px !important;
}
/* ── RESULTS CARD ── */
.result-card {
    background: #13131F;
    border: 1px solid #1E1E35;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.result-transcript {
    font-size: 1.05rem;
    color: #CBD5E1;
    font-style: italic;
    padding: 0.75rem 1rem;
    background: #0F0F1A;
    border-left: 3px solid #4F46E5;
    border-radius: 4px;
    margin-bottom: 1.25rem;
}
/* ── GLOSS TOKENS ── */
.gloss-tag {
    display: inline-block;
    padding: 0.4em 0.85em;
    margin: 0.25em;
    background: #1E1E35;
    color: #818CF8;
    border: 1px solid #2D2D4E;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    font-size: 0.88em;
    font-weight: 700;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.gloss-tag:hover {
    background: #252550;
    border-color: #4F46E5;
    color: white;
}
/* ── PIPELINE BADGE ── */
.pipeline-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    background: #0F0F1A;
    border: 1px solid #1E1E35;
    border-radius: 999px;
    padding: 0.3em 0.8em;
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 1rem;
}
.pipeline-badge span { color: #4ADE80; }
/* ── FILE UPLOADER ── */
.stFileUploader {
    background: #13131F !important;
    border: 1px dashed #2D2D4E !important;
    border-radius: 10px !important;
}
/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13131F !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #1E1E35 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #4F46E5 !important;
    color: white !important;
}
/* ── WARNING / ERROR ── */
.stAlert {
    background: #1A1225 !important;
    border: 1px solid #2D2D4E !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
/* ── DIVIDER ── */
hr {
    border-color: #1E1E35 !important;
    margin: 1.5rem 0 !important;
}
</style>
<!-- HERO -->
<div class="hero">
    <div class="hero-badge">🤟 Open Source · CC0 · Hugging Face</div>
    <h1>SignLink</h1>
    <p>Voice to American Sign Language — powered by Whisper ASR, Gemma LLM, and StudioGalt motion capture.</p>
    <div class="hero-links">
        <a class="hero-link primary" href="https://github.com/gaurannggg7/signlink" target="_blank">
            ⭐ GitHub
        </a>
        <a class="hero-link secondary" href="https://huggingface.co/datasets/gaurannggg7/asl-dictionary" target="_blank">
            🗂️ Dataset
        </a>
        <a class="hero-link secondary" href="https://huggingface.co/spaces/gaurannggg7/asl-translator-website" target="_blank">
            🌐 About
        </a>
    </div>
</div>
<!-- STATS -->
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-num">1,566</span>
        <div class="stat-label">Signs + Letters</div>
    </div>
    <div class="stat-item">
        <span class="stat-num">A–Z</span>
        <div class="stat-label">Fingerspelling</div>
    </div>
    <div class="stat-item">
        <span class="stat-num">3-Stage</span>
        <div class="stat-label">OOV Fallback</div>
    </div>
    <div class="stat-item">
        <span class="stat-num">CC0</span>
        <div class="stat-label">Open Assets</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── DATASET INIT ──
LOCAL_DATA_PATH = "content/asl_app_data"
CSV_TARGET = "asl_video_index_final_with_path_cleaned.csv"

if not os.path.exists(os.path.join(LOCAL_DATA_PATH, CSV_TARGET)):
    with st.status("⚙️ Initializing SignLink Engine...", expanded=True) as status:
        st.write("📥 Syncing ASL motion capture dataset...")
        try:
            snapshot_download(
                repo_id="gaurannggg7/asl-dictionary",
                repo_type="dataset",
                local_dir=LOCAL_DATA_PATH,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN")
            )
            status.update(label="✅ Engine Ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Initialization Failed", state="error")
            st.error(f"Dataset sync failed: {e}")

# ── MAIN CONTENT ──
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Demo sentences
DEMO_SENTENCES = [
    "Please clean",
    "Come here",
    "Hello",
    "I appreciate you",
    "Announce the answer",
    "Arrive and attend",
    "Ask anyone",
    "Around the apartment",
]

st.markdown('<div class="section-label">Quick Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Try a sentence</div>', unsafe_allow_html=True)

# Demo pills as buttons
cols = st.columns(4)
selected_demo = None
for idx, sentence in enumerate(DEMO_SENTENCES):
    with cols[idx % 4]:
        if st.button(sentence, key=f"demo_{idx}"):
            selected_demo = sentence

st.markdown("<hr>", unsafe_allow_html=True)

# Input tabs
st.markdown('<div class="section-label">Translate</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📝 Text Input", "🎙️ Audio Upload"])
transcript = ""
mode = ""

with tab1:
    default_text = selected_demo if selected_demo else ""
    txt = st.text_area(
        "Enter English text",
        value=default_text,
        placeholder="Type anything — e.g. 'Please come here'",
        height=100,
        label_visibility="collapsed"
    )
    if st.button("🤟 Translate to ASL", key="text_btn") and txt:
        transcript = txt
        mode = "Text"

with tab2:
    upload = st.file_uploader(
        "Upload audio (WAV/MP3)",
        type=["wav", "mp3"],
        label_visibility="collapsed"
    )
    if upload and st.button("🎙️ Transcribe + Translate", key="audio_btn"):
        tmp = "temp_input.wav"
        with open(tmp, "wb") as f:
            f.write(upload.getbuffer())
        with st.spinner("Transcribing with Whisper..."):
            try:
                transcript = transcribe_file(tmp)
                mode = "Audio"
            except Exception as e:
                st.error(f"Transcription failed: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

if selected_demo and not transcript:
    transcript = selected_demo
    mode = "Text"

# ── PIPELINE ──
if transcript:
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.status("⚡ Running SignLink Pipeline...", expanded=True) as status:
        st.write("🧠 Generating ASL gloss tokens...")
        gloss = build_gloss_sequence(transcript, "gemma3n_E2B")

        st.write("🗂️ Retrieving motion capture assets...")
        dictionary = ASLDictionary()
        paths = dictionary.get_paths(gloss)

        status.update(label="✅ Translation Complete", state="complete", expanded=False)

    if paths:
        try:
            out_video = stitch_clips(paths)
            st.video(out_video)
        except Exception as e:
            st.error(f"Video rendering error: {e}")

        # Results card
        tags_html = "".join([f"<span class='gloss-tag'>{t}</span>" for t in gloss])
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Original Input</div>
            <div class="result-transcript">"{transcript}"</div>
            <div class="result-label">ASL Gloss Tokens</div>
            <div>{tags_html}</div>
            <div class="pipeline-badge">
                <span>✓</span> {len(paths)} clips assembled &nbsp;·&nbsp;
                Gemma → Lemmatization → Fingerspelling fallback
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No ASL clips found for this input.")

st.markdown('</div>', unsafe_allow_html=True)
