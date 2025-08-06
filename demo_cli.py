import argparse
from whisper_transcribe import transcribe_file, transcribe_from_mic
from gloss_builder import build_gloss_sequence
from mapping import ASLDictionary
from renderer import stitch_clips

def main():
    parser = argparse.ArgumentParser("Offline ASL Translator CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--file", type=str)
    group.add_argument("--mic", action="store_true")
    parser.add_argument("--duration", type=int, default=5)
    args = parser.parse_args()

    if args.text:
        transcript = args.text
    elif args.file:
        transcript = transcribe_file(args.file)
    else:
        transcript = transcribe_from_mic(args.duration)
    print("📝 Transcript:", transcript)

    gloss = build_gloss_sequence(transcript)
    print("🔤 Gloss tokens:", gloss)

    dictionary = ASLDictionary()
    paths = dictionary.get_paths(gloss)
    print("🎞️ Clip paths:", paths)

    if paths:
        out = stitch_clips(paths)
        print(f"✅ Video at {out}")
    else:
        print("❌ No clips found.")

if __name__ == "__main__":
    main()
