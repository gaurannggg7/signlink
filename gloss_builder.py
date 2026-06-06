import os
import requests
import pandas as pd

def load_known_tokens(csv_path: str = "content/asl_app_data/asl_video_index_final_with_path_cleaned.csv") -> list[str]:
    """Load vocabulary from CSV — single words only for prompt constraint."""
    try:
        df = pd.read_csv(csv_path)
        tokens = df['token'].dropna().str.upper().str.strip().unique().tolist()
        # Single words only — more useful for Gemma constraint
        single = sorted([t for t in tokens if ' ' not in t and len(t) > 1])
        return single
    except Exception as e:
        print(f"⚠️ Could not load tokens from CSV: {e}")
        # Fallback: words we know we have
        return [
            "HELLO", "PLEASE", "COME", "CLEAN", "YOU", "AND", "ANY",
            "ASK", "ANSWER", "ANOTHER", "ANYONE", "ANYTHING", "ANYWHERE",
            "APOLOGY", "APPRECIATE", "APPROACH", "ARMY", "AROUND",
            "ARRIVE", "ART", "ASSIST", "ATTEND", "ATTENTION", "ATTITUDE",
            "ANGER", "ANIMAL", "ANNOUNCE", "ANNOY", "ANNOYED", "ANXIOUS",
            "APART", "APARTMENT", "APPLE", "ARROGANT", "ASSOCIATE",
            "ASSUME", "ATTRACT", "AUNT"
        ]


def build_gloss_sequence(
    transcript: str,
    model_name: str,
    max_tokens: int = 60,
    csv_path: str = "content/asl_app_data/asl_video_index_final_with_path_cleaned.csv"
) -> list[str]:
    """
    Convert English transcript to ASL gloss tokens using Gemma.
    Uses constrained prompt to maximize hits against known vocabulary.
    Falls back to simple uppercase split if API fails.
    """
    known_tokens = load_known_tokens(csv_path)
    
    # Take first 300 tokens for prompt — keep it under context limit
    vocab_sample = ', '.join(known_tokens[:300])

    prompt = f"""You are an expert ASL (American Sign Language) gloss translator.
STRICT RULES:
1. DROP all articles: a, an, the
2. DROP all linking verbs: is, are, was, were, am, be, been, being  
3. DROP all prepositions: to, of, for, in, on, at, by, with, from, into
4. USE present tense root form only: CLEAN not CLEANING, DELAY not DELAYED, ARRIVE not ARRIVED
5. ONLY output words from this vocabulary list: {vocab_sample}
6. If a word has no match, find the CLOSEST SYNONYM from the vocabulary
7. Output ONLY uppercase tokens separated by spaces — no punctuation, no explanation
EXAMPLES:
English: "Hello, how are you doing today?"
ASL Gloss: HELLO YOU
English: "Please clean the dishes"  
ASL Gloss: PLEASE CLEAN
English: "I am anxious about the appointment"
ASL Gloss: ANXIOUS APPOINT
English: "Can you come here and help me?"
ASL Gloss: COME ASSIST
English: "{transcript}"
ASL Gloss:"""

    api_url = f"https://api-inference.huggingface.co/models/google/gemma-2b-it"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": 0.1,  # Low temp = more deterministic/constrained
            "do_sample": False
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        output = response.json()

        raw = ""
        if isinstance(output, list) and len(output) > 0:
            raw = output[0].get("generated_text", "").strip()
        elif isinstance(output, dict):
            raw = output.get("generated_text", "").strip()

        if not raw:
            print(f"⚠️ Empty Gemma response, falling back. Raw: {output}")
            return _simple_gloss(transcript)

        # Clean output — take only first line, strip non-alpha
        first_line = raw.split('\n')[0].strip()
        tokens = [
            t.upper() for t in first_line.split()
            if t.isalpha() and t.upper() not in {"ASL", "GLOSS", "ENGLISH"}
        ]

        if not tokens:
            return _simple_gloss(transcript)

        print(f"✅ Gemma gloss: {tokens}")
        return tokens

    except Exception as e:
        print(f"⚠️ Gemma API failed: {e}. Falling back to simple gloss.")
        return _simple_gloss(transcript)


def _simple_gloss(transcript: str) -> list[str]:
    """
    Fallback: strip stopwords and return uppercase tokens.
    Used when Gemma API is unavailable.
    """
    STOPWORDS = {
        "A", "AN", "THE", "IS", "ARE", "WAS", "WERE", "AM", "BE", "BEEN",
        "BEING", "TO", "OF", "FOR", "IN", "ON", "AT", "BY", "WITH", "FROM",
        "INTO", "AND", "BUT", "OR", "SO", "IF", "AS", "IT", "ITS", "THIS",
        "THAT", "THESE", "THOSE", "MY", "YOUR", "HIS", "HER", "OUR", "THEIR"
    }
    import re
    words = re.findall(r"[A-Za-z]+", transcript)
    return [w.upper() for w in words if w.upper() not in STOPWORDS]
