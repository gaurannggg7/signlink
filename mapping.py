import os
import re
import pandas as pd


class ASLDictionary:
    """
    Load token->video-filename map from CSV and dynamically resolve paths.
    Lookup chain: exact match -> lemmatization -> fingerspelling fallback.
    """

    LEMMA_RULES = [
        (r'ING$', ''),      # CLEANING -> CLEAN
        (r'INGS$', ''),     # CLEANINGS -> CLEAN
        (r'ED$', ''),       # DELAYED -> DELAY
        (r'LY$', ''),       # QUICKLY -> QUICK
        (r'ER$', ''),       # FASTER -> FAST
        (r'EST$', ''),      # FASTEST -> FAST
        (r'ION$', ''),      # ATTENTION -> ATTEND (approximate)
        (r'TION$', ''),     # ATTENTION -> ATTEN (then try)
        (r'LY$', ''),       # SLOWLY -> SLOW
        (r'S$', ''),        # FRIENDS -> FRIEND
    ]

    def __init__(
        self,
        csv_path: str = "content/asl_app_data/asl_video_index_final_with_path_cleaned.csv",
        dictionary_dir: str = "content/asl_app_data/dictionary",
        fingerspelling_dir: str = "content/asl_app_data/Letters"
    ):
        self.directory = dictionary_dir
        self.fingerspelling_dir = fingerspelling_dir

        # 1. Build token -> filename map from CSV
        df = pd.read_csv(csv_path)
        self.token_to_filename = {}
        for _, row in df.iterrows():
            filepath = row['path']
            fname = os.path.basename(filepath)
            for col in ('token', 'phrase', 'word'):
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip():
                    key = str(val).upper().strip()
                    self.token_to_filename[key] = fname

        # Sort keys by word-count descending for greedy phrase matching
        self.keys = sorted(
            self.token_to_filename.keys(),
            key=lambda x: len(x.split()),
            reverse=True
        )

        # 2. RAG-style indexer: scan all subfolders for absolute paths
        self.actual_file_paths = {}
        if os.path.exists(self.directory):
            for root, dirs, files in os.walk(self.directory):
                for file in files:
                    if not file.startswith('.'):
                        self.actual_file_paths[file] = os.path.join(root, file)

        # 3. Fingerspelling index: A-Z letter paths
        self.letter_paths = {}
        if os.path.exists(self.fingerspelling_dir):
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                path = os.path.join(self.fingerspelling_dir, f"{letter}.mp4")
                if os.path.exists(path):
                    self.letter_paths[letter] = path

        print(f"✅ Dictionary loaded: {len(self.token_to_filename)} tokens")
        print(f"✅ Fingerspelling loaded: {sorted(self.letter_paths.keys())}")

    def _resolve(self, token: str) -> str | None:
        """Look up token in CSV map and resolve to absolute file path."""
        fname = self.token_to_filename.get(token)
        if fname:
            full = self.actual_file_paths.get(fname)
            if full and os.path.exists(full):
                return full
        return None

    def _lemmatize(self, token: str) -> str | None:
        """Try stripping suffixes to find a base form in vocabulary."""
        for pattern, replacement in self.LEMMA_RULES:
            candidate = re.sub(pattern, replacement, token)
            if candidate != token and len(candidate) > 2:
                if candidate in self.token_to_filename:
                    return candidate
        return None

    def _fingerspell(self, token: str) -> list[str]:
        """Break token into individual letter video paths."""
        paths = []
        for char in token.upper():
            if char in self.letter_paths:
                paths.append(self.letter_paths[char])
            elif char == ' ':
                pass
            else:
                print(f"⚠️ No fingerspelling for: '{char}'")
        return paths

    def get_paths(self, gloss_tokens: list[str]) -> list[str]:
        out_paths = []
        tokens = [t.upper() for t in gloss_tokens]
        n = len(tokens)
        i = 0

        while i < n:
            # 1. Greedy phrase match (longest first)
            match = None
            match_len = 0
            for key in self.keys:
                parts = key.split()
                L = len(parts)
                if L > 1 and i + L <= n and tokens[i:i + L] == parts:
                    match, match_len = key, L
                    break

            if match:
                full = self._resolve(match)
                if full:
                    out_paths.append(full)
                    print(f"✅ Phrase match: '{match}'")
                else:
                    print(f"⚠️ Phrase found but file missing: '{match}'")
                i += match_len
                continue

            tok = tokens[i]

            # 2. Exact single token match
            full = self._resolve(tok)
            if full:
                out_paths.append(full)
                print(f"✅ Exact match: '{tok}'")
                i += 1
                continue

            # 3. Lemmatization fallback
            lemma = self._lemmatize(tok)
            if lemma:
                full = self._resolve(lemma)
                if full:
                    out_paths.append(full)
                    print(f"✅ Lemma match: '{tok}' -> '{lemma}'")
                    i += 1
                    continue

            # 4. Fingerspelling fallback
            print(f"⚠️ No mapping for '{tok}' — fingerspelling...")
            spelled = self._fingerspell(tok)
            if spelled:
                out_paths.extend(spelled)
            else:
                print(f"⚠️ Could not fingerspell '{tok}'")

            i += 1

        return out_paths
