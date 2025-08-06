import os
import pandas as pd

class ASLDictionary:
    """
    Load token→video-filename map from CSV and resolve paths under dictionary_dir.
    Supports greedy multi-word matching; no fuzzy fallbacks.
    """
    def __init__(
        self,
        csv_path: str = "content/asl_app_data/asl_video_index_final_with_path_cleaned.csv",
        dictionary_dir: str = "content/asl_app_data/dictionary"
    ):
        self.directory = dictionary_dir
        df = pd.read_csv(csv_path)

        # build map: each non-empty value in ['token','phrase','word'] → filename
        self.map = {}
        for _, row in df.iterrows():
            filepath = row['path']
            fname = os.path.basename(filepath)
            for col in ('token', 'phrase', 'word'):
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip():
                    key = str(val).upper().strip()
                    self.map[key] = fname  # later rows override earlier if duplicate

        # sort keys by word-count descending for greedy phrase matching
        self.keys = sorted(self.map.keys(), key=lambda x: len(x.split()), reverse=True)

    def get_paths(self, gloss_tokens: list[str]) -> list[str]:
        """
        Given a list of gloss tokens (all-caps single words),
        matches the longest multi-word phrases first,
        then exact single tokens. No fuzzy matching.
        """
        out_paths = []
        tokens = [t.upper() for t in gloss_tokens]
        n = len(tokens)
        i = 0

        while i < n:
            # Attempt multi-word phrase match
            match = None
            match_len = 0
            for key in self.keys:
                parts = key.split()
                L = len(parts)
                if L > 1 and i + L <= n and tokens[i:i+L] == parts:
                    match, match_len = key, L
                    break

            if match:
                fname = self.map[match]
                full = os.path.join(self.directory, fname)
                if os.path.exists(full):
                    out_paths.append(full)
                else:
                    print(f"⚠️ File not found for phrase '{match}': {full}")
                i += match_len
                continue

            # Exact single-token match only
            tok = tokens[i]
            fname = self.map.get(tok)
            if fname:
                full = os.path.join(self.directory, fname)
                if os.path.exists(full):
                    out_paths.append(full)
                else:
                    print(f"⚠️ File not found for token '{tok}': {full}")
            else:
                print(f"⚠️ No mapping for token: {tok}")

            i += 1

        return out_paths
