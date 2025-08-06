import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from utils import get_best_device

class Gemma3nEdge:
    """
    Gemma3nEdge wrapper: loads model with device fallback and auto-download support.
    """
    def __init__(self, model_dir: str = "models/gemma3n_E2B"):
        # Determine best device (cuda > mps > cpu)
        self.device = get_best_device()
        # Map special alias names to Hugging Face repository IDs
        hf_alias = {
            "gemma3n_E2B": "models/gemma3n_E2B",
            "gemma-3n-E4B": "google/gemma-3n-E4B"
        }
        repo = hf_alias.get(model_dir, model_dir)
        # Attempt to load tokenizer and model locally, else download from HF
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.float16 if self.device.type == "cuda" else None,
                load_in_8bit=self.device.type == "cuda",
                local_files_only=True
            )
        except Exception:
            # Download and cache from Hugging Face
            repo_path = snapshot_download(repo)
            self.tokenizer = AutoTokenizer.from_pretrained(repo_path)
            self.model = AutoModelForCausalLM.from_pretrained(repo_path)
        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def generate_gloss(self, transcript: str, max_tokens: int = 100) -> str:
        # Build prompt for ASL gloss
        prompt = f"English: {transcript}\nASL Gloss:"
        # Tokenize and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        # Generate response, with fallback to CPU if OOM
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        except RuntimeError:
            # Fallback: run on CPU
            inputs_cpu = inputs.to("cpu")
            outputs = self.model.to("cpu").generate(
                **inputs_cpu,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        # Strip prompt from output
        gen_ids = outputs[0][input_ids.shape[-1]:]
        # Decode gloss tokens
        gloss = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return gloss
