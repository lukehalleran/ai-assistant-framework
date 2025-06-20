"""
TokenizerManager - Manages tokenizers for local models.

Skips loading tokenizer if the model is an API-based model (e.g. OpenAI).
Caches tokenizers to avoid reloading.
"""


from transformers import AutoTokenizer
from models import model_manager  # so it can check if a model is OpenAI

class TokenizerManager:
    def __init__(self):
        self.tokenizers = {}

    def get_tokenizer(self, model_name):
        if model_name is None:
            return None

        # 🔥 NEW: Skip HuggingFace loading if OpenAI API model
        if model_manager.is_api_model(model_name):
            # print(f"[DEBUG] Skipping local tokenizer load for OpenAI model '{model_name}'")
            return None

        if model_name not in self.tokenizers:
            # print(f"[DEBUG] Loading tokenizer for {model_name}")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizers[model_name]
