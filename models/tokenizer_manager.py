# core/tokenizer_manager.py
"""
TokenizerManager - Manages tokenizers for local & API models.
- Local models: Hugging Face AutoTokenizer (cached)
- API models (OpenAI/Claude/Gemini/etc.):
    * Try tiktoken (OpenAI-like) for realistic counts
    * Fallback to whitespace tokenizer
Never returns None; always provides an object with .encode(text) -> List[int]
"""

from __future__ import annotations
import logging

from utils.logging_utils import log_and_time

logger = logging.getLogger(__name__)
logger.debug("tokenizer_manager.py is alive")

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # allow environments without HF

# Optional dependency for OpenAI tokenization
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def _looks_like_api_model(name: str) -> bool:
    """Heuristic if ModelManager is absent or undecided."""
    n = (name or "").lower()
    return any(s in n for s in ["gpt-", "gpt4", "gpt-4", "gpt-3.5", "openai", "claude", "gemini", "cohere", "groq"])


def _choose_tiktoken_encoding(model_name: str) -> str:
    """
    Pick a tiktoken encoding. o200k_base works for new OpenAI "o" series;
    cl100k_base is compatible with GPT-3.5/4 families.
    """
    n = (model_name or "").lower()
    if any(k in n for k in ["o3", "o4", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]):
        # If tiktoken lacks o200k_base in your env, we'll fall back to cl100k_base below.
        return "o200k_base"
    return "cl100k_base"


class _WhitespaceTokenizer:
    def encode(self, text: str):
        return (text or "").split()


class _TiktokenWrapper:
    def __init__(self, enc):
        self._enc = enc
    def encode(self, text: str):
        return self._enc.encode(text or "")


class TokenizerManager:
    """Manages tokenizers - always used as an instance, never as a class"""

    def __init__(self, model_manager):
        if model_manager is None:
            raise ValueError("model_manager is required for TokenizerManager")

        self._cache = {}
        self.model_manager = model_manager

    def _is_api_model(self, model_name: str) -> bool:
        if self.model_manager and hasattr(self.model_manager, "is_api_model"):
            try:
                return bool(self.model_manager.is_api_model(model_name))
            except Exception:
                pass
        return _looks_like_api_model(model_name)

    def _load_hf_tokenizer(self, model_name: str):
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed; cannot load local tokenizer.")
        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"[TokenizerManager] HF load failed for '{model_name}': {e}. Falling back to gpt2.")
            return AutoTokenizer.from_pretrained("gpt2")

    def _load_api_tokenizer(self, model_name: str):
        # Prefer tiktoken if present
        if tiktoken is not None:
            enc_name = _choose_tiktoken_encoding(model_name)
            try:
                enc = tiktoken.get_encoding(enc_name)
            except Exception:
                # fallback to widely available base
                enc = tiktoken.get_encoding("cl100k_base")
            return _TiktokenWrapper(enc)
        # Final fallback: whitespace
        return _WhitespaceTokenizer()

    @log_and_time("get tokenizer")
    def get_tokenizer(self, model_name: str):
        if not model_name:
            raise ValueError("[TokenizerManager] No model_name provided.")

        key = model_name.lower().strip()

        if key in self._cache:
            return self._cache[key]

        # Prefer API tokenizer for API models (tiktoken when available)
        if self._is_api_model(model_name):
            tok = self._load_api_tokenizer(model_name)
            self._cache[key] = tok
            return tok

        # Special-case common OpenAI/Anthropic names if someone passes them as local
        if key in {"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4.5-turbo", "claude-opus"}:
            logger.debug(f"[TokenizerManager] Remapping '{model_name}' -> 'gpt2' for HF tokenizer (local fallback).")
            tok = self._load_hf_tokenizer("gpt2")
            self._cache[key] = tok
            return tok

        # Local/HF model
        tok = self._load_hf_tokenizer(model_name)
        self._cache[key] = tok
        return tok

    def count_tokens(self, text: str, model_name: str = None) -> int:
        """Single method for token counting"""
        if not text:
            return 0

        model_name = model_name or "gpt-4-turbo"  # default

        try:
            tokenizer = self.get_tokenizer(model_name)
            if hasattr(tokenizer, 'encode'):
                return len(tokenizer.encode(text))
            else:
                # Fallback for whitespace tokenizer
                return len(tokenizer.encode(text))
        except:
            # Ultimate fallback
            return max(1, len(text) // 4)
# ✅ Manual test
if __name__ == "__main__":
    try:
        from models.model_manager import ModelManager  # optional
    except Exception:
        ModelManager = None

    sample = "Hello! This is a test message for token counting. How many tokens is it?"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    mm = ModelManager() if ModelManager else None
    tm = TokenizerManager(model_manager=mm)

    tok = tm.get_tokenizer(model_name)
    ids = tok.encode(sample)
    print(f"[{model_name}] token count:", len(ids))

    # Simulate API model
    for api_model in ["gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet-20240620"]:
        tok = tm.get_tokenizer(api_model)
        print(f"[{api_model}] token count:", len(tok.encode(sample)))
