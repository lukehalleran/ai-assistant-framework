"""
# models/model_manager.py

Module Contract
- Purpose: Single interface over local HF models and API chat models (OpenRouter/OpenAI). Handles registration, switching, sync/async generation, and embedding model access.
- Inputs:
  - generate(prompt, model_name=?, system_prompt=?, …)
  - generate_async(prompt, raw=?, system_prompt=?)
  - generate_once(prompt, model_name=?, system_prompt=?, max_tokens=?)
- Outputs:
  - Text responses (sync) or async stream of ChatCompletion chunks; stub output when API unavailable.
- Key methods:
  - load_model(), load_openai_model(), switch_model(), get_active_model_name(), get_embedder()
  - truncate_prompt(): ensures local prompts fit context window
- Dependencies:
  - transformers, sentence-transformers, httpx, environment OPENAI_API_KEY
- Side effects:
  - Maintains HTTP clients; exposes aclose/close to release resources.
"""
# Import dependencies and config defaults
from utils.logging_utils import log_and_time, get_logger
# Use the root logger or create a child logger that will inherit handlers
logger = get_logger("main")
import os
from config.app_config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time
import torch
# Optional OpenAI dependency (tests may run without the package installed)
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:  # pragma: no cover - triggered in trimmed test envs
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore
import httpx
import os
import asyncio
import json

# Global embedding model cache to prevent re-loading SentenceTransformer
_global_embed_model = None
_embed_model_lock = asyncio.Lock()

# Global cross-encoder model cache to prevent re-loading CrossEncoder models
_global_cross_encoders = {}
_cross_encoder_lock = asyncio.Lock()


# Set OpenAI API key for API calls
class ModelManager:
    """Manager class for handling both local and API-based language models."""

    def __init__(self, api_key: str = None):
        # Active model name (local or API)
        self.active_model_name = None
        # Dictionary of loaded local models
        self.models = {}
        self.allow_fallback = False  # Disable fallback to unknown API models
        # Dictionary of loaded tokenizers for local models
        self.tokenizers = {}
        # Dictionary mapping registered API models
        self.api_models = {}
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.base_url = "https://openrouter.ai/api/v1"

        # Initialize OpenAI clients only when the package is available AND an API key is provided.
        if OpenAI is not None and AsyncOpenAI is not None and self.api_key:
            sync_http_client = httpx.Client(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=10),
                headers={"Connection": "keep-alive"},
            )

            async_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=10),
                headers={"Connection": "keep-alive"},
            )

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=sync_http_client,
            )

            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=async_http_client,
            )
        else:
            # No API key or OpenAI package unavailable — operate in stub/offline mode.
            self.client = None  # type: ignore[assignment]
            self.async_client = None  # type: ignore[assignment]

        self.default_model = "gpt-4-turbo"
        # Embedding model used across memory/gating paths. Use global cache to prevent re-loading
        self.embed_model = self._get_cached_embedder()
        # Runtime-overridable defaults (mutable via GUI and persisted to config)
        self.default_temperature = DEFAULT_TEMPERATURE
        self.default_max_tokens = DEFAULT_MAX_TOKENS
        # Common API model aliases
        self.api_models["claude-opus"] = "anthropic/claude-3-opus"
        self.api_models["sonnet-4.5"] = "anthropic/claude-sonnet-4.5"
        # OpenAI models via OpenRouter
        self.api_models["gpt-4o-mini"] = "openai/gpt-4o-mini"
        self.api_models["gpt-4o"] = "openai/gpt-4o"
        self.api_models["gpt-4.1"] = "openai/gpt-4.1"
        # Add GPT‑5 support (via OpenRouter naming)
        self.api_models["gpt-5"] = "openai/gpt-5"
        # GLM model
        self.api_models["glm-4.6"] = "z-ai/glm-4.6"
        # DeepSeek models
        self.api_models["deepseek-v3.1"] = "deepseek/deepseek-chat-v3.1"
        self.api_models["deepseek-r1"] = "deepseek/deepseek-r1-0528"

    def _stub_response(self, prompt: str) -> str:
        snippet = (prompt or "").strip().splitlines()[0] if prompt else ""
        if snippet and len(snippet) > 120:
            snippet = snippet[:117] + "..."
        return f"[OpenAI unavailable] {snippet or 'stub response'}"

    def _stub_stream(self, prompt: str):
        async def _gen():
            yield self._stub_response(prompt)

        return _gen()
    def list_provider_models(self, vendor_prefix=None):
        """
        Returns a list of model ids exposed by the provider (OpenRouter).
        Optionally filter by a vendor prefix, e.g. 'anthropic/claude-3-opus'.
        """
        if self.client is None:
            return []
        r = self.client._client.get(f"{self.base_url}/models", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })
        r.raise_for_status()
        data = r.json()
        models = [m["id"] for m in data.get("data", []) if "id" in m]
        if vendor_prefix:
            models = [m for m in models if m.startswith(vendor_prefix)]
        return models

    def get_context_limit(self):
        """Get the maximum context window based on active model."""
        if self.is_api_model(self.get_active_model_name()):
            # For API models (OpenAI), assume known context window
            # logger.debug(f" Using OpenAI model, assuming 128000 token context window.")
            return 128000  # Default GPT-4 Turbo context
        model = self.get_model()
        if model:
            return model.config.max_position_embeddings
        else:
            raise ValueError("[ERROR] No model loaded. Cannot determine context limit.")

    @log_and_time("Get Embedder")
    def get_embedder(self):
        return self.embed_model

    @staticmethod
    def _get_cached_embedder():
        """Get or create a cached SentenceTransformer model to avoid re-loading"""
        global _global_embed_model

        if _global_embed_model is not None:
            return _global_embed_model

        try:
            logger.debug("Loading SentenceTransformer model (first time only)...")
            _global_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("SentenceTransformer model cached successfully")
            return _global_embed_model
        except Exception:
            logger.warning("Failed to load SentenceTransformer, using stub embedder")
            class _StubEmbedder:
                def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
                    import numpy as np
                    n = len(texts or [])
                    return np.zeros((n, 384), dtype=np.float32)

            # Cache the stub embedder too to avoid re-creating it
            _global_embed_model = _StubEmbedder()
            return _global_embedder

    @log_and_time("Get Cross-Encoder")
    def get_cross_encoder(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Get or create a cached CrossEncoder model to avoid re-loading.

        Args:
            model_name: Name of the cross-encoder model to load

        Returns:
            CrossEncoder: Cached or newly created cross-encoder instance
        """
        global _global_cross_encoders

        # Check cache first
        if model_name in _global_cross_encoders:
            logger.debug(f"Using cached cross-encoder: {model_name}")
            return _global_cross_encoders[model_name]

        try:
            logger.debug(f"Loading CrossEncoder model (first time only): {model_name}")
            from sentence_transformers import CrossEncoder

            cross_encoder = CrossEncoder(model_name)

            # Cache the model for future use
            _global_cross_encoders[model_name] = cross_encoder
            logger.debug(f"CrossEncoder model cached successfully: {model_name}")

            return cross_encoder

        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder '{model_name}': {e}")
            # Return a stub cross-encoder that always returns neutral scores
            class _StubCrossEncoder:
                def predict(self, pairs, batch_size=32, show_progress_bar=False):
                    import numpy as np
                    n = len(pairs or [])
                    return np.ones(n, dtype=np.float32) * 0.5  # Neutral score

            # Cache the stub cross-encoder too to avoid re-creating it
            stub_encoder = _StubCrossEncoder()
            _global_cross_encoders[model_name] = stub_encoder
            return stub_encoder

    @log_and_time("Load Model")
    def load_model(self, model_name, model_path):
        """Load a local Huggingface model and tokenizer."""
        try:
            # Determine if using local files only
            local_files_only = model_path.startswith("./")
            # logger.debug( Loading local model '{model_name}' from '{model_path}'")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                device_map="auto",
                trust_remote_code=True
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True,
                use_fast=True
            )

            # Ensure tokenizer has a pad token set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Set tokenizer max length to model context window
            tokenizer.model_max_length = model.config.max_position_embeddings

            # Store model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            # logger.debug( Successfully loaded model: {model.__class__.__name__}")
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {str(e)}")
    @log_and_time("Load OpenAI")
    def load_openai_model(self, model_name, api_model_name):
        """Register an OpenAI API model (no local loading)."""
        # logger.debug( Registering OpenAI model '{model_name}'")
        self.api_models[model_name] = api_model_name
    def close(self):
        """Gracefully close the HTTP client to avoid socket leak."""
        if hasattr(self.client, "_client"):
            self.client._client.close()
    async def aclose(self):
        if hasattr(self.async_client, "_client"):
            await self.async_client._client.aclose()
    def is_api_model(self, model_name):
        """Check if a given model is an API-based model."""
        return model_name in self.api_models

    @log_and_time("Generate with openAI")
    def generate_with_openai(self, prompt, model_name, system_prompt=None, max_tokens=None, temperature=None, top_p=None):
        """Generate text using OpenAI API, with global defaults fallback."""
        # Apply global defaults if not provided (allow runtime override)
        max_tokens = (self.default_max_tokens if max_tokens is None else max_tokens)
        temperature = (self.default_temperature if temperature is None else temperature)
        top_p = DEFAULT_TOP_P if top_p is None else top_p

        if self.client is None:
            return self._stub_response(prompt)

        try:
            # Stop sequences to prevent hallucinating user responses (less restrictive)
            stop_sequences = [
                "\n\nUser:",
                "\n\nUSER:",
                "\n\nHuman:",
            ]

            # logger.debug(  Calling OpenAI API: {model_name}")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[OpenAI API Error] {str(e)}"

    def switch_model(self, model_name):
        """Switch active model (local or API)."""
        self.active_model_name = model_name

    def get_model(self):
        """Return active local model instance (if any)."""
        return self.models.get(self.active_model_name)

    def get_tokenizer(self):
        """Return active tokenizer instance (if any)."""
        return self.tokenizers.get(self.active_model_name)

    def get_active_model_name(self):
        """Return the name of the currently active model."""
        return self.active_model_name

    @staticmethod
    def truncate_prompt(prompt, tokenizer, max_input_tokens, preserve_prefix="You are Daemon"):
        """Ensure prompt fits within model's input size (optional prefix preservation)."""
        if preserve_prefix in prompt:
            prefix_index = prompt.index(preserve_prefix)
            prefix = prompt[:prefix_index + len(preserve_prefix)]
            rest = prompt[prefix_index + len(preserve_prefix):]
        else:
            prefix = ""
            rest = prompt

        prefix_tokens = tokenizer.encode(prefix)
        rest_tokens = tokenizer.encode(rest)

        # No truncation needed
        if len(prefix_tokens) + len(rest_tokens) <= max_input_tokens:
            return prompt

        # logger.debug( Truncating prompt: {len(prefix_tokens) + len(rest_tokens)} → {max_input_tokens} tokens")

        # If prefix alone is too long, truncate from the end of entire prompt
        allowed_rest_tokens = max_input_tokens - len(prefix_tokens)
        if allowed_rest_tokens <= 0:
            print("[WARN] Prefix alone exceeds max input size.")
            return tokenizer.decode((prefix_tokens + rest_tokens)[-max_input_tokens:])

        # Truncate rest of prompt and return combined prompt
        truncated_rest_tokens = rest_tokens[-allowed_rest_tokens:]
        return tokenizer.decode(prefix_tokens + truncated_rest_tokens)



    @log_and_time("ModelManager Generate Call")
    def generate(self, prompt, model_name="gpt-4-turbo", max_tokens=None, temperature=None, top_p=None, top_k=None, no_repeat_ngram_size=None, pad_token_id=None, system_prompt=None):
        """Main generate function for both local and OpenAI models."""

        # IMPORTANT: Use the provided model_name if given, otherwise use active model
        logger.debug(f"[generate] Model received: {model_name}")
        logger.debug(f"[generate] Known local models: {list(self.models.keys())}")
        logger.debug(f"[generate] Known API models: {list(self.api_models.keys())}")

        target_model = model_name or self.active_model_name

        if not target_model:
            raise ValueError("No model specified. Pass model_name or use switch_model() first.")

        # Check if this is a local model FIRST
        if target_model in self.models:  # This is a local model
            logger.debug(f"Using local model: {target_model}")

            # Get the specific local model and tokenizer
            model = self.models[target_model]
            tokenizer = self.tokenizers[target_model]

            # Use defaults where needed
            max_tokens = 64 if max_tokens is None else max_tokens
            temperature = self.default_temperature if temperature is None else temperature
            top_p = DEFAULT_TOP_P if top_p is None else top_p
            top_k = DEFAULT_TOP_K if top_k is None else top_k

            # Check input length and truncate if needed
            tokens = tokenizer.encode(prompt)
            max_len = model.config.max_position_embeddings

            if len(tokens) > max_len:
                tokens = tokens[-max_len:]
                prompt = tokenizer.decode(tokens)

            # Prepare safe prompt for max input size
            context_limit = model.config.max_position_embeddings
            max_input_tokens = context_limit - max_tokens
            safe_prompt = self.truncate_prompt(prompt, tokenizer, max_input_tokens)

            # Tokenize and move inputs to model device
            inputs = tokenizer(safe_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate output with Huggingface model
            with torch.no_grad():
                start = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    # Enable sampling so temperature has an effect on local models
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=1,
                    pad_token_id=pad_token_id or tokenizer.pad_token_id
            )
                end = time.time()
                logger.debug(f"Local generation finished in {end - start:.2f} seconds")

            # Decode and return output text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            return generated_text[len(safe_prompt):].strip()

        # API model generation path
        elif target_model in self.api_models:
            logger.debug(f"Using OpenAI model alias: {self.api_models[target_model]}")
            return self.generate_with_openai(
                prompt,
                self.api_models[target_model],
                system_prompt=system_prompt,
                max_tokens=max_tokens or 500,
                temperature=self.default_temperature if temperature is None else temperature,
                top_p=top_p or 1.0
            )

        else:
            if not self.allow_fallback:
                raise ValueError(f"[ModelManager] Model '{target_model}' is not recognized as a local or registered API model. Fallback is disabled.")

            logger.warning(f"[ModelManager] Fallback triggered for unknown model: {target_model}")
            return self.generate_with_openai(
                prompt,
                target_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens or 500,
                temperature=temperature or 0.7,
                top_p=top_p or 1.0
            )
    async def generate_once(self,
                            prompt: str,
                            model_name: str = None,
                            system_prompt: str = "You are a concise and helpful assistant.",
                            max_tokens: int = 256,
                            temperature: float = None,
                            top_p: float = None) -> str:
        """
        Generates a single, complete response asynchronously (non-streaming).
        Ideal for internal tasks like query rewriting or classification.
        """
        target_model = model_name or self.active_model_name
        if not target_model:
            logger.error("[generate_once] No model specified or active.")
            raise ValueError("No model specified. Pass model_name or use switch_model() first.")

        # --- Handle Local Models ---
        if target_model in self.models:
            # Use the existing synchronous generate method in a separate thread
            return await asyncio.to_thread(
                self.generate,
                prompt,
                model_name=target_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else self.default_temperature,
                top_p=top_p if top_p is not None else DEFAULT_TOP_P
            )

        # --- Handle API Models ---
        elif target_model in self.api_models:
            if self.async_client is None:
                return self._stub_response(prompt)
            try:
                # Stop sequences to prevent hallucinating user responses (less restrictive)
                stop_sequences = [
                    "\n\nUser:",
                    "\n\nUSER:",
                    "\n\nHuman:",
                ]

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                response = await self.async_client.chat.completions.create(
                    model=self.api_models[target_model],
                    messages=messages,
                    max_tokens=(max_tokens if max_tokens is not None else self.default_max_tokens),
                    temperature=temperature if temperature is not None else self.default_temperature,
                    top_p=top_p if top_p is not None else DEFAULT_TOP_P,
                    stop=stop_sequences,
                    stream=False  # Key change for non-streaming response
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"[ModelManager] OpenAI generate_once error: {e}")
                return self._stub_response(prompt)

        # --- Handle Unrecognized Models ---
        else:
            logger.error(f"[generate_once] Model '{target_model}' is not recognized.")
            raise ValueError(f"[ModelManager] Model '{target_model}' is not recognized as a local or registered API model.")

    @log_and_time("ModelManager Generate Async")
    async def generate_async(self, prompt, raw=False, **kwargs):
        """Async wrapper for generation using the active model"""
        target_model = self.active_model_name  # No longer allows override
        logger.debug(f"[generate_async] Active model: {target_model}")
        logger.debug(f"[generate_async] Registered OpenAI models: {self.api_models}")
        logger.debug(f"[generate_async] Registered local models: {self.models}")

        if target_model in self.models:
            return await asyncio.to_thread(
                self.generate, prompt, model_name=target_model, **kwargs
            )
        elif target_model in self.api_models:
            if self.async_client is None:
                return self._stub_stream(prompt)

            try:
                logger.debug(f"[ModelManager] Using OpenAI async model: {target_model}")
                if raw:
                    messages = [{"role": "user", "content": prompt}]
                else:
                    system_prompt = kwargs.get('system_prompt')
                    if system_prompt is not None:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ]
                for i, msg in enumerate(messages):
                    logger.debug(f"--- Prompt Message {i} ---")
                    logger.debug(f"Role: {msg['role']}")
                    logger.debug(f"Content:\n{msg['content']}")
                # Stop sequences to prevent hallucinating user responses (less restrictive)
                stop_sequences = [
                    "\n\nUser:",
                    "\n\nUSER:",
                    "\n\nHuman:",
                ]

                stream = await self.async_client.chat.completions.create(
                    model=self.api_models[target_model],
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', self.default_max_tokens),
                    temperature=kwargs.get('temperature', self.default_temperature),
                    top_p=kwargs.get('top_p', DEFAULT_TOP_P),
                    stop=stop_sequences,
                    stream=True
                )
                return stream
            except Exception as e:
                logger.error(f"[ModelManager] OpenAI streaming error: {e}")
                return self._stub_stream(prompt)
        else:
            return await asyncio.to_thread(
                self.generate, prompt, model_name=target_model, **kwargs
            )
