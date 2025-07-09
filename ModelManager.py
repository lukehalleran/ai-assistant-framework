# Import dependencies and config defaults
import logging
from logging_utils import log_and_time

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("Model Manager.py is alive")

import re
import json
import os
from datetime import datetime, timedelta
from config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, OpenAPIKey, SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time
import torch
from openai import OpenAI, AsyncOpenAI
import httpx
import os
import asyncio



# Set OpenAI API key for API calls
class ModelManager:
    """Manager class for handling both local and API-based language models."""

    def __init__(self, api_key: str = None):
        # Active model name (local or API)
        self.active_model_name = None
        # Dictionary of loaded local models
        self.models = {}
        # Dictionary of loaded tokenizers for local models
        self.tokenizers = {}
        # Dictionary mapping registered API models
        self.api_models = {}
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key,   http_client=httpx.Client(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=10
        ),
        headers={"Connection": "keep-alive"},
    )
    )
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.default_model = "gpt-4-turbo"
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
        if hasattr(self.client, "_client"):
            await self.client._client.aclose()
    def is_api_model(self, model_name):
        """Check if a given model is an API-based model."""
        return model_name in self.api_models

    @log_and_time("Generate with openAI")
    def generate_with_openai(self, prompt, model_name, system_prompt=None, max_tokens=None, temperature=None, top_p=None):
        """Generate text using OpenAI API, with global defaults fallback."""
        # Apply global defaults if not provided
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        top_p = DEFAULT_TOP_P if top_p is None else top_p

        try:
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
            temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
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
                    do_sample=False,
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
                temperature=temperature or 0.7,
                top_p=top_p or 1.0
            )

        else:
            # Try as a direct API model name
            logger.warning(f"[ModelManager] Model name fallback triggered. target_model = {target_model}")

            logger.debug(f"Using API model directly: {target_model}")
            return self.generate_with_openai(
                prompt,
                target_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens or 500,
                temperature=temperature or 0.7,
                top_p=top_p or 1.0
            )

    @log_and_time("ModelManager Generate Async")
    async def generate_async(self, prompt, raw=False, **kwargs):
        """Async wrapper for generation using the active model"""
        target_model = self.active_model_name  # No longer allows override
        logger.debug(f"[generate_async] Active model: {target_model}")
        logger.debug(f"[generate_async] Registered OpenAI models: {self.api_models}")
        logger.debug(f"[generate_async] Registered local models: {self.models}")
        self.switch_model("gpt-4-turbo")
        logger.debug(f"[generate_async] Forcing model to: {target_model}")


        if target_model in self.models:
            return await asyncio.to_thread(
                self.generate, prompt, model_name=target_model, **kwargs
            )

        elif target_model in self.api_models:
            try:
                logger.debug(f"[ModelManager] Using OpenAI async model: {target_model}")

                if raw:
                    # Bypass all prompt-building logic
                    messages = [{"role": "user", "content": prompt}]
                else:
                    # Inject system prompt as usual
                    if kwargs.get('system_prompt', SYSTEM_PROMPT) is not None:
                        messages = [
                            {"role": "system", "content": kwargs.get('system_prompt', SYSTEM_PROMPT)},
                            {"role": "user", "content": prompt}
                        ]
                    else:
                        messages = [{"role": "user", "content": prompt}]

                stream = await self.async_client.chat.completions.create(
                    model=self.api_models[target_model],
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', DEFAULT_MAX_TOKENS),
                    temperature=kwargs.get('temperature', DEFAULT_TEMPERATURE),
                    top_p=kwargs.get('top_p', DEFAULT_TOP_P),
                    stream=True
                )
                return stream

            except Exception as e:
                logger.error(f"[ModelManager] OpenAI streaming error: {e}")
                return f"[OpenAI Async API Error] {str(e)}"

        else:
            return await asyncio.to_thread(
                self.generate, prompt, model_name=target_model, **kwargs
            )



