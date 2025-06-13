# Import dependencies and config defaults
from config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, OpenAPIKey, SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import openai
import os

# Set OpenAI API key for API calls
openai.api_key = OpenAPIKey


class ModelManager:
    """Manager class for handling both local and API-based language models."""

    def __init__(self):
        # Active model name (local or API)
        self.active_model_name = None
        # Dictionary of loaded local models
        self.models = {}
        # Dictionary of loaded tokenizers for local models
        self.tokenizers = {}
        # Dictionary mapping registered API models
        self.api_models = {}

    def get_context_limit(self):
        """Get the maximum context window based on active model."""
        if self.is_api_model(self.get_active_model_name()):
            # For API models (OpenAI), assume known context window
            # print("[DEBUG] Using OpenAI model, assuming 128000 token context window.")
            return 128000  # Default GPT-4 Turbo context
        model = self.get_model()
        if model:
            return model.config.max_position_embeddings
        else:
            raise ValueError("[ERROR] No model loaded. Cannot determine context limit.")

    def load_model(self, model_name, model_path):
        """Load a local Huggingface model and tokenizer."""
        try:
            # Determine if using local files only
            local_files_only = model_path.startswith("./")
            # print(f"[DEBUG] Loading local model '{model_name}' from '{model_path}'")

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

            # print(f"[DEBUG] Successfully loaded model: {model.__class__.__name__}")
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {str(e)}")

    def load_openai_model(self, model_name, api_model_name):
        """Register an OpenAI API model (no local loading)."""
        # print(f"[DEBUG] Registering OpenAI model '{model_name}'")
        self.api_models[model_name] = api_model_name

    def is_api_model(self, model_name):
        """Check if a given model is an API-based model."""
        return model_name in self.api_models
    def generate_with_openai(self, prompt, model_name, system_prompt=None, max_tokens=None, temperature=None, top_p=None):
        """Generate text using OpenAI API, with global defaults fallback."""
        # Apply global defaults if not provided
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        top_p = DEFAULT_TOP_P if top_p is None else top_p

        try:
            # print(f"[DEBUG]  Calling OpenAI API: {model_name}")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt or SYSTEM_PROMPT},  # ← this is correct
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response['choices'][0]['message']['content'].strip()

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

        # print(f"[DEBUG] Truncating prompt: {len(prefix_tokens) + len(rest_tokens)} → {max_input_tokens} tokens")

        # If prefix alone is too long, truncate from the end of entire prompt
        allowed_rest_tokens = max_input_tokens - len(prefix_tokens)
        if allowed_rest_tokens <= 0:
            print("[WARN] Prefix alone exceeds max input size.")
            return tokenizer.decode((prefix_tokens + rest_tokens)[-max_input_tokens:])

        # Truncate rest of prompt and return combined prompt
        truncated_rest_tokens = rest_tokens[-allowed_rest_tokens:]
        return tokenizer.decode(prefix_tokens + truncated_rest_tokens)

    def generate(self, prompt, model_name=None, max_tokens=None, temperature=None, top_p=None, top_k=None, no_repeat_ngram_size=None, pad_token_id=None, system_prompt=None):
        """Main generate function for both local and OpenAI models."""
        # Ensure an active model is selected
        if not self.active_model_name:
            if model_name:
                self.switch_model(model_name)
            else:
                raise ValueError("No active model. Use switch_model() first or pass model_name.")

        # API model generation path
        if self.is_api_model(self.active_model_name):
            # print(f"[DEBUG] Using OpenAI model: {self.api_models[self.active_model_name]}")
            return self.generate_with_openai(
                prompt,
                self.api_models[self.active_model_name],
                max_tokens=max_tokens or 500,
                temperature=temperature or 0.7,
                top_p=top_p or 1.0
            )

        # Local model generation path
        # print(f"[DEBUG] Using local model: {self.active_model_name}")
        model = self.get_model()
        tokenizer = self.get_tokenizer()

        # Use defaults where needed
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        top_p = DEFAULT_TOP_P if top_p is None else top_p
        top_k = DEFAULT_TOP_K if top_k is None else top_k

        # Check input length and truncate if needed
        tokens = tokenizer.encode(prompt)
        max_len = model.config.max_position_embeddings

        if len(tokens) > max_len:
            # print(f"[DEBUG] Trimming input from {len(tokens)} to {max_len} tokens")
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
        # print("[DEBUG] 🚀 Starting Huggingface model generation...")
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=pad_token_id or tokenizer.pad_token_id,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            end = time.time()
            # print(f"[DEBUG]  Finished generation in {end - start:.2f} seconds")

        # Decode and return output text
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
