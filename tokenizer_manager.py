"""
TokenizerManager - Manages tokenizers for local models.

Skips loading tokenizer if the model is an API-based model (e.g. OpenAI).
Caches tokenizers to avoid reloading.
"""
import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("tokenizer manager.py is alive")

import re
import json
import os
from datetime import datetime, timedelta
from transformers import AutoTokenizer
from models import model_manager  # so it can check if a model is OpenAI
from logging_utils import log_and_time

class TokenizerManager:
    def __init__(self):
        self.tokenizers = {}
    @log_and_time("get tokenizer")
    def get_tokenizer(self, model_name):
        if model_name is None:
            return None

        # 🔥 NEW: Skip HuggingFace loading if OpenAI API model
        if model_manager.is_api_model(model_name):
            # logger.debug(f" Skipping local tokenizer load for OpenAI model '{model_name}'")
            return None
        if model_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"}:
            model_name = "gpt2"

        if model_name not in self.tokenizers:
            # logger.debug(f" Loading tokenizer for {model_name}")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizers[model_name]
