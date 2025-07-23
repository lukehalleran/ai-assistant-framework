import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("Personality manager.py is alive")

import re
import json
import os
from datetime import datetime, timedelta
import os

class PersonalityManager:
    def __init__(self):
        self.personalities = {
            "default": {
                "system_prompt_file": "system_prompt_default.txt",
                "directives_file": "structured_directives.txt",
                "num_memories": 5,
                "include_wiki": True,
                "include_semantic_search": True,
                "include_summaries": True,
                "summary_limit": 5

            },
            "therapy": {
                "system_prompt_file": "system_prompt_therapy.txt",
                "directives_file": "structured_directives_therapy.txt",
                "num_memories": 30,
                "include_wiki": False,
                "include_semantic_search": False,
                "include_summaries": True,  # Important for therapy mode
                "summary_limit": 5

            },
            "snarky": {
                "system_prompt_file": "system_prompt_snarky.txt",
                "directives_file": "structured_directives_snarky.txt",
                "num_memories": 3,
                "include_wiki": True,
                "include_semantic_search": True,
                "include_summaries": True,  # Important for therapy mode
                "summary_limit": 5

            }
        }
        self.current_personality = "default"

    def switch_personality(self, name):
        if name in self.personalities:
            self.current_personality = name
            logger.debug(f"[PersonalityManager] Switched to: {name}")
        else:
            logger.debug(f"[PersonalityManager] Personality '{name}' not found. Using default.")

    def get_current_config(self):
        return self.personalities[self.current_personality]

