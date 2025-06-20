import os

class PersonalityManager:
    def __init__(self):
        self.personalities = {
            "default": {
                "system_prompt_file": "system_prompt_default.txt",
                "directives_file": "structured_directives.txt",
                "num_memories": 5,
                "include_wiki": True,
                "include_semantic_search": True

            },
            "therapy": {
                "system_prompt_file": "system_prompt_therapy.txt",
                "directives_file": "structured_directives_therapy.txt",
                "num_memories": 30,
                "include_wiki": False,
                "include_semantic_search": False

            },
            "snarky": {
                "system_prompt_file": "system_prompt_snarky.txt",
                "directives_file": "structured_directives_snarky.txt",
                "num_memories": 3,
                "include_wiki": True,
                "include_semantic_search": True,


            }
        }
        self.current_personality = "default"

    def switch_personality(self, name):
        if name in self.personalities:
            self.current_personality = name
            print(f"[PersonalityManager] Switched to: {name}")
        else:
            print(f"[PersonalityManager] Personality '{name}' not found. Using default.")

    def get_current_config(self):
        return self.personalities[self.current_personality]

