# daemon_7_11_25_refactor/core/orchestrator.py
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from utils.logging_utils import get_logger
from core.prompt import PromptBuilder
from core.prompt_builder import UnifiedHierarchicalPromptBuilder
from memory.corpus_manager import CorpusManager
from personality.personality_manager import PersonalityManager
from knowledge.topic_manager import TopicManager
from knowledge.WikiManager import WikiManager
logger = get_logger("orchestrator")

class DaemonOrchestrator:
    """Central coordinator for the AI assistant system"""

    def __init__(self,
                 model_manager,
                 response_generator,
                 file_processor,
                 prompt_builder: UnifiedHierarchicalPromptBuilder = None,
                 personality_manager: PersonalityManager = None):
        self.model_manager = model_manager
        self.response_generator = response_generator
        self.file_processor = file_processor
        self.topic_manager= TopicManager()
        self.WikiManager= WikiManager()
        self.personality_manager = personality_manager or PersonalityManager()

        base_pb = PromptBuilder(
            model_manager,           # tokenizer_manager
            "gpt-4-turbo",           # model_name
            WikiManager(),           # wiki
            model_manager,           # model_manager
            TopicManager()           # topic_manager
        )

        self.prompt_builder = prompt_builder or UnifiedHierarchicalPromptBuilder(
            prompt_builder=base_pb,
            model_manager=model_manager
        )


        # **Expose the MemoryCoordinator directly on the orchestrator**
        self.memory_system = self.prompt_builder.memory_system
    def get_recent_conversation(self, n: int = 5):
        return self.memory_system.corpus_manager.get_recent_memories(count=n)

    async def process_user_query(self,
                                user_input: str,
                                files: List[Any] = None,
                                use_raw_mode: bool = False) -> Tuple[str, Dict]:
        """
        Main entry point for processing user queries.
        Returns: (response, debug_info)
        """
        debug_info = {
            'start_time': datetime.now(),
            'user_input': user_input[:100],
            'files_count': len(files) if files else 0,
            'mode': 'raw' if use_raw_mode else 'enhanced'
        }

        try:
            # Step 1: Process files if any
            combined_text = user_input
            if files and not use_raw_mode:
                combined_text = await self.file_processor.process_files(user_input, files)
                debug_info['combined_text_length'] = len(combined_text)

            # Step 2: Build prompt
            if use_raw_mode:
                prompt = combined_text
            else:
                # Use prompt builder if available, otherwise just use the text
                config = self.personality_manager.get_current_config()

                if self.prompt_builder:
                    recent = self.get_recent_conversation(n=5)
                    prompt = await self.prompt_builder.build_hierarchical_prompt(
                        user_input=combined_text,
                        include_dreams=True,
                        include_wiki=config.get("include_wiki", True),
                        include_semantic=config.get("include_semantic_search", True),
                        recent_conversations=recent,
                        system_prompt=open(config["system_prompt_file"], "r").read(),
                        directives_file=config["directives_file"]
                    )

            # Ensure we have a model name
            model_name = self.model_manager.get_active_model_name() if hasattr(self.model_manager, 'get_active_model_name') else "gpt-4-turbo"
            #instante vars
            full_response = ""
            chunk_count = 0

            async for chunk in self.response_generator.generate_streaming_response(prompt, model_name):
                full_response += chunk + " "
                chunk_count += 1

            debug_info['response_length'] = len(full_response)
            debug_info['chunk_count'] = chunk_count
            debug_info['end_time'] = datetime.now()
            debug_info['duration'] = (debug_info['end_time'] - debug_info['start_time']).total_seconds()
            debug_info['prompt'] = prompt

            return full_response.strip(), debug_info

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            debug_info['error'] = str(e)
            raise
