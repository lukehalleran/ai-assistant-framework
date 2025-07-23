from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from utils.logging_utils import get_logger
from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.memory_interface import HierarchicalMemorySystem
from personality.personality_manager import PersonalityManager
from knowledge.topic_manager import TopicManager
from knowledge.WikiManager import WikiManager
from processing.gate_system import MultiStageGateSystem
from models.tokenizer_manager import TokenizerManager
from core.prompt_builder_v2 import UnifiedPromptBuilder

logger = get_logger("orchestrator")


class DaemonOrchestrator:
    """Central coordinator for the AI assistant system"""

    def __init__(
        self,
        model_manager,
        response_generator,
        file_processor,
        prompt_builder,
        memory_system,
        personality_manager: PersonalityManager = None
    ):
        self.model_manager = model_manager
        self.response_generator = response_generator
        self.file_processor = file_processor
        self.personality_manager = personality_manager or PersonalityManager()
        self.prompt_builder = prompt_builder

        # Initialize components
        self.topic_manager = TopicManager()
        self.wiki_manager = WikiManager()
        self.tokenizer_manager = TokenizerManager(model_manager)

        # Initialize memory system
        self.corpus_manager = CorpusManager()
        self.chroma_store = MultiCollectionChromaStore()
        self.hierarchical_memory = HierarchicalMemorySystem(
            model_manager=model_manager,
            chroma_store=self.chroma_store
        )

        # Initialize memory coordinator
        self.memory_system = MemoryCoordinator(
            corpus_manager=self.corpus_manager,
            chroma_store=self.chroma_store,
            hierarchical_memory=self.hierarchical_memory,
            gate_system=MultiStageGateSystem(model_manager)
        )



    def get_recent_conversation(self, n: int = 5):
        """Get recent conversation history"""
        return self.memory_system.corpus_manager.get_recent_memories(count=n)

    async def process_user_query(
        self,
        user_input: str,
        files: List[Any] = None,
        use_raw_mode: bool = False
    ) -> Tuple[str, Dict]:
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
                # Get current personality configuration
                config = self.personality_manager.get_current_config()

                # Read system prompt from file
                system_prompt = ""
                if config.get("system_prompt_file"):
                    try:
                        with open(config["system_prompt_file"], "r") as f:
                            system_prompt = f.read()
                    except:
                        logger.warning(f"Could not read system prompt file: {config['system_prompt_file']}")

                # Build prompt using simplified builder
                prompt = await self.prompt_builder.build_prompt(
                    user_input=combined_text,
                    include_dreams=True,
                    include_wiki=config.get("include_wiki", True),
                    include_semantic=config.get("include_semantic_search", True),
                    include_summaries=True,
                    system_prompt=system_prompt,
                    directives_file=config.get("directives_file", "structured_directives.txt"),
                    personality_config=config
                )

            # Step 3: Generate response
            model_name = self.model_manager.get_active_model_name()
            if not model_name:
                model_name = "gpt-4-turbo"
                self.model_manager.switch_model(model_name)

            full_response = ""
            chunk_count = 0

            async for chunk in self.response_generator.generate_streaming_response(prompt, model_name):
                full_response += chunk + " "
                chunk_count += 1

            # Step 4: Store the interaction in memory
            if not use_raw_mode:
                await self.memory_system.store_interaction(
                    query=user_input,
                    response=full_response.strip(),
                    tags=["conversation"]
                )

            debug_info['response_length'] = len(full_response)
            debug_info['chunk_count'] = chunk_count
            debug_info['end_time'] = datetime.now()
            debug_info['duration'] = (debug_info['end_time'] - debug_info['start_time']).total_seconds()
            debug_info['prompt_length'] = len(prompt)

            return full_response.strip(), debug_info

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            debug_info['error'] = str(e)
            raise
