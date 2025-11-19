#!/usr/bin/env python3
"""
migrate_corpus_to_chromadb.py

Migrate existing conversations from corpus manager to ChromaDB collections.
This will populate the vector store with all historical conversations for semantic search.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from models.model_manager import ModelManager
from memory.memory_coordinator import MemoryCoordinator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CorpusToChromaMigrator:
    """Migrate corpus data to ChromaDB collections"""

    def __init__(self):
        # Initialize components
        self.model_manager = ModelManager()
        self.corpus_manager = CorpusManager()

        # Get ChromaDB path
        chroma_path = os.getenv('CHROMA_PATH', './chroma_db')
        logger.info(f"Using ChromaDB path: {chroma_path}")

        # Initialize ChromaDB store
        self.chroma_store = MultiCollectionChromaStore(persist_directory=chroma_path)

        # Initialize memory coordinator for proper metadata handling
        self.memory_coordinator = MemoryCoordinator(
            corpus_manager=self.corpus_manager,
            chroma_store=self.chroma_store,
            model_manager=self.model_manager
        )

    def get_corpus_stats(self) -> Dict[str, int]:
        """Get statistics about corpus data"""
        logger.info("Analyzing corpus data...")

        # Get recent memories (all conversations)
        all_conversations = self.corpus_manager.get_recent_memories(count=10000)  # Get all

        stats = {
            'total_conversations': len(all_conversations),
            'with_thread_id': 0,
            'with_timestamps': 0,
            'date_range': {'earliest': None, 'latest': None}
        }

        for conv in all_conversations:
            if conv.get('thread_id'):
                stats['with_thread_id'] += 1
            if conv.get('timestamp'):
                stats['with_timestamps'] += 1

                # Track date range
                ts = conv['timestamp']
                if isinstance(ts, datetime):
                    if stats['date_range']['earliest'] is None or ts < stats['date_range']['earliest']:
                        stats['date_range']['earliest'] = ts
                    if stats['date_range']['latest'] is None or ts > stats['date_range']['latest']:
                        stats['date_range']['latest'] = ts
                elif isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        if stats['date_range']['earliest'] is None or dt < stats['date_range']['earliest']:
                            stats['date_range']['earliest'] = dt
                        if stats['date_range']['latest'] is None or dt > stats['date_range']['latest']:
                            stats['date_range']['latest'] = dt
                    except:
                        pass

        return stats

    def migrate_conversations(self, batch_size: int = 100, dry_run: bool = False):
        """Migrate conversations from corpus to ChromaDB"""

        # Check current ChromaDB state
        current_counts = {}
        for name, collection in self.chroma_store.collections.items():
            current_counts[name] = collection.count()

        logger.info(f"Current ChromaDB state: {current_counts}")

        # Get all conversations from corpus
        all_conversations = self.corpus_manager.get_recent_memories(count=10000)
        total_to_migrate = len(all_conversations)

        logger.info(f"Found {total_to_migrate} conversations to migrate")

        if dry_run:
            logger.info("DRY RUN: Would migrate conversations but not actually storing them")
            return

        # Migrate in batches
        migrated_count = 0
        failed_count = 0

        for i in range(0, total_to_migrate, batch_size):
            batch = all_conversations[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_to_migrate-1)//batch_size + 1} ({len(batch)} conversations)")

            for conv in batch:
                try:
                    # Extract conversation data
                    query = conv.get('query', '')
                    response = conv.get('response', '')

                    if not query or not response:
                        logger.debug(f"Skipping conversation {i+migrated_count}: missing query or response")
                        continue

                    # Prepare metadata using memory coordinator's logic
                    metadata = self._prepare_metadata(conv)

                    # Add to ChromaDB conversations collection
                    memory_id = self.chroma_store.add_conversation_memory(query, response, metadata)
                    migrated_count += 1

                    if migrated_count % 50 == 0:
                        logger.info(f"Migrated {migrated_count}/{total_to_migrate} conversations")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to migrate conversation {i+migrated_count}: {e}")
                    continue

        # Final counts
        final_counts = {}
        for name, collection in self.chroma_store.collections.items():
            final_counts[name] = collection.count()

        logger.info(f"Migration completed:")
        logger.info(f"  Successfully migrated: {migrated_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Final ChromaDB counts: {final_counts}")

    def _prepare_metadata(self, conv: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata using memory coordinator's logic"""

        # Use memory coordinator's metadata preparation logic
        try:
            # Extract thread information
            thread_id = conv.get('thread_id', f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            thread_depth = conv.get('thread_depth', 1)
            thread_started = conv.get('thread_started', conv.get('timestamp', datetime.now()))
            thread_topic = conv.get('thread_topic', 'general')
            is_heavy_topic = conv.get('is_heavy_topic', False)

            # Create metadata similar to store_interaction
            metadata = {
                "timestamp": conv.get('timestamp', datetime.now().isoformat()),
                "thread_id": thread_id,
                "thread_depth": thread_depth,
                "thread_started": thread_started.isoformat() if isinstance(thread_started, datetime) else str(thread_started),
                "thread_topic": thread_topic,
                "is_heavy_topic": is_heavy_topic,
                "memory_type": "EPISODIC",
                "truth_score": 1.0,  # Default for migrated data
                "importance_score": 1.0,  # Default for migrated data
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "tags": ",".join(conv.get('tags', [])) if conv.get('tags') else "",
                "migration_timestamp": datetime.now().isoformat()
            }

            return metadata

        except Exception as e:
            logger.error(f"Error preparing metadata: {e}")
            # Fallback minimal metadata
            return {
                "timestamp": conv.get('timestamp', datetime.now().isoformat()),
                "thread_id": conv.get('thread_id', 'migrated'),
                "memory_type": "EPISODIC",
                "migration_timestamp": datetime.now().isoformat()
            }

def main():
    """Main migration function"""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate corpus data to ChromaDB")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for migration")
    parser.add_argument("--dry-run", action="store_true", help="Do a dry run without actually migrating")
    parser.add_argument("--stats", action="store_true", help="Only show statistics, don't migrate")

    args = parser.parse_args()

    logger.info("=== Corpus to ChromaDB Migration ===")

    # Initialize migrator
    migrator = CorpusToChromaMigrator()

    # Show statistics
    stats = migrator.get_corpus_stats()
    logger.info(f"Corpus statistics: {stats}")

    if args.stats:
        logger.info("Statistics only mode, exiting.")
        return

    # Confirm migration (auto-confirmed for CLI use)
    if not args.dry_run:
        logger.info(f"Auto-confirming migration of {stats['total_conversations']} conversations to ChromaDB...")
        # response = input(f"\nReady to migrate {stats['total_conversations']} conversations to ChromaDB. Continue? (y/N): ")
        # if response.lower() not in ['y', 'yes']:
        #     logger.info("Migration cancelled.")
        #     return

    # Run migration
    try:
        migrator.migrate_conversations(batch_size=args.batch_size, dry_run=args.dry_run)
        logger.info("Migration completed successfully!")

    except KeyboardInterrupt:
        logger.info("Migration interrupted by user.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()