# migrate_memory.py - Migrate existing memories to hierarchical system

import json
import asyncio
from datetime import datetime
from typing import List, Dict
import os

from hierarchical_memory import HierarchicalMemorySystem, MemoryType, MemoryNode
from memory import load_corpus
from chromadb import PersistentClient

async def migrate_existing_memories(model_manager):
    """Migrate existing corpus and ChromaDB to hierarchical system"""

    print("Starting memory migration...")

    # Initialize new hierarchical system
    hierarchical_memory = HierarchicalMemorySystem(model_manager)

    # Load existing corpus
    print("Loading existing corpus...")
    corpus = load_corpus()
    print(f"Found {len(corpus)} entries in corpus")

    # Load ChromaDB
    print("Loading ChromaDB...")
    client = PersistentClient(path="chroma_db")
    try:
        collection = client.get_collection("assistant-memory")
        chroma_data = collection.get()
        print(f"Found {len(chroma_data['ids'])} entries in ChromaDB")
    except:
        print("ChromaDB collection not found")
        chroma_data = {'ids': [], 'documents': [], 'metadatas': []}

    # Migration statistics
    stats = {
        'episodic': 0,
        'summaries': 0,
        'semantic': 0,
        'errors': 0
    }

    # Migrate corpus entries
    print("\nMigrating corpus entries...")
    for i, entry in enumerate(corpus):
        try:
            # Determine memory type
            if "@summary" in entry.get("tags", []):
                memory_type = MemoryType.SUMMARY
                stats['summaries'] += 1
            else:
                memory_type = MemoryType.EPISODIC
                stats['episodic'] += 1

            # Create memory node
            memory_id = entry.get('id', f"corpus_{i}")

            # Parse timestamp
            timestamp = entry.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except:
                    timestamp = datetime.now()

            # Construct content
            content = f"User: {entry.get('query', '')}\nAssistant: {entry.get('response', '')}"

            # Calculate importance (summaries get higher importance)
            importance = 0.8 if memory_type == MemoryType.SUMMARY else 0.5

            # Create memory node
            memory = MemoryNode(
                id=memory_id,
                content=content,
                type=memory_type,
                timestamp=timestamp,
                importance_score=importance,
                tags=entry.get('tags', []),
                metadata={'source': 'corpus_migration'}
            )

            # Add to hierarchical system
            hierarchical_memory.memories[memory_id] = memory
            hierarchical_memory.type_index[memory_type].append(memory_id)

            # Update tag index
            for tag in memory.tags:
                hierarchical_memory.tag_index[tag].append(memory_id)

            # Extract semantic knowledge from recent episodic memories
            if memory_type == MemoryType.EPISODIC and i >= len(corpus) - 50:  # Last 50 entries
                semantic_memories = await hierarchical_memory._extract_semantic_knowledge(
                    entry.get('query', ''),
                    entry.get('response', '')
                )
                for sem_mem in semantic_memories:
                    hierarchical_memory._add_child_memory(memory_id, sem_mem)
                    stats['semantic'] += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(corpus)} corpus entries...")

        except Exception as e:
            print(f"  Error migrating entry {i}: {e}")
            stats['errors'] += 1

    # Save migrated memories
    print("\nSaving hierarchical memories...")
    hierarchical_memory.save_memories()

    # Print migration summary
    print("\n=== Migration Summary ===")
    print(f"Episodic memories: {stats['episodic']}")
    print(f"Summary memories: {stats['summaries']}")
    print(f"Semantic facts extracted: {stats['semantic']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total memories: {len(hierarchical_memory.memories)}")

    return hierarchical_memory

async def verify_migration(hierarchical_memory):
    """Verify the migration with some test queries"""

    print("\n=== Verification Tests ===")

    test_queries = [
        "What have we discussed recently?",
        "Tell me about artificial intelligence",
        "What do you remember about our conversations?"
    ]

    for query in test_queries:
        print(f"\nTest query: '{query}'")
        memories = await hierarchical_memory.retrieve_relevant_memories(query, max_memories=3)

        print(f"Retrieved {len(memories)} memories:")
        for i, mem in enumerate(memories):
            memory = mem['memory']
            print(f"  {i+1}. [{memory.type.value}] Score: {mem['final_score']:.3f}")
            print(f"     Preview: {memory.content[:100]}...")

def create_backup():
    """Create backup of existing memory files"""
    import shutil
    from datetime import datetime

    backup_dir = f"memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)

    # Backup corpus file
    if os.path.exists("corpus.json"):
        shutil.copy2("corpus.json", os.path.join(backup_dir, "corpus.json"))
        print(f"✓ Backed up corpus.json to {backup_dir}")

    # Backup ChromaDB
    if os.path.exists("chroma_db"):
        shutil.copytree("chroma_db", os.path.join(backup_dir, "chroma_db"))
        print(f"✓ Backed up chroma_db to {backup_dir}")

    return backup_dir

if __name__ == "__main__":
    print("=== DAEMON Memory Migration Tool ===\n")

    # Check prerequisites
    print("Checking prerequisites...")

    if not os.path.exists("hierarchical_memory.py"):
        print("ERROR: hierarchical_memory.py not found!")
        exit(1)

    if not os.path.exists("corpus.json"):
        print("WARNING: corpus.json not found. No existing memories to migrate.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

    print("✓ Prerequisites satisfied\n")

    # Create backup
    response = input("Create backup of existing memories? (recommended) (y/n): ")
    if response.lower() == 'y':
        backup_dir = create_backup()
        print(f"\nBackup created in: {backup_dir}\n")

    # Initialize model manager
    print("Initializing model manager...")
    from ModelManager import ModelManager
    model_manager = ModelManager()

    # Check if we need to load a model
    try:
        model_manager.load_openai_model('gpt-4-turbo', 'gpt-4-turbo')
        model_manager.switch_model('gpt-4-turbo')
        print("✓ Using GPT-4 Turbo for migration")
    except:
        print("WARNING: Could not load OpenAI model. Migration will use defaults.")

    # Run migration
    print("\nStarting migration...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        hierarchical_memory = loop.run_until_complete(migrate_existing_memories(model_manager))

        # Verify migration
        response = input("\nRun verification tests? (y/n): ")
        if response.lower() == 'y':
            loop.run_until_complete(verify_migration(hierarchical_memory))

        print("\n✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Review the migrated memories in 'hierarchical_memory/' directory")
        print("2. Test the system with gui_hierarchical.py")
        print("3. Once verified, replace gui.py with gui_hierarchical.py")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Your original files are unchanged. Check the error and try again.")

    finally:
        loop.close()
