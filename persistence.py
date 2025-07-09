import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("persistence.y.py is alive")
from config import IN_HARM_TEST
import uuid
from datetime import datetime
import asyncio

# In persistence.py

# Keep the original single-item function
def add_to_chroma(content, uid, tags, collection, entry_type="memory"):
    """Add a single item to ChromaDB"""
    try:
        collection.add(
            documents=[content],
            metadatas=[{
                "tags": ", ".join(tags) if isinstance(tags, list) else tags,
                "entry_type": entry_type,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[uid]
        )
    except Exception as e:
        logger.error(f"Error adding to ChromaDB: {e}")

# Add the batch version as a separate function
def add_to_chroma_batch(items):
    """Batch add multiple items to ChromaDB"""
    if not items:
        return

    documents = []
    metadatas = []
    ids = []

    for item in items:
        documents.append(item['document'])
        metadatas.append(item['metadata'])
        ids.append(item['id'])

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        logger.error(f"Error batch adding to ChromaDB: {e}")
