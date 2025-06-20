from config import IN_HARM_TEST
import uuid
from datetime import datetime

def add_to_chroma(text, id, tags,collection, entry_type="memory", truth_scalar=None, emotional_valence=None):
    if IN_HARM_TEST:
        return  # 🚫 Block writes during harm test

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "timestamp": timestamp,
        "tags": ", ".join(tags),
        "type": entry_type
    }
    if truth_scalar: metadata["truth_scalar"] = truth_scalar
    if emotional_valence: metadata["emotional_valence"] = emotional_valence
    collection.add(documents=[text], metadatas=[metadata], ids=[id])