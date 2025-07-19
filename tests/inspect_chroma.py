# inspect_chroma.py

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import onnxruntime as ort

def inspect_chroma(path="chroma_db", collection_name="assistant-memory"):
    print(f"🔍 Inspecting ChromaDB at: {path}")

    try:
        client = PersistentClient(path=path)
        collection = client.get_collection(collection_name)
        data = collection.get()

        print(f"✅ Collection '{collection_name}' loaded successfully.")
        print(f"📄 Total documents: {len(data['documents'])}")
        print(f"🆔 First 5 IDs: {data['ids'][:5]}")
        print(f"📝 First 1 document preview:\n{data['documents'][0][:300]}...\n")

    except Exception as e:
        print(f"❌ Failed to load Chroma collection: {e}")

if __name__ == "__main__":
    inspect_chroma()
