import chromadb
from chromadb.config import Settings
from app.core.config import settings
from typing import Dict, List, Optional
from datetime import datetime
import json

class MemoryStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        ))
        self.collection = self.client.get_or_create_collection(
            name="user_memories",
            metadata={"hnsw:space": "cosine"}
        )

    def add_memory(self, user_id: str, content: str, metadata: Dict = None) -> str:
        """Add a new memory for a user"""
        memory_id = f"{user_id}_{datetime.now().isoformat()}"
        self.collection.add(
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }],
            ids=[memory_id]
        )
        return memory_id

    def get_relevant_memories(self, user_id: str, query: str, n_results: int = 5) -> List[Dict]:
        """Get memories relevant to the current context"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        memories = []
        for i in range(len(results['ids'][0])):
            memories.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return memories

    def delete_memory(self, memory_id: str) -> None:
        """Delete a specific memory"""
        self.collection.delete(ids=[memory_id])

    def get_user_memories(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get all memories for a user"""
        results = self.collection.get(
            where={"user_id": user_id},
            limit=limit
        )
        
        memories = []
        for i in range(len(results['ids'])):
            memories.append({
                "id": results['ids'][i],
                "content": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
        return memories

    def update_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """Update an existing memory"""
        self.collection.update(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata] if metadata else None
        )

# Create a singleton instance
memory_store = MemoryStore()

def get_memory_store():
    return memory_store 