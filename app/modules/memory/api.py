from fastapi import APIRouter, HTTPException
from app.modules.memory.memory_store import get_memory_store
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class MemoryCreate(BaseModel):
    content: str
    metadata: Optional[Dict] = None

class MemoryUpdate(BaseModel):
    content: str
    metadata: Optional[Dict] = None

@router.post("/users/{user_id}/memories")
async def create_memory(user_id: str, memory: MemoryCreate):
    """Create a new memory for a user"""
    try:
        memory_store = get_memory_store()
        memory_id = memory_store.add_memory(
            user_id,
            memory.content,
            memory.metadata
        )
        return {"memory_id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/memories")
async def get_memories(user_id: str, query: Optional[str] = None, limit: int = 100):
    """Get user memories, optionally filtered by query"""
    try:
        memory_store = get_memory_store()
        if query:
            memories = memory_store.get_relevant_memories(user_id, query, limit)
        else:
            memories = memory_store.get_user_memories(user_id, limit)
        return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/memories/{memory_id}")
async def update_memory(memory_id: str, memory: MemoryUpdate):
    """Update an existing memory"""
    try:
        memory_store = get_memory_store()
        memory_store.update_memory(
            memory_id,
            memory.content,
            memory.metadata
        )
        return {"message": "Memory updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory"""
    try:
        memory_store = get_memory_store()
        memory_store.delete_memory(memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 