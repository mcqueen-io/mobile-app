from fastapi import APIRouter, HTTPException
from app.modules.user_info.user_graph import get_user_graph
from typing import Dict, List
from pydantic import BaseModel

router = APIRouter()

class UserCreate(BaseModel):
    name: str
    preferences: Dict = {}
    voice_embedding: str = None

class RelationshipCreate(BaseModel):
    user2_id: str
    relationship_type: str

@router.post("/users/{user_id}")
async def create_user(user_id: str, user_data: UserCreate):
    """Create a new user"""
    try:
        user_graph = get_user_graph()
        user_graph.create_user(user_id, user_data.dict())
        return {"message": "User created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user1_id}/relationships")
async def create_relationship(user1_id: str, relationship: RelationshipCreate):
    """Create a relationship between two users"""
    try:
        user_graph = get_user_graph()
        user_graph.create_relationship(
            user1_id,
            relationship.user2_id,
            relationship.relationship_type
        )
        return {"message": "Relationship created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/preferences")
async def get_preferences(user_id: str):
    """Get user preferences"""
    try:
        user_graph = get_user_graph()
        preferences = user_graph.get_user_preferences(user_id)
        return preferences
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/users/{user_id}/preferences")
async def update_preferences(user_id: str, preferences: Dict):
    """Update user preferences"""
    try:
        user_graph = get_user_graph()
        user_graph.update_user_preferences(user_id, preferences)
        return {"message": "Preferences updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/family")
async def get_family_tree(user_id: str):
    """Get user's family tree"""
    try:
        user_graph = get_user_graph()
        family = user_graph.get_family_tree(user_id)
        return family
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 