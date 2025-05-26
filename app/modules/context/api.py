from fastapi import APIRouter, HTTPException, Depends
from app.modules.context.context_manager import get_context_manager
from app.core.security import require_auth
from typing import Dict, Optional
from pydantic import BaseModel
import uuid

router = APIRouter()

class SessionData(BaseModel):
    users: Optional[list] = None
    driver: Optional[str] = None
    location: Optional[Dict] = None
    destination: Optional[Dict] = None

class QueryRequest(BaseModel):
    session_id: str
    query: str

@router.post("/sessions")
@require_auth()
async def create_session():
    """Create a new session"""
    try:
        context_manager = get_context_manager()
        session_id = str(uuid.uuid4())
        session = context_manager.create_session(session_id)
        return {
            "session_id": session_id,
            "start_time": session.start_time.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/sessions/{session_id}")
@require_auth()
async def update_session(session_id: str, data: SessionData):
    """Update session data"""
    try:
        context_manager = get_context_manager()
        context_manager.update_session_data(session_id, data.dict())
        return {"message": "Session updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/query")
@require_auth()
async def get_query_context(session_id: str, request: QueryRequest):
    """Get context for a specific query"""
    try:
        context_manager = get_context_manager()
        context = context_manager.get_context_for_query(session_id, request.query)
        return context
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
@require_auth()
async def end_session(session_id: str):
    """End a session"""
    try:
        context_manager = get_context_manager()
        context_manager.end_session(session_id)
        return {"message": "Session ended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 