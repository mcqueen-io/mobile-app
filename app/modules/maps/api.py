from fastapi import APIRouter, HTTPException
from app.modules.maps.maps_optimizer import get_maps_optimizer
from typing import Dict, Optional
from pydantic import BaseModel

router = APIRouter()

class Location(BaseModel):
    latitude: float
    longitude: float

class RouteRequest(BaseModel):
    start: Location
    end: Location
    preferences: Optional[Dict] = None

class NavigationQuery(BaseModel):
    current_location: Location
    query: str

@router.post("/optimize")
async def optimize_route(request: RouteRequest):
    """Optimize route based on user preferences and current conditions"""
    try:
        maps_optimizer = get_maps_optimizer()
        result = await maps_optimizer.optimize_route(
            start_location=request.start.dict(),
            end_location=request.end.dict(),
            preferences=request.preferences
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/next-instruction")
async def get_next_instruction(
    current_location: Location,
    context: Optional[Dict] = None
):
    """Get the next navigation instruction"""
    try:
        maps_optimizer = get_maps_optimizer()
        result = await maps_optimizer.get_next_instruction(
            current_location=current_location.dict(),
            context=context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/handle-confusion")
async def handle_confusion(query: NavigationQuery):
    """Handle user confusion about navigation"""
    try:
        maps_optimizer = get_maps_optimizer()
        result = await maps_optimizer.handle_confusion(
            current_location=query.current_location.dict(),
            user_query=query.query
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 