from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
from app.services.smart_navigation_service import get_smart_navigation_service
from app.services.offline_navigation_cache import get_offline_navigation_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/navigation", tags=["navigation"])

class NavigationRequest(BaseModel):
    origin: str
    destination: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class LocationFeedbackRequest(BaseModel):
    location: Dict[str, float]  # {"lat": 12.345, "lng": -67.890}
    user_feedback: str
    original_instruction: str
    user_id: str

class NavigationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TripCacheRequest(BaseModel):
    origin: Dict[str, Any]  # {'lat': float, 'lng': float, 'address': str}
    destination: Dict[str, Any]
    hotel_address: Optional[str] = None
    waypoints: Optional[List[Dict[str, Any]]] = None
    trip_duration_days: Optional[int] = None

class OfflineGuidanceRequest(BaseModel):
    current_location: Dict[str, float]  # {'lat': float, 'lng': float}
    need_type: Optional[str] = "general"  # 'emergency', 'fuel', 'accommodation', 'supplies', 'general'
    user_context: Optional[Dict[str, Any]] = None

@router.post("/enhanced-directions", response_model=NavigationResponse)
async def get_enhanced_directions(request: NavigationRequest):
    """
    Get AI-enhanced navigation directions with natural language guidance
    and proactive clarifications for confusing areas.
    """
    try:
        navigation_service = await get_smart_navigation_service()
        
        # Get enhanced directions
        enhanced_directions = await navigation_service.enhance_directions(
            origin=request.origin,
            destination=request.destination,
            user_context=request.context or {}
        )
        
        if "error" in enhanced_directions:
            return NavigationResponse(
                success=False,
                error=enhanced_directions["error"]
            )
        
        return NavigationResponse(
            success=True,
            data=enhanced_directions
        )
        
    except Exception as e:
        logger.error(f"Error getting enhanced directions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report-confusing-location")
async def report_confusing_location(request: LocationFeedbackRequest):
    """
    Report a confusing location to help improve navigation guidance.
    Community-driven feedback to enhance directions for everyone.
    """
    try:
        navigation_service = await get_smart_navigation_service()
        
        await navigation_service.report_confusing_location(
            location=request.location,
            user_feedback=request.user_feedback,
            original_instruction=request.original_instruction,
            user_id=request.user_id
        )
        
        return {"success": True, "message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error reporting confusing location: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/confusing-locations")
async def get_confusing_locations():
    """
    Get list of currently known confusing locations.
    Useful for debugging and understanding problem areas.
    """
    try:
        navigation_service = await get_smart_navigation_service()
        
        return {
            "success": True,
            "locations": list(navigation_service.confusing_locations_db.values()),
            "count": len(navigation_service.confusing_locations_db)
        }
        
    except Exception as e:
        logger.error(f"Error getting confusing locations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quick-clarification")
async def get_quick_clarification(
    instruction: str,
    location: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Get quick AI clarification for a specific navigation instruction.
    Useful for real-time help when user is confused.
    """
    try:
        navigation_service = await get_smart_navigation_service()
        
        # Create a mock step for the AI enhancement
        mock_step = {
            "html_instructions": instruction,
            "distance": {"text": "ahead"},
            "duration": {"text": "shortly"},
            "start_location": location or {"lat": 0, "lng": 0}
        }
        
        enhanced_instruction = await navigation_service.generate_enhanced_instruction(
            step=mock_step,
            step_index=0,
            all_steps=[mock_step],
            user_context=context or {}
        )
        
        tips = await navigation_service.generate_clarification_tips(mock_step, context)
        
        return {
            "success": True,
            "original_instruction": instruction,
            "ai_enhanced_instruction": enhanced_instruction,
            "helpful_tips": tips
        }
        
    except Exception as e:
        logger.error(f"Error getting quick clarification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preload-trip-cache")
async def preload_trip_cache(request: TripCacheRequest):
    """
    Preload offline cache for an upcoming trip.
    Caches prominent places, hotels, gas stations along the route for offline use.
    """
    try:
        offline_cache = await get_offline_navigation_cache()
        
        trip_data = {
            'origin': request.origin,
            'destination': request.destination,
            'hotel_address': request.hotel_address,
            'waypoints': request.waypoints or [],
            'trip_duration_days': request.trip_duration_days or 1
        }
        
        cache_summary = await offline_cache.preload_trip_cache(trip_data)
        
        return {
            "success": True,
            "message": "Trip cache preloaded successfully",
            "cache_summary": cache_summary
        }
        
    except Exception as e:
        logger.error(f"Error preloading trip cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/offline-guidance")
async def get_offline_guidance(request: OfflineGuidanceRequest):
    """
    Get offline navigation guidance when connection is lost.
    Returns cached locations and simplified directions based on current location.
    """
    try:
        offline_cache = await get_offline_navigation_cache()
        
        user_context = request.user_context or {}
        user_context['need_type'] = request.need_type
        
        guidance = await offline_cache.get_offline_guidance(
            current_location=request.current_location,
            user_context=user_context
        )
        
        return {
            "success": True,
            "offline_guidance": guidance
        }
        
    except Exception as e:
        logger.error(f"Error getting offline guidance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache-status")
async def get_cache_status():
    """
    Get current status of the offline navigation cache.
    Shows what's cached and when it was last updated.
    """
    try:
        offline_cache = await get_offline_navigation_cache()
        
        # Get cache statistics from database
        cache_stats = await offline_cache.mongo_manager.db.offline_navigation_cache.count_documents({})
        last_update = await offline_cache._get_last_cache_update()
        
        return {
            "success": True,
            "cache_status": {
                "total_cached_trips": cache_stats,
                "last_update": last_update,
                "cache_expiry_hours": offline_cache.cache_expiry_hours,
                "max_cached_locations": offline_cache.max_cached_locations
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 