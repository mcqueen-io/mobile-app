import logging
from typing import Dict, Any, List, Optional
from app.services.unified_service import get_unified_service, UnifiedService
from app.core.cache_manager import get_cache_manager, CacheManager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AIWrapper:
    """
    Wrapper for AI model interactions.
    Handles context preparation and response generation.
    """
    def __init__(self, unified_service: UnifiedService, cache_manager: CacheManager):
        self.unified_service = unified_service
        self.cache = cache_manager

    @property
    def cache_decorator(self):
        """Get the cache decorator from the cache manager"""
        return self.cache.cache_decorator

    @cache_decorator("ai_response", timedelta(minutes=5))
    async def generate_response(self, user_ids: List[str], user_input: str) -> Dict[str, Any]:
        """Generate AI response with context"""
        try:
            # Get relevant context
            context = await self.unified_service.get_relevant_context(user_ids, user_input)
            
            # TODO: Implement actual AI model call here
            # For now, return a mock response
            return {
                "response": f"Mock response for input: {user_input}",
                "context_used": context
            }
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return {
                "error": "Failed to generate response",
                "details": str(e)
            }

    async def search_memories(self, user_ids: List[str], query: str) -> Dict[str, Any]:
        """Search memories through unified service"""
        return await self.unified_service.search_memories(user_ids, query)

# Singleton instance
_ai_wrapper = None

async def get_ai_wrapper() -> AIWrapper:
    """Get singleton instance of AIWrapper"""
    global _ai_wrapper
    if _ai_wrapper is None:
        unified_service = await get_unified_service()
        cache_manager = await get_cache_manager()
        _ai_wrapper = AIWrapper(unified_service, cache_manager)
    return _ai_wrapper 