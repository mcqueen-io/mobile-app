import logging
from typing import Dict, Any, Optional, List
from app.services.user_service import UserService, get_user_service
from app.db.mongo_manager import get_mongo_manager
import asyncio
import time
# Potentially import memory module later

logger = logging.getLogger(__name__)

class ContextService:
    """
    Service responsible for gathering and preparing relevant context for the AI model.
    Acts as the CPL (Context Preparation Layer).
    """
    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self.mongo_manager = get_mongo_manager()
        self._context_cache = {}
        self._cache_ttl = 60  # 1 minute cache TTL
        self._last_cache_update = 0
        self._cache_lock = asyncio.Lock()

    async def _get_cached_context(self, user_ids: List[str], user_input: str) -> Optional[Dict[str, Any]]:
        """Get cached context if available and not expired based on multiple user IDs"""
        # Sort user_ids to ensure consistent cache key
        sorted_user_ids = sorted(user_ids)
        cache_key = f"{'-'.join(sorted_user_ids)}:{hash(user_input)}"
        current_time = time.time()
        
        if cache_key in self._context_cache:
            cached_data, timestamp = self._context_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                return cached_data
        return None

    async def _update_context_cache(self, user_ids: List[str], user_input: str, context: Dict[str, Any]):
        """Update context cache with new data for multiple user IDs"""
        # Sort user_ids to ensure consistent cache key
        sorted_user_ids = sorted(user_ids)
        cache_key = f"{'-'.join(sorted_user_ids)}:{hash(user_input)}"
        self._context_cache[cache_key] = (context, time.time())

    async def _get_family_ids_for_users(self, user_ids: List[str]) -> set:
        """Helper to get unique family_tree_ids for a list of user IDs"""
        family_ids = set()
        for user_id in user_ids:
            user = await self.user_service.get_user(user_id)
            if user and hasattr(user, 'family_tree_id') and user.family_tree_id:
                family_ids.add(user.family_tree_id)
        return family_ids

    async def get_relevant_context(self, user_ids: List[str], user_input: str) -> Dict[str, Any]:
        """Optimized context gathering with caching for multiple users, prioritizing primary family context"""
        if not user_ids:
            logger.warning("ContextService: No user IDs provided for getting relevant context.")
            return {}

        # Check cache first (using sorted user_ids for cache key consistency)
        cached_context = await self._get_cached_context(user_ids, user_input)
        if cached_context:
            return cached_context

        context: Dict[str, Any] = {}
        primary_user_id = user_ids[0] # Assumption: First user is primary for prototype
        all_identified_user_ids = user_ids

        try:
            # Get the primary user's family ID
            primary_user = await self.user_service.get_user(primary_user_id)
            primary_family_id = primary_user.family_tree_id if primary_user and hasattr(primary_user, 'family_tree_id') else None

            if not primary_family_id:
                 logger.warning(f"Primary user {primary_user_id} not found or missing family_tree_id. Cannot gather primary family context.")
                 # Still proceed to gather individual context for identified users

            # Get data for all identified users in parallel
            user_data_futures = [self.user_service.get_user(uid) for uid in all_identified_user_ids]
            identified_users = [user for user in await asyncio.gather(*user_data_futures) if user]

            # Gather context based on primary family and identified individuals
            context["participating_users_info"] = [
                {
                    "user_id": str(u.id),
                    "username": u.username,
                    "name": u.name,
                    # Include individual preferences for all identified users
                    "preferences": {k: v for k, v in u.preferences.model_dump().items() if v is not None}
                } for u in identified_users
            ]

            # Gather recent family memories from the primary family
            if primary_family_id:
                 # Need a MongoManager method to get family memories by family_id
                 # Temporarily using the existing method with primary user ID - needs refinement
                 # TODO: Update MongoManager to get family memories by family_id list
                 relevant_family_memories = await self.mongo_manager.get_family_memories(
                     family_ids=[primary_family_id], # Pass primary family ID
                     limit=5 # Limit recent memories
                 )
                 if relevant_family_memories:
                     context["recent_family_memories"] = [
                         {
                             "content": m.get('content', ''),
                             "created_by": await self._get_username(m.get('created_by')), # Use helper
                             "created_at": m.get('created_at', ''),
                             "type": m.get('type', 'family') # Assume family type for these
                         } for m in relevant_family_memories
                     ]

            # Search memories relevant to the query for identified users and primary family
            # This will include: user's own, memories from primary family, shared with user
            relevant_memories_search = await self.mongo_manager.search_memories(
                 user_ids=all_identified_user_ids, # Search memories created by or shared with these users
                 family_ids=[primary_family_id] if primary_family_id else [], # Include primary family memories
                 query=user_input,
                 limit=10 # Limit search results
            )
            
            if relevant_memories_search:
                 context["relevant_memories_search_results"] = [
                     {
                         "content": m.get('content', ''),
                         "created_by": await self._get_username(m.get('created_by')), # Use helper
                         "created_at": m.get('created_at', ''),
                         "type": m.get('type', 'unknown')
                     } for m in relevant_memories_search
                 ]

            # Cache the context
            await self._update_context_cache(user_ids, user_input, context)
            return context

        except Exception as e:
            logger.error(f"ContextService: Error in context gathering for users '{user_ids}': {str(e)}")
            return context

    async def search_memories(self, user_ids: List[str], query: str) -> Dict[str, Any]:
        """Searches memories relevant to the given users, prioritizing primary family and shared memories."""
        if not user_ids:
            logger.warning("ContextService: No user IDs provided for memory search.")
            return {"memories_found": False, "query": query, "results": "No user IDs provided.", "message": "No user IDs provided for memory search."}

        primary_user_id = user_ids[0] # Assumption: First user is primary for prototype
        all_identified_user_ids = user_ids

        try:
            # Get the primary user's family ID
            primary_user = await self.user_service.get_user(primary_user_id)
            primary_family_id = primary_user.family_tree_id if primary_user and hasattr(primary_user, 'family_tree_id') else None

            if not primary_family_id:
                 logging.warning(f"Primary user {primary_user_id} not found or missing family_tree_id for memory search.")
                 # Still proceed to search memories created by or shared with the users

            # Search memories relevant to the query for identified users and primary family
            # This will include: user's own, memories from primary family, shared with user
            memory_results = await self.mongo_manager.search_memories(
                 user_ids=all_identified_user_ids, # Search memories created by or shared with these users
                 family_ids=[primary_family_id] if primary_family_id else [], # Include primary family memories
                 query=query,
                 limit=10 # Limit search results
            )

            if memory_results:
                # Format results efficiently
                formatted_memories = [
                    f"- [{m.get('type', 'unknown').upper()}] {m.get('content', 'No content')} "
                    f"(Created by: {await self._get_username(m.get('created_by'))}, Time: {m.get('created_at', 'N/A')})"
                    for m in memory_results
                ]
                
                return {
                    "memories_found": True,
                    "query": query,
                    "results": "\n".join(formatted_memories),
                    "total_count": len(memory_results)
                }
            else:
                return {
                    "memories_found": False,
                    "query": query,
                    "results": "None found.",
                    "message": "No relevant memories found for this query."
                }
        except Exception as e:
            logger.error(f"ContextService: Error searching memories for users {user_ids}: {str(e)}")
            return {
                "memories_found": False,
                "query": query,
                "results": "Error during memory search.",
                "error": str(e)
            }

    async def _get_username(self, user_id: Optional[str]) -> str:
        """Helper to get username for display in memory results"""
        if not user_id:
            return "unknown"
        try:
            user = await self.user_service.get_user(user_id)
            return user.username if user else "unknown"
        except Exception:
            return "unknown"

# Singleton instance
_context_service: Optional[ContextService] = None

async def get_context_service() -> ContextService:
    """Get singleton instance of ContextService"""
    global _context_service
    if _context_service is None:
        user_service = await get_user_service()
        _context_service = ContextService(user_service)
    return _context_service 