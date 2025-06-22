from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import asyncio
from app.db.mongo_manager import get_mongo_manager, MongoManager
from app.models.user import User, Relationship
from app.db.schema_validator import get_schema_validator, SchemaValidator
from app.core.cache_manager import get_cache_manager, CacheManager
from app.models.data_transformer import DataTransformer

logger = logging.getLogger(__name__)

class UnifiedService:
    """
    Unified service combining user management and context gathering functionality.
    Provides a single point of interaction for user-related operations and context management.
    """
    def __init__(self, mongo_manager: MongoManager, schema_validator: SchemaValidator, cache_manager: CacheManager):
        self.db = mongo_manager
        self.validator = schema_validator
        self.cache = cache_manager
        self.transformer = DataTransformer()

    async def ensure_initialized(self):
        """Ensure all dependencies are initialized"""
        await self.db.ensure_initialized()

    # User Management Methods
    async def create_user(self, user_data: Dict) -> User:
        """Create a new user"""
        await self.ensure_initialized()
        self.validator.validate_user(user_data)
        user = User(**user_data)
        await self.db.create_user(self.transformer.pydantic_to_mongo(user))
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID with caching"""
        await self.ensure_initialized()
        user_data = await self.db.get_user(user_id)
        return self.transformer.mongo_to_pydantic(user_data, User)

    async def update_user(self, user_id: str, update_data: Dict) -> Optional[User]:
        """Update user data"""
        user = await self.get_user(user_id)
        if not user:
            return None

        user_dict = user.model_dump()
        user_dict.update(update_data)
        
        if '_id' in user_dict:
            del user_dict['_id']

        self.validator.validate_user(user_dict)
        await self.db.update_user(user_id, user_dict)
        
        # Invalidate cache
        await self.cache.delete(self.cache.generate_key("user", user_id))
        return await self.get_user(user_id)

    # Context Management Methods
    async def get_relevant_context(self, user_ids: List[str], user_input: str) -> Dict[str, Any]:
        """Get relevant context for users"""
        if not user_ids:
            logger.warning("No user IDs provided for context gathering")
            return {}

        try:
            context = {}
            primary_user_id = user_ids[0]
            all_identified_user_ids = user_ids

            # Get primary user's family ID
            primary_user = await self.get_user(primary_user_id)
            if primary_user:
                if isinstance(primary_user, dict):
                    primary_family_id = primary_user.get('family_tree_id')
                else:
                    primary_family_id = getattr(primary_user, 'family_tree_id', None)
            else:
                primary_family_id = None

            # Get all users in parallel
            user_data_futures = [self.get_user(uid) for uid in all_identified_user_ids]
            identified_users = [user for user in await asyncio.gather(*user_data_futures) if user]

            # Format user context
            context["participating_users_info"] = [
                self.transformer.format_user_context(u) for u in identified_users
            ]

            # Get family memories
            if primary_family_id:
                family_memories = await self.db.get_family_memories(
                    family_ids=[primary_family_id],
                    limit=5
                )
                if family_memories:
                    context["recent_family_memories"] = [
                        await self.transformer.format_memory(m, self._get_username)
                        for m in family_memories
                    ]

            # Search relevant memories
            relevant_memories = await self.db.search_memories(
                user_ids=all_identified_user_ids,
                family_ids=[primary_family_id] if primary_family_id else [],
                query=user_input,
                limit=10
            )
            
            if relevant_memories:
                context["relevant_memories_search_results"] = [
                    await self.transformer.format_memory(m, self._get_username)
                    for m in relevant_memories
                ]

            return context

        except Exception as e:
            logger.error(f"Error gathering context for users '{user_ids}': {str(e)}")
            return {}

    async def search_memories(self, user_ids: List[str], query: str) -> Dict[str, Any]:
        """Search memories for users"""
        if not user_ids:
            return {
                "memories_found": False,
                "query": query,
                "results": "No user IDs provided.",
                "message": "No user IDs provided for memory search."
            }

        try:
            primary_user = await self.get_user(user_ids[0])
            if primary_user:
                primary_family_id = primary_user['family_tree_id'] if isinstance(primary_user, dict) else primary_user.family_tree_id
            else:
                primary_family_id = None

            memory_results = await self.db.search_memories(
                user_ids=user_ids,
                family_ids=[primary_family_id] if primary_family_id else [],
                query=query,
                limit=10
            )

            if memory_results:
                result = {
                    "memories_found": True,
                    "query": query,
                    "results": self.transformer.format_memory_search_results(memory_results, self._get_username),
                    "total_count": len(memory_results)
                }
            else:
                result = {
                    "memories_found": False,
                    "query": query,
                    "results": "None found.",
                    "message": "No relevant memories found for this query."
                }

            return result

        except Exception as e:
            logger.error(f"Error searching memories for users {user_ids}: {str(e)}")
            return {
                "memories_found": False,
                "query": query,
                "results": "Error during memory search.",
                "error": str(e)
            }

    async def _get_username(self, user_id: Optional[str]) -> str:
        """Helper to get username for display"""
        if not user_id:
            return "unknown"
        try:
            user = await self.get_user(user_id)
            return user.username if user else "unknown"
        except Exception:
            return "unknown"

    # Additional User Management Methods
    async def add_relationship(self, user_id: str, relationship_data: Dict) -> None:
        """Add a relationship between users"""
        await self.ensure_initialized()
        await self.db.add_relationship(
            user_id=user_id,
            related_user_id=relationship_data["user_id"],
            relationship_type=relationship_data["type"],
            metadata=relationship_data.get("metadata", {}),
            since=relationship_data.get("since")
        )

    async def update_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences"""
        await self.ensure_initialized()
        return await self.db.update_preferences(user_id, preferences)

    async def get_preference_history(self, user_id: str, preference_type: Optional[str] = None, limit: int = 10) -> List:
        """Get preference change history"""
        await self.ensure_initialized()
        return await self.db.get_preference_history(user_id, preference_type, limit)

    async def suggest_connections(self, user_id: str, max_depth: int = 2, min_common_interests: int = 2) -> List:
        """Suggest connections for a user"""
        await self.ensure_initialized()
        return await self.db.suggest_connections(user_id, max_depth, min_common_interests)

# Singleton instance
_unified_service = None

async def get_unified_service() -> UnifiedService:
    """Get singleton instance of UnifiedService"""
    global _unified_service
    if _unified_service is None:
        mongo_manager = await get_mongo_manager()
        schema_validator = get_schema_validator()
        cache_manager = await get_cache_manager()
        _unified_service = UnifiedService(mongo_manager, schema_validator, cache_manager)
        
        # Apply decorators after instance creation
        _unified_service.get_user = _unified_service.cache.cache_decorator(
            "user", timedelta(minutes=5)
        )(_unified_service.get_user)
        _unified_service.get_relevant_context = _unified_service.cache.cache_decorator(
            "context", timedelta(minutes=5)
        )(_unified_service.get_relevant_context)
        _unified_service.search_memories = _unified_service.cache.cache_decorator(
            "memory_search", timedelta(minutes=5)
        )(_unified_service.search_memories)
        
    return _unified_service 