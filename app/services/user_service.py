from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from app.db.mongo_manager import get_mongo_manager, MongoManager
from app.models.user import User, Relationship
from app.db.schema_validator import get_schema_validator, SchemaValidator
import asyncio
from functools import lru_cache
import logging

class UserService:
    def __init__(self, mongo_manager: MongoManager, schema_validator: SchemaValidator):
        self.db = mongo_manager
        self.validator = schema_validator
        self._cache = {}
        self._cache_lock = asyncio.Lock()
        self._cache_ttl = timedelta(minutes=5)

    async def _get_cached(self, key: str) -> Optional[Tuple[datetime, any]]:
        """Get cached value if not expired"""
        async with self._cache_lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if datetime.utcnow() - timestamp < self._cache_ttl:
                    return value
                del self._cache[key]
        return None

    async def _set_cached(self, key: str, value: any):
        """Set cached value with timestamp"""
        async with self._cache_lock:
            self._cache[key] = (datetime.utcnow(), value)

    async def create_user(self, user_data: Dict) -> User:
        """Create a new user"""
        self.validator.validate_user(user_data)
        user = User(**user_data)
        # Convert Pydantic model to dictionary before saving to MongoDB
        await self.db.create_user(user.model_dump())
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID with caching"""
        cache_key = f"user:{user_id}"
        cached_user = await self._get_cached(cache_key)
        if cached_user:
            return cached_user

        user_data = await self.db.get_user(user_id)
        if user_data:
            # Convert MongoDB ObjectId to string for Pydantic
            if '_id' in user_data:
                 user_data['id'] = str(user_data.pop('_id'))
            user = User.model_validate(user_data)
            await self._set_cached(cache_key, user)
            return user
        return None

    async def update_user(self, user_id: str, update_data: Dict) -> Optional[User]:
        """Update user data"""
        user = await self.get_user(user_id)
        if not user:
            return None

        # Update user data using Pydantic's update method if possible, or dict update
        # Simple dict update for demonstration, more robust logic might be needed for nested fields
        user_dict = user.model_dump()
        user_dict.update(update_data)
        
        # Ensure _id is not in the update data from Pydantic
        if '_id' in user_dict:
            del user_dict['_id']

        self.validator.validate_user(user_dict)
        
        # Convert Pydantic model back to dictionary for MongoDB update
        await self.db.update_user(user_id, user_dict)

        # Fetch updated user to ensure cache consistency
        updated_user = await self.get_user(user_id)
        return updated_user

    async def update_preferences(self, user_id: str, preferences: Dict) -> Optional[User]:
        """Update user preferences with history tracking"""
        user = await self.get_user(user_id)
        if not user:
            return None

        self.validator.validate_preferences(preferences)
        
        # Track changes in preferences history
        for pref_type, new_value in preferences.items():
            old_value = user.preferences.model_dump().get(pref_type) # Use model_dump for Pydantic v2
            if old_value != new_value:
                # Note: Storing dict values directly. Consider deep copying if values are mutable.
                await self.db.track_preference_change(
                    user_id, pref_type, old_value, new_value
                )
        
        # Update preferences on the model
        current_prefs_dict = user.preferences.model_dump()
        current_prefs_dict.update(preferences)
        # Re-validate the updated preferences structure
        user.preferences = user.preferences.model_validate(current_prefs_dict)
        user.updated_at = datetime.utcnow()
        
        await self.db.update_user(user_id, user.model_dump())
        await self._set_cached(f"user:{user_id}", user)
        return user

    async def get_preference_history(self, user_id: str, 
                                   preference_type: Optional[str] = None,
                                   limit: int = 10) -> List[Dict]:
        """Get preference change history"""
        cache_key = f"pref_history:{user_id}:{preference_type}:{limit}"
        cached_history = await self._get_cached(cache_key)
        if cached_history:
            return cached_history

        history = await self.db.get_preference_history(user_id, preference_type, limit)
        await self._set_cached(cache_key, history)
        return history

    async def suggest_connections(self, user_id: str, 
                                max_depth: int = 2,
                                min_common_interests: int = 2) -> List[Dict]:
        """Suggest connections using graph lookup with caching"""
        cache_key = f"connections:{user_id}:{max_depth}:{min_common_interests}"
        cached_suggestions = await self._get_cached(cache_key)
        if cached_suggestions:
            return cached_suggestions

        # MongoDB graphLookup uses the _id field by default for connecting, but our user model uses 'id'
        # Need to ensure the graphLookup pipeline uses the correct field ('_id' in MongoDB)
        # The MongoManager.suggest_connections method already handles this mapping.
        suggestions = await self.db.suggest_connections(
            user_id, max_depth, min_common_interests
        )
        
        # Convert _id in suggested_user sub-documents to 'id' strings for consistency with User model
        for suggestion in suggestions:
            if 'suggested_user' in suggestion and '_id' in suggestion['suggested_user']:
                 suggestion['suggested_user']['id'] = str(suggestion['suggested_user'].pop('_id'))

        await self._set_cached(cache_key, suggestions)
        return suggestions

    async def find_similar_voice_profiles(self, embedding_version: str,
                                        quality_threshold: float = 0.8) -> List[Dict]:
        """Find users with similar voice profiles"""
        cache_key = f"voice_profiles:{embedding_version}:{quality_threshold}"
        cached_profiles = await self._get_cached(cache_key)
        if cached_profiles:
            return cached_profiles

        profiles = await self.db.find_similar_voice_profiles(
            embedding_version, quality_threshold
        )
        
        # Convert _id in profile sub-documents to 'id' strings
        for profile in profiles:
             if '_id' in profile:
                 profile['id'] = str(profile.pop('_id'))

        await self._set_cached(cache_key, profiles)
        return profiles

    async def add_relationship(self, user_id: str, relationship_data: Dict) -> Optional[User]:
        """Add a relationship to user"""
        user = await self.get_user(user_id)
        if not user:
            return None

        self.validator.validate_relationship(relationship_data)
        relationship = Relationship(**relationship_data)
        user.add_relationship(relationship)
        
        await self.db.update_user(user_id, user.model_dump())
        # Invalidate cache for the user since relationships changed
        async with self._cache_lock:
            cache_key = f"user:{user_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
             # Also invalidate related_users cache for this user
            related_users_cache_key = f"related_users:{user_id}:None" # assuming None for all types
            if related_users_cache_key in self._cache:
                del self._cache[related_users_cache_key]
             # Invalidate specific relationship type caches as well
            for rel_type in ["PARENT_OF", "CHILD_OF", "SPOUSE_OF", "SIBLING_OF", "CONNECTED_TO", "FRIEND_OF", "COLLEAGUE_OF"]:
                 specific_rel_cache_key = f"related_users:{user_id}:{rel_type}"
                 if specific_rel_cache_key in self._cache:
                     del self._cache[specific_rel_cache_key]

        # Fetch the updated user to return and potentially recache on next get
        updated_user = await self.get_user(user_id)
        return updated_user

    async def update_voice_profile(self, user_id: str, 
                                 embedding_version: str,
                                 quality_score: float) -> Optional[User]:
        """Update user's voice profile"""
        user = await self.get_user(user_id)
        if not user:
            return None

        user.update_voice_profile(embedding_version, quality_score)
        await self.db.update_user(user_id, user.model_dump())
        # Invalidate cache for the user
        async with self._cache_lock:
            cache_key = f"user:{user_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
        # Fetch the updated user
        updated_user = await self.get_user(user_id)
        return updated_user

    async def get_related_users(self, user_id: str, 
                              relationship_type: Optional[str] = None) -> List[User]:
        """Get users related to the given user with caching"""
        cache_key = f"related_users:{user_id}:{relationship_type}"
        cached_users = await self._get_cached(cache_key)
        if cached_users:
            return cached_users

        user = await self.get_user(user_id)
        if not user:
            return []

        related_ids = [
            rel.user_id for rel in user.relationships
            if not relationship_type or rel.type == relationship_type
        ]

        related_users = []
        for rel_id in related_ids:
            # Use get_user here to leverage its caching
            related_user = await self.get_user(rel_id)
            if related_user:
                related_users.append(related_user)

        await self._set_cached(cache_key, related_users)
        return related_users

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user and clear related caches"""
        success = await self.db.delete_user(user_id)
        if success:
            async with self._cache_lock:
                # Clear all caches related to this user
                keys_to_delete = [
                    key for key in self._cache.keys()
                    if key.startswith(f"user:{user_id}") or
                    key.startswith(f"related_users:{user_id}") or
                    key.startswith(f"connections:{user_id}") or
                    key.startswith(f"pref_history:{user_id}") or
                    # Also clear caches where this user might be a related user
                    f":{user_id}" in key # A simple heuristic, might need refinement
                ]
                for key in keys_to_delete:
                    del self._cache[key]
        return success

# Create singleton instance
_user_service: Optional[UserService] = None

async def get_user_service() -> UserService:
    """Get or initialize singleton instance of UserService"""
    global _user_service
    if _user_service is None:
        mongo_manager = await get_mongo_manager()
        schema_validator = get_schema_validator()
        _user_service = UserService(mongo_manager, schema_validator)
    return _user_service 