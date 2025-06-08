import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from typing import Optional, Dict, List, Any
import os
from datetime import datetime
import asyncio
import time
from app.core.cache_manager import get_cache_manager, CacheManager
from bson import ObjectId

logger = logging.getLogger(__name__)

class MongoManager:
    _instance = None
    _client: Optional[AsyncIOMotorClient] = None
    _db = None
    _family_tree_cache: Dict[str, Dict[str, set]] = {}  # Cache: {family_id: {user_id: {related_user_ids}}}
    _memory_cache = {}  # LRU cache for frequently accessed memories
    _cache_ttl = 300  # 5 minutes cache TTL
    _last_cache_update: float = 0
    _cache_lock = asyncio.Lock()  # Add missing cache lock

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoManager, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._init_lock = asyncio.Lock()
        return cls._instance

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._init_lock = asyncio.Lock()

    async def ensure_initialized(self):
        """Ensure the manager is initialized before use"""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await self.initialize()

    async def initialize(self):
        """Initialize MongoDB connection"""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    try:
                        # Get MongoDB URI from environment variable or use default
                        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
                        
                        # Initialize async client
                        self._client = AsyncIOMotorClient(
                            mongo_uri,
                            maxPoolSize=50,
                            minPoolSize=10,
                            maxIdleTimeMS=30000,
                            waitQueueTimeoutMS=10000,
                            connectTimeoutMS=20000,
                            serverSelectionTimeoutMS=5000
                        )
                        
                        # Get database
                        self._db = self._client.family_assistant
                        
                        # Test connection
                        await self._client.admin.command('ping')
                        logger.info("Successfully connected to MongoDB")
                        
                        self._initialized = True
                    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                        logger.error(f"Failed to connect to MongoDB: {str(e)}")
                        raise

    async def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")

    @property
    def db(self):
        """Get database instance"""
        if not self._initialized:
            raise ConnectionFailure("MongoDB client not initialized. Call initialize() first.")
        return self._db

    async def create_collections(self):
        """Create collections with proper indexes"""
        try:
            # Ensure db is accessed after initialization
            db = self.db

            # Drop collections if they exist to ensure a clean test state
            for collection_name in ['users', 'events', 'preferences_history', 'memories']:
                if collection_name in await db.list_collection_names():
                    logger.info(f"Dropping existing collection: {collection_name}")
                    await db.drop_collection(collection_name)

            # Users collection with enhanced indexes
            await db.create_collection('users')
            user_indexes = [
                IndexModel([('voice_id', ASCENDING)], unique=True),
                IndexModel([('relationships.user_id', ASCENDING)]),
                IndexModel([('relationships.type', ASCENDING)]),
                IndexModel([('name', 'text')]),
                IndexModel([('preferences.cuisine', ASCENDING)]),
                IndexModel([('preferences.genre', ASCENDING)]),
                IndexModel([('activities', ASCENDING)]),
                IndexModel([('voice_profile.embedding_version', ASCENDING)])
            ]
            await db.users.create_indexes(user_indexes)
            
            # Preferences history collection
            await db.create_collection('preferences_history')
            pref_history_indexes = [
                IndexModel([('user_id', ASCENDING), ('timestamp', DESCENDING)]),
                IndexModel([('preference_type', ASCENDING)]),
                IndexModel([('timestamp', DESCENDING)])
            ]
            await db.preferences_history.create_indexes(pref_history_indexes)
            
            # Events collection
            await db.create_collection('events')
            event_indexes = [
                IndexModel([('datetime', ASCENDING)]),
                IndexModel([('participants', ASCENDING)]),
                IndexModel([('status', ASCENDING)]),
                IndexModel([('created_at', ASCENDING)], expireAfterSeconds=31536000)
            ]
            await db.events.create_indexes(event_indexes)
            
            # Enhanced Memories collection with family/individual support
            await db.create_collection('memories')
            memory_indexes = [
                IndexModel([('created_by', ASCENDING)]),
                IndexModel([('type', ASCENDING)]),
                IndexModel([('created_at', DESCENDING)]),
                IndexModel([('content', 'text')]),  # Text index for searching
                IndexModel([('visibility.type', ASCENDING)]),
                IndexModel([('visibility.shared_with', ASCENDING)]),
                IndexModel([('visibility.family_branch', ASCENDING)]),
                IndexModel([('metadata.tags', ASCENDING)]),
                IndexModel([('metadata.participants', ASCENDING)])
            ]
            # Create indexes. MongoDB's create_indexes is idempotent.
            await db.memories.create_indexes(memory_indexes)
            
            logger.info("Successfully created collections and indexes")
            
        except Exception as e:
            logger.error(f"Error creating collections: {str(e)}")
            raise

    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create a new user with proper timestamps and family_tree_id"""
        try:
            # Ensure family_tree_id is provided or generated (placeholder for now)
            if 'family_tree_id' not in user_data or not user_data['family_tree_id']:
                # In a real application, you'd have logic to assign/create a family_tree_id
                # For now, let's raise an error or assign a default/placeholder
                logger.warning("Creating user without family_tree_id. Assigning a placeholder.")
                # Example placeholder (replace with proper logic later)
                user_data['family_tree_id'] = 'default_family'

            # Convert id to _id for MongoDB
            if 'id' in user_data:
                user_data['_id'] = user_data.pop('id')

            user_data.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            result = await self.db.users.insert_one(user_data)
            logger.info(f"User created with ID: {result.inserted_id} and family_tree_id: {user_data['family_tree_id']}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise

    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user data with proper timestamp"""
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = await self.db.users.update_one(
                {'_id': user_id},
                {'$set': update_data}
            )
            # Invalidate cache
            await self.cache.delete(self.cache.generate_key("user", user_id))
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            raise

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        # Generate cache key
        cache_key = self.cache.generate_key("user", user_id)
        
        # Try to get from cache
        cached_user = await self.cache.get(cache_key)
        if cached_user:
            return cached_user

        try:
            user = await self.db.users.find_one({'_id': user_id})
            
            # Cache the result
            if user:
                await self.cache.set(cache_key, user)
            
            return user
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            raise

    async def get_user_by_voice_id(self, voice_id: str) -> Optional[dict]:
        """Get user by voice ID"""
        try:
            return await self.db.users.find_one({'voice_id': voice_id})
        except Exception as e:
            logger.error(f"Error getting user by voice ID: {str(e)}")
            raise

    async def add_relationship(self, user_id: str, related_user_id: str, 
                             relationship_type: str, metadata: dict = None) -> bool:
        """Add a relationship between users"""
        try:
            relationship = {
                'user_id': related_user_id,
                'type': relationship_type,
                'since': datetime.utcnow(),
                'metadata': metadata or {}
            }
            
            result = await self.db.users.update_one(
                {'_id': user_id},
                {
                    '$push': {'relationships': relationship},
                    '$set': {'updated_at': datetime.utcnow()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error adding relationship: {str(e)}")
            raise

    async def update_preferences(self, user_id: str, preferences: dict) -> bool:
        """Update user preferences"""
        try:
            result = await self.db.users.update_one(
                {'_id': user_id},
                {
                    '$set': {
                        'preferences': preferences,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
            raise

    async def create_event(self, event_data: dict) -> str:
        """Create a new event"""
        try:
            event_data.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            result = await self.db.events.insert_one(event_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            raise

    async def track_preference_change(self, user_id: str, preference_type: str, old_value: dict, new_value: dict):
        """Track preference changes in history"""
        try:
            history_entry = {
                'user_id': user_id,
                'preference_type': preference_type,
                'old_value': old_value,
                'new_value': new_value,
                'timestamp': datetime.utcnow()
            }
            await self.db.preferences_history.insert_one(history_entry)
        except Exception as e:
            logger.error(f"Error tracking preference change: {str(e)}")
            raise

    async def get_preference_history(self, user_id: str, preference_type: Optional[str] = None, 
                                   limit: int = 10) -> list:
        """Get preference change history"""
        try:
            query = {'user_id': user_id}
            if preference_type:
                query['preference_type'] = preference_type
                
            cursor = self.db.preferences_history.find(query)
            cursor.sort('timestamp', -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Error getting preference history: {str(e)}")
            raise

    async def suggest_connections(self, user_id: str, max_depth: int = 2, 
                                min_common_interests: int = 2) -> list:
        """Suggest connections using graph lookup, scoped by family_tree_id"""
        try:
            user = await self.get_user(user_id)
            if not user or 'family_tree_id' not in user:
                 logger.warning(f"User {user_id} not found or missing family_tree_id for connection suggestions.")
                 return []

            family_id = user['family_tree_id']

            pipeline = [
                {
                    '$match': {
                        '_id': user_id,
                        'family_tree_id': family_id # Match user within their family
                    }
                },
                {
                    '$graphLookup': {
                        'from': 'users',
                        'startWith': '$relationships.user_id',
                        'connectFromField': 'relationships.user_id',
                        'connectToField': '_id',
                        'as': 'suggested_connections',
                        'maxDepth': max_depth,
                        'depthField': 'depth',
                        # Add a filter to constrain graph traversal within the same family
                         'restrictSearchWithAlias': {'family_tree_id': family_id}
                    }
                },
                {
                    '$unwind': '$suggested_connections'
                },
                {
                    '$match': {
                        'suggested_connections._id': {'$ne': user_id}
                    }
                },
                {
                    '$project': {
                        'suggested_user': '$suggested_connections',
                        'depth': '$suggested_connections.depth',
                        'common_interests': {
                            '$size': {
                                '$setIntersection': [
                                    '$preferences.cuisine',
                                    '$suggested_connections.preferences.cuisine'
                                ]
                            }
                        }
                    }
                },
                {
                    '$match': {
                        'common_interests': {'$gte': min_common_interests}
                    }
                },
                {
                    '$sort': {
                        'common_interests': -1,
                        'depth': 1
                    }
                }
            ]
            
            result = await self.db.users.aggregate(pipeline).to_list(length=None)
            return result
        except Exception as e:
            logger.error(f"Error suggesting connections: {str(e)}")
            raise

    async def find_similar_voice_profiles(self, embedding_version: str, 
                                        quality_threshold: float = 0.8) -> list:
        """Find users with similar voice profiles"""
        try:
            pipeline = [
                {
                    '$match': {
                        'voice_profile.embedding_version': embedding_version,
                        'voice_profile.quality_score': {'$gte': quality_threshold}
                    }
                },
                {
                    '$project': {
                        'name': 1,
                        'voice_id': 1,
                        'voice_profile': 1
                    }
                }
            ]
            
            result = await self.db.users.aggregate(pipeline).to_list(length=None)
            return result
        except Exception as e:
            logger.error(f"Error finding similar voice profiles: {str(e)}")
            raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            result = await self.db.users.delete_one({'_id': user_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            raise

    async def create_memory(self, memory_data: Dict) -> str:
        """Create a new memory"""
        # Validate using schema
        # self.validator.validate_memory(memory_data) # Uncomment and implement schema validation if needed

        # Get the family_tree_id from the user who created the memory
        created_by_user_id = memory_data.get('created_by')
        if not created_by_user_id:
            raise ValueError("Memory data must include 'created_by' user_id")

        user = await self.get_user(created_by_user_id)
        if not user or 'family_tree_id' not in user:
            raise ValueError(f"User {created_by_user_id} not found or has no associated family_tree_id.")
        memory_data['family_tree_id'] = user['family_tree_id']

        # Assign a new ObjectId if not provided (e.g., for manually created memories)
        if 'memory_id' not in memory_data:
            memory_data['memory_id'] = str(ObjectId())
            
        # Ensure _id is set for MongoDB. If memory_id is provided, use it for _id.
        memory_data['_id'] = memory_data.get('memory_id', str(ObjectId()))

        # Ensure timestamps exist
        if 'created_at' not in memory_data:
            memory_data['created_at'] = datetime.utcnow()
        if 'updated_at' not in memory_data:
            memory_data['updated_at'] = datetime.utcnow()

        # Set default visibility based on type (simplified)
        if 'visibility' not in memory_data:
            memory_data['visibility'] = 'family' if memory_data.get('type') == 'family' else 'private'

        result = await self.db.memories.insert_one(memory_data)
        logging.info(f"Memory created with ID: {result.inserted_id}")
        return str(result.inserted_id)

    async def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a memory by its ID"""
        return await self.db.memories.find_one({'_id': memory_id})

    async def search_memories(self, user_ids: List[str], query: str, family_ids: List[str], limit: int = 5) -> List[Dict]:
        """Search memories using text index, prioritizing user's own, family, and shared memories"""
        # TODO: Implement proper text search with scoring and prioritization
        # This is a simplified placeholder using basic filtering
        
        # Build query filters
        filters = []

        # Filter by user_ids (memories created by or shared with these users)
        # This requires a 'shared_with' field in memory documents
        # For now, let's just filter by 'created_by'
        if user_ids:
            filters.append({'created_by': {'$in': user_ids}})
            # TODO: Add logic for 'shared_with'

        # Filter by family_ids (memories belonging to these families)
        if family_ids:
            filters.append({'family_tree_id': {'$in': family_ids}})

        # Combine filters with OR for now - needs more sophisticated logic for relevance
        combined_filter = {'$or': filters} if filters else {}
        
        # Add text search (assuming text index is on 'content')
        if query:
            # Ensure text index exists on the 'content' field of the 'memories' collection
            # Example: db.collection('memories').create_index([('content', 'text')])
            if combined_filter:
                combined_filter['$text'] = {'$search': query}
            else:
                combined_filter = {'$text': {'$search': query}}
                
        logging.info(f"Searching memories with filter: {combined_filter}")

        # Execute query
        # The cursor will return documents as dictionaries
        cursor = self.db.memories.find(combined_filter).limit(limit)
        memories = await cursor.to_list(length=limit) # Use to_list for async driver
        
        # TODO: Rerank results based on relevance (e.g., recency, user interaction, explicit links)
        
        return memories

    async def _update_family_tree_cache(self):
        """Update family tree cache with current relationships, organized by family_tree_id"""
        async with self._cache_lock:
            current_time = time.time()
            if current_time - self._last_cache_update < self._cache_ttl:
                return

            # Clear old cache
            self._family_tree_cache.clear()
            
            # Build family tree from relationships, grouped by family_tree_id
            pipeline = [
                {
                    '$match': {
                        'family_tree_id': { '$exists': True, '$ne': None }
                    }
                },
                {
                    '$project': {
                        '_id': 1,
                        'family_tree_id': 1,
                        'relationships': 1
                    }
                }
            ]
            
            async for user in self.db.users.aggregate(pipeline):
                family_id = user['family_tree_id']
                user_id = str(user['_id'])

                if family_id not in self._family_tree_cache:
                    self._family_tree_cache[family_id] = {}

                self._family_tree_cache[family_id][user_id] = set()
                
                for rel in user.get('relationships', []):
                    if rel['type'] in ['parent', 'child', 'spouse']:
                         self._family_tree_cache[family_id][user_id].add(rel['user_id'])
            
            self._last_cache_update = current_time

    async def _get_family_members(self, user_id: str) -> set:
        """Get all family members for a user using cached tree"""
        await self._update_family_tree_cache()
        
        user = await self.get_user(user_id)
        if not user or 'family_tree_id' not in user:
            return set()

        family_id = user['family_tree_id']

        # Simple traversal within the cached tree for the user's family
        family_members = set()
        to_visit = {user_id}
        visited = set()

        family_subgraph = self._family_tree_cache.get(family_id, {})

        while to_visit:
            current_user = to_visit.pop()
            if current_user in visited:
                continue

            visited.add(current_user)
            family_members.add(current_user)

            # Add connected users from the cached subgraph
            to_visit.update(family_subgraph.get(current_user, set()))
        
        return family_members

    async def get_family_memories(self, family_ids: List[str], limit: int = 10) -> list:
        """Get recent family memories for specified families"""
        try:
            if not family_ids:
                logging.warning("get_family_memories called without family_ids.")
                return []
            # family_ids is a list of strings, so no attribute access issue here
            cursor = self.db.memories.find({
                'family_tree_id': {'$in': family_ids},
                'type': 'family',
                'visibility.type': 'family'
            })
            cursor.sort('created_at', DESCENDING).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logging.error(f"Error getting family memories for families {family_ids}: {str(e)}")
            return []

    async def share_memory(self, memory_id: str, shared_with: list) -> bool:
        """Share a memory with specific users"""
        try:
            result = await self.db.memories.update_one(
                {'_id': memory_id},
                {
                    '$set': {
                        'visibility.type': 'custom',
                        'visibility.shared_with': shared_with,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error sharing memory: {str(e)}")
            raise

    async def create_family_tree(self, family_tree_data: Dict) -> str:
        """Create a new family tree"""
        try:
            # Ensure family_tree_id is provided
            if 'family_tree_id' not in family_tree_data or not family_tree_data['family_tree_id']:
                raise ValueError("Family tree data must include 'family_tree_id'")

            # Convert id to _id for MongoDB
            if 'id' in family_tree_data:
                family_tree_data['_id'] = family_tree_data.pop('id')

            family_tree_data.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            result = await self.db.family_trees.insert_one(family_tree_data)
            logger.info(f"Family tree created with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating family tree: {str(e)}")
            raise

# Create singleton instance
_mongo_manager = None

async def get_mongo_manager() -> MongoManager:
    """Get or initialize singleton instance of MongoManager"""
    global _mongo_manager
    if _mongo_manager is None:
        # Get the CacheManager singleton
        cache_manager = await get_cache_manager()
        # Pass the cache_manager to the MongoManager constructor
        _mongo_manager = MongoManager(cache_manager)
        await _mongo_manager.initialize()
    return _mongo_manager 