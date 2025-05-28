from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from typing import Optional, Dict, List
import os
from datetime import datetime
import asyncio
import time

class MongoManager:
    _instance = None
    _client: Optional[AsyncIOMotorClient] = None
    _db = None
    _family_tree_cache: Dict[str, Dict[str, set]] = {}  # Cache: {family_id: {user_id: {related_user_ids}}}
    _memory_cache = {}  # LRU cache for frequently accessed memories
    _cache_ttl = 300  # 5 minutes cache TTL

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self):
        """Initialize MongoDB client with connection pooling (async)"""
        if self._client is None:
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
                logging.info("Successfully connected to MongoDB")
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logging.error(f"Failed to connect to MongoDB: {str(e)}")
                raise

    async def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logging.info("MongoDB connection closed")

    @property
    def db(self):
        """Get database instance"""
        if self._db is None:
            logging.error("MongoDB client not initialized when accessing db property!")
            raise ConnectionFailure("MongoDB client not initialized.")
        return self._db

    async def create_collections(self):
        """Create collections with proper indexes"""
        try:
            # Ensure db is accessed after initialization
            db = self.db

            # Drop collections if they exist to ensure a clean test state
            for collection_name in ['users', 'events', 'preferences_history', 'memories']:
                if collection_name in await db.list_collection_names():
                    logging.info(f"Dropping existing collection: {collection_name}")
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
            await db.memories.create_indexes(memory_indexes)
            
            logging.info("Successfully created collections and indexes")
            
        except Exception as e:
            logging.error(f"Error creating collections: {str(e)}")
            raise

    async def create_user(self, user_data: dict) -> str:
        """Create a new user with proper timestamps and family_tree_id"""
        try:
            # Ensure family_tree_id is provided or generated (placeholder for now)
            if 'family_tree_id' not in user_data or not user_data['family_tree_id']:
                # In a real application, you'd have logic to assign/create a family_tree_id
                # For now, let's raise an error or assign a default/placeholder
                logging.warning("Creating user without family_tree_id. Assigning a placeholder.")
                # Example placeholder (replace with proper logic later)
                user_data['family_tree_id'] = 'default_family'

            user_data.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            result = await self.db.users.insert_one(user_data)
            logging.info(f"User created with ID: {result.inserted_id} and family_tree_id: {user_data['family_tree_id']}")
            return str(result.inserted_id)
        except Exception as e:
            logging.error(f"Error creating user: {str(e)}")
            raise

    async def update_user(self, user_id: str, update_data: dict) -> bool:
        """Update user data with proper timestamp"""
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = await self.db.users.update_one(
                {'_id': user_id},
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error updating user: {str(e)}")
            raise

    async def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID"""
        try:
            return await self.db.users.find_one({'_id': user_id})
        except Exception as e:
            logging.error(f"Error getting user: {str(e)}")
            raise

    async def get_user_by_voice_id(self, voice_id: str) -> Optional[dict]:
        """Get user by voice ID"""
        try:
            return await self.db.users.find_one({'voice_id': voice_id})
        except Exception as e:
            logging.error(f"Error getting user by voice ID: {str(e)}")
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
            logging.error(f"Error adding relationship: {str(e)}")
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
            logging.error(f"Error updating preferences: {str(e)}")
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
            logging.error(f"Error creating event: {str(e)}")
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
            logging.error(f"Error tracking preference change: {str(e)}")
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
            logging.error(f"Error getting preference history: {str(e)}")
            raise

    async def suggest_connections(self, user_id: str, max_depth: int = 2, 
                                min_common_interests: int = 2) -> list:
        """Suggest connections using graph lookup, scoped by family_tree_id"""
        try:
            user = await self.get_user(user_id)
            if not user or 'family_tree_id' not in user:
                 logging.warning(f"User {user_id} not found or missing family_tree_id for connection suggestions.")
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
            logging.error(f"Error suggesting connections: {str(e)}")
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
            logging.error(f"Error finding similar voice profiles: {str(e)}")
            raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            result = await self.db.users.delete_one({'_id': user_id})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Error deleting user: {str(e)}")
            raise

    async def create_memory(self, memory_data: dict) -> str:
        """Create a new memory with proper structure and linked family_tree_id"""
        try:
            # Ensure required fields
            if 'content' not in memory_data or 'type' not in memory_data or 'created_by' not in memory_data:
                raise ValueError("Memory must have content, type, and created_by fields")

            created_by_user_id = memory_data['created_by']

            # Get the creator's family_tree_id
            creator_user = await self.get_user(created_by_user_id)
            if not creator_user or 'family_tree_id' not in creator_user:
                # Handle case where creator user is not found or has no family_tree_id
                # Depending on logic, you might raise an error or assign to a default/orphan family
                logging.warning(f"Creator user {created_by_user_id} not found or missing family_tree_id. Cannot link memory to a family.")
                # Optionally assign to a default/orphan family or handle differently
                # memory_data['family_tree_id'] = 'orphan'
                pass # Or raise an error if linking is mandatory
            else:
                memory_data['family_tree_id'] = creator_user['family_tree_id']

            # Set default visibility based on type if not specified
            if 'visibility' not in memory_data:
                memory_data['visibility'] = {
                    'type': 'family' if memory_data['type'] == 'family' else 'individual',
                    'shared_with': [],
                    'family_branch': None # This might be used later for sub-family sharing
                }

            # Add timestamps
            memory_data.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })

            result = await self.db.memories.insert_one(memory_data)
            logging.info(f"Memory created with ID: {result.inserted_id} and family_tree_id: {memory_data.get('family_tree_id', 'N/A')}")
            return str(result.inserted_id)
        except Exception as e:
            logging.error(f"Error creating memory: {str(e)}")
            raise

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
            return set() # User not found or no family_tree_id

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

    async def search_memories(self, user_ids: List[str], query: str, family_ids: List[str], limit: int = 5) -> list:
        """Optimized memory search for multiple users across specified families"""
        try:
            if not user_ids and not family_ids:
                logging.warning("search_memories called without user_ids or family_ids.")
                return []

            # Build query based on visibility rules and provided user/family IDs
            visibility_query = {
                '$or': [
                    # Memories created by any of the specified users
                    {'created_by': {'$in': user_ids}},
                    # Family memories from any of the specified families
                    {'family_tree_id': {'$in': family_ids}, 'type': 'family', 'visibility.type': 'family'},
                     # Memories explicitly shared with any of the specified users
                    {'visibility.shared_with': {'$in': user_ids}}
                ]
            }

            # Add text search if query provided
            if query:
                # Combine text search with visibility using $and
                final_query = {
                    '$and': [
                        visibility_query,
                        {'$text': {'$search': query}}
                    ]
                }
            else:
                final_query = visibility_query # No text search, just visibility filters

            # Use compound index for efficient querying
            # Note: The text index is separate, MongoDB will combine search results.
            cursor = self.db.memories.find(
                final_query,
                {'score': {'$meta': 'textScore'}} if query else {} # Include score only if text searching
            )

            # Sort by relevance (if text searching) and timestamp
            sort_criteria = [('created_at', DESCENDING)]
            if query:
                sort_criteria.insert(0, ('score', {'$meta': 'textScore'}))

            cursor.sort(sort_criteria).limit(limit)

            return await cursor.to_list(length=limit)
        except Exception as e:
            logging.error(f"Error searching memories for users {user_ids} and families {family_ids}: {str(e)}")
            # Consider a fallback if text search fails or handle differently
            return []

    async def get_family_memories(self, family_ids: List[str], limit: int = 10) -> list:
        """Get recent family memories for specified families"""
        try:
            if not family_ids:
                logging.warning("get_family_memories called without family_ids.")
                return []

            # Query family memories for the specified family IDs
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

# Create singleton instance
_mongo_manager = None

async def get_mongo_manager() -> MongoManager:
    """Get or initialize singleton instance of MongoManager"""
    global _mongo_manager
    if _mongo_manager is None:
        _mongo_manager = MongoManager()
        await _mongo_manager.initialize() # Await initialization
    return _mongo_manager 