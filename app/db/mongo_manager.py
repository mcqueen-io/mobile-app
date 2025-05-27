from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from typing import Optional
import os
from datetime import datetime

class MongoManager:
    _instance = None
    _client: Optional[AsyncIOMotorClient] = None
    _db = None

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
            for collection_name in ['users', 'events', 'preferences_history']:
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
            
            logging.info("Successfully created collections and indexes")
            
        except Exception as e:
            logging.error(f"Error creating collections: {str(e)}")
            raise

    async def create_user(self, user_data: dict) -> str:
        """Create a new user with proper timestamps"""
        try:
            user_data.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            result = await self.db.users.insert_one(user_data)
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
        """Suggest connections using graph lookup"""
        try:
            pipeline = [
                {
                    '$match': {'_id': user_id}
                },
                {
                    '$graphLookup': {
                        'from': 'users',
                        'startWith': '$relationships.user_id',
                        'connectFromField': 'relationships.user_id',
                        'connectToField': '_id',
                        'as': 'suggested_connections',
                        'maxDepth': max_depth,
                        'depthField': 'depth'
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

# Create singleton instance
_mongo_manager = None

async def get_mongo_manager() -> MongoManager:
    """Get or initialize singleton instance of MongoManager"""
    global _mongo_manager
    if _mongo_manager is None:
        _mongo_manager = MongoManager()
        await _mongo_manager.initialize() # Await initialization
    return _mongo_manager 