import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from app.db.mongo_manager import get_mongo_manager
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_family():
    """Set up a test family tree with 5 members and their relationships"""
    mongo_manager = await get_mongo_manager()
    
    # Drop collections if they exist to ensure a clean test state
    for collection_name in ['users', 'memories']:
        if collection_name in await mongo_manager.db.list_collection_names():
            logger.info(f"Dropping existing collection: {collection_name}")
            await mongo_manager.db.drop_collection(collection_name)

    # Create a text index on the 'content' field of the memories collection
    logger.info("Creating text index on 'memories.content'...")
    await mongo_manager.db.memories.create_index([('content', 'text')])
    logger.info("Text index created.")

    # Create family members
    family_members = [
        {
            "_id": "dad_123",
            "user_id": "dad_123",
            "name": "John Smith",
            "username": "john_smith",
            "email": "john.smith@example.com",
            "family_tree_id": "smith_family_001",
            "preferences": {
                "favorite_music_genre": "classic rock",
                "favorite_restaurant": "Italian",
                "home_location": "123 Main St",
                "car_preferences": {
                    "brand": "Toyota",
                    "model": "Camry",
                    "color": "silver"
                }
            },
            "voice_id": "voice_dad_123"
        },
        {
            "_id": "mom_123",
            "user_id": "mom_123",
            "name": "Mary Smith",
            "username": "mary_smith",
            "email": "mary.smith@example.com",
            "family_tree_id": "smith_family_001",
            "preferences": {
                "favorite_music_genre": "jazz",
                "favorite_restaurant": "Thai",
                "home_location": "123 Main St",
                "car_preferences": {
                    "brand": "Honda",
                    "model": "CR-V",
                    "color": "blue"
                }
            },
            "voice_id": "voice_mom_123"
        },
        {
            "_id": "son_123",
            "user_id": "son_123",
            "name": "Mike Smith",
            "username": "mike_smith",
            "email": "mike.smith@example.com",
            "family_tree_id": "smith_family_001",
            "preferences": {
                "favorite_music_genre": "hip hop",
                "favorite_restaurant": "Mexican",
                "car_preferences": {
                    "brand": "Tesla",
                    "model": "Model 3",
                    "color": "red"
                }
            },
            "voice_id": "voice_son_123"
        },
        {
            "_id": "daughter_123",
            "user_id": "daughter_123",
            "name": "Sarah Smith",
            "username": "sarah_smith",
            "email": "sarah.smith@example.com",
            "family_tree_id": "smith_family_001",
            "preferences": {
                "favorite_music_genre": "pop",
                "favorite_restaurant": "Japanese",
                "car_preferences": {
                    "brand": "Subaru",
                    "model": "Crosstrek",
                    "color": "green"
                }
            },
            "voice_id": "voice_daughter_123"
        },
        {
            "_id": "youngest_123",
            "user_id": "youngest_123",
            "name": "Tom Smith",
            "username": "tom_smith",
            "email": "tom.smith@example.com",
            "family_tree_id": "smith_family_001",
            "preferences": {
                "favorite_music_genre": "electronic",
                "favorite_restaurant": "Pizza",
                "car_preferences": {
                    "brand": "Ford",
                    "model": "Mustang",
                    "color": "black"
                }
            },
            "voice_id": "voice_youngest_123"
        }
    ]

    # Insert family members
    for member in family_members:
        await mongo_manager.create_user(member)
        logger.info(f"Created user: {member['name']}")

    # Create some initial memories
    memories = [
        {
            "memory_id": "mem_001",
            "user_id": "dad_123",
            "family_tree_id": "smith_family_001",
            "type": "family",
            "content": "Family vacation to Hawaii last summer",
            "created_by": "dad_123",
            "timestamp": datetime.now() - timedelta(days=180),
            "metadata": {
                "location": "Hawaii",
                "participants": ["dad_123", "mom_123", "son_123", "daughter_123", "youngest_123"],
                "tags": ["vacation", "beach", "family"]
            }
        },
        {
            "memory_id": "mem_002",
            "user_id": "son_123",
            "family_tree_id": "smith_family_001",
            "type": "individual",
            "content": "Got my driver's license",
            "created_by": "son_123",
            "timestamp": datetime.now() - timedelta(days=30),
            "metadata": {
                "location": "DMV",
                "tags": ["milestone", "driving"]
            }
        }
    ]

    # Insert memories
    for memory in memories:
        await mongo_manager.create_memory(memory)
        logger.info(f"Created memory: {memory['content']}")

    logger.info("Test family tree setup complete!")

if __name__ == "__main__":
    asyncio.run(setup_test_family()) 