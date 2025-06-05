import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.db.mongo_manager import get_mongo_manager
# from app.services.context_service import get_context_service
# from app.services.user_service import get_user_service
from app.services.unified_service import get_unified_service
from app.models.user import User, UserPreferences, SchedulePreferences, CommunicationPreferences
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_family():
    """Set up a test family with rich preferences"""
    mongo_manager = await get_mongo_manager()
    
    # First, drop all collections to start fresh
    await mongo_manager.create_collections()
    
    # Create father with preferences
    father = User(
        id=str(ObjectId()),  # Generate a new ObjectId
        name="John Smith",
        username="johnsmith",
        voice_id="voice_dad_001",
        family_tree_id="smith_family_001",
        preferences=UserPreferences(
            music=["classic_rock", "The Beatles", "Led Zeppelin", "Pink Floyd", "Road Trip Classics", "Sunday Morning Coffee"],
            cuisine=["italian", "Luigi's", "Pasta Palace"],
            activities=["no_spicy_food"],
            schedule=SchedulePreferences(
                preferred_times=["morning", "afternoon"],
                timezone="America/New_York"
            ),
            communication=CommunicationPreferences(
                preferred_language="en",
                formality_level=3,
                notification_preferences={"email": True, "push": True}
            )
        )
    )
    
    # Create mother with preferences
    mother = User(
        id=str(ObjectId()),  # Generate a new ObjectId
        name="Mary Smith",
        username="marysmith",
        voice_id="voice_mom_001",
        family_tree_id="smith_family_001",
        preferences=UserPreferences(
            music=["jazz", "Norah Jones", "Diana Krall", "Relaxing Jazz", "Coffee Shop Vibes"],
            cuisine=["mediterranean", "Olive Garden", "Greek Delight"],
            activities=["vegetarian"],
            schedule=SchedulePreferences(
                preferred_times=["evening", "night"],
                timezone="America/New_York"
            ),
            communication=CommunicationPreferences(
                preferred_language="en",
                formality_level=4,
                notification_preferences={"email": True, "push": False}
            )
        )
    )
    
    # Create son with preferences
    son = User(
        id=str(ObjectId()),  # Generate a new ObjectId
        name="Mike Smith",
        username="mikesmith",
        voice_id="voice_son_001",
        family_tree_id="smith_family_001",
        preferences=UserPreferences(
            music=["electronic", "Daft Punk", "The Chemical Brothers", "Workout Mix", "Party Time"],
            cuisine=["mexican", "Taco Bell", "Chipotle"],
            activities=[],
            schedule=SchedulePreferences(
                preferred_times=["afternoon", "evening"],
                timezone="America/New_York"
            ),
            communication=CommunicationPreferences(
                preferred_language="en",
                formality_level=2,
                notification_preferences={"email": False, "push": True}
            )
        )
    )
    
    # Create shared memories
    family_memory = {
        "memory_id": "mem_001",
        "user_id": father.id,  # Use the generated ID
        "family_tree_id": "smith_family_001",
        "type": "family",
        "content": "Family road trip to the beach last summer. We played classic rock and jazz, stopping at Italian and Mediterranean restaurants along the way.",
        "created_by": father.id,  # Use the generated ID
        "timestamp": datetime.now(),
        "metadata": {
            "location": "Coastal Highway",
            "participants": [father.id, mother.id, son.id],  # Use the generated IDs
            "tags": ["road trip", "beach", "family time"],
            "music_played": ["The Beatles", "Norah Jones"],
            "restaurants_visited": ["Luigi's", "Greek Delight"]
        }
    }
    
    # Insert all documents
    await mongo_manager.create_user(father.model_dump())
    await mongo_manager.create_user(mother.model_dump())
    await mongo_manager.create_user(son.model_dump())
    await mongo_manager.create_memory(family_memory)
    
    logger.info("Test family setup complete with rich preferences")
    
    # Return the user IDs for use in tests
    return {
        "father_id": father.id,
        "mother_id": mother.id,
        "son_id": son.id
    }

async def test_preference_context():
    """Test how the AI handles preference-based queries"""
    mongo_manager = await get_mongo_manager()
    # context_service = await get_context_service()
    unified_service = await get_unified_service()
    gemini = await get_gemini_wrapper()
    
    # First, set up our test family and get the IDs
    user_ids = await setup_test_family()
    
    print("\n=== Testing Preference Context Gathering ===")
    
    # Test 1: Query about music preferences
    print("\nTest 1: Music Preferences Query")
    response = await gemini.generate_response(
        user_id=user_ids["son_id"],
        user_input="What kind of music should we play for our next road trip?",
        context=await unified_service.get_relevant_context(
            [user_ids["son_id"], user_ids["father_id"], user_ids["mother_id"]], 
            "What kind of music should we play for our next road trip?"
        )
    )
    print(f"Response: {response}")
    
    # Test 2: Query about food preferences
    print("\nTest 2: Food Preferences Query")
    response = await gemini.generate_response(
        user_id=user_ids["mother_id"],
        user_input="Where should we stop for dinner?",
        context=await unified_service.get_relevant_context(
            [user_ids["mother_id"], user_ids["father_id"], user_ids["son_id"]], 
            "Where should we stop for dinner?"
        )
    )
    print(f"Response: {response}")
    
    # Test 3: Query about car preferences
    print("\nTest 3: Car Preferences Query")
    response = await gemini.generate_response(
        user_id=user_ids["father_id"],
        user_input="What temperature should I set in the car?",
        context=await unified_service.get_relevant_context(
            [user_ids["father_id"], user_ids["mother_id"], user_ids["son_id"]], 
            "What temperature should I set in the car?"
        )
    )
    print(f"Response: {response}")
    
    # Test 4: Query about shared memories with preferences
    print("\nTest 4: Shared Memories with Preferences")
    response = await gemini.generate_response(
        user_id=user_ids["son_id"],
        user_input="What music did we play on our last road trip?",
        context=await unified_service.get_relevant_context(
            [user_ids["son_id"], user_ids["father_id"], user_ids["mother_id"]], 
            "What music did we play on our last road trip?"
        )
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_preference_context()) 