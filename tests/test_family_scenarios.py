import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.db.mongo_manager import get_mongo_manager
from app.services.context_service import ContextService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scenarios():
    """Test various family scenarios with the AI"""
    mongo_manager = await get_mongo_manager()
    context_service = ContextService(mongo_manager)
    gemini = await get_gemini_wrapper()
    
    # Scenario 1: Son gets a new car and connects it
    print("\n=== Scenario 1: Son gets a new car ===")
    new_car_memory = {
        "memory_id": "mem_003",
        "user_id": "son_123",
        "family_tree_id": "smith_family_001",
        "type": "individual",
        "content": "Got my new Tesla Model 3! First electric car in the family.",
        "created_by": "son_123",
        "timestamp": datetime.now(),
        "metadata": {
            "location": "Tesla Dealership",
            "tags": ["car", "tesla", "electric", "milestone"],
            "car_details": {
                "brand": "Tesla",
                "model": "Model 3",
                "color": "red",
                "vin": "TESLA123456"
            }
        }
    }
    await mongo_manager.create_memory(new_car_memory)
    
    # Test AI response about the new car
    response = await gemini.generate_response(
        user_id="son_123",
        user_input="I just got a new Tesla! What do you think?",
        context=await context_service.get_relevant_context(["son_123"], "I just got a new Tesla! What do you think?")
    )
    print(f"AI Response about new car: {response}")
    
    # Scenario 2: Son adds girlfriend to family tree
    print("\n=== Scenario 2: Adding girlfriend ===")
    girlfriend = {
        "user_id": "gf_123",
        "name": "Emma Wilson",
        "email": "emma.wilson@example.com",
        "family_tree_id": "smith_family_001",
        "preferences": {
            "favorite_music_genre": "indie",
            "favorite_restaurant": "Vegan",
            "car_preferences": {
                "brand": "Tesla",
                "model": "Model Y",
                "color": "white"
            }
        }
    }
    await mongo_manager.create_user(girlfriend)
    
    # Create shared memory
    shared_memory = {
        "memory_id": "mem_004",
        "user_id": "son_123",
        "family_tree_id": "smith_family_001",
        "type": "shared",
        "content": "First road trip with Emma in the new Tesla",
        "created_by": "son_123",
        "timestamp": datetime.now(),
        "metadata": {
            "location": "Coastal Highway",
            "participants": ["son_123", "gf_123"],
            "tags": ["road trip", "tesla", "couple"],
            "shared_with": ["gf_123"]
        }
    }
    await mongo_manager.create_memory(shared_memory)
    
    # Test AI response about shared experiences
    response = await gemini.generate_response(
        user_id="son_123",
        user_input="What memories do Emma and I share?",
        context=await context_service.get_relevant_context(["son_123", "gf_123"], "What memories do Emma and I share?")
    )
    print(f"AI Response about shared memories: {response}")
    
    # Scenario 3: Memory reflection with rich preferences
    print("\n=== Scenario 3: Memory reflection ===")
    response = await gemini.generate_response(
        user_id="son_123",
        user_input="What kind of music would be perfect for our next road trip?",
        context=await context_service.get_relevant_context(["son_123", "gf_123"], "What kind of music would be perfect for our next road trip?")
    )
    print(f"AI Response about music preferences: {response}")
    
    # Scenario 4: Web search integration
    print("\n=== Scenario 4: Web search ===")
    response = await gemini.generate_response(
        user_id="son_123",
        user_input="What are the best charging stations along the Pacific Coast Highway?",
        context=await context_service.get_relevant_context(["son_123"], "What are the best charging stations along the Pacific Coast Highway?")
    )
    print(f"AI Response with web search: {response}")

if __name__ == "__main__":
    asyncio.run(test_scenarios()) 