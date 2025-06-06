import asyncio
import logging
from datetime import datetime
from app.db.mongo_manager import get_mongo_manager
from app.services.unified_service import get_unified_service
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_families():
    """Set up two test families with some shared experiences"""
    mongo_manager = await get_mongo_manager()
    
    # Clean up previous test data
    db = mongo_manager.db
    await db.users.drop()
    await db.family_trees.drop()
    await db.memories.drop()
    
    # Create collections with indexes
    await mongo_manager.create_collections()
    
    # Create Smith Family
    smith_family = {
        "family_tree_id": "smith_family_001",
        "name": "Smith Family",
        "members": ["john_smith", "mary_smith", "jimmy_smith"]
    }
    await mongo_manager.create_family_tree(smith_family)
    
    # Create Johnson Family
    johnson_family = {
        "family_tree_id": "johnson_family_001",
        "name": "Johnson Family",
        "members": ["bob_johnson", "sarah_johnson", "emma_johnson"]
    }
    await mongo_manager.create_family_tree(johnson_family)
    
    # Create users for both families
    users = [
        {
            "id": "john_smith",
            "username": "johnsmith",
            "name": "John Smith",
            "family_tree_id": "smith_family_001",
            "voice_id": "voice_john_smith"
        },
        {
            "id": "mary_smith",
            "username": "marysmith",
            "name": "Mary Smith",
            "family_tree_id": "smith_family_001",
            "voice_id": "voice_mary_smith"
        },
        {
            "id": "jimmy_smith",
            "username": "jimmysmith",
            "name": "Jimmy Smith",
            "family_tree_id": "smith_family_001",
            "voice_id": "voice_jimmy_smith"
        },
        {
            "id": "bob_johnson",
            "username": "bobjohnson",
            "name": "Bob Johnson",
            "family_tree_id": "johnson_family_001",
            "voice_id": "voice_bob_johnson"
        },
        {
            "id": "sarah_johnson",
            "username": "sarahjohnson",
            "name": "Sarah Johnson",
            "family_tree_id": "johnson_family_001",
            "voice_id": "voice_sarah_johnson"
        },
        {
            "id": "emma_johnson",
            "username": "emmajohnson",
            "name": "Emma Johnson",
            "family_tree_id": "johnson_family_001",
            "voice_id": "voice_emma_johnson"
        }
    ]
    
    for user in users:
        await mongo_manager.create_user(user)
    
    # Create shared experience (trip) between families
    trip_memory = {
        "memory_id": "shared_trip_001",
        "type": "shared_experience",
        "content": "Annual summer camping trip at Yellowstone with the Johnson family",
        "created_by": "john_smith",
        "timestamp": datetime.now(),
        "visibility": {
            "type": "shared_experience",
            "shared_with": ["smith_family_001", "johnson_family_001"],
            "participants": ["john_smith", "mary_smith", "jimmy_smith", "bob_johnson", "sarah_johnson", "emma_johnson"]
        }
    }
    await mongo_manager.create_memory(trip_memory)
    
    # Create family-specific memories (not shared)
    smith_memory = {
        "memory_id": "smith_private_001",
        "type": "family_event",
        "content": "Smith family dinner at home",
        "created_by": "mary_smith",
        "timestamp": datetime.now(),
        "visibility": {
            "type": "family_only",
            "shared_with": ["smith_family_001"]
        }
    }
    await mongo_manager.create_memory(smith_memory)
    
    johnson_memory = {
        "memory_id": "johnson_private_001",
        "type": "family_event",
        "content": "Johnson family movie night",
        "created_by": "sarah_johnson",
        "timestamp": datetime.now(),
        "visibility": {
            "type": "family_only",
            "shared_with": ["johnson_family_001"]
        }
    }
    await mongo_manager.create_memory(johnson_memory)
    
    logger.info("Test families setup complete with shared and private memories")

async def test_shared_experience_context_and_response():
    """Test how context is gathered and used by the AI for shared experiences vs private family events"""
    unified_service = await get_unified_service()
    gemini = await get_gemini_wrapper()
    
    # Test 1: Query about shared trip (should include both families context)
    user_ids_shared = ["john_smith", "bob_johnson"]
    user_input_shared = "Remember our camping trip at Yellowstone?"
    trip_context = await unified_service.get_relevant_context(
        user_ids_shared,
        user_input_shared
    )
    logger.info(f"Shared trip context: {trip_context}")
    assert any(m.get('memory_id') == 'shared_trip_001' for m in trip_context.get('relevant_memories_search_results', [])), "Shared context should include shared trip memory ID"

    # Get AI response for the shared trip query
    if 'user_ids' not in trip_context:
        trip_context['user_ids'] = user_ids_shared
    response_shared = await gemini.generate_response(
        user_input=user_input_shared,
        context=trip_context
    )
    logger.info(f"Shared trip AI response: {response_shared}")
    # Assert AI response mentions the trip based on context
    assert isinstance(response_shared, str) and ("Yellowstone" in response_shared or "camping trip" in response_shared), "AI response should acknowledge the shared trip based on context"
    # In a real scenario, you'd assert for content related to the trip using the memory data

    logger.info("Shared experience test passed.")

    # Test 2: Query about private family event (should only include own family context)
    user_ids_smith = ["john_smith"]
    user_input_smith = "What did we do for dinner last week?"
    smith_context = await unified_service.get_relevant_context(
        user_ids_smith,
        user_input_smith
    )
    logger.info(f"Smith family context: {smith_context}")
    # Assert context includes the Smith private memory ID but not the Johnson private memory ID
    assert any(m.get('memory_id') == 'smith_private_001' for m in smith_context.get('relevant_memories_search_results', [])), "Smith context should include Smith private memory ID"
    assert not any(m.get('memory_id') == 'john_private_001' for m in smith_context.get('relevant_memories_search_results', [])), "Smith context should not include Johnson private memory ID"

    # Get AI response for the private Smith event query
    if 'user_ids' not in smith_context:
        smith_context['user_ids'] = user_ids_smith
    response_smith = await gemini.generate_response(
        user_input=user_input_smith,
        context=smith_context
    )
    logger.info(f"Smith family AI response: {response_smith}")
    # Assert AI response mentions the Smith family dinner based on context
    assert isinstance(response_smith, str) and ("dinner" in response_smith or "last week" in response_smith), "AI response should acknowledge the Smith family dinner based on context"
    # In a real scenario, you'd assert for content related to the Smith dinner

    logger.info("Private Smith family event test passed.")

    # Test 3: Query about unrelated families (should not share private memories)
    user_ids_unrelated = ["john_smith", "bob_johnson"]
    user_input_unrelated = "What did you do last weekend?"
    unrelated_context = await unified_service.get_relevant_context(
        user_ids_unrelated,
        user_input_unrelated
    )
    logger.info(f"Unrelated context: {unrelated_context}")
    # Assert context includes neither private memory ID
    assert not any(m.get('memory_id') == 'smith_private_001' for m in unrelated_context.get('relevant_memories_search_results', [])), "Unrelated context should not include private Smith memory ID"
    assert not any(m.get('memory_id') == 'johnson_private_001' for m in unrelated_context.get('relevant_memories_search_results', [])), "Unrelated context should not include private Johnson memory ID"

    # Get AI response for the unrelated query
    if 'user_ids' not in unrelated_context:
        unrelated_context['user_ids'] = user_ids_unrelated
    response_unrelated = await gemini.generate_response(
        user_input=user_input_unrelated,
        context=unrelated_context
    )
    logger.info(f"Unrelated AI response: {response_unrelated}")
    # Assert AI response is generic and doesn't mention private details
    assert isinstance(response_unrelated, str) and \
           ("Smith family dinner" not in response_unrelated and "Johnson family movie night" not in response_unrelated), \
           "AI response should be generic and not mention private family details"

    logger.info("Unrelated families test passed.")

async def test_web_search():
    """Test how the AI handles web search queries"""
    unified_service = await get_unified_service()
    gemini = await get_gemini_wrapper()

    # Test web search query
    user_input_web = "What is the weather like in New York?"
    web_context = await unified_service.get_relevant_context(
        ["john_smith"],  # Example user ID
        user_input_web
    )
    logger.info(f"Web search context: {web_context}")

    # Get AI response for the web search query
    response_web = await gemini.generate_response(
        user_input=user_input_web,
        context=web_context
    )
    logger.info(f"Web search AI response: {response_web}")

    # Check if the response is a function call
    if isinstance(response_web, list) and len(response_web) > 0 and 'name' in response_web[0]:
        # Handle the function call (e.g., execute the web search function)
        logger.info("AI returned a function call for web search.")
        # Here you would typically execute the web search function and get the actual weather information
        # For now, we will just log it
        logger.info(f"Function call details: {response_web}")
        # Assert that the function call is for web_search
        assert response_web[0]['name'] == 'web_search', "AI should return a web_search function call"
    else:
        # Assert AI response is relevant to the web search query
        assert isinstance(response_web, str) and "weather" in response_web.lower(), "AI response should acknowledge the web search query"

    logger.info("Web search test passed.")

async def main():
    await setup_test_families()
    await test_shared_experience_context_and_response()
    await test_web_search()
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 