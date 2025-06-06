import asyncio
import sys
import os
import logging # Import logging
from typing import Dict, Any, List, Union
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env

# Add the parent directory of 'app' to the sys.path to allow importing 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.modules.ai_wrapper.gemini_wrapper import GeminiWrapper, get_gemini_wrapper
from app.core.config import settings # Import the settings object
# from django.conf import settings # Remove this incorrect import

logger = logging.getLogger(__name__) # Initialize logger

async def test_gemini_wrapper_responses():
    """Tests the GeminiWrapper's response capabilities in different scenarios."""
    logger.info("Initializing Gemini Wrapper...")
    # Note: This requires GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and
    # GOOGLE_APPLICATION_CREDENTIALS environment variables to be set.
    # Mocking might be needed for full independence from GCP.
    try:
        wrapper = await get_gemini_wrapper()
        logger.info("Wrapper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Wrapper: {e}")
        logger.error("Please ensure Google Cloud credentials environment variables are set correctly.")
        return

    # Placeholder user ID - in a real scenario, this would come from authentication/identification
    test_user_id = "test_user_123"
    logger.info(f"Using placeholder user ID: {test_user_id}")

    logger.info("\n--- Scenario 1: Simple conversational input ---")
    user_input_1 = "Hello, how are you today?"
    context_1 = {}
    logger.info(f"Input: {user_input_1}")
    # Call the chat method
    response_1 = await wrapper.chat(test_user_id, user_input_1, context=context_1)
    logger.info(f"Response: {response_1}")
    # Assertion for simple text response
    assert isinstance(response_1, str) and len(response_1) > 0, "Expected a non-empty string response for Scenario 1"

    logger.info("\n--- Scenario 2: Input asking for user preference (should trigger tool call handled internally) ---")
    user_input_2 = "What's my favorite music genre?"
    context_2 = {
        "participating_users_info": [
            {
                "user_id": test_user_id,
                "username": "TestUser",
                "name": "Test User",
                "preferences": {} # Preference not in context
            }
        ]
    }
    logger.info(f"Input: {user_input_2}")
    # Call the chat method, which will handle the tool call
    response_2 = await wrapper.chat(test_user_id, user_input_2, context=context_2)
    logger.info(f"Response: {response_2}")
    # Assertion for response after tool call (will depend on the mock/real tool result)
    assert isinstance(response_2, str) and len(response_2) > 0, "Expected a non-empty string response after preference tool call"
    # Further assertions could check if the response mentions preferences if the tool returned them

    logger.info("\n--- Scenario 3: Input asking to search memories (should trigger tool call handled internally) ---")
    user_input_3 = "Tell me about our last family trip."
    context_3 = {
         "participating_users_info": [
            {
                "user_id": test_user_id,
                "username": "TestUser",
                "name": "Test User",
                "preferences": {}
            }
        ],
        "recent_family_memories": [] # Memories not in context
    }
    logger.info(f"Input: {user_input_3}")
    # Call the chat method, which will handle the tool call
    response_3 = await wrapper.chat(test_user_id, user_input_3, context=context_3)
    logger.info(f"Response: {response_3}")
    # Assertion for response after tool call (will depend on the mock/real tool result)
    assert isinstance(response_3, str) and len(response_3) > 0, "Expected a non-empty string response after memory search tool call"
    # Further assertions could check if the response mentions trip details if the tool returned them

    logger.info("\n--- Scenario 4: Input asking to search memories, but relevant info IS in context (should get text response directly) ---")
    user_input_4 = "What did we do on our trip last summer?"
    context_4 = {
         "participating_users_info": [
            {
                "user_id": test_user_id,
                "username": "TestUser",
                "name": "Test User",
                "preferences": {}
            }
        ],
        "relevant_memories_search_results": [
             # Renamed from recent_family_memories in previous test to match UnifiedService output format
            {
                "memory_id": "mem_summer_trip",
                "content": "We went camping in the mountains last summer. It rained a lot!",
                "created_by": test_user_id,
                "created_at": "<timestamp>",
                "type": "family"
            }
        ] # Relevant memory IS in context
    }
    logger.info(f"Input: {user_input_4}")
    # Call the chat method
    response_4 = await wrapper.chat(test_user_id, user_input_4, context=context_4)
    logger.info(f"Response: {response_4}")
    # Assertion for text response based on context
    assert isinstance(response_4, str) and ("camping" in response_4 or "mountains" in response_4 or "summer trip" in response_4), "Expected text response mentioning the summer trip from context"

    logger.info("\n--- Scenario 5: Input suggesting low mood (should trigger comforting tone) ---")
    # Note: Sentiment analysis isn't implemented in CPL yet, so AI relies on text cues.
    user_input_5 = "Feeling a bit down today..."
    context_5 = {
         "participating_users_info": [
            {
                "user_id": test_user_id,
                "username": "TestUser",
                "name": "Test User",
                "preferences": {}
            }
        ]
    }
    logger.info(f"Input: {user_input_5}")
    # Call the chat method
    response_5 = await wrapper.chat(test_user_id, user_input_5, context=context_5)
    logger.info(f"Response: {response_5}")
    # Assertion for comforting tone (will depend on AI's response)
    assert isinstance(response_5, str) and len(response_5) > 0, "Expected a non-empty string response with potentially comforting tone"
    # More specific assertion here would require robust sentiment analysis or mock AI

    logger.info("\n--- Scenario 6: Input requiring web search (should trigger tool call handled internally) ---")
    user_input_6 = "What is the weather forecast for London tomorrow?"
    context_6 = {} # No relevant context for weather forecast
    logger.info(f"Input: {user_input_6}")
    # Call the chat method, which will handle the tool call
    response_6 = await wrapper.chat(test_user_id, user_input_6, context=context_6)
    logger.info(f"Response: {response_6}")
    # Assertion for response after web search (will depend on the real search results and AI's synthesis)
    assert isinstance(response_6, str) and ("weather" in response_6 or "forecast" in response_6 or "London" in response_6), "Expected a string response mentioning weather/forecast for London after web search"

    logger.info("\n--- Test complete ---")
    logger.info("Review the output above to see how the AI responded in each scenario.")
    logger.info("Specifically, check if:")
    logger.info("- Simple inputs get text responses.")
    logger.info("- Queries requiring data (preference, memory, web) trigger tool calls that are handled internally, resulting in a final text response.")
    logger.info("- Queries where data is ALREADY in context lead to text responses directly.")
    logger.info("- The response for Scenario 5 reflects a comforting tone based on the user input.")
    logger.info("- The response for Scenario 6 provides information related to the weather forecast for London.")

# Boilerplate to run the async function
if __name__ == "__main__":
    # Ensure environment variables are loaded if using .env file
    # from dotenv import load_dotenv # Uncomment if using dotenv
    # load_dotenv() # Uncomment if using dotenv
    
    # Set placeholder env vars if not using .env and running locally for quick test setup
    # WARNING: Running this without valid GCP credentials and Search API keys
    # will result in errors when initializing the wrapper or executing web search.
    if not os.getenv('GOOGLE_CLOUD_PROJECT'):
        # Replace with your actual project ID and location if testing with GCP
        os.environ['GOOGLE_CLOUD_PROJECT'] = settings.GOOGLE_CLOUD_PROJECT # Use settings value
        os.environ['GOOGLE_CLOUD_LOCATION'] = settings.GOOGLE_CLOUD_LOCATION # Use settings value
        # Set GOOGLE_APPLICATION_CREDENTIALS environment variable
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/your/service/account.json'
        # Also ensure GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID are set in .env
        
        logger.warning("Google Cloud Project/Location env vars not fully set via this script. Relying on settings and .env. Ensure GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_SEARCH_API_KEY, and GOOGLE_SEARCH_ENGINE_ID are properly configured.")
    
    logger.info(f"Attempting to initialize Gemini Wrapper with model: {settings.GEMINI_MODEL_NAME}")
    asyncio.run(test_gemini_wrapper_responses()) 