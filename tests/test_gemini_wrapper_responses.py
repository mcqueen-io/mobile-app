import asyncio
import sys
import os
from typing import Dict, Any, List, Union

# Add the parent directory of 'app' to the sys.path to allow importing 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.modules.ai_wrapper.gemini_wrapper import GeminiWrapper, get_gemini_wrapper
from app.core.config import settings # Import the settings object
# from django.conf import settings # Remove this incorrect import

async def test_gemini_wrapper_responses():
    """Tests the GeminiWrapper's response capabilities in different scenarios."""
    print("Initializing Gemini Wrapper...")
    # Note: This requires GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and
    # GOOGLE_APPLICATION_CREDENTIALS environment variables to be set.
    # Mocking might be needed for full independence from GCP.
    try:
        wrapper = await get_gemini_wrapper()
        print("Wrapper initialized.")
    except Exception as e:
        print(f"Failed to initialize Gemini Wrapper: {e}")
        print("Please ensure Google Cloud credentials environment variables are set correctly.")
        return

    # Placeholder user ID - in a real scenario, this would come from authentication/identification
    test_user_id = "test_user_123"
    print(f"Using placeholder user ID: {test_user_id}")

    print("\n--- Scenario 1: Simple conversational input ---")
    user_input_1 = "Hello, how are you today?"
    context_1 = {}
    print(f"Input: {user_input_1}")
    response_1 = await wrapper.generate_response(test_user_id, user_input_1, context=context_1)
    print(f"Response: {response_1}")
    print(f"Response Type: {'Tool Call(s)' if isinstance(response_1, list) else 'Text'}")

    print("\n--- Scenario 2: Input asking for user preference (should trigger tool call) ---")
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
    print(f"Input: {user_input_2}")
    response_2 = await wrapper.generate_response(test_user_id, user_input_2, context=context_2)
    print(f"Response: {response_2}")
    print(f"Response Type: {'Tool Call(s)' if isinstance(response_2, list) else 'Text'}")

    print("\n--- Scenario 3: Input asking to search memories (should trigger tool call) ---")
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
    print(f"Input: {user_input_3}")
    response_3 = await wrapper.generate_response(test_user_id, user_input_3, context=context_3)
    print(f"Response: {response_3}")
    print(f"Response Type: {'Tool Call(s)' if isinstance(response_3, list) else 'Text'}")

    print("\n--- Scenario 4: Input asking to search memories, but relevant info IS in context ---")
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
        "recent_family_memories": [
            {
                "content": "We went camping in the mountains last summer. It rained a lot!",
                "created_by": test_user_id,
                "created_at": "<timestamp>",
                "type": "family"
            }
        ] # Relevant memory IS in context
    }
    print(f"Input: {user_input_4}")
    response_4 = await wrapper.generate_response(test_user_id, user_input_4, context=context_4)
    print(f"Response: {response_4}")
    print(f"Response Type: {'Tool Call(s)' if isinstance(response_4, list) else 'Text'}")

    print("\n--- Scenario 5: Input suggesting low mood (should trigger comforting tone) ---")
    # Note: Sentiment analysis isn't implemented in CPL yet, so AI relies on text cues.
    # The response type will likely be Text.
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
    print(f"Input: {user_input_5}")
    response_5 = await wrapper.generate_response(test_user_id, user_input_5, context=context_5)
    print(f"Response: {response_5}")
    print(f"Response Type: {'Tool Call(s)' if isinstance(response_5, list) else 'Text'}")

    print("\n--- Test complete ---")
    print("Review the output above to see how the AI responded in each scenario.")
    print("Specifically, check if:"+ 
          "\n- Simple inputs get text responses." + 
          "\n- Queries requiring data (preference, memory) trigger tool calls (Scenario 2 & 3)." +
          "\n- Queries where data is ALREADY in context lead to text responses, not tool calls (Scenario 4)." +
          "\n- The response for Scenario 5 reflects a comforting tone based on the user input.")

# Boilerplate to run the async function
if __name__ == "__main__":
    # Ensure environment variables are loaded if using .env file
    # from dotenv import load_dotenv # Uncomment if using dotenv
    # load_dotenv() # Uncomment if using dotenv
    
    # Set placeholder env vars if not using .env and running locally for quick test setup
    if not os.getenv('GOOGLE_CLOUD_PROJECT'):
        os.environ['GOOGLE_CLOUD_PROJECT'] = 'YOUR_PROJECT_ID' # Replace with a dummy or real project ID
        os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1' # Replace with a valid location
        # Warning: Running this without valid credentials will cause initialization failure.
        print("Warning: Google Cloud Project/Location env vars not set. Using placeholders.")
        print("Set GOOGLE_APPLICATION_CREDENTIALS for actual API calls.")
    
    print(f"Attempting to initialize Gemini Wrapper with model: {settings.GEMINI_MODEL_NAME}")
    asyncio.run(test_gemini_wrapper_responses()) 