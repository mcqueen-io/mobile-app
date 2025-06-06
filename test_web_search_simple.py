import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to sys.path to import app modules
sys.path.insert(0, os.path.abspath('.'))

from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper

async def test_web_search_simple():
    """Simple test for web search functionality"""
    print("Testing Web Search Functionality...")
    
    try:
        # Get the Gemini wrapper
        wrapper = await get_gemini_wrapper()
        print("✅ Gemini wrapper initialized successfully")
        
        # Test 1: Simple web search query
        print("\n--- Test 1: Weather Query ---")
        user_input = "What is the weather like in New York?"
        context = {}
        
        response = await wrapper.chat("test_user", user_input, context)
        print(f"User Input: {user_input}")
        print(f"AI Response: {response}")
        
        # Check if the response mentions weather or contains relevant information
        if "weather" in response.lower() or "new york" in response.lower():
            print("✅ Test 1 PASSED: Response contains relevant weather information")
        else:
            print("❌ Test 1 FAILED: Response doesn't seem to contain weather information")
            
        # Test 2: Current events query
        print("\n--- Test 2: News Query ---")
        user_input = "What's happening in technology news today?"
        
        response = await wrapper.chat("test_user", user_input, context)
        print(f"User Input: {user_input}")
        print(f"AI Response: {response}")
        
        if "technology" in response.lower() or "news" in response.lower():
            print("✅ Test 2 PASSED: Response contains relevant news information")
        else:
            print("❌ Test 2 FAILED: Response doesn't seem to contain news information")
            
        print("\n🎉 Web search tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_web_search_simple()) 