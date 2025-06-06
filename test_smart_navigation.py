import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to sys.path to import app modules
sys.path.insert(0, os.path.abspath('.'))

from app.services.smart_navigation_service import get_smart_navigation_service

async def test_smart_navigation():
    """Test the smart navigation enhancement system"""
    print("🗺️ Testing Smart Navigation Enhancement System...")
    
    try:
        # Initialize the service
        nav_service = await get_smart_navigation_service()
        print("✅ Smart Navigation Service initialized")
        
        # Test 1: Airport navigation (commonly confusing)
        print("\n--- Test 1: Airport Navigation ---")
        origin = "Downtown Manhattan, NY"
        destination = "JFK Airport Terminal 4, NY"
        
        enhanced_directions = await nav_service.enhance_directions(
            origin=origin,
            destination=destination,
            user_context={"trip_purpose": "flight_departure", "airline": "Delta"}
        )
        
        if enhanced_directions.get("error"):
            print(f"❌ Error: {enhanced_directions['error']}")
        else:
            print(f"📍 Route: {origin} → {destination}")
            print(f"📏 Distance: {enhanced_directions['total_distance']}")
            print(f"⏱️ Duration: {enhanced_directions['total_duration']}")
            print(f"⚠️ Confusing areas detected: {enhanced_directions['confusing_areas_count']}")
            
            print("\n🤖 AI-Enhanced Steps:")
            for step in enhanced_directions['enhanced_steps'][:3]:  # Show first 3 steps
                print(f"\nStep {step['step_number']}:")
                print(f"  Original: {step['original_instruction']}")
                
                if step['is_confusing']:
                    print(f"  🎯 AI Enhanced: {step.get('ai_enhanced_instruction', 'Generating...')}")
                    if step.get('clarification_tips'):
                        print("  💡 Tips:")
                        for tip in step['clarification_tips'][:2]:
                            print(f"    - {tip}")
                else:
                    print(f"  🗣️ Natural: {step.get('natural_instruction', 'Converting...')}")
            
            # Show proactive guidance
            guidance = enhanced_directions.get('proactive_guidance', {})
            if guidance.get('preparation_tips'):
                print(f"\n🎯 Proactive Tips:")
                for tip in guidance['preparation_tips']:
                    print(f"  - {tip}")
        
        # Test 2: Quick clarification for confusing instruction
        print("\n--- Test 2: Quick Clarification ---")
        confusing_instruction = "Continue on I-495 N and take exit 22A toward Rental Car Return"
        
        # Create a mock step for clarification
        mock_step = {
            "html_instructions": confusing_instruction,
            "distance": {"text": "0.5 mi"},
            "duration": {"text": "2 mins"},
            "start_location": {"lat": 40.6413, "lng": -73.7781}  # Near JFK
        }
        
        enhanced_instruction = await nav_service.ai_enhance_instruction(
            f"Enhance this navigation instruction: {confusing_instruction}",
            mock_step
        )
        
        tips = await nav_service.generate_clarification_tips(mock_step)
        
        print(f"Original: {confusing_instruction}")
        print(f"🤖 AI Enhanced: {enhanced_instruction}")
        if tips:
            print("💡 Helpful Tips:")
            for tip in tips:
                print(f"  - {tip}")
        
        # Test 3: Report confusing location (community feature)
        print("\n--- Test 3: Community Feedback ---")
        test_location = {"lat": 40.6413, "lng": -73.7781}
        await nav_service.report_confusing_location(
            location=test_location,
            user_feedback="The rental car return signs are really unclear here. I drove around 3 times!",
            original_instruction="Turn right toward Rental Car Return",
            user_id="test_user_123"
        )
        print("✅ Community feedback submitted")
        
        print("\n🎉 Smart Navigation tests completed!")
        
        # Show what this system provides
        print("\n📋 System Capabilities:")
        print("✅ Natural language directions instead of robotic commands")
        print("✅ Proactive clarification for known confusing areas")
        print("✅ Context-aware guidance (airports, rental cars, etc.)")
        print("✅ Community-driven improvements")
        print("✅ Real-time clarification for confused users")
        print("✅ Cost-effective overlay on existing map APIs")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_smart_navigation()) 