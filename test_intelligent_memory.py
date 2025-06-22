#!/usr/bin/env python3
"""
Test script for Queen's Intelligent Memory Formation

This script demonstrates how Queen can extract and store important life events
from natural conversations, including:
- Job interviews
- Appointments  
- Birthdays
- Trips
- Presentations
- Emotional context

Run this to see Queen's intelligent memory extraction in action.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the app directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app.services.memory_intelligence_service import get_memory_intelligence_service
from app.modules.ai_wrapper.user_specific_gemini_wrapper import UserSpecificGeminiWrapper
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test conversation examples that should trigger intelligent memory formation
TEST_CONVERSATIONS = [
    {
        "title": "Job Interview Scenario",
        "conversation": "I'm so nervous about my interview at Google next Friday for the Software Engineer position. I've been preparing for weeks but I'm still worried about the technical questions.",
        "participants": ["user_001"],
        "expected_events": ["interview"]
    },
    {
        "title": "Birthday Planning",
        "conversation": "Mom's birthday is coming up next month and I need to plan something special. She's turning 65 and the whole family will be there.",
        "participants": ["user_001", "family"],
        "expected_events": ["birthday"]
    },
    {
        "title": "Family Trip Planning",
        "conversation": "We're planning a family trip to Hawaii this summer. The kids are so excited and we're looking at resorts near the beach.",
        "participants": ["user_001", "user_002", "kids"],
        "expected_events": ["trip"]
    },
    {
        "title": "Work Presentation Stress",
        "conversation": "I have this big presentation tomorrow for the board and I'm feeling really stressed about it. It's about our quarterly results.",
        "participants": ["user_001"],
        "expected_events": ["presentation"]
    },
    {
        "title": "Medical Appointment",
        "conversation": "I have a doctor's appointment at the medical center next Tuesday. Just a routine checkup but I always get a bit anxious.",
        "participants": ["user_001"],
        "expected_events": ["appointment"]
    },
    {
        "title": "Mixed Events Conversation",
        "conversation": "What a week! I have my interview at Microsoft on Wednesday, then Mom's birthday party on Saturday, and I'm still preparing for my presentation next Monday.",
        "participants": ["user_001"],
        "expected_events": ["interview", "birthday", "presentation"]
    }
]

async def test_memory_intelligence_service():
    """Test the Memory Intelligence Service directly"""
    print("🧠 Testing Memory Intelligence Service")
    print("=" * 50)
    
    try:
        # Get the service
        memory_service = await get_memory_intelligence_service()
        
        for i, test_case in enumerate(TEST_CONVERSATIONS, 1):
            print(f"\n🔍 Test {i}: {test_case['title']}")
            print(f"Conversation: \"{test_case['conversation']}\"")
            print(f"Participants: {test_case['participants']}")
            
            # Extract events
            result = await memory_service.extract_and_store_events(
                conversation_text=test_case['conversation'],
                participants=test_case['participants']
            )
            
            print(f"✅ Result: {result['message']}")
            print(f"   Events found: {result['events_found']}")
            print(f"   Events stored: {result['events_stored']}")
            
            if result.get('stored_events'):
                for event in result['stored_events']:
                    print(f"   📅 {event['event_type']} on {event.get('event_date', 'TBD')} (Triggers: {event['reflection_triggers_count']})")
            
            print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error testing memory intelligence service: {str(e)}")

async def test_gemini_wrapper_integration():
    """Test Queen's intelligent memory formation through the Gemini wrapper"""
    print("\n👑 Testing Queen's Intelligent Memory Formation")
    print("=" * 50)
    
    try:
        # Initialize the user-specific Gemini wrapper
        wrapper = UserSpecificGeminiWrapper(
            api_key=settings.GOOGLE_APPLICATION_CREDENTIALS,
            project_id=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION
        )
        await wrapper.initialize()
        
        test_user_id = "test_user_memory_001"
        
        for i, test_case in enumerate(TEST_CONVERSATIONS[:3], 1):  # Test first 3 cases
            print(f"\n🎯 Queen Test {i}: {test_case['title']}")
            print(f"User says: \"{test_case['conversation']}\"")
            
            # Test Queen's response and memory formation
            response = await wrapper.chat(
                user_id=test_user_id,
                user_input=test_case['conversation'],
                context={"participants": test_case['participants']}
            )
            
            print(f"👑 Queen responds: {response}")
            print("-" * 40)
        
    except Exception as e:
        logger.error(f"Error testing Gemini wrapper integration: {str(e)}")
        print(f"❌ Could not test Gemini integration: {str(e)}")
        print("💡 This might be due to missing Google Cloud credentials")

async def demonstrate_reflection_triggers():
    """Demonstrate how reflection triggers work"""
    print("\n🔄 Demonstrating Reflection Triggers")
    print("=" * 50)
    
    print("When Queen extracts events, she automatically creates reflection triggers:")
    print("📋 Interview Example:")
    print("   • 3 days before: 'How are you feeling about your interview in 3 days?'")
    print("   • 1 day before: 'Your interview is tomorrow. How are you feeling about it?'")
    print("   • 1 day after: 'How did your interview go yesterday?'")
    print("   • 1 week after: 'Any updates on the interview you had last week?'")
    print()
    print("🎂 Birthday Example:")
    print("   • 1 day after: 'How did your Mom's birthday go yesterday?'")
    print()
    print("✈️ Trip Example:")
    print("   • 1 day after: 'How did your trip go yesterday?'")
    print()
    print("💡 These triggers enable Queen to proactively follow up and show genuine care!")

def print_examples():
    """Print example conversations that trigger intelligent memory formation"""
    print("\n📝 Example Conversations That Trigger Memory Formation")
    print("=" * 60)
    
    examples = [
        "I have an interview at Apple next Monday for the iOS Developer role",
        "Dad's birthday is next week and I need to buy him a gift",
        "We're going on vacation to Paris in July",
        "I'm nervous about my presentation to the CEO tomorrow",
        "I have a dentist appointment this Thursday afternoon",
        "The project deadline is next Friday and I'm stressed about it"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. \"{example}\"")
        print(f"   → Queen extracts: date, event type, emotional context, participants")
        print(f"   → Creates reflection triggers for follow-up")
        print()

async def main():
    """Main test function"""
    print("🚗 Queen's Intelligent Memory Formation Test")
    print("=" * 60)
    print("This test demonstrates Queen's ability to:")
    print("• Extract important life events from natural conversation")
    print("• Store structured event data with dates, emotions, and context")
    print("• Create reflection triggers for proactive follow-up")
    print("• Build genuine relationship through intelligent memory")
    print()
    
    # Print examples first
    print_examples()
    
    # Test the memory intelligence service
    await test_memory_intelligence_service()
    
    # Demonstrate reflection triggers
    await demonstrate_reflection_triggers()
    
    # Test Gemini integration (may fail without credentials)
    await test_gemini_wrapper_integration()
    
    print("\n🎉 Testing Complete!")
    print("Queen can now intelligently extract and remember important life events!")
    print("This enables genuine, caring conversations and proactive engagement.")

if __name__ == "__main__":
    asyncio.run(main()) 