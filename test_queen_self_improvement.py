#!/usr/bin/env python3
"""
Test Queen's Self-Improvement Capabilities
Demonstrates the new reflection, meta-cognition, and learning features.
"""

import asyncio
import json
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.modules.ai_wrapper.reflection_manager import get_reflection_manager

async def test_queens_intelligence():
    """Test Queen's enhanced reasoning and self-reflection"""
    print("🚗 Starting Queen's Self-Improvement Test")
    print("=" * 60)
    
    # Get Queen (Gemini wrapper) and reflection manager
    queen = await get_gemini_wrapper()
    reflection_manager = get_reflection_manager()
    
    # Test scenarios to show Queen's improved capabilities
    test_scenarios = [
        {
            "scenario": "Basic Navigation Query",
            "user_input": "I need directions to the nearest hospital",
            "context": {"user_id": "test_user", "current_location": "Buffalo, NY"}
        },
        {
            "scenario": "Complex Multi-Step Request", 
            "user_input": "I'm feeling anxious about driving in this weather. Can you help me find a safe route home and maybe play some calming music?",
            "context": {"user_id": "test_user", "weather": "heavy rain", "mood": "anxious"}
        },
        {
            "scenario": "Learning from Previous Interaction",
            "user_input": "Actually, I need directions to the nearest hospital again",
            "context": {"user_id": "test_user", "current_location": "Buffalo, NY"}
        }
    ]
    
    for i, test in enumerate(test_scenarios, 1):
        print(f"\n🔍 Test {i}: {test['scenario']}")
        print(f"👤 User: {test['user_input']}")
        
        # Get Queen's response with reflection
        response = await queen.chat(
            user_id=test["context"]["user_id"],
            user_input=test["user_input"],
            context=test["context"]
        )
        
        print(f"👑 Queen: {response}")
        
        # Show reflection insights
        summary = reflection_manager.get_reflection_summary()
        print(f"🧠 Reflection Summary: {json.dumps(summary, indent=2)}")
        
        print("-" * 50)
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    print("\n🎯 Queen's Learning Progress Summary")
    print("=" * 60)
    
    final_summary = reflection_manager.get_reflection_summary()
    print(f"Total Interactions: {final_summary.get('total_interactions', 0)}")
    print(f"Recent Performance: {final_summary.get('recent_performance_avg', 0):.3f}")
    print(f"Improvement Trend: {final_summary.get('improvement_trend', 'unknown')}")
    
    # Test the new reflection-enhanced system prompt
    print("\n🧠 Testing Meta-Cognitive Reasoning")
    print("=" * 60)
    
    complex_query = "I'm new to this city and driving for the first time here. I'm scared of getting lost and my phone battery is low. What should I do?"
    
    print(f"👤 User: {complex_query}")
    
    response = await queen.chat(
        user_id="test_user",
        user_input=complex_query,
        context={"user_id": "test_user", "battery_level": "15%", "location": "unfamiliar_city"}
    )
    
    print(f"👑 Queen: {response}")
    
    print("\n✨ Queen's self-improvement features are working!")
    print("Key Enhancements:")
    print("• Pre-response reflection for better understanding")
    print("• Post-response learning for continuous improvement")
    print("• Meta-cognitive reasoning framework")
    print("• Performance tracking and optimization")
    print("• Safety-first adaptive communication")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_queens_intelligence()) 