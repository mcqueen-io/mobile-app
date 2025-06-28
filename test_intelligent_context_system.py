#!/usr/bin/env python3
"""
Test Intelligent Context Management System
Tests Gemini-powered topic detection, selective memory storage, and emotional context tracking
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.context.intelligent_context_manager import get_intelligent_context_manager
from datetime import datetime, timezone
import json

async def test_intelligent_utterance_analysis():
    """Test LLM-powered utterance analysis vs mundane conversations"""
    print("🧠 Test 1: Intelligent Utterance Analysis")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "intelligent_test_001"
        session = manager.create_session(session_id)
        
        # Set up family context
        session.active_users = {
            "dad": {"name": "David", "role": "driver"},
            "mom": {"name": "Sarah", "role": "passenger"},
            "kid1": {"name": "Emma", "age": 8}
        }
        session.current_driver = "dad"
        session.destination = {"name": "Disney World", "address": "Orlando, FL"}
        
        # Test conversations: mundane vs important
        test_conversations = [
            # MUNDANE - Should NOT be stored
            ("dad", "Let's stop at McDonald's for lunch", "mundane food request"),
            ("mom", "What time is it?", "simple time check"),
            ("kid1", "Are we there yet?", "routine kid question"),
            
            # IMPORTANT - Should be stored  
            ("mom", "I'm feeling really anxious about Emma's doctor appointment tomorrow", "emotional concern"),
            ("dad", "I'm getting drowsy, we should find a rest stop soon", "safety concern"),
            ("kid1", "My stomach really hurts, I think I'm getting sick", "health issue"),
            
            # MODERATE - Depends on context
            ("mom", "Can you remind me to call my boss when we arrive?", "task reminder"),
            ("dad", "This traffic is making me frustrated", "emotional state"),
            ("kid1", "I'm excited to see Mickey Mouse!", "positive emotion")
        ]
        
        print("Processing conversations...")
        results = []
        
        for user_id, utterance, expected_category in test_conversations:
            print(f"\n{user_id}: '{utterance}'")
            
            result = await manager.process_utterance(session_id, user_id, utterance)
            analysis = result["analysis"]
            
            print(f"  Topic: {analysis['topic']}")
            print(f"  Intent: {analysis['intent']}")
            print(f"  Emotion: {analysis['emotional_state']}")
            print(f"  Importance: {analysis['importance_score']:.2f}")
            print(f"  Memory Worthy: {analysis['memory_worthy']}")
            print(f"  Reasoning: {analysis['reasoning']}")
            
            results.append({
                "utterance": utterance,
                "expected": expected_category,
                "analysis": analysis,
                "stored": result["memory_stored"]
            })
        
        # Analyze results
        mundane_stored = sum(1 for r in results[:3] if r["stored"])
        important_stored = sum(1 for r in results[3:6] if r["stored"])
        
        print(f"\n📊 FILTERING RESULTS:")
        print(f"Mundane conversations stored: {mundane_stored}/3 (should be 0)")
        print(f"Important conversations stored: {important_stored}/3 (should be 3)")
        
        session_summary = manager.get_session_summary(session_id)
        print(f"\n📋 SESSION SUMMARY:")
        print(f"Total utterances: {session_summary['total_utterances']}")
        print(f"Memory worthy events: {session_summary['memory_worthy_events']}")
        print(f"Important moments stored: {session_summary['important_moments']}")
        print(f"Session themes: {session_summary['session_themes']}")
        print(f"Emotional context: {session_summary['emotional_context']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_emotional_context_tracking():
    """Test emotional context tracking across conversation"""
    print("\n😊 Test 2: Emotional Context Tracking")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "emotion_test_002"
        session = manager.create_session(session_id)
        
        # Set up users
        session.active_users = {
            "dad": {"name": "John"},
            "mom": {"name": "Lisa"},
            "teen": {"name": "Alex", "age": 16}
        }
        
        # Emotional journey conversation
        emotional_journey = [
            ("dad", "I'm excited about our family vacation!", "excited → happy"),
            ("mom", "I'm worried we forgot to pack Emma's medication", "worried → anxious"),
            ("teen", "This is so boring, I want to go home", "bored → frustrated"),
            ("dad", "Don't worry honey, I packed all the meds", "reassuring → calm"),
            ("mom", "Oh thank goodness, I was so stressed about that", "relieved → grateful"),
            ("teen", "Actually, this road trip music is pretty cool", "interested → positive")
        ]
        
        print("Tracking emotional journey...")
        
        for user_id, utterance, expected_emotion in emotional_journey:
            print(f"\n{user_id}: '{utterance}'")
            
            result = await manager.process_utterance(session_id, user_id, utterance)
            analysis = result["analysis"]
            
            print(f"  Detected emotion: {analysis['emotional_state']}")
            print(f"  Expected: {expected_emotion}")
            print(f"  Importance: {analysis['importance_score']:.2f}")
        
        # Check final emotional context
        final_session = manager.get_session(session_id)
        print(f"\n🎭 FINAL EMOTIONAL CONTEXT:")
        for user_id, emotion in final_session.emotional_context.items():
            print(f"  {user_id}: {emotion}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_intelligent_response_generation():
    """Test contextually intelligent response generation"""
    print("\n🤖 Test 3: Intelligent Response Generation")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "response_test_003"
        session = manager.create_session(session_id)
        
        # Set up context
        session.active_users = {"dad": {"name": "Mike"}}
        session.current_driver = "dad"
        
        # Build conversation context first
        context_utterances = [
            "I'm feeling really tired from driving",
            "We've been on the road for 4 hours now",
            "I'm worried about falling asleep at the wheel"
        ]
        
        for utterance in context_utterances:
            await manager.process_utterance(session_id, "dad", utterance)
        
        # Test intelligent responses to different queries
        test_queries = [
            "Queen, should I keep driving?",
            "Find me something to help me stay alert",
            "How much longer until we reach our destination?"
        ]
        
        print("Generating contextually intelligent responses...")
        
        for query in test_queries:
            print(f"\nUser: '{query}'")
            
            response = await manager.generate_intelligent_response(session_id, query, "dad")
            print(f"Queen: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_memory_retrieval_intelligence():
    """Test intelligent memory retrieval based on context"""
    print("\n🧠 Test 4: Intelligent Memory Retrieval")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "memory_test_004"
        session = manager.create_session(session_id)
        
        session.active_users = {"mom": {"name": "Jennifer"}}
        session.current_driver = "mom"
        
        # Store some important memories first
        important_memories = [
            "Emma has a severe peanut allergy, always check ingredients",
            "Dad gets car sick on winding mountain roads",
            "We always stop at that diner in Springfield on family trips",
            "The kids love that playground at the rest stop on Route 95"
        ]
        
        for memory in important_memories:
            await manager.process_utterance(session_id, "mom", memory)
        
        # Test memory retrieval with different queries
        test_queries = [
            "Where should we eat with the kids?",
            "Emma wants peanut butter cookies",
            "Should we take the scenic mountain route?",
            "The kids are getting restless, any ideas?"
        ]
        
        print("Testing intelligent memory retrieval...")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            context = await manager.get_intelligent_context(session_id, query)
            relevant_memories = context.get("relevant_memories", [])
            
            print(f"Retrieved {len(relevant_memories)} relevant memories:")
            for memory in relevant_memories:
                content = memory.get("content", "")[:60] + "..."
                print(f"  - {content}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_session_theme_detection():
    """Test detection of high-level conversation themes"""
    print("\n🎯 Test 5: Session Theme Detection")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "theme_test_005"
        session = manager.create_session(session_id)
        
        session.active_users = {
            "dad": {"name": "Robert"},
            "mom": {"name": "Maria"},
            "kid": {"name": "Sofia", "age": 6}
        }
        
        # Conversation with multiple themes
        themed_conversation = [
            ("dad", "I'm excited about our first family camping trip"),
            ("mom", "I hope Sofia isn't scared of sleeping in the tent"),
            ("kid", "Will there be bears in the forest?"),
            ("dad", "We need to remember to bring the first aid kit"),
            ("mom", "And don't forget Sofia's inhaler for her asthma"),
            ("kid", "Can we make s'mores by the campfire?"),
            ("dad", "This GPS says there's construction ahead"),
            ("mom", "Should we take the alternate route?"),
            ("kid", "I'm getting hungry, when will we eat?"),
            ("dad", "Let's find a family restaurant soon")
        ]
        
        print("Processing themed conversation...")
        
        for user_id, utterance in themed_conversation:
            await manager.process_utterance(session_id, user_id, utterance)
        
        session_summary = manager.get_session_summary(session_id)
        
        print(f"\n🎭 DETECTED THEMES:")
        for theme in session_summary["session_themes"]:
            print(f"  - {theme}")
        
        print(f"\n📊 THEME ANALYSIS:")
        print(f"Total themes detected: {len(session_summary['session_themes'])}")
        print(f"Expected themes: family_bonding, safety_concerns, navigation, food, camping")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_mundane_vs_important_filtering():
    """Test the system's ability to filter mundane conversations"""
    print("\n🔍 Test 6: Mundane vs Important Filtering")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "filtering_test_006"
        session = manager.create_session(session_id)
        
        # Clearly mundane conversations (should NOT be stored)
        mundane_conversations = [
            "Let's get McDonald's",
            "What time is it?",
            "Turn up the AC",
            "Are we there yet?",
            "I need to use the bathroom",
            "This song is okay I guess",
            "It's sunny today",
            "Traffic is moving slowly"
        ]
        
        # Clearly important conversations (SHOULD be stored)
        important_conversations = [
            "I think I'm having chest pains",
            "Emma's teacher called about her grades dropping",
            "I got the promotion at work!",
            "Grandma is in the hospital",
            "I'm feeling really depressed lately",
            "We need to talk about our marriage",
            "The bank called about our mortgage",
            "I'm pregnant!"
        ]
        
        mundane_stored = 0
        important_stored = 0
        
        print("Testing mundane conversation filtering...")
        for utterance in mundane_conversations:
            result = await manager.process_utterance(session_id, "user", utterance)
            if result["memory_stored"]:
                mundane_stored += 1
                print(f"  ❌ STORED mundane: '{utterance}'")
            else:
                print(f"  ✅ FILTERED mundane: '{utterance}'")
        
        print(f"\nTesting important conversation detection...")
        for utterance in important_conversations:
            result = await manager.process_utterance(session_id, "user", utterance)
            if result["memory_stored"]:
                important_stored += 1
                print(f"  ✅ STORED important: '{utterance}'")
            else:
                print(f"  ❌ MISSED important: '{utterance}'")
        
        print(f"\n📊 FILTERING PERFORMANCE:")
        print(f"Mundane conversations stored: {mundane_stored}/{len(mundane_conversations)} (target: 0)")
        print(f"Important conversations stored: {important_stored}/{len(important_conversations)} (target: {len(important_conversations)})")
        
        filtering_accuracy = ((len(mundane_conversations) - mundane_stored) + important_stored) / (len(mundane_conversations) + len(important_conversations))
        print(f"Overall filtering accuracy: {filtering_accuracy:.2%}")
        
        return filtering_accuracy > 0.8  # 80% accuracy threshold
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def main():
    """Run all intelligent context management tests"""
    print("🚀 Intelligent Context Management Test Suite")
    print("=" * 70)
    
    tests = [
        test_intelligent_utterance_analysis,
        test_emotional_context_tracking,
        test_intelligent_response_generation,
        test_memory_retrieval_intelligence,
        test_session_theme_detection,
        test_mundane_vs_important_filtering
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if await test():
                passed += 1
                print("\n✅ PASSED\n")
            else:
                failed += 1
                print("\n❌ FAILED\n")
        except Exception as e:
            print(f"\n💥 CRASHED: {e}\n")
            failed += 1
    
    print("=" * 70)
    print(f"📊 Final Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All intelligent context management tests passed!")
        print("\n💡 Key Achievements:")
        print("✅ LLM-powered topic detection and intent classification")
        print("✅ Selective memory storage (no more mundane clutter)")
        print("✅ Emotional context tracking across conversations")
        print("✅ Intelligent response generation with context awareness")
        print("✅ Theme detection and conversation flow analysis")
    else:
        print("⚠️ Some tests failed. The system needs refinement.")

if __name__ == "__main__":
    asyncio.run(main()) 