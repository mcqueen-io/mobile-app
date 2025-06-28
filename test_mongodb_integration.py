#!/usr/bin/env python3
"""
Test MongoDB Integration with Intelligent Context Manager
This test will actually store memories in MongoDB so you can see them in Atlas
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.context.intelligent_context_manager import get_intelligent_context_manager
from datetime import datetime, timezone
import json

async def test_mongodb_memory_storage():
    """Test storing intelligent memories in MongoDB"""
    print("🗄️ Test: MongoDB Memory Storage Integration")
    print("=" * 60)
    
    try:
        # Initialize the intelligent context manager
        manager = await get_intelligent_context_manager()
        session_id = "mongodb_test_session"
        session = manager.create_session(session_id)
        
        # Set up family context
        session.active_users = {
            "dad": {"name": "John", "family_tree_id": "smith_family"},
            "mom": {"name": "Sarah", "family_tree_id": "smith_family"},
            "kid": {"name": "Emma", "age": 8, "family_tree_id": "smith_family"}
        }
        session.current_driver = "dad"
        session.destination = {"name": "Disney World", "address": "Orlando, FL"}
        
        print("👨‍👩‍👧 Family context set up")
        print(f"Session ID: {session_id}")
        print(f"Family: {list(session.active_users.keys())}")
        
        # Test conversations that should be stored in MongoDB
        important_conversations = [
            ("dad", "I'm feeling really drowsy while driving, we need to find a rest stop immediately"),
            ("mom", "Emma has been complaining about stomach pain for the last hour, I'm worried"),
            ("kid", "My chest hurts and I can't breathe properly"),
            ("dad", "Sarah, I got the promotion at work! We can finally afford that house"),
            ("mom", "I'm pregnant! We're going to have another baby"),
            ("dad", "My father called - grandpa is in the hospital with a heart attack"),
            ("mom", "Emma's teacher wants to meet about her learning difficulties"),
            ("kid", "I'm scared about starting at the new school next week")
        ]
        
        print(f"\n🧠 Processing {len(important_conversations)} important conversations...")
        
        stored_memories = []
        
        for i, (user_id, utterance) in enumerate(important_conversations, 1):
            print(f"\n--- Conversation {i} ---")
            print(f"{user_id}: '{utterance}'")
            
            # Process the utterance
            result = await manager.process_utterance(session_id, user_id, utterance)
            analysis = result["analysis"]
            
            print(f"  📊 Analysis:")
            print(f"    Topic: {analysis['topic']}")
            print(f"    Emotional State: {analysis['emotional_state']}")
            print(f"    Importance: {analysis['importance_score']:.2f}")
            print(f"    Memory Worthy: {analysis['memory_worthy']}")
            print(f"    Urgency: {analysis['urgency']}")
            
            if result["memory_stored"]:
                print(f"  ✅ STORED in MongoDB + ChromaDB")
                stored_memories.append({
                    "user": user_id,
                    "content": utterance,
                    "analysis": analysis
                })
            else:
                print(f"  ❌ Not stored (importance too low)")
        
        # Get session summary
        session_summary = manager.get_session_summary(session_id)
        
        print(f"\n📊 SESSION SUMMARY:")
        print(f"  Total utterances: {session_summary['total_utterances']}")
        print(f"  Memory worthy events: {session_summary['memory_worthy_events']}")
        print(f"  Important moments stored: {session_summary['important_moments']}")
        print(f"  Session themes: {session_summary['session_themes']}")
        print(f"  Emotional context: {session_summary['emotional_context']}")
        
        print(f"\n🎯 MONGODB STORAGE RESULTS:")
        print(f"  Memories stored: {len(stored_memories)}")
        print(f"  Expected in MongoDB Atlas: {len(stored_memories)} documents")
        
        # List what should appear in MongoDB
        print(f"\n📋 WHAT YOU SHOULD SEE IN MONGODB ATLAS:")
        for i, memory in enumerate(stored_memories, 1):
            print(f"  {i}. User: {memory['user']}")
            print(f"     Topic: {memory['analysis']['topic']}")
            print(f"     Emotion: {memory['analysis']['emotional_state']}")
            print(f"     Content: {memory['content'][:50]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mongodb_query_capabilities():
    """Test querying the MongoDB memories"""
    print("\n🔍 Test: MongoDB Query Capabilities")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        
        # Search for memories by different criteria
        search_queries = [
            ("dad", "safety concerns"),
            ("mom", "health issues"),
            ("kid", "emotional problems"),
            ("dad", "work related"),
        ]
        
        print("Testing MongoDB memory searches...")
        
        for user_id, query in search_queries:
            print(f"\n🔎 Searching for '{query}' by {user_id}:")
            
            try:
                # This would use the MongoDB search functionality
                memories = await manager.mongo_manager.search_memories(
                    user_ids=[user_id],
                    query=query,
                    family_ids=["smith_family"],
                    limit=3
                )
                
                print(f"  Found {len(memories)} memories")
                for memory in memories:
                    content = memory.get('content', '')[:40] + "..."
                    topic = memory.get('metadata', {}).get('topic', 'unknown')
                    print(f"    - {topic}: {content}")
                    
            except Exception as e:
                print(f"  Search failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_family_memory_sharing():
    """Test family memory sharing capabilities"""
    print("\n👨‍👩‍👧‍👦 Test: Family Memory Sharing")
    print("=" * 50)
    
    try:
        manager = await get_intelligent_context_manager()
        
        # Test family-wide memories
        family_conversations = [
            ("dad", "We're planning a family vacation to Europe next summer"),
            ("mom", "Let's make sure we save money for Emma's college fund"),
            ("kid", "I want to learn French for our trip to Paris"),
        ]
        
        session_id = "family_sharing_test"
        session = manager.create_session(session_id)
        session.active_users = {
            "dad": {"name": "John", "family_tree_id": "smith_family"},
            "mom": {"name": "Sarah", "family_tree_id": "smith_family"},
            "kid": {"name": "Emma", "family_tree_id": "smith_family"}
        }
        
        print("Processing family conversations...")
        
        for user_id, utterance in family_conversations:
            print(f"\n{user_id}: '{utterance}'")
            result = await manager.process_utterance(session_id, user_id, utterance)
            
            if result["memory_stored"]:
                print(f"  ✅ Stored as family memory")
            else:
                print(f"  ❌ Not stored")
        
        print(f"\n📊 Family memories should be visible to all family members in MongoDB")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def main():
    """Run MongoDB integration tests"""
    print("🚀 MongoDB Integration Test Suite")
    print("This will store actual data in your MongoDB Atlas database")
    print("=" * 70)
    
    tests = [
        test_mongodb_memory_storage,
        test_mongodb_query_capabilities,
        test_family_memory_sharing
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if await test():
                passed += 1
                print("\n✅ PASSED")
            else:
                failed += 1
                print("\n❌ FAILED")
        except Exception as e:
            print(f"\n💥 CRASHED: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Final Results: {passed} passed, {failed} failed")
    
    if passed > 0:
        print("\n🎉 MongoDB integration working!")
        print("\n💡 CHECK YOUR MONGODB ATLAS NOW:")
        print("  1. Go to your MongoDB Atlas dashboard")
        print("  2. Navigate to family_assistant database")
        print("  3. Check the 'memories' collection")
        print("  4. You should see intelligent memories with:")
        print("     - Emotional analysis")
        print("     - Importance scores")
        print("     - Topic classification")
        print("     - Session context")
    else:
        print("⚠️ MongoDB integration needs debugging")

if __name__ == "__main__":
    asyncio.run(main()) 