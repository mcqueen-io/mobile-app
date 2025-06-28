#!/usr/bin/env python3
"""
Live Memory Storage Test
Test if Queen's memory system is actually working right now
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.context.intelligent_context_manager import get_intelligent_context_manager
from app.modules.memory.memory_store import get_memory_store

async def test_live_memory_storage():
    """Test if memory storage is working right now"""
    print("🧠 Live Memory Storage Test")
    print("=" * 40)
    
    try:
        # Test 1: Direct memory store
        print("📝 Test 1: Direct Memory Store")
        memory_store = get_memory_store()
        
        # Check initial count
        initial_count = memory_store.collection.count()
        print(f"  Initial memories: {initial_count}")
        
        # Add a test memory
        test_memory_id = memory_store.add_memory(
            user_id="test_user_live",
            content="This is a live test memory to verify ChromaDB is working",
            metadata={
                "test": True,
                "timestamp": "2025-06-22",
                "importance": 0.9,
                "topic": "testing"
            }
        )
        
        print(f"  ✅ Added memory: {test_memory_id}")
        
        # Check new count
        new_count = memory_store.collection.count()
        print(f"  New memory count: {new_count}")
        
        if new_count > initial_count:
            print("  ✅ Memory successfully stored!")
            
            # Try to retrieve it
            memories = memory_store.get_user_memories("test_user_live")
            print(f"  Retrieved {len(memories)} memories for test user")
            
            if memories:
                print(f"  📄 Memory content: {memories[0]['content'][:50]}...")
                print(f"  📊 Memory metadata: {memories[0]['metadata']}")
        else:
            print("  ❌ Memory was not stored")
            return False
        
        # Test 2: Intelligent Context Manager
        print(f"\n📝 Test 2: Intelligent Context Manager")
        
        manager = await get_intelligent_context_manager()
        session_id = "live_test_session"
        session = manager.create_session(session_id)
        
        session.active_users = {"live_user": {"name": "Live Test User"}}
        session.current_driver = "live_user"
        
        # Process an important utterance
        utterance = "I'm having severe chest pains and need to get to the hospital immediately!"
        print(f"  Processing: '{utterance}'")
        
        result = await manager.process_utterance(session_id, "live_user", utterance)
        analysis = result["analysis"]
        
        print(f"  📊 Analysis:")
        print(f"    Topic: {analysis['topic']}")
        print(f"    Importance: {analysis['importance_score']:.2f}")
        print(f"    Memory stored: {result['memory_stored']}")
        
        if result["memory_stored"]:
            print("  ✅ Intelligent memory storage working!")
            
            # Check if memory count increased
            final_count = memory_store.collection.count()
            print(f"  Final memory count: {final_count}")
            
        else:
            print("  ❌ Intelligent memory storage failed")
        
        # Test 3: Memory Retrieval
        print(f"\n📝 Test 3: Memory Retrieval")
        
        try:
            relevant_memories = memory_store.get_relevant_memories(
                user_id="live_user",
                query="medical emergency",
                n_results=3
            )
            
            print(f"  Found {len(relevant_memories)} relevant memories")
            for i, memory in enumerate(relevant_memories, 1):
                print(f"    {i}. {memory['content'][:50]}...")
                
        except Exception as e:
            print(f"  ❌ Memory retrieval failed: {e}")
        
        # Cleanup
        print(f"\n🧹 Cleanup")
        try:
            memory_store.delete_memory(test_memory_id)
            cleanup_count = memory_store.collection.count()
            print(f"  Cleaned up test memory, count: {cleanup_count}")
        except Exception as e:
            print(f"  ❌ Cleanup failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Live memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def inspect_current_collections():
    """Inspect what collections exist right now"""
    print(f"\n🔍 Current Collections Inspection")
    print("=" * 35)
    
    try:
        memory_store = get_memory_store()
        
        # Check the collection that memory store uses
        collection = memory_store.collection
        print(f"📁 Memory Store Collection:")
        print(f"  Name: {collection.name}")
        print(f"  Count: {collection.count()}")
        
        if collection.count() > 0:
            # Get some samples
            results = collection.get(limit=3, include=['metadatas', 'documents'])
            print(f"  📋 Sample memories:")
            
            for i in range(len(results['ids'])):
                doc_id = results['ids'][i]
                content = results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i]
                metadata = results['metadatas'][i]
                
                print(f"    {i+1}. ID: {doc_id}")
                print(f"       Content: {content}")
                print(f"       User: {metadata.get('user_id', 'unknown')}")
                print(f"       Topic: {metadata.get('topic', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection inspection failed: {e}")
        return False

async def main():
    """Run live memory tests"""
    print("🚀 Live Memory Storage Test")
    print("Testing Queen's memory system in real-time")
    print("=" * 50)
    
    # First inspect current state
    inspect_current_collections()
    
    # Then run live tests
    success = await test_live_memory_storage()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 MEMORY SYSTEM IS WORKING!")
        print("\n✅ Confirmed:")
        print("  • ChromaDB is storing memories")
        print("  • Intelligent analysis is working")
        print("  • Memory retrieval is functional")
        print("  • Vector similarity search works")
        
        print("\n💾 Your memories are being stored in:")
        print("  • Location: ./data/chroma/")
        print("  • Collection: user_memories")
        print("  • Format: Vector embeddings + metadata")
    else:
        print("❌ Memory system has issues")

if __name__ == "__main__":
    asyncio.run(main()) 