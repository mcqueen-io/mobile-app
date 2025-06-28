#!/usr/bin/env python3
"""
Create Sample Family Memories & Examine Storage
Show exactly how Queen stores conversation and family-specific data
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.context.intelligent_context_manager import get_intelligent_context_manager
from app.modules.memory.memory_store import get_memory_store

async def create_sample_family_memories():
    """Create realistic family conversation memories"""
    print("👨‍👩‍👧‍👦 Creating Sample Family Memories")
    print("=" * 45)
    
    try:
        manager = await get_intelligent_context_manager()
        session_id = "family_car_trip"
        session = manager.create_session(session_id)
        
        # Set up family members
        session.active_users = {
            "dad_mike": {"name": "Mike Johnson", "family_tree_id": "johnson_family"},
            "mom_sarah": {"name": "Sarah Johnson", "family_tree_id": "johnson_family"},
            "kid_emma": {"name": "Emma Johnson", "family_tree_id": "johnson_family", "age": 8},
            "teen_alex": {"name": "Alex Johnson", "family_tree_id": "johnson_family", "age": 16}
        }
        session.current_driver = "dad_mike"
        
        # Sample family conversations (important ones that should be stored)
        family_conversations = [
            ("dad_mike", "Sarah, I got the promotion! We can finally afford that house we looked at"),
            ("mom_sarah", "Oh my god Mike! That's amazing! Kids, daddy got promoted at work!"),
            ("kid_emma", "Does this mean we can get a puppy now? You promised!"),
            ("teen_alex", "This is great dad! Does this mean I can get a car for my 17th birthday?"),
            ("mom_sarah", "I'm feeling really dizzy and nauseous. Can we pull over?"),
            ("dad_mike", "Emergency - there's been an accident ahead. Everyone buckle up tight!"),
            ("kid_emma", "I'm scared daddy! That car looks really hurt!"),
            ("mom_sarah", "Emma, I'm pregnant! We're going to have another baby!"),
            ("teen_alex", "Wait, seriously? I'm going to be a big brother again?"),
            ("dad_mike", "Grandpa called - he's in the hospital. We need to visit him this weekend"),
        ]
        
        print(f"🎯 Processing {len(family_conversations)} family conversations...")
        
        stored_count = 0
        for i, (user_id, utterance) in enumerate(family_conversations, 1):
            print(f"\n--- Conversation {i} ---")
            print(f"{session.active_users[user_id]['name']}: '{utterance}'")
            
            result = await manager.process_utterance(session_id, user_id, utterance)
            analysis = result["analysis"]
            
            print(f"  📊 Topic: {analysis['topic']}")
            print(f"  😊 Emotion: {analysis['emotional_state']}")
            print(f"  ⭐ Importance: {analysis['importance_score']:.2f}")
            
            if result["memory_stored"]:
                stored_count += 1
                print(f"  ✅ STORED as important memory")
            else:
                print(f"  ❌ Not stored (importance too low)")
        
        print(f"\n📊 Results: {stored_count}/{len(family_conversations)} conversations stored")
        return stored_count > 0
        
    except Exception as e:
        print(f"❌ Sample creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def examine_stored_family_data():
    """Examine how the family data is actually stored"""
    print(f"\n🔍 Examining Stored Family Data")
    print("=" * 35)
    
    try:
        memory_store = get_memory_store()
        collection = memory_store.collection
        
        total_count = collection.count()
        print(f"📊 Total family memories: {total_count}")
        
        if total_count == 0:
            print("❌ No memories to examine")
            return False
        
        # Get all memories with full details
        results = collection.get(
            include=['embeddings', 'metadatas', 'documents']
        )
        
        print(f"\n📋 Detailed Memory Storage Analysis:")
        print("=" * 40)
        
        for i in range(len(results['ids'])):
            memory_id = results['ids'][i]
            content = results['documents'][i]
            metadata = results['metadatas'][i]
            embedding = results['embeddings'][i] if results['embeddings'] else []
            
            print(f"\n🔸 Memory {i+1}:")
            print(f"  📄 Memory ID: {memory_id}")
            print(f"  📝 Full Content:")
            print(f"      {content}")
            print(f"  🧠 Vector Embedding:")
            print(f"      Dimensions: {len(embedding)}")
            if embedding:
                print(f"      Sample values: {[round(x, 4) for x in embedding[:10]]}...")
            
            print(f"  📊 Rich Metadata:")
            if metadata:
                for key, value in metadata.items():
                    print(f"      {key}: {value}")
            
            print(f"  🔍 Family Context Extracted:")
            # Extract family-specific information
            family_keywords = ['family', 'mom', 'dad', 'child', 'kids', 'parent', 'grandpa', 'grandma', 'brother', 'sister', 'baby', 'pregnant', 'promotion', 'house', 'puppy', 'car', 'hospital']
            found_keywords = [word for word in family_keywords if word.lower() in content.lower()]
            if found_keywords:
                print(f"      Family keywords: {found_keywords}")
            else:
                print(f"      No specific family keywords detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Data examination failed: {e}")
        return False

def test_family_memory_search():
    """Test semantic search on family memories"""
    print(f"\n🔍 Family Memory Search Test")
    print("=" * 30)
    
    try:
        memory_store = get_memory_store()
        
        # Test family-specific searches
        family_searches = [
            ("pregnancy announcement", "mom_sarah"),
            ("job promotion celebration", "dad_mike"),
            ("pet requests", "kid_emma"),
            ("car birthday gift", "teen_alex"),
            ("medical emergency", "mom_sarah"),
            ("family accident concern", "dad_mike"),
            ("hospital visit plans", "dad_mike")
        ]
        
        for query, user_id in family_searches:
            print(f"\n🔎 Search: '{query}' (user: {user_id})")
            
            try:
                results = memory_store.get_relevant_memories(
                    user_id=user_id,
                    query=query,
                    n_results=2
                )
                
                if results:
                    print(f"  ✅ Found {len(results)} relevant memories:")
                    for j, result in enumerate(results, 1):
                        content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                        print(f"    {j}. {content_preview}")
                        
                        # Show metadata relevance
                        metadata = result.get('metadata', {})
                        topic = metadata.get('topic', 'unknown')
                        emotion = metadata.get('emotional_state', 'unknown')
                        print(f"       Topic: {topic}, Emotion: {emotion}")
                else:
                    print(f"  ❌ No relevant memories found")
            
            except Exception as e:
                print(f"  ❌ Search failed: {e}")
    
    except Exception as e:
        print(f"❌ Family search test failed: {e}")

def analyze_family_conversation_patterns():
    """Analyze patterns in family conversations"""
    print(f"\n📈 Family Conversation Pattern Analysis")
    print("=" * 40)
    
    try:
        memory_store = get_memory_store()
        results = memory_store.collection.get(include=['metadatas', 'documents'])
        
        # Analyze by family member
        family_members = {}
        topics_by_member = {}
        emotions_by_member = {}
        
        for content, metadata in zip(results['documents'], results['metadatas']):
            if metadata:
                user_id = metadata.get('user_id', 'unknown')
                topic = metadata.get('topic', 'unknown')
                emotion = metadata.get('emotional_state', 'neutral')
                importance = metadata.get('importance_score', 0)
                
                # Count by family member
                if user_id not in family_members:
                    family_members[user_id] = []
                family_members[user_id].append({
                    'content': content,
                    'topic': topic,
                    'emotion': emotion,
                    'importance': importance
                })
                
                # Track topics by member
                if user_id not in topics_by_member:
                    topics_by_member[user_id] = {}
                topics_by_member[user_id][topic] = topics_by_member[user_id].get(topic, 0) + 1
                
                # Track emotions by member
                if user_id not in emotions_by_member:
                    emotions_by_member[user_id] = {}
                emotions_by_member[user_id][emotion] = emotions_by_member[user_id].get(emotion, 0) + 1
        
        print(f"👨‍👩‍👧‍👦 Family Member Analysis:")
        for member, memories in family_members.items():
            print(f"\n  👤 {member}:")
            print(f"    Stored memories: {len(memories)}")
            avg_importance = sum(m['importance'] for m in memories) / len(memories)
            print(f"    Average importance: {avg_importance:.2f}")
            print(f"    Main topics: {list(topics_by_member[member].keys())}")
            print(f"    Emotions expressed: {list(emotions_by_member[member].keys())}")
    
    except Exception as e:
        print(f"❌ Pattern analysis failed: {e}")

async def main():
    """Run complete family memory analysis"""
    print("🚀 Family Memory Storage Analysis")
    print("How Queen stores conversation & family-specific data")
    print("=" * 60)
    
    # Create sample family memories
    success = await create_sample_family_memories()
    
    if success:
        # Examine how they're stored
        examine_stored_family_data()
        
        # Test family-specific searches
        test_family_memory_search()
        
        # Analyze conversation patterns
        analyze_family_conversation_patterns()
        
        print("\n" + "=" * 60)
        print("🎉 Family Memory Analysis Complete!")
        
        print("\n💡 Key Storage Insights:")
        print("  📝 Content: Full conversation text with user attribution")
        print("  🧠 Embeddings: 384-dimensional semantic vectors")
        print("  📊 Metadata: Rich context (topic, emotion, importance, user)")
        print("  👨‍👩‍👧‍👦 Family Context: User IDs, family relationships preserved")
        print("  🔍 Search: Semantic similarity + metadata filtering")
        print("  ⭐ Filtering: Only important conversations stored (0.7+ importance)")
        
        print("\n🏗️ Storage Architecture:")
        print("  • Each memory = Content + Vector + Metadata")
        print("  • Family members identified by user_id")
        print("  • Conversations linked to family_tree_id")
        print("  • Emotional context tracked per person")
        print("  • Topics categorized (personal, safety, etc.)")
        print("  • Timestamps preserve conversation chronology")
    else:
        print("\n❌ Could not create sample memories")

if __name__ == "__main__":
    asyncio.run(main()) 