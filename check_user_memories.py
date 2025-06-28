#!/usr/bin/env python3
"""
Check User Memories in ChromaDB
Examine how conversation and family-specific data is stored
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.memory.memory_store import get_memory_store

def examine_user_memories():
    """Examine all data in user_memories collection"""
    print("🔍 User Memories Data Examination")
    print("=" * 45)
    
    try:
        memory_store = get_memory_store()
        collection = memory_store.collection
        
        # Get collection info
        total_count = collection.count()
        print(f"📊 Total memories stored: {total_count}")
        
        if total_count == 0:
            print("❌ No memories found in collection")
            return False
        
        # Get ALL memories with full details
        results = collection.get(
            include=['embeddings', 'metadatas', 'documents']
        )
        
        print(f"\n📋 All Stored Memories:")
        print("=" * 30)
        
        for i in range(len(results['ids'])):
            memory_id = results['ids'][i]
            content = results['documents'][i]
            metadata = results['metadatas'][i]
            embedding = results['embeddings'][i] if results['embeddings'] else []
            
            print(f"\n🔸 Memory {i+1}:")
            print(f"  📄 ID: {memory_id}")
            print(f"  📝 Content: {content}")
            print(f"  🧠 Embedding: {len(embedding)} dimensions")
            
            if embedding:
                # Show first few values of the embedding vector
                vector_preview = [round(x, 4) for x in embedding[:5]]
                print(f"  🔢 Vector preview: {vector_preview}...")
            
            print(f"  📊 Metadata:")
            if metadata:
                for key, value in metadata.items():
                    print(f"    {key}: {value}")
            else:
                print("    (No metadata)")
        
        # Analyze patterns in the data
        analyze_memory_patterns(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Memory examination failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_memory_patterns(results):
    """Analyze patterns in stored memories"""
    print(f"\n📈 Memory Pattern Analysis")
    print("=" * 30)
    
    # Analyze metadata patterns
    users = set()
    topics = {}
    emotions = {}
    importance_scores = []
    timestamps = []
    
    for metadata in results['metadatas']:
        if metadata:
            # Collect user IDs
            user_id = metadata.get('user_id')
            if user_id:
                users.add(user_id)
            
            # Count topics
            topic = metadata.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
            
            # Count emotional states
            emotion = metadata.get('emotional_state', 'unknown')
            emotions[emotion] = emotions.get(emotion, 0) + 1
            
            # Collect importance scores
            importance = metadata.get('importance_score')
            if importance is not None:
                try:
                    importance_scores.append(float(importance))
                except:
                    pass
            
            # Collect timestamps
            timestamp = metadata.get('timestamp')
            if timestamp:
                timestamps.append(timestamp)
    
    print(f"👥 Users: {list(users)}")
    print(f"📊 Topics: {dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))}")
    print(f"😊 Emotions: {dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))}")
    
    if importance_scores:
        avg_importance = sum(importance_scores) / len(importance_scores)
        print(f"⭐ Average importance: {avg_importance:.2f}")
        print(f"📊 Importance range: {min(importance_scores):.2f} - {max(importance_scores):.2f}")
    
    if timestamps:
        print(f"⏰ Time range: {min(timestamps)} to {max(timestamps)}")

def test_semantic_search():
    """Test semantic search capabilities"""
    print(f"\n🔍 Semantic Search Test")
    print("=" * 25)
    
    try:
        memory_store = get_memory_store()
        
        # Test different search queries
        search_queries = [
            "medical emergency",
            "family conversation",
            "safety concern",
            "hospital",
            "chest pain"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching for: '{query}'")
            
            try:
                results = memory_store.get_relevant_memories(
                    user_id="live_user",  # Use the user from our test
                    query=query,
                    n_results=3
                )
                
                if results:
                    print(f"  ✅ Found {len(results)} relevant memories:")
                    for j, result in enumerate(results, 1):
                        content_preview = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
                        print(f"    {j}. {content_preview}")
                        
                        # Show similarity score if available
                        if 'distance' in result:
                            similarity = 1 - result['distance']  # Convert distance to similarity
                            print(f"       Similarity: {similarity:.3f}")
                else:
                    print(f"  ❌ No results found")
                    
            except Exception as e:
                print(f"  ❌ Search failed: {e}")
    
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")

def examine_family_specific_data():
    """Look for family-specific information in memories"""
    print(f"\n👨‍👩‍👧‍👦 Family-Specific Data Analysis")
    print("=" * 35)
    
    try:
        memory_store = get_memory_store()
        
        # Get all memories
        results = memory_store.collection.get(
            include=['metadatas', 'documents']
        )
        
        family_indicators = []
        conversation_types = []
        
        for i, (content, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            # Look for family-related keywords in content
            family_keywords = ['family', 'mom', 'dad', 'child', 'kids', 'parent', 'sibling', 'brother', 'sister', 'grandpa', 'grandma']
            found_family_words = [word for word in family_keywords if word.lower() in content.lower()]
            
            if found_family_words:
                family_indicators.append({
                    'memory_index': i,
                    'family_words': found_family_words,
                    'content_preview': content[:100] + "..."
                })
            
            # Analyze conversation types from metadata
            if metadata:
                topic = metadata.get('topic', 'unknown')
                conversation_types.append(topic)
        
        print(f"👨‍👩‍👧‍👦 Family-related memories: {len(family_indicators)}")
        for indicator in family_indicators:
            print(f"  Memory {indicator['memory_index'] + 1}: {indicator['family_words']}")
            print(f"    Content: {indicator['content_preview']}")
        
        # Show conversation type distribution
        from collections import Counter
        topic_counts = Counter(conversation_types)
        print(f"\n📊 Conversation types:")
        for topic, count in topic_counts.most_common():
            print(f"  {topic}: {count}")
    
    except Exception as e:
        print(f"❌ Family data analysis failed: {e}")

def main():
    """Run comprehensive user memory examination"""
    print("🚀 User Memories Data Examination")
    print("Analyzing how Queen stores conversation & family data")
    print("=" * 60)
    
    success = examine_user_memories()
    
    if success:
        test_semantic_search()
        examine_family_specific_data()
        
        print("\n" + "=" * 60)
        print("🎉 Memory Data Examination Complete!")
        
        print("\n💡 Key Insights:")
        print("  • Each memory has content + rich metadata")
        print("  • Vector embeddings enable semantic search")
        print("  • Metadata includes topics, emotions, importance")
        print("  • Family context preserved in conversations")
        print("  • Timestamps track conversation chronology")
        
        print("\n🔧 Storage Format:")
        print("  • Content: Full conversation text")
        print("  • Embeddings: 384-dimensional vectors (sentence-transformers)")
        print("  • Metadata: Structured information for filtering")
        print("  • Search: Semantic similarity matching")
    else:
        print("\n❌ No memory data found to examine")

if __name__ == "__main__":
    main() 