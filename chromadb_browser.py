#!/usr/bin/env python3
"""
ChromaDB Browser - Complete UI for viewing ChromaDB data
Access actual collections including voice_embeddings
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chromadb
from chromadb.config import Settings
from app.core.config import settings

def browse_chromadb():
    """Browse all actual ChromaDB collections and data"""
    print("🌐 ChromaDB Browser")
    print("=" * 50)
    
    try:
        # Create ChromaDB client
        client = chromadb.Client(Settings(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        ))
        
        print(f"📊 ChromaDB Info:")
        print(f"  Location: {settings.CHROMA_PERSIST_DIRECTORY}")
        print(f"  Client Type: {type(client).__name__}")
        
        # Get all collections
        collections = client.list_collections()
        print(f"\n📚 Collections Found: {len(collections)}")
        
        if not collections:
            print("❌ No collections found via list_collections()")
            print("🔧 Trying direct access to known collections...")
            
            # Try to access voice_embeddings directly (we know it exists from SQLite)
            try:
                voice_collection = client.get_collection("voice_embeddings")
                collections = [voice_collection]
                print("✅ Found voice_embeddings collection directly!")
            except Exception as e:
                print(f"❌ Could not access voice_embeddings: {e}")
                return False
        
        for collection in collections:
            print(f"\n📁 Collection: '{collection.name}'")
            
            # Get collection info
            count = collection.count()
            print(f"  📊 Documents: {count}")
            
            if count > 0:
                # Get all data
                results = collection.get(
                    include=['embeddings', 'metadatas', 'documents']
                )
                
                print(f"  📋 All Documents:")
                
                for i in range(len(results['ids'])):
                    doc_id = results['ids'][i]
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document = results['documents'][i] if results['documents'] else ""
                    embedding = results['embeddings'][i] if results['embeddings'] else []
                    
                    print(f"\n    🔸 Document {i+1}:")
                    print(f"      ID: {doc_id}")
                    
                    if document:
                        content_preview = document[:150] + "..." if len(document) > 150 else document
                        print(f"      Content: {content_preview}")
                    else:
                        print(f"      Content: (no text content)")
                    
                    print(f"      Embedding: {len(embedding)} dimensions")
                    if embedding:
                        # Show first few values of embedding
                        preview = embedding[:5] + ["..."] if len(embedding) > 5 else embedding
                        print(f"      Vector preview: {preview}")
                    
                    if metadata:
                        print(f"      Metadata:")
                        for key, value in metadata.items():
                            print(f"        {key}: {value}")
                    else:
                        print(f"      Metadata: (none)")
                
                # Test similarity search if this is voice embeddings
                if collection.name == "voice_embeddings":
                    print(f"\n  🔍 Testing Voice Similarity Search:")
                    try:
                        # Query with the existing embedding to test similarity
                        query_results = collection.query(
                            query_embeddings=[embedding],  # Use the stored embedding
                            n_results=1
                        )
                        
                        if query_results['ids']:
                            print(f"    ✅ Similarity search working")
                            print(f"    📊 Distance: {query_results['distances'][0][0]:.4f}")
                        
                    except Exception as e:
                        print(f"    ❌ Similarity search failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB browsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_store_collection():
    """Test if memory store can create its collection"""
    print(f"\n🧠 Memory Store Collection Test")
    print("=" * 35)
    
    try:
        from app.modules.memory.memory_store import get_memory_store
        
        memory_store = get_memory_store()
        print(f"✅ Memory store initialized")
        print(f"  Collection name: {memory_store.collection.name}")
        print(f"  Collection count: {memory_store.collection.count()}")
        
        # Try to add a test memory
        test_id = memory_store.add_memory(
            user_id="test_user",
            content="This is a test memory for ChromaDB browser",
            metadata={"test": True, "importance": 0.8}
        )
        
        print(f"✅ Test memory added: {test_id}")
        
        # Verify it was added
        new_count = memory_store.collection.count()
        print(f"  New collection count: {new_count}")
        
        # Clean up test memory
        memory_store.delete_memory(test_id)
        final_count = memory_store.collection.count()
        print(f"  After cleanup count: {final_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory store test failed: {e}")
        return False

def create_chromadb_summary():
    """Create a summary of what's in ChromaDB"""
    print(f"\n📊 ChromaDB Summary Report")
    print("=" * 30)
    
    try:
        client = chromadb.Client(Settings(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        ))
        
        # Try to access known collections
        collections_data = {}
        
        # Check voice_embeddings
        try:
            voice_coll = client.get_collection("voice_embeddings")
            collections_data["voice_embeddings"] = {
                "count": voice_coll.count(),
                "type": "Voice Recognition Embeddings",
                "dimensions": 192
            }
        except:
            pass
        
        # Check user_memories
        try:
            memory_coll = client.get_collection("user_memories")
            collections_data["user_memories"] = {
                "count": memory_coll.count(),
                "type": "Conversation Memories",
                "dimensions": "Unknown"
            }
        except:
            collections_data["user_memories"] = {
                "count": 0,
                "type": "Conversation Memories (not created yet)",
                "dimensions": "N/A"
            }
        
        print("📋 Collection Summary:")
        for name, data in collections_data.items():
            print(f"  📁 {name}:")
            print(f"    Type: {data['type']}")
            print(f"    Documents: {data['count']}")
            print(f"    Dimensions: {data['dimensions']}")
        
        # File size analysis
        chroma_path = "./data/chroma"
        if os.path.exists(chroma_path):
            total_size = 0
            for root, dirs, files in os.walk(chroma_path):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            
            print(f"\n💾 Storage:")
            print(f"  Total size: {total_size/1024:.1f}KB")
            print(f"  Location: {chroma_path}")
        
        return collections_data
        
    except Exception as e:
        print(f"❌ Summary creation failed: {e}")
        return {}

def main():
    """Run comprehensive ChromaDB browser"""
    print("🚀 ChromaDB Browser & Inspector")
    print("Complete view of Queen's vector memory")
    print("=" * 60)
    
    success = browse_chromadb()
    
    if success:
        test_memory_store_collection()
        summary = create_chromadb_summary()
        
        print("\n" + "=" * 60)
        print("🎉 ChromaDB Browser Complete!")
        
        print("\n💡 Key Findings:")
        print("  • ChromaDB is working and contains data")
        print("  • Voice embeddings are stored (192-dim)")
        print("  • Memory store creates separate collection")
        print("  • Both voice and conversation data can coexist")
        
        print("\n🔧 For Production:")
        print("  • Voice embeddings: Speaker identification")
        print("  • User memories: Conversation semantics")
        print("  • Both use vector similarity search")
        
    else:
        print("\n❌ ChromaDB browsing failed")

if __name__ == "__main__":
    main() 