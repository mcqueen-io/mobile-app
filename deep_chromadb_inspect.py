#!/usr/bin/env python3
"""
Deep ChromaDB Inspector - Find ALL data in ChromaDB
Look for voice embeddings, hidden collections, and any stored data
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chromadb
from chromadb.config import Settings
from app.core.config import settings

def deep_inspect_chromadb():
    """Deep inspection of ChromaDB to find all data"""
    print("🔍 Deep ChromaDB Inspector")
    print("=" * 50)
    
    try:
        # Create direct ChromaDB client
        client = chromadb.Client(Settings(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        ))
        
        print(f"📊 ChromaDB Database Info:")
        print(f"  Location: {settings.CHROMA_PERSIST_DIRECTORY}")
        print(f"  Client: {type(client).__name__}")
        
        # List ALL collections
        collections = client.list_collections()
        print(f"\n📚 ALL Collections ({len(collections)}):")
        
        total_documents = 0
        
        for collection in collections:
            print(f"\n  📁 Collection: '{collection.name}'")
            
            # Get collection stats
            count = collection.count()
            total_documents += count
            print(f"    📊 Document count: {count}")
            
            if count > 0:
                # Get all documents with full details
                try:
                    results = collection.get(
                        include=['embeddings', 'metadatas', 'documents']
                    )
                    
                    print(f"    📋 Documents:")
                    
                    # Show all documents (not just first 3)
                    for i in range(len(results['ids'])):
                        doc_id = results['ids'][i]
                        metadata = results['metadatas'][i] if results['metadatas'] else {}
                        document = results['documents'][i] if results['documents'] else ""
                        embedding_size = len(results['embeddings'][i]) if results['embeddings'] and results['embeddings'][i] else 0
                        
                        print(f"\n      🔸 Document {i+1}:")
                        print(f"        ID: {doc_id}")
                        print(f"        Content: {document[:200]}{'...' if len(document) > 200 else ''}")
                        print(f"        Embedding: {embedding_size} dimensions")
                        
                        if metadata:
                            print(f"        Metadata:")
                            for key, value in metadata.items():
                                print(f"          {key}: {value}")
                        else:
                            print(f"        Metadata: (none)")
                            
                except Exception as e:
                    print(f"    ❌ Error reading collection: {e}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total Collections: {len(collections)}")
        print(f"  Total Documents: {total_documents}")
        
        # Check for specific collections we expect
        expected_collections = ['user_memories', 'voice_embeddings', 'conversation_memories']
        print(f"\n🔍 Looking for expected collections:")
        for expected in expected_collections:
            found = any(c.name == expected for c in collections)
            status = "✅ Found" if found else "❌ Missing"
            print(f"  {expected}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deep ChromaDB inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_chromadb_files():
    """Check ChromaDB files for more clues"""
    print(f"\n📁 ChromaDB File Analysis")
    print("=" * 30)
    
    chroma_path = "./data/chroma"
    
    if os.path.exists(chroma_path):
        print(f"📂 {chroma_path}/")
        
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(chroma_path):
            level = root.replace(chroma_path, '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            print(f"{indent}📁 {folder_name}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
                
                size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                print(f"{subindent}📄 {file} ({size_str})")
                
                # Special analysis for data files
                if file.endswith('.bin') and 'data' in file:
                    print(f"{subindent}   💾 This looks like vector data!")
                elif file.endswith('.sqlite3'):
                    print(f"{subindent}   🗄️ SQLite metadata database")
        
        print(f"\n📊 File Summary:")
        print(f"  Total files: {file_count}")
        print(f"  Total size: {total_size/1024:.1f}KB")
        
        if total_size > 100000:  # > 100KB
            print(f"  💡 Significant data present - there should be memories stored!")
    else:
        print("❌ ChromaDB directory not found")

def try_direct_collection_access():
    """Try to access collections by known names"""
    print(f"\n🔧 Direct Collection Access Test")
    print("=" * 35)
    
    try:
        client = chromadb.Client(Settings(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        ))
        
        # Try to access collections that might exist
        collection_names = [
            'user_memories',
            'voice_embeddings', 
            'conversation_memories',
            'memories',
            'embeddings',
            'default'
        ]
        
        for name in collection_names:
            try:
                collection = client.get_collection(name)
                count = collection.count()
                print(f"  ✅ '{name}': {count} documents")
                
                if count > 0:
                    # Get a sample
                    sample = collection.get(limit=1, include=['metadatas', 'documents'])
                    if sample['ids']:
                        print(f"    📄 Sample ID: {sample['ids'][0]}")
                        if sample['documents']:
                            content = sample['documents'][0][:100]
                            print(f"    📝 Sample content: {content}...")
                        
            except Exception as e:
                print(f"  ❌ '{name}': Not found")
    
    except Exception as e:
        print(f"❌ Direct access failed: {e}")

def main():
    """Run deep ChromaDB inspection"""
    print("🚀 Deep ChromaDB Data Inspector")
    print("Find ALL data stored in ChromaDB")
    print("=" * 60)
    
    success = deep_inspect_chromadb()
    check_chromadb_files()
    try_direct_collection_access()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Deep ChromaDB Inspection Complete!")
        print("\n💡 Key Findings:")
        print("  • ChromaDB files exist with significant data")
        print("  • Check if data is in different collections")
        print("  • Voice embeddings might be stored separately")
        print("  • Recent memories should be in user_memories collection")
    else:
        print("❌ Deep inspection failed")

if __name__ == "__main__":
    main() 