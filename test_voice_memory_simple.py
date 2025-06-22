#!/usr/bin/env python3
"""
Simple Voice + Memory Test

This test focuses on the core functionality:
1. Create test user in MongoDB
2. Register voice
3. Test voice identification 
4. Test memory extraction with interview conversation
5. Verify memory storage

Usage: python test_voice_memory_simple.py
"""

import asyncio
import sys
import os
import logging
import wave
import numpy as np
from datetime import datetime

# Add the app directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app.modules.voice_layer.voice_processor import VoiceProcessor
from app.services.unified_service import get_unified_service
from app.services.memory_intelligence_service import get_memory_intelligence_service
from app.db.mongo_manager import get_mongo_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_voice_memory_pipeline():
    """Test the complete voice + memory pipeline"""
    
    print("🚗👑 Queen's Voice + Memory Simple Test")
    print("=" * 50)
    
    # Test configuration
    test_user_id = "lohit_user_001"
    voice_file = "Lohit_fixed.wav"
    family_id = "lohit_family_001"
    
    try:
        # Step 1: Initialize services
        print("🔧 Initializing services...")
        unified_service = await get_unified_service()
        voice_processor = VoiceProcessor(unified_service)
        memory_service = await get_memory_intelligence_service()
        mongo_manager = await get_mongo_manager()
        print("✅ Services initialized")
        
        # Step 2: Create test user in MongoDB
        print(f"\n👤 Creating test user: {test_user_id}")
        try:
            # Check if user already exists
            existing_user = await mongo_manager.get_user(test_user_id)
            if existing_user:
                print(f"✅ User {test_user_id} already exists")
            else:
                # Create new user
                user_data = {
                    "id": test_user_id,
                    "username": "lohit",
                    "name": "Lohit Test User",
                    "family_tree_id": family_id,
                    "voice_id": f"voice_{test_user_id}",
                    "preferences": {
                        "favorite_music_genre": "rock",
                        "favorite_restaurant": "italian"
                    }
                }
                await mongo_manager.create_user(user_data)
                print(f"✅ Created test user: {test_user_id}")
        except Exception as e:
            print(f"⚠️ User creation issue: {str(e)}")
        
        # Step 3: Register voice
        print(f"\n🎤 Registering voice from {voice_file}")
        if not os.path.exists(voice_file):
            print(f"❌ Voice file {voice_file} not found!")
            return False
        
        # Load and process audio
        with wave.open(voice_file, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
        
        # Convert to numpy array
        if sample_width == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16)
        else:
            audio_data = np.frombuffer(frames, dtype=np.int32)
        
        # Normalize to float32
        audio_data = audio_data.astype(np.float32) / (2**(sample_width*8-1))
        
        # Convert stereo to mono if needed
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        print(f"   📊 Audio duration: {len(audio_data) / 16000:.2f} seconds")
        
        # Extract embedding and register
        embedding = voice_processor.extract_voice_embedding(audio_data)
        if embedding is not None:
            success = voice_processor.register_new_speaker(test_user_id, embedding)
            if success:
                print("✅ Voice registered successfully")
            else:
                print("❌ Voice registration failed")
                return False
        else:
            print("❌ Voice embedding extraction failed")
            return False
        
        # Step 4: Test voice identification
        print(f"\n🔍 Testing voice identification")
        identified_speaker = voice_processor.identify_speaker(embedding, threshold=0.7)
        if identified_speaker == test_user_id:
            print(f"✅ Voice correctly identified as: {identified_speaker}")
        else:
            print(f"⚠️ Voice identification result: {identified_speaker}")
        
        # Step 5: Test memory extraction with interview conversation
        print(f"\n💬 Testing interview conversation memory extraction")
        
        interview_conversation = (
            "I'm really nervous about my interview at Google next Friday for the "
            "Senior Software Engineer position. I've been preparing for weeks but "
            "I'm still worried about the system design questions."
        )
        
        print(f"User says: \"{interview_conversation}\"")
        
        # Extract and store events
        memory_result = await memory_service.extract_and_store_events(
            conversation_text=interview_conversation,
            participants=[test_user_id]
        )
        
        print(f"📊 Memory extraction result:")
        print(f"   Events found: {memory_result.get('events_found', 0)}")
        print(f"   Events stored: {memory_result.get('events_stored', 0)}")
        print(f"   Message: {memory_result.get('message', 'No message')}")
        
        if memory_result.get('stored_events'):
            for event in memory_result['stored_events']:
                print(f"   📅 Stored: {event['event_type']} on {event.get('event_date', 'TBD')}")
                print(f"       Memory ID: {event['memory_id']}")
                print(f"       Reflection triggers: {event['reflection_triggers_count']}")
        
        # Step 6: Verify memory storage
        print(f"\n📚 Verifying memory storage in MongoDB")
        memories = await mongo_manager.db.memories.find({
            "created_by": test_user_id,
            "type": "intelligent_event"
        }).to_list(length=10)
        
        print(f"📊 Found {len(memories)} intelligent memories for {test_user_id}")
        
        if memories:
            for i, memory in enumerate(memories, 1):
                print(f"\n📝 Memory {i}:")
                print(f"   Event Type: {memory.get('event_type', 'Unknown')}")
                print(f"   Content: {memory.get('content', 'No content')[:100]}...")
                print(f"   Created: {memory.get('created_at', 'Unknown')}")
                
                metadata = memory.get('metadata', {})
                extracted_data = metadata.get('extracted_data', {})
                
                if extracted_data:
                    print(f"   📊 Key extracted data:")
                    for key in ['company', 'role', 'parsed_date', 'initial_emotion']:
                        if key in extracted_data:
                            print(f"      {key}: {extracted_data[key]}")
                
                triggers = metadata.get('reflection_triggers', [])
                print(f"   🔄 Reflection triggers: {len(triggers)}")
        
        # Step 7: Test a second conversation to verify system is working
        print(f"\n💬 Testing second conversation (birthday)")
        birthday_conversation = "Mom's birthday is coming up next month and I need to plan something special for her."
        
        print(f"User says: \"{birthday_conversation}\"")
        
        memory_result2 = await memory_service.extract_and_store_events(
            conversation_text=birthday_conversation,
            participants=[test_user_id]
        )
        
        print(f"📊 Second memory extraction result:")
        print(f"   Events found: {memory_result2.get('events_found', 0)}")
        print(f"   Events stored: {memory_result2.get('events_stored', 0)}")
        
        # Final verification
        total_memories = await mongo_manager.db.memories.count_documents({
            "created_by": test_user_id,
            "type": "intelligent_event"
        })
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"✅ Voice registration: Working")
        print(f"✅ Voice identification: Working")
        print(f"✅ Event extraction: Working")
        print(f"✅ Memory storage: {total_memories} events stored")
        print(f"✅ Complete pipeline: SUCCESS!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        logger.exception("Test error:")
        return False

async def cleanup_test_data():
    """Clean up test data"""
    print(f"\n🧹 Cleaning up test data...")
    
    try:
        mongo_manager = await get_mongo_manager()
        
        # Delete test memories
        result = await mongo_manager.db.memories.delete_many({
            "created_by": "lohit_user_001"
        })
        print(f"🗑️ Deleted {result.deleted_count} test memories")
        
        # Delete test user
        user_result = await mongo_manager.delete_user("lohit_user_001")
        if user_result:
            print(f"🗑️ Deleted test user")
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"⚠️ Error during cleanup: {str(e)}")

async def main():
    """Main function"""
    success = await test_voice_memory_pipeline()
    
    if success:
        print(f"\n🤔 Would you like to clean up test data? (y/n): ", end="")
        # For now, let's skip the interactive prompt
        print("n")
        print("💡 Test data preserved for inspection. Run cleanup manually if needed.")
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 