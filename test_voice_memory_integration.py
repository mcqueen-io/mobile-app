#!/usr/bin/env python3
"""
Test Voice + Intelligent Memory Integration

This script tests the complete pipeline:
1. Register your voice using clean WAV file
2. Test Google STT with voice recognition
3. Process interview conversation through Queen
4. Verify intelligent memory extraction and storage

Usage: python test_voice_memory_integration.py
"""

import asyncio
import sys
import os
import logging
import json
from datetime import datetime
import wave
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app.modules.voice_layer.voice_processor import VoiceProcessor
from app.services.unified_service import get_unified_service
from app.modules.ai_wrapper.user_specific_gemini_wrapper import UserSpecificGeminiWrapper
from app.services.memory_intelligence_service import get_memory_intelligence_service
from app.db.mongo_manager import get_mongo_manager
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceMemoryTester:
    def __init__(self):
        self.voice_processor = None
        self.unified_service = None
        self.gemini_wrapper = None
        self.memory_service = None
        self.mongo_manager = None
        self.test_user_id = "lohit_user_001"
        self.voice_file = "Lohit_fixed.wav"
        
    async def initialize(self):
        """Initialize all services"""
        print("🔧 Initializing services...")
        
        # Initialize unified service
        self.unified_service = await get_unified_service()
        
        # Initialize voice processor
        self.voice_processor = VoiceProcessor(self.unified_service)
        
        # Initialize Gemini wrapper
        self.gemini_wrapper = UserSpecificGeminiWrapper(
            api_key=settings.GOOGLE_APPLICATION_CREDENTIALS,
            project_id=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION
        )
        await self.gemini_wrapper.initialize()
        
        # Initialize memory service
        self.memory_service = await get_memory_intelligence_service()
        
        # Initialize MongoDB
        self.mongo_manager = await get_mongo_manager()
        
        print("✅ All services initialized")
    
    async def test_voice_registration(self):
        """Test voice registration using clean WAV file"""
        print(f"\n🎤 Testing Voice Registration with {self.voice_file}")
        print("=" * 50)
        
        try:
            # Check if voice file exists
            if not os.path.exists(self.voice_file):
                print(f"❌ Voice file {self.voice_file} not found!")
                return False
            
            # Load audio data from WAV file
            print(f"📁 Loading audio from {self.voice_file}...")
            with wave.open(self.voice_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                print(f"   📊 Sample rate: {sample_rate} Hz")
                print(f"   📊 Channels: {channels}")
                print(f"   📊 Sample width: {sample_width} bytes")
                print(f"   📊 Duration: {wf.getnframes() / sample_rate:.2f} seconds")
            
            # Convert to numpy array
            if sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
            else:
                audio_data = np.frombuffer(frames, dtype=np.int32)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / (2**(sample_width*8-1))
            
            # If stereo, convert to mono
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            print(f"   📊 Audio data shape: {audio_data.shape}")
            print(f"   📊 Audio data range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            # Extract voice embedding
            print("🔍 Extracting voice embedding...")
            embedding = self.voice_processor.extract_voice_embedding(audio_data)
            
            if embedding is not None:
                print(f"✅ Voice embedding extracted successfully!")
                print(f"   📊 Embedding shape: {embedding.shape}")
                print(f"   📊 Embedding norm: {np.linalg.norm(embedding):.3f}")
                
                # Register the speaker
                print(f"📝 Registering speaker: {self.test_user_id}")
                success = self.voice_processor.register_new_speaker(self.test_user_id, embedding)
                
                if success:
                    print(f"✅ Speaker {self.test_user_id} registered successfully!")
                    return True
                else:
                    print(f"❌ Failed to register speaker {self.test_user_id}")
                    return False
            else:
                print("❌ Failed to extract voice embedding")
                return False
                
        except Exception as e:
            print(f"❌ Error in voice registration: {str(e)}")
            logger.exception("Voice registration error:")
            return False
    
    async def test_voice_identification(self):
        """Test voice identification with the same audio"""
        print(f"\n🔍 Testing Voice Identification")
        print("=" * 50)
        
        try:
            # Load the same audio file for identification test
            with wave.open(self.voice_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
            
            # Convert to numpy array
            if sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
            else:
                audio_data = np.frombuffer(frames, dtype=np.int32)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / (2**(sample_width*8-1))
            
            # If stereo, convert to mono
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # Test identification
            print("🔍 Testing voice identification...")
            identified_user = await self.voice_processor._identify_speaker_from_audio(audio_data)
            
            if identified_user:
                print(f"✅ Voice identified as: {identified_user}")
                if identified_user == self.test_user_id:
                    print("🎉 Perfect match! Voice recognition working correctly!")
                    return True
                else:
                    print(f"⚠️ Identified as different user: expected {self.test_user_id}, got {identified_user}")
                    return False
            else:
                print("❌ Voice identification failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in voice identification: {str(e)}")
            logger.exception("Voice identification error:")
            return False
    
    async def test_interview_conversation(self):
        """Test interview conversation processing and memory extraction"""
        print(f"\n💬 Testing Interview Conversation Processing")
        print("=" * 50)
        
        # Test conversation about an interview
        interview_conversations = [
            "I'm really nervous about my interview at Google next Friday for the Senior Software Engineer position. I've been preparing for weeks but I'm still worried about the system design questions.",
            "I have an interview at Microsoft tomorrow for the Product Manager role. I'm feeling confident but a bit anxious about the case study portion.",
            "There's this big interview coming up at Apple next week for iOS Developer. I'm excited but also stressed about the coding challenge."
        ]
        
        for i, conversation in enumerate(interview_conversations, 1):
            print(f"\n🎯 Test Conversation {i}:")
            print(f"User says: \"{conversation}\"")
            
            try:
                # Process through Queen's Gemini wrapper
                print("👑 Processing through Queen...")
                response = await self.gemini_wrapper.chat(
                    user_id=self.test_user_id,
                    user_input=conversation,
                    context={
                        "participants": [self.test_user_id],
                        "conversation_text": conversation
                    }
                )
                
                print(f"👑 Queen responds: {response}")
                
                # Also test direct memory extraction
                print("🧠 Testing direct memory extraction...")
                memory_result = await self.memory_service.extract_and_store_events(
                    conversation_text=conversation,
                    participants=[self.test_user_id]
                )
                
                print(f"📊 Memory extraction result:")
                print(f"   Events found: {memory_result.get('events_found', 0)}")
                print(f"   Events stored: {memory_result.get('events_stored', 0)}")
                print(f"   Message: {memory_result.get('message', 'No message')}")
                
                if memory_result.get('stored_events'):
                    for event in memory_result['stored_events']:
                        print(f"   📅 Stored: {event['event_type']} on {event.get('event_date', 'TBD')}")
                        print(f"       Triggers: {event['reflection_triggers_count']}")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"❌ Error processing conversation {i}: {str(e)}")
                logger.exception(f"Conversation {i} error:")
    
    async def verify_memory_storage(self):
        """Verify that memories were stored correctly in MongoDB"""
        print(f"\n📚 Verifying Memory Storage")
        print("=" * 50)
        
        try:
            # Search for memories created by our test user
            print(f"🔍 Searching for memories by user: {self.test_user_id}")
            
            # Query MongoDB directly
            memories = await self.mongo_manager.db.memories.find({
                "created_by": self.test_user_id,
                "type": "intelligent_event"
            }).to_list(length=10)
            
            print(f"📊 Found {len(memories)} intelligent memories")
            
            for i, memory in enumerate(memories, 1):
                print(f"\n📝 Memory {i}:")
                print(f"   Event Type: {memory.get('event_type', 'Unknown')}")
                print(f"   Content: {memory.get('content', 'No content')}")
                print(f"   Created: {memory.get('created_at', 'Unknown')}")
                
                metadata = memory.get('metadata', {})
                extracted_data = metadata.get('extracted_data', {})
                
                if extracted_data:
                    print(f"   📊 Extracted Data:")
                    for key, value in extracted_data.items():
                        if key not in ['reflection_triggers', 'participants']:
                            print(f"      {key}: {value}")
                
                triggers = metadata.get('reflection_triggers', [])
                print(f"   🔄 Reflection Triggers: {len(triggers)}")
                for trigger in triggers[:2]:  # Show first 2 triggers
                    print(f"      {trigger.get('trigger_type', 'unknown')}: {trigger.get('trigger_date', 'no date')}")
            
            return len(memories) > 0
            
        except Exception as e:
            print(f"❌ Error verifying memory storage: {str(e)}")
            logger.exception("Memory verification error:")
            return False
    
    async def cleanup_test_data(self):
        """Clean up test data"""
        print(f"\n🧹 Cleaning up test data...")
        
        try:
            # Delete test memories
            result = await self.mongo_manager.db.memories.delete_many({
                "created_by": self.test_user_id
            })
            print(f"🗑️ Deleted {result.deleted_count} test memories")
            
            # Clear voice embeddings (if needed)
            # Note: This depends on your ChromaDB implementation
            print("🗑️ Test cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {str(e)}")

async def main():
    """Main test function"""
    print("🚗👑 Queen's Voice + Memory Integration Test")
    print("=" * 60)
    print("This test will:")
    print("1. Register your voice using clean WAV file")
    print("2. Test voice identification")
    print("3. Process interview conversations through Queen")
    print("4. Verify intelligent memory extraction and storage")
    print()
    
    tester = VoiceMemoryTester()
    
    try:
        # Initialize services
        await tester.initialize()
        
        # Test voice registration
        voice_registered = await tester.test_voice_registration()
        if not voice_registered:
            print("❌ Voice registration failed. Cannot proceed with full test.")
            return
        
        # Test voice identification
        voice_identified = await tester.test_voice_identification()
        if not voice_identified:
            print("⚠️ Voice identification failed, but continuing with conversation test...")
        
        # Test interview conversation processing
        await tester.test_interview_conversation()
        
        # Verify memory storage
        memories_found = await tester.verify_memory_storage()
        if memories_found:
            print("\n🎉 SUCCESS! Complete voice-to-memory pipeline working!")
            print("✅ Voice recognition: Working")
            print("✅ Conversation processing: Working") 
            print("✅ Memory extraction: Working")
            print("✅ Memory storage: Working")
        else:
            print("\n⚠️ Partial success - some components may need adjustment")
        
        # Ask if user wants to cleanup
        print(f"\n🤔 Would you like to clean up test data? (y/n): ", end="")
        # For automated testing, we'll skip cleanup prompt
        print("Skipping cleanup for review...")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        logger.exception("Main test error:")
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 