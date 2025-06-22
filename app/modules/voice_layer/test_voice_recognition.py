import asyncio
import os
from pathlib import Path
from app.modules.voice_layer.voice_processor import get_voice_processor

async def test_voice_recognition():
    """Test the voice recognition system with noise suppression"""
    try:
        # Get voice processor instance
        processor = await get_voice_processor()
        
        # Create test recordings directory
        test_dir = Path("test_recordings")
        test_dir.mkdir(exist_ok=True)
        
        print("\n=== Voice Recognition Test with Noise Suppression ===")
        print("This test will:")
        print("1. Record a sample with noise suppression")
        print("2. Extract voice features using Resemblyzer")
        print("3. Register you as a new speaker")
        print("4. Test speaker identification")
        print("\nPress Enter to start the test...")
        input()
        
        # Test 1: Record sample
        print("\nTest 1: Recording sample")
        print("Speak clearly into your microphone...")
        audio_data = processor.record_audio()
        if audio_data is not None:
            processor.save_audio(audio_data, "test_recordings/sample.wav")
            print("Sample recorded and saved successfully")
        else:
            print("Failed to record sample")
            return
            
        # Test 2: Extract voice embedding
        print("\nTest 2: Extracting voice embedding")
        embedding = processor.extract_voice_embedding(audio_data)
        if embedding is not None:
            print(f"Voice embedding extracted successfully (shape: {embedding.shape})")
        else:
            print("Failed to extract voice embedding")
            return
            
        # Test 3: Register new speaker
        print("\nTest 3: Registering new speaker")
        processor.register_new_speaker("test_user", embedding)
        
        # Test 4: Test speaker identification
        print("\nTest 4: Testing speaker identification")
        print("Please speak again to verify your identity...")
        test_audio = processor.record_audio()
        if test_audio is not None:
            processor.save_audio(test_audio, "test_recordings/test.wav")
            test_embedding = processor.extract_voice_embedding(test_audio)
            if test_embedding is not None:
                speaker = processor.identify_speaker(test_embedding)
                if speaker:
                    print(f"Successfully identified speaker: {speaker}")
                else:
                    print("Could not identify speaker")
            else:
                print("Failed to extract test embedding")
        else:
            print("Failed to record test audio")
            
        print("\nTest completed!")
        print("Voice samples saved in test_recordings directory")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_voice_recognition()) 