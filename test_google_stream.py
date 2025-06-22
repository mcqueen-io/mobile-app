import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
import wave
import os
from dotenv import load_dotenv
import queue
import threading
import struct
import traceback

# Load environment variables
load_dotenv()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.03  # 30ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Audio queue for thread-safe audio passing
audio_queue = queue.Queue()

# Set the default input device to index 1 (Microphone - Realtek High Definition Audio, MME)
sd.default.device = 1
print(f"[INFO] Using input device: {sd.query_devices(1)['name']} (index 1)")

def audio_callback(indata, frames, time, status):
    """Callback for continuous audio stream"""
    if status:
        print(f"Audio callback status: {status}")
    # Put audio data in queue
    audio_queue.put(indata.copy())

async def audio_generator():
    """Generate audio chunks from the queue"""
    print("Recording started. Press Ctrl+C to stop...")
    
    # For debugging - save first few seconds of audio
    debug_audio = []
    debug_duration = 3  # seconds
    debug_samples = int(SAMPLE_RATE * debug_duration)
    
    # Use MacBook Pro Microphone explicitly (device index 1 from the test)
    device_info = sd.query_devices(1, 'input')
    print(f"Using audio device: {device_info['name']}")
    
    # Start the audio stream
    stream = sd.InputStream(
        device=1,  # MacBook Pro Microphone
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',  # Changed to int16 directly
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    
    try:
        with stream:
            chunk_count = 0
            total_samples = 0
            silent_chunks = 0
            while True:
                # Get audio from queue
                if not audio_queue.empty():
                    audio_chunk = audio_queue.get()
                    
                    # Check if chunk is silent
                    max_val = np.max(np.abs(audio_chunk))
                    if max_val < 100:
                        silent_chunks += 1
                    else:
                        if silent_chunks > 0:
                            print(f"Had {silent_chunks} silent chunks, now getting audio (max: {max_val})")
                            silent_chunks = 0
                    
                    # No conversion needed, already int16
                    audio_bytes = audio_chunk.tobytes()
                    
                    # Save debug audio
                    if total_samples < debug_samples:
                        debug_audio.extend(audio_chunk.flatten())
                        total_samples += len(audio_chunk.flatten())
                        
                        if total_samples >= debug_samples and debug_audio:
                            # Save debug audio to file
                            with wave.open('debug_google_audio.wav', 'wb') as wav_file:
                                wav_file.setnchannels(CHANNELS)
                                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                                wav_file.setframerate(SAMPLE_RATE)
                                wav_file.writeframes(struct.pack(f'{len(debug_audio[:debug_samples])}h', *debug_audio[:debug_samples]))
                            print(f"Debug audio saved to debug_google_audio.wav (first {debug_duration} seconds)")
                    
                    chunk_count += 1
                    if chunk_count % 100 == 0:  # Log every 100 chunks
                        print(f"Sent {chunk_count} audio chunks (chunk size: {len(audio_bytes)} bytes)")
                    
                    yield audio_bytes
                else:
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except Exception as e:
        print(f"Error in audio stream: {e}")
    finally:
        stream.close()

async def handle_websocket_messages(websocket):
    """Handle incoming WebSocket messages"""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get('type')
                
                if event_type == 'transcript':
                    # Handle transcription results
                    if data.get('is_final'):
                        print("\n🔥 Google Final Transcript:")
                    else:
                        print("\n📝 Google Interim Transcript:")
                    
                    # Print the full text
                    print(f"Text: {data.get('text', '')}")
                    
                    # Print speaker segments if available
                    if data.get('speakers'):
                        print("\n👥 Google Speaker Segments:")
                        for segment in data['speakers']:
                            print(f"Speaker {segment['speaker']}: {segment['text']}")
                    
                elif event_type == 'error':
                    print("❌ Google Error:", data.get('message', 'Unknown error'))
                else:
                    print("❓ Unknown event type:", event_type)
                    
            except json.JSONDecodeError:
                print("Error decoding message:", message)
            except Exception as e:
                print(f"Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print(f"Error in WebSocket message handler: {e}")

async def main():
    """Main function to handle WebSocket connection and audio streaming"""
    # Use Google transcriber endpoint - connecting to local server with provider parameter
    uri = "ws://localhost:8000/api/v1/voice/ws/stream?provider=google"
    websocket = None
    
    try:
        websocket = await websockets.connect(uri)
        print("🌐 Connected to Google Cloud Speech WebSocket server")
        print("🎙️ Starting Google Cloud Speech real-time transcription...")
        print("🗣️ Speak into your microphone. Press Ctrl+C to stop.")
        
        # Start message handler
        message_handler = asyncio.create_task(handle_websocket_messages(websocket))
        
        # Wait a moment for the connection to be fully established
        await asyncio.sleep(1)
        
        # Stream audio
        async for audio_chunk in audio_generator():
            try:
                await websocket.send(audio_chunk)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed while sending audio")
                break
            except Exception as e:
                print(f"Error sending audio: {e}")
                break
        
        # Wait for message handler to complete
        await message_handler
        
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print('Error in main loop:')
        traceback.print_exc()
    finally:
        if websocket:
            await websocket.close()
            print("WebSocket connection closed properly")

if __name__ == "__main__":
    try:
        print("🚀 Starting Google Cloud Speech-to-Text Test")
        print("📋 Configuration:")
        print(f"   Sample Rate: {SAMPLE_RATE}Hz")
        print(f"   Channels: {CHANNELS}")
        print(f"   Chunk Duration: {CHUNK_DURATION}s")
        print(f"   Chunk Size: {CHUNK_SIZE} samples")
        print("-" * 50)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Program terminated by user")
    except Exception as e:
        print(f"💥 Program terminated with error: {e}") 