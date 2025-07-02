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
import base64
import time
from collections import deque

# Load environment variables
load_dotenv()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.03  # 30ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Audio queue for thread-safe audio passing
audio_queue = queue.Queue()

# Audio playback using robust buffering
playback_queue = queue.Queue()
playback_active = False
audio_stream = None

# Audio buffer for smooth playback
audio_buffer = deque()
buffer_lock = threading.Lock()
min_buffer_samples = SAMPLE_RATE * 0.5  # 0.5 seconds minimum buffer (much more aggressive)
target_buffer_samples = SAMPLE_RATE * 1.5  # 1.5 seconds target buffer (reduced for faster startup)
audio_started = False  # Flag to track if audio playback has started
tts_in_progress = False  # Flag to track if TTS is currently sending chunks

# Session tracking for debugging
tts_session_counter = 0

# Auto-detect suitable audio devices
def get_suitable_input_device():
    """Find a suitable input device"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Has input capability
            return i, device
    return None, None

def get_suitable_output_device():
    """Find a suitable output device"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:  # Has output capability
            return i, device
    return None, None

# Get suitable devices
input_device_idx, input_device_info = get_suitable_input_device()
output_device_idx, output_device_info = get_suitable_output_device()

if input_device_info:
    print(f"[INFO] Using input device: {input_device_info['name']} (index {input_device_idx})")
else:
    print("[ERROR] No suitable input device found!")

if output_device_info:
    print(f"[INFO] Using output device: {output_device_info['name']} (index {output_device_idx})")
else:
    print("[ERROR] No suitable output device found!")

def audio_callback(indata, frames, time, status):
    """Callback for continuous audio stream"""
    if status:
        print(f"Audio callback status: {status}")
    # Put audio data in queue
    audio_queue.put(indata.copy())

def buffer_manager_thread():
    """Thread to manage audio buffer from TTS chunks"""
    global playback_active
    
    print("🔧 Audio buffer manager started")
    
    while playback_active:
        try:
            # Get audio chunk from playback queue with minimal timeout for faster processing
            audio_chunk = playback_queue.get(timeout=0.01)
            
            if audio_chunk is not None:
                with buffer_lock:
                    # Add samples to buffer
                    audio_buffer.extend(audio_chunk)
                    buffer_size = len(audio_buffer)
                    
                    # CRITICAL FIX: Much more conservative buffer trimming for long audio
                    # Only trim if buffer becomes extremely large (30+ seconds) to prevent loss of early chunks
                    max_buffer_samples = SAMPLE_RATE * 30  # 30 seconds max (much larger to keep long audio)
                    if buffer_size > max_buffer_samples:
                        # Only remove small amount to prevent aggressive trimming
                        excess_samples = int(buffer_size - (SAMPLE_RATE * 25))  # Keep 25 seconds
                        for _ in range(min(excess_samples, len(audio_buffer))):
                            audio_buffer.popleft()
                        session_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                        print(f"⚠️ CONSERVATIVE trim: reduced buffer from {buffer_size} to {len(audio_buffer)} samples (session had {session_chunks} chunks)")
                    
                    # Debug info (reduced frequency)
                    if hasattr(buffer_manager_thread, 'chunk_count'):
                        buffer_manager_thread.chunk_count += 1
                    else:
                        buffer_manager_thread.chunk_count = 1
                    
                    # More frequent logging for long audio debugging
                    if buffer_manager_thread.chunk_count % 10 == 0:
                        buffer_seconds = buffer_size / SAMPLE_RATE
                        session_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                        print(f"🔊 Buffer manager: {buffer_size} samples ({buffer_seconds:.1f}s) from {session_chunks} TTS chunks")
                        
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in buffer manager: {e}")
    
    print("🔧 Audio buffer manager stopped")

def audio_playback_callback(outdata, frames, time, status):
    """Callback for continuous audio output stream with ZERO-DELAY immediate startup"""
    global audio_started, tts_in_progress
    
    if status:
        print(f"Audio playback callback status: {status}")
    
    # Initialize output with silence
    outdata.fill(0)
    
    try:
        with buffer_lock:
            buffer_size = len(audio_buffer)
            
            # ULTRA-EARLY startup - start immediately on ANY audio if TTS is active
            should_start_immediately = (
                not audio_started and 
                tts_in_progress and 
                buffer_size > 0  # Start on ANY audio chunks, no minimum threshold
            )
            
            # For TTS that's already complete, start with much smaller buffer
            should_start_buffered = (
                not audio_started and 
                not tts_in_progress and 
                buffer_size >= 512  # Very small threshold when TTS is done
            )
            
            # Continue playing if already started and we have any audio
            should_continue = audio_started and buffer_size > 0
            
            if should_start_immediately:
                audio_started = True
                tts_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                print(f"🚀 SESSION #{tts_session_counter} ULTRA-EARLY START: Audio with {buffer_size} samples ({buffer_size/SAMPLE_RATE:.3f}s) - TTS chunk #{tts_chunks}")
            elif should_start_buffered:
                audio_started = True
                print(f"🎵 SESSION #{tts_session_counter} SMALL-BUFFER START: Playing {buffer_size} samples ({buffer_size/SAMPLE_RATE:.1f}s) - TTS complete")
            
            if (should_start_immediately or should_start_buffered or should_continue):
                # Get samples from buffer
                samples_to_play = min(frames, buffer_size)
                
                if samples_to_play > 0:
                    # Extract samples from buffer
                    audio_samples = []
                    for _ in range(samples_to_play):
                        if audio_buffer:
                            audio_samples.append(audio_buffer.popleft())
                    
                    # Fill output buffer
                    if audio_samples:
                        audio_array = np.array(audio_samples, dtype=np.float32)
                        outdata[:len(audio_array), 0] = audio_array
                        
                        # Debug output (very reduced frequency)
                        if hasattr(audio_playback_callback, 'play_count'):
                            audio_playback_callback.play_count += 1
                        else:
                            audio_playback_callback.play_count = 1
                        
                        # Log first 10 callbacks of each session to track initial chunks
                        if audio_playback_callback.play_count <= 10:
                            remaining_samples = len(audio_buffer)
                            tts_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                            print(f"🎵 SESSION #{tts_session_counter} Callback #{audio_playback_callback.play_count}: played {len(audio_array)} samples, {remaining_samples} remaining (TTS chunks: {tts_chunks})")
                        elif audio_playback_callback.play_count % 300 == 0:  # Every 300 callbacks
                            remaining_samples = len(audio_buffer)
                            remaining_seconds = remaining_samples / SAMPLE_RATE
                            tts_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                            print(f"🎵 Audio streaming (callback #{audio_playback_callback.play_count}), buffer: {remaining_samples} samples ({remaining_seconds:.1f}s), TTS chunks: {tts_chunks}")
                
                # Handle low buffer situations - but continue playing to avoid gaps
                if audio_started and buffer_size < 256 and buffer_size > 0:  # Very low buffer
                    if hasattr(audio_playback_callback, 'underrun_count'):
                        audio_playback_callback.underrun_count += 1
                    else:
                        audio_playback_callback.underrun_count = 1
                    
                    if audio_playback_callback.underrun_count % 100 == 0:
                        tts_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                        print(f"⚠️ SESSION #{tts_session_counter} Buffer very low #{audio_playback_callback.underrun_count} - {buffer_size} samples remaining (TTS chunks: {tts_chunks})")
            
            # Stop playing only if buffer is completely empty and TTS is done
            elif audio_started and buffer_size == 0 and not tts_in_progress:
                audio_started = False
                total_callbacks = getattr(audio_playback_callback, 'play_count', 0)
                total_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                print(f"🔇 SESSION #{tts_session_counter} END: Audio playback completed - {total_callbacks} callbacks played {total_chunks} TTS chunks")
                        
    except Exception as e:
        print(f"Error in audio playback callback: {e}")

def start_audio_playback():
    """Start continuous audio playback stream with buffer management"""
    global audio_stream, playback_active, audio_started, tts_in_progress
    
    if output_device_idx is None:
        print("[ERROR] No suitable output device found!")
        return False
    
    try:
        print(f"[INFO] Starting fresh audio playback on device: {output_device_info['name']} (index {output_device_idx})")
        
        # Initialize audio state for new session
        audio_started = False
        tts_in_progress = False
        
        # Ensure buffer is clear for new session
        with buffer_lock:
            audio_buffer.clear()
        
        # Clear any remaining queue items
        while not playback_queue.empty():
            try:
                playback_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start buffer manager thread
        playback_active = True
        buffer_thread = threading.Thread(target=buffer_manager_thread, daemon=True)
        buffer_thread.start()
        
        # Create continuous output stream with optimized settings for immediate response
        audio_stream = sd.OutputStream(
            device=output_device_idx,
            channels=1,  # Mono output
            samplerate=SAMPLE_RATE,
            callback=audio_playback_callback,
            blocksize=256,  # Even smaller blocks for better responsiveness
            dtype=np.float32,
            latency='low'  # Use low latency mode
        )
        
        audio_stream.start()
        print("✅ Fresh audio playback stream started with session isolation")
        return True
        
    except Exception as e:
        print(f"Error starting audio playback: {e}")
        playback_active = False
        return False

def stop_audio_playback():
    """Stop continuous audio playback stream"""
    global audio_stream, playback_active
    
    print("🛑 Stopping audio playback...")
    playback_active = False
    
    # Give buffer manager time to finish
    time.sleep(0.5)
    
    if audio_stream:
        try:
            audio_stream.stop()
            audio_stream.close()
            print("✅ Audio playback stream stopped")
        except Exception as e:
            print(f"Error stopping audio playback: {e}")
        finally:
            audio_stream = None
    
    # Clear buffer
    with buffer_lock:
        audio_buffer.clear()
        print("🔧 Audio buffer cleared")

async def audio_generator():
    """Generate audio chunks from the queue"""
    print("Recording started. Press Ctrl+C to stop...")
    
    if input_device_idx is None:
        print("[ERROR] No suitable input device found!")
        return
    
    # For debugging - save first few seconds of audio
    debug_audio = []
    debug_duration = 3  # seconds
    debug_samples = int(SAMPLE_RATE * debug_duration)
    
    # Use detected input device
    device_info = input_device_info
    print(f"Using audio input device: {device_info['name']} (index {input_device_idx})")
    
    # Start the audio stream
    stream = sd.InputStream(
        device=input_device_idx,  # Use detected input device
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
    global audio_started, tts_in_progress
    
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
                
                elif event_type == 'tts_start':
                    # Handle TTS start notification
                    global tts_session_counter
                    tts_session_counter += 1
                    
                    print(f"\n🎵 TTS SESSION #{tts_session_counter} Starting: {data.get('text_preview', '')}")
                    print("🔄 COMPLETE AUDIO RESTART for session isolation...")
                    
                    # COMPLETE AUDIO STREAM RESTART for perfect session isolation
                    global audio_started, tts_in_progress, playback_active, audio_stream
                    
                    # 1. Force stop current audio
                    audio_started = False
                    tts_in_progress = False
                    
                    # 2. Stop and restart the entire audio playback system
                    print("🛑 Stopping audio stream for session restart...")
                    if audio_stream:
                        try:
                            audio_stream.stop()
                            audio_stream.close()
                        except Exception as e:
                            print(f"Warning stopping audio stream: {e}")
                        audio_stream = None
                    
                    # 3. Clear all buffers completely
                    playback_active = False
                    time.sleep(0.2)  # Give time for threads to stop
                    
                    # Clear queue
                    queue_size = 0
                    while not playback_queue.empty():
                        try:
                            playback_queue.get_nowait()
                            queue_size += 1
                        except queue.Empty:
                            break
                    
                    # Clear buffer
                    with buffer_lock:
                        buffer_size = len(audio_buffer)
                        audio_buffer.clear()
                    
                    # 4. Reset all counters
                    if hasattr(audio_playback_callback, 'play_count'):
                        audio_playback_callback.play_count = 0
                    if hasattr(audio_playback_callback, 'underrun_count'):
                        audio_playback_callback.underrun_count = 0
                    if hasattr(handle_websocket_messages, 'audio_chunk_count'):
                        handle_websocket_messages.audio_chunk_count = 0
                    
                    print(f"🧹 SESSION #{tts_session_counter} COMPLETE RESTART: Cleared {queue_size} queued chunks, {buffer_size} buffered samples")
                    
                    # 5. Restart audio playback system
                    print("🚀 Restarting fresh audio stream...")
                    if not start_audio_playback():
                        print("❌ Failed to restart audio playback")
                        return
                    
                    # 6. Set new session as active
                    tts_in_progress = True
                    print(f"✅ SESSION #{tts_session_counter} fresh audio system ready - ZERO contamination!")
                
                elif event_type == 'tts_audio':
                    # Handle TTS audio chunk
                    try:
                        audio_data = base64.b64decode(data['audio_data'])
                        audio_size = data.get('size', len(audio_data))
                        
                        # TTS returns linear16 PCM data (16-bit signed integers)
                        # Convert from bytes to int16 array, then to float32 for playback
                        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Convert to float32 in range [-1, 1] with proper scaling
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        
                        # Apply volume normalization if audio is too quiet
                        max_amplitude = np.max(np.abs(audio_float32))
                        if max_amplitude > 0:
                            # Boost quiet audio slightly
                            if max_amplitude < 0.05:  # Very quiet audio
                                volume_boost = min(3.0, 0.2 / max_amplitude)  # Cap at 3x boost
                                audio_float32 *= volume_boost
                            
                            # Ensure we don't clip
                            max_after_boost = np.max(np.abs(audio_float32))
                            if max_after_boost > 0.95:  # Leave some headroom
                                audio_float32 *= (0.95 / max_after_boost)
                        
                        # Track chunks with extra logging for first few
                        if hasattr(handle_websocket_messages, 'audio_chunk_count'):
                            handle_websocket_messages.audio_chunk_count += 1
                        else:
                            handle_websocket_messages.audio_chunk_count = 1
                        
                        chunk_num = handle_websocket_messages.audio_chunk_count
                        
                        # Enhanced logging to track where chunks go
                        with buffer_lock:
                            buffer_samples_before = len(audio_buffer)
                        
                        # Add to playback queue for buffer manager (consistent processing for all chunks)
                        playback_queue.put(audio_float32)
                        
                        # Check buffer immediately (no delay for zero-latency processing)  
                        with buffer_lock:
                            buffer_samples_after = len(audio_buffer)
                        
                        # Log first 10 chunks in detail to debug truncation
                        if chunk_num <= 10:
                            queue_size = playback_queue.qsize()
                            print(f"🎯 SESSION #{tts_session_counter} Chunk #{chunk_num}: {len(audio_float32)} samples")
                            print(f"   Buffer before: {buffer_samples_before}, after: {buffer_samples_after} ({buffer_samples_after/SAMPLE_RATE:.2f}s)")
                            print(f"   Queue size: {queue_size}, Audio started: {audio_started}")
                        
                        # Enhanced logging for long audio: Log every 10th chunk to track progression
                        elif chunk_num % 10 == 0:
                            buffer_seconds = buffer_samples_after / SAMPLE_RATE
                            queue_size = playback_queue.qsize()
                            print(f"🔊 SESSION #{tts_session_counter}: Chunk #{chunk_num} processed, buffer: {buffer_samples_after} samples ({buffer_seconds:.1f}s), queue: {queue_size}, audio_started: {audio_started}")
                            
                            # Special alert for potential long audio
                            if chunk_num >= 50:  # Likely a long audio response
                                print(f"🚨 LONG AUDIO DETECTED: Session #{tts_session_counter} has {chunk_num} chunks - monitoring for truncation")
                        
                        # Log every 50th chunk for very long responses
                        elif chunk_num % 50 == 0:
                            buffer_seconds = buffer_samples_after / SAMPLE_RATE
                            print(f"🔊 SESSION #{tts_session_counter}: LONG AUDIO - Chunk #{chunk_num}, buffer: {buffer_samples_after} samples ({buffer_seconds:.1f}s)")
                        
                        # Minimal rate limiting - let it flow fast for immediate startup
                        current_queue_size = playback_queue.qsize()
                        if current_queue_size > 100:  # Only limit if queue gets extremely large
                            time.sleep(0.0001)  # Minimal delay only for extreme overflow
                            print(f"⚠️ SESSION #{tts_session_counter}: Rate limiting at chunk #{chunk_num} due to large queue ({current_queue_size})")
                    
                    except Exception as e:
                        print(f"Error processing TTS audio: {e}")
                        import traceback
                        traceback.print_exc()
                
                elif event_type == 'tts_end':
                    # Handle TTS completion
                    tts_in_progress = False  # Mark TTS as completed
                    
                    success = data.get('success', False)
                    total_chunks = getattr(handle_websocket_messages, 'audio_chunk_count', 0)
                    
                    with buffer_lock:
                        buffer_samples = len(audio_buffer)
                        buffer_seconds = buffer_samples / SAMPLE_RATE
                    
                    print(f"\n🎵 TTS SESSION #{tts_session_counter} {'Completed' if success else 'Failed'}: {data.get('message', '')}")
                    print(f"📊 Session #{tts_session_counter} Summary: {total_chunks} chunks processed")
                    print(f"🔊 Buffer Status: {buffer_samples} samples ({buffer_seconds:.1f}s) ready for playback")
                    print(f"🎯 Audio State: Started={audio_started}, TTS Active={tts_in_progress}")
                    print(f"✅ Session #{tts_session_counter} ended - waiting for audio to complete before next session")
                    
                    if buffer_samples > 0:
                        estimated_playback_time = buffer_seconds
                        print(f"⏳ Estimated playback time: {estimated_playback_time:.1f} seconds")
                    
                    # Add a session separator for debugging
                    print("=" * 60)
                    print("🎤 Ready for next conversation...")
                    print("=" * 60)
                
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
    global playback_active
    
    # Use Google transcriber endpoint - connecting to local server with provider parameter
    uri = "ws://localhost:8000/api/v1/voice/ws/stream?provider=google"
    websocket = None
    
    # Start continuous audio playback
    if not start_audio_playback():
        print("❌ Failed to start audio playback")
        return
    
    try:
        websocket = await websockets.connect(uri)
        print("🌐 Connected to Google Cloud Speech WebSocket server")
        print("🎙️ Starting Google Cloud Speech real-time transcription...")
        print("🗣️ Speak into your microphone. Press Ctrl+C to stop.")
        print("🔊 TTS audio will be captured and played IMMEDIATELY - no delays!")
        print("🚀 Ultra-aggressive audio system - captures every word from the start!")
        print("-" * 50)
        
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
        # Stop audio playback
        stop_audio_playback()
        
        if websocket:
            await websocket.close()
            print("WebSocket connection closed properly")

if __name__ == "__main__":
    try:
        print("🚀 Starting Google Cloud Speech-to-Text Test with TTS Playback")
        print("📋 Configuration:")
        print(f"   Sample Rate: {SAMPLE_RATE}Hz")
        print(f"   Channels: {CHANNELS}")
        print(f"   Chunk Duration: {CHUNK_DURATION}s")
        print(f"   Chunk Size: {CHUNK_SIZE} samples")
        print("-" * 50)
        
        # List available audio devices for debugging
        print("🎵 Available Audio Devices:")
        print("Input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default_marker = " (SELECTED)" if i == input_device_idx else ""
                print(f"   [{i}] {device['name']} - {device['max_input_channels']} input channels{default_marker}")
        
        print("\nOutput devices:")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                default_marker = " (SELECTED)" if i == output_device_idx else ""
                print(f"   [{i}] {device['name']} - {device['max_output_channels']} output channels{default_marker}")
        print("-" * 50)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Program terminated by user")
    except Exception as e:
        print(f"💥 Program terminated with error: {e}")
        import traceback
        traceback.print_exc() 