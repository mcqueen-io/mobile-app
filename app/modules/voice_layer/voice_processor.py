import asyncio
import json
from app.core.config import settings
import numpy as np
import torch
import torchaudio
from typing import Optional, Dict, Any, List
import sounddevice as sd
import wave
import os
from pathlib import Path
import librosa
import queue
import threading
import time
import tempfile
from speechbrain.pretrained import EncoderClassifier
import webrtcvad

class VoiceProcessor:
    def __init__(self):
        self.deepgram = None
        if hasattr(settings, 'DEEPGRAM_API_KEY') and settings.DEEPGRAM_API_KEY:
            try:
                from deepgram import Deepgram
                self.deepgram = Deepgram(settings.DEEPGRAM_API_KEY)
            except Exception as e:
                print(f"Warning: Could not initialize Deepgram: {e}")
        else:
            print("Warning: Deepgram API key not found. Deepgram features will be disabled.")
        
        self.websocket = None
        self.voice_encoder = None
        self.vad = None
        self.current_speaker = None
        self.speaker_embeddings = {}
        self.sample_rate = 16000
        self.chunk_duration = 0.03  # 30ms chunks for VAD
        self.embedding_size = 256  # Resemblyzer embedding size
        
        # Initialize voice encoder and VAD
        self._initialize_voice_models()
        
        # List available audio devices
        self._list_audio_devices()
        
    def _list_audio_devices(self):
        """List available audio input devices"""
        try:
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Only show input devices
                    print(f"{i}: {device['name']}")
            print("\nUsing default input device")
        except Exception as e:
            print(f"Error listing audio devices: {str(e)}")
        
    def _initialize_voice_models(self):
        """Initialize the voice recognition models"""
        try:
            # Initialize SpeechBrain ECAPA-TDNN
            self.voice_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
            print("Voice encoder initialized successfully")
            
            # Initialize WebRTC VAD
            self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
            if self.vad:
                print("Voice Activity Detection initialized successfully")
            else:
                print("Error: Voice Activity Detection failed to initialize.")
            
        except Exception as e:
            print(f"Error initializing voice models: {str(e)}")
            raise

    def record_audio(self, duration: int = 5) -> np.ndarray:
        """Record audio from microphone with user-controlled duration."""
        try:
            print("Press Enter to start recording...")
            input()
            print("Recording started. Press Enter to stop recording...")
            
            # Configure audio settings
            sd.default.samplerate = self.sample_rate
            sd.default.channels = 1
            sd.default.dtype = 'float32'
            
            # Start recording
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocking=False
            )
            
            # Wait for user to press Enter to stop recording
            input()
            sd.stop()
            print("Recording stopped.")
            
            # Apply VAD and process only speech frames
            processed_audio = self._process_with_vad(recording)
            
            return processed_audio
            
        except Exception as e:
            print(f"Error recording audio: {str(e)}")
            return None

    def _process_with_vad(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply VAD to audio data and return only speech frames."""
        try:
            if self.vad is None:
                print("VAD not initialized.")
                return audio_data # Return original data if VAD is not available

            # Convert to 16-bit PCM for VAD and ensure correct format
            # Check if audio_data is float32 and within valid range [-1.0, 1.0]
            if audio_data.dtype != np.float32 or np.max(np.abs(audio_data)) > 1.0:
                print("Warning: Audio data format not ideal for VAD. Attempting conversion.")
                # Attempt conversion if not in expected format
                audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
                audio_data = np.clip(audio_data, -1.0, 1.0)

            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Ensure sample rate is supported by VAD (8000, 16000, or 32000 Hz)
            if self.sample_rate not in [8000, 16000, 32000]:
                print(f"Unsupported sample rate for VAD: {self.sample_rate}. Skipping VAD.")
                return audio_data # Return original data if sample rate is not supported

            # Process in 30ms chunks (required by WebRTC VAD)
            frame_duration_ms = 30
            frame_size = int(self.sample_rate * frame_duration_ms / 1000)
            
            processed_frames = []
            for i in range(0, len(audio_int16), frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    try:
                        is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                        if is_speech:
                            processed_frames.append(frame)
                    except Exception as e:
                        # This catch is important if webrtcvad throws errors on certain frames
                        print(f"VAD processing error: {str(e)}")
                        continue # Skip problematic frames
            
            if not processed_frames:
                print("VAD: No speech detected in processed frames.")
                return np.array([]) # Return empty array if no speech

            # Combine processed frames and convert back to float32
            processed_audio_int16 = np.concatenate(processed_frames)
            processed_audio_float32 = processed_audio_int16.astype(np.float32) / 32767.0

            print(f"Processed audio length after VAD: {len(processed_audio_float32)} samples")
            return processed_audio_float32

        except Exception as e:
            print(f"Error in VAD processing: {str(e)}")
            # In case of a VAD processing error, it's safer to return an empty array
            # or None, rather than potentially corrupted data.
            return np.array([]) # Indicate failure to process with VAD

    def save_audio(self, audio_data: np.ndarray, filename: str):
        """Save audio data to WAV file with error handling"""
        try:
            if audio_data is None:
                print("No audio data to save")
                return
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for float32
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            print(f"Audio saved to {filename}")
            
        except Exception as e:
            print(f"Error saving audio: {str(e)}")

    def extract_voice_embedding(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract voice embedding using Resemblyzer"""
        try:
            if self.voice_encoder is None:
                print("Voice encoder not initialized")
                return None
                
            if audio_data is None:
                print("No audio data provided")
                return None
            
            if len(audio_data) == 0:
                print("Audio data is empty after VAD. Skipping embedding extraction.")
                return None
            
            # Save audio data to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 2 bytes for float32
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            try:
                # Load and preprocess audio using librosa
                wav, sr = librosa.load(temp_file.name, sr=self.sample_rate)
                
                # Ensure audio is mono
                if len(wav.shape) > 1:
                    wav = librosa.to_mono(wav)
                
                # Normalize audio
                wav = librosa.util.normalize(wav)
                
                print("Computing voice embedding...")
                embedding = self.voice_encoder.encode_batch(torch.tensor(wav).unsqueeze(0).to(self.voice_encoder.device))
                print("Voice embedding computed successfully.")
                
                # Convert tensor to numpy array and squeeze dimensions
                embedding = embedding.squeeze().detach().cpu().numpy()
                print(f"Embedding shape after processing: {embedding.shape}")
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return embedding
                
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                return None
            
        except Exception as e:
            print(f"Error extracting voice embedding: {str(e)}")
            return None

    def identify_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """Identify speaker from voice embedding using cosine similarity"""
        try:
            if not self.speaker_embeddings:
                print("No registered speakers found")
                return None
                
            if embedding is None:
                print("No embedding provided")
                return None
                
            # Calculate cosine similarity with all registered speakers
            best_match = None
            best_score = -1
            
            for speaker_id, stored_embedding in self.speaker_embeddings.items():
                # Ensure embeddings are 1D numpy arrays for dot product
                embedding_flat = embedding.flatten()
                stored_embedding_flat = stored_embedding.flatten()
                
                similarity = np.dot(embedding_flat, stored_embedding_flat) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                if similarity > best_score:
                    best_score = similarity
                    best_match = speaker_id
                    
            # Only return a match if similarity is above threshold
            # Adjusting threshold for SpeechBrain ECAPA-TDNN
            return best_match if best_score > 0.6 else None
            
        except Exception as e:
            print(f"Error identifying speaker: {str(e)}")
            return None

    def register_new_speaker(self, speaker_id: str, embedding: np.ndarray) -> None:
        """Register a new speaker's voice embedding"""
        try:
            if embedding is None:
                print("No embedding provided")
                return
                
            self.speaker_embeddings[speaker_id] = embedding
            print(f"Speaker {speaker_id} registered successfully")
            
        except Exception as e:
            print(f"Error registering speaker: {str(e)}")

    async def start_stream(self, callback):
        """Start a WebSocket stream with Deepgram for real-time transcription"""
        if not self.deepgram:
            print("Deepgram is not initialized. Streaming is unavailable.")
            return None
        try:
            from deepgram import Deepgram
            self.websocket = await self.deepgram.transcription.live({
                'punctuate': True,
                'interim_results': False,
                'language': 'en-US',
                'model': 'nova',
                'diarize': True
            })

            self.websocket.registerHandler(
                self.websocket.event.TRANSCRIPT_RECEIVED,
                lambda data: callback(json.loads(data))
            )

            return self.websocket

        except Exception as e:
            print(f"Error starting stream: {str(e)}")
            raise

    async def process_audio(self, audio_data: bytes) -> None:
        """Process audio data through the WebSocket"""
        if self.websocket:
            await self.websocket.send(audio_data)

    async def stop_stream(self) -> None:
        """Stop the WebSocket stream"""
        if self.websocket:
            await self.websocket.finish()
            self.websocket = None

# Create a singleton instance
voice_processor = VoiceProcessor()

def get_voice_processor():
    return voice_processor 