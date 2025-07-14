import json
import numpy as np
import torch
from typing import Optional, Dict, Any
import sounddevice as sd
import wave
import io
import os
import librosa
import tempfile
from speechbrain.pretrained import EncoderClassifier
import webrtcvad
import logging
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.db.chroma_manager import get_chroma_manager, ChromaManager
from app.services.unified_service import get_unified_service, UnifiedService
from fastapi import WebSocket, UploadFile

logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self, unified_service: UnifiedService):
        self.websocket: Optional[WebSocket] = None
        self.voice_encoder = None
        self.vad = None
        self.current_speaker = None
        self.speaker_embeddings = {}
        self.sample_rate = 16000
        self.chunk_duration = 0.03  # 30ms chunks for VAD
        self.embedding_size = 192  # ECAPA-TDNN embedding size
        
        self.unified_service = unified_service
        
        # Initialize voice encoder and VAD
        self._initialize_voice_models()

        # List available audio devices
        self._list_audio_devices()
        
        # Initialize ChromaDB Manager
        self.chroma_manager: ChromaManager = get_chroma_manager()
        self.embeddings_collection_name = "voice_embeddings"
        # Ensure the collection exists with cosine similarity for voice embeddings
        self._initialize_voice_embeddings_collection()
        
    def _initialize_voice_embeddings_collection(self):
        """Initialize voice embeddings collection with cosine similarity"""
        try:
            # Check if collection exists
            existing_collections = [col.name for col in self.chroma_manager.client.list_collections()]
            if self.embeddings_collection_name in existing_collections:
                logger.info(f"Retrieved existing voice embeddings collection: {self.embeddings_collection_name}")
            else:
                # Create collection with cosine similarity for voice embeddings
                collection = self.chroma_manager.client.create_collection(
                    name=self.embeddings_collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity like memory store
                )
                logger.info(f"Created new voice embeddings collection with cosine similarity: {self.embeddings_collection_name}")
        except Exception as e:
            logger.error(f"Error initializing voice embeddings collection: {str(e)}")
            raise

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
            # Initialize SpeechBrain ECAPA-TDNN for speaker recognition
            self.voice_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            print("ECAPA-TDNN voice encoder initialized successfully")
            
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
                return audio_data

            # Convert to 16-bit PCM for VAD
            if audio_data.dtype != np.float32 or np.max(np.abs(audio_data)) > 1.0:
                print("Warning: Audio data format not ideal for VAD. Attempting conversion.")
                audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
                audio_data = np.clip(audio_data, -1.0, 1.0)

            audio_int16 = (audio_data * 32767).astype(np.int16)

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
                        print(f"VAD processing error: {str(e)}")
                        continue
            
            if not processed_frames:
                print("VAD: No speech detected in processed frames.")
                return np.array([])

            # Combine processed frames and convert back to float32
            processed_audio_int16 = np.concatenate(processed_frames)
            processed_audio_float32 = processed_audio_int16.astype(np.float32) / 32767.0

            print(f"Processed audio length after VAD: {len(processed_audio_float32)} samples")
            return processed_audio_float32

        except Exception as e:
            print(f"Error in VAD processing: {str(e)}")
            return np.array([])

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
        """Extract voice embedding using ECAPA-TDNN"""
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
                    wf.setsampwidth(2)
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
                
                print("Computing voice embedding using ECAPA-TDNN...")
                embedding = self.voice_encoder.encode_batch(torch.tensor(wav).unsqueeze(0).to(self.voice_encoder.device))
                print("Voice embedding computed successfully.")
                
                # Convert tensor to numpy array and squeeze dimensions
                embedding = embedding.squeeze().detach().cpu().numpy()
                print(f"Embedding shape: {embedding.shape}")
                
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

    def register_new_speaker(self, name: str, embedding: np.ndarray) -> bool:
        """
        Register a new speaker by storing their embedding in ChromaDB.
        
        Args:
            name: The name or ID of the speaker (preferably unique user_id from MongoDB).
            embedding: The voice embedding as a numpy array.
        
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        try:
            # Use name as ID for ChromaDB entry for now. Ideally, this should be the MongoDB user_id.
            chroma_id = name # TODO: Use MongoDB user_id here
            metadata = {"speaker_name": name} # Store relevant metadata
            
            # Add embedding to ChromaDB collection
            self.chroma_manager.add_embedding(
                collection_name=self.embeddings_collection_name,
                embedding=embedding.tolist(), # Convert numpy array to list for ChromaDB
                metadata=metadata,
                id=chroma_id
            )
            logger.info(f"Registered speaker {name} with embedding ID {chroma_id} in ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error registering speaker {name} in ChromaDB: {str(e)}")
            return False

    def identify_speaker(self, embedding: np.ndarray, threshold: float = 0.5) -> Optional[str]:
        """
        Identify a speaker by querying similar embeddings in ChromaDB.
        
        Args:
            embedding: The voice embedding to identify.
            threshold: Cosine distance threshold for identification (lower is better, typical range 0-2).
        
        Returns:
            Optional[str]: The name/ID of the identified speaker (or None if no match above threshold).
        """
        try:
            # Query ChromaDB for similar embeddings
            results = self.chroma_manager.query_embeddings(
                collection_name=self.embeddings_collection_name,
                query_embedding=embedding.tolist(), # Convert numpy array to list
                n_results=1 # Get the single most similar result
            )
            print("results: ", results)
            
            logger.debug(f"ChromaDB query results: {results}")

            if results and results['ids'] and results['ids'][0]:
                # Get the most similar result
                most_similar_id = results['ids'][0][0]
                distance = results['distances'][0][0]
                print("distance: ", distance)
                metadata = results['metadatas'][0][0]
                print("metadata: ", metadata)
                
                # ChromaDB returns cosine distance (lower is better, range 0-2)
                # For cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
                # Typical good match threshold: 0.3-0.7 depending on voice quality
                
                logger.info(f"Voice similarity check: distance={distance:.4f}, threshold={threshold}")
                
                if distance is not None and distance < threshold:
                    identified_speaker_name = metadata.get("speaker_name")
                    logger.info(f"Identified speaker {identified_speaker_name} with cosine distance {distance:.4f} (ID: {most_similar_id})")
                    print("identified_speaker_name: ", identified_speaker_name)
                    return identified_speaker_name # TODO: Return MongoDB user_id here
                else:
                    logger.info(f"No speaker identified above threshold {threshold} (closest cosine distance: {distance:.4f})")
                    print(f"Speaker not identified: distance {distance:.4f} > threshold {threshold}")
                    return None
            else:
                logger.info("No results from ChromaDB query")
                return None
            
        except Exception as e:
            logger.error(f"Error identifying speaker from ChromaDB: {str(e)}")
            return None

    def set_websocket(self, websocket: WebSocket):
        self.websocket = websocket

    async def process_audio(self, audio_file: UploadFile) -> Dict[str, Any]:
        """
        Process incoming audio data.
        """
        
        # TODO: Implement actual audio processing
        # Read the audio file
        try:
            contents = await audio_file.read()
            logger.info(f"Read {len(contents)} bytes from audio file")
            
            # Convert to numpy array using wave
            with wave.open(io.BytesIO(contents), 'rb') as wf:
                # Get audio parameters
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                logger.info(f"Audio parameters: channels={n_channels}, sample_width={sample_width}, "
                        f"frame_rate={frame_rate}, n_frames={n_frames}")

                print(f"Audio parameters: channels={n_channels}, sample_width={sample_width}, "
                        f"frame_rate={frame_rate}, n_frames={n_frames}")
                
                # Read audio data
                audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                
                # Convert to float32 and normalize
                audio_data = audio_data.astype(np.float32) / 32767.0
                
                # Convert stereo to mono if needed
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                processed_audio = self._process_with_vad(audio_data)
                logger.info(f"Processed audio length: {len(processed_audio)}")
                
                if len(processed_audio) == 0:
                    logger.warning("No speech detected in the audio")
                    return {
                        "success": False,
                        "message": "No speech detected in the audio. Please ensure the audio contains clear speech.",
                        "status_code": 400
                    }
                
                embedding = self.extract_voice_embedding(processed_audio)

                if embedding is None:
                    logger.warning("Failed to extract voice features")
                    return {
                        "success": False,
                        "message": "Failed to extract voice features. Please ensure the audio is clear and contains speech.",
                        "status_code": 400
                    }
                
                return {
                    "success": True,
                    "message": "Audio processed successfully",
                    "status_code": 200,
                    "embedding": embedding
                }
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing audio: {str(e)}",
                "status_code": 500
            }

    async def get_user_voice_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve or determine user-specific voice settings/preferences.
        This could involve looking up preferences in the database via UnifiedService.
        """
        # Example: Fetch user preferences (simplified)
        user = await self.unified_service.get_user(user_id)
        if user and user.preferences:
            # Assuming 'voice_settings' is a dictionary within UserPreferences
            return user.preferences.voice_settings.model_dump() # Use model_dump if voice_settings is a Pydantic model
        return {} # Default empty settings

    async def stop_stream(self) -> None:
        """Stop the WebSocket stream"""
        if self.websocket:
            try:
                # Send a close message to Deepgram's WebSocket
                # TODO: Confirm the correct method to close Deepgram stream gracefully
                # await self.websocket.finish() # This might be the method based on deepgram-python SDK docs
                await self.websocket.close() # Standard WebSocket close
                logger.info("Deepgram WebSocket stream stopped.")
            except Exception as e:
                logger.error(f"Error stopping Deepgram stream: {str(e)}")
            finally:
                self.websocket = None # Ensure the websocket object is cleared

# Singleton instance
_voice_processor = None

async def get_voice_processor() -> VoiceProcessor:
    """Get or initialize singleton instance of VoiceProcessor"""
    global _voice_processor
    if _voice_processor is None:
        unified_service = await get_unified_service()
        # Pass only unified_service to the constructor
        _voice_processor = VoiceProcessor(unified_service)
    return _voice_processor 