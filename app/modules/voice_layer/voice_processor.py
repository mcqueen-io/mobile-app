import json
import numpy as np
import torch
from typing import Optional, Dict, Any
import sounddevice as sd
import wave
import os
import librosa
import tempfile
from speechbrain.pretrained import EncoderClassifier
import webrtcvad
import logging
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.db.chroma_manager import get_chroma_manager, ChromaManager
from app.services.unified_service import get_unified_service, UnifiedService
from fastapi import WebSocket

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
        # Ensure the collection exists
        self.chroma_manager.get_or_create_collection(self.embeddings_collection_name)
        
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

    async def _process_transcript(self, data: Dict):
        """
        Process the Deepgram transcript, identify user, fetch context, and send to Gemini.
        Handles potential tool calls from Gemini.
        """
        try:
            if self.websocket is None:
                logger.warning("WebSocket not set in VoiceProcessor.")
                return

            # Extract transcript
            transcript = None
            if data.get('channel') and data['channel'].get('alternatives'):
                transcript = data['channel']['alternatives'][0].get('transcript')
                confidence = data['channel']['alternatives'][0].get('confidence')

            if not transcript or not transcript.strip():
                logger.debug("Received empty or whitespace transcript.")
                return # Ignore empty or just whitespace transcripts

            logger.info(f"Processing transcript: \"{transcript}\" (Confidence: {confidence}) ")

            # TODO: Refine user identification during streaming.
            # Identified user ID (using self.current_speaker or by processing audio chunk)
            identified_user_id = self.current_speaker or "default_user" # Fallback
            logger.info(f"Identified user ID for transcript: {identified_user_id}")

            # --- Initial call to Gemini with proactive context ---

            # Get proactive context from UnifiedService
            proactive_context = await self.unified_service.get_relevant_context([identified_user_id], transcript)
            
            logger.info("Calling Gemini wrapper with initial context...")
            gemini_response = await get_gemini_wrapper().generate_response(
                user_id=identified_user_id, # Pass user ID
                user_input=transcript, # Pass the transcript
                context=proactive_context # Pass proactive context
            )

            # --- Process Gemini's response (text or tool call) ---

            if isinstance(gemini_response, list):
                # Gemini returned tool calls - Execute them via UnifiedService/CPL
                logger.info(f"Gemini requested tool calls: {gemini_response}")
                
                tool_results = []
                for tool_call in gemini_response:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    logger.info(f"Executing tool call: {tool_name} with args {tool_args}")

                    result = {
                        "name": tool_name,
                        "content": f"Error executing tool {tool_name}" # Default error
                    }
                    
                    try:
                        # --- Execute tool call using UnifiedService (or MCP for others) ---
                        if tool_name == "get_user_preference":
                            # Ensure user_id from identification is passed if not already in args
                            if "user_id" not in tool_args:
                                tool_args["user_id"] = identified_user_id
                                logger.warning(f"user_id not in tool_args for {tool_name}, using identified_user_id: {identified_user_id}")

                            preference_result = await self.unified_service.get_user_preference(**tool_args)
                            # Ensure the result content is a JSON string
                            result["content"] = json.dumps(preference_result)
                            
                        elif tool_name == "search_memories":
                             # Ensure user_id from identification is passed if not already in args
                            if "user_id" not in tool_args:
                                tool_args["user_id"] = identified_user_id
                                logger.warning(f"user_id not in tool_args for {tool_name}, using identified_user_id: {identified_user_id}")

                            memories_result = await self.unified_service.search_memories(**tool_args)
                             # Ensure the result content is a JSON string
                            result["content"] = json.dumps(memories_result)

                        # TODO: Add handling for other tools like 'web_search' (via MCP)
                        # elif tool_name == "web_search":
                        #    # Assuming MCP client is available in VoiceProcessor or passed in
                        #    mcp_client = await get_mcp_client() # Example getter
                        #    mcp_result = await mcp_client.execute_tool(tool_name, tool_args)
                        #    result["content"] = json.dumps(mcp_result)

                        else:
                            logger.warning(f"Unknown tool requested by Gemini: {tool_name}")
                            result["content"] = f"Unknown tool: {tool_name}"
                            
                    except Exception as tool_e:
                        logger.error(f"Error executing tool {tool_name}: {str(tool_e)}")
                        result["content"] = json.dumps({"error": f"Error executing tool {tool_name}: {str(tool_e)}"})
                    
                    tool_results.append(result)
                
                # --- Send tool results back to Gemini for follow-up ---
                
                if tool_results:
                    logger.info("Sending tool results back to Gemini for follow-up response.")
                    # Call generate_response again, this time with tool_results.
                    # We also pass the original transcript again to maintain context, although
                    # the chat history in ChatSession should handle the turn structure.
                    final_response = await get_gemini_wrapper().generate_response(
                        user_id=identified_user_id, # Pass user ID
                        user_input=transcript, # Pass original transcript
                        tool_results=tool_results # Pass the results of tool execution
                    )

                    # Process the final response (expecting text this time, but handle potential further tool calls)
                    if isinstance(final_response, str):
                         response_text = final_response
                         logger.info(f"Received final Gemini text response (first 50 chars): {response_text[:50]}...")
                         # Send final text response to client
                         response_payload = {
                             "type": "gemini_response",
                             "user_id": identified_user_id,
                             "text": response_text,
                         }
                         await self.websocket.send_json(response_payload)
                    elif isinstance(final_response, list):
                         # Handle consecutive tool calls if necessary
                         logger.warning(f"Gemini requested consecutive tool calls after receiving tool results: {final_response}")
                         # TODO: Implement robust handling for consecutive tool calls (e.g., loop back or limit depth)
                         response_text = "I need to perform another action based on the previous results. (Consecutive tool call requested)"
                         response_payload = {
                            "type": "tool_call_requested",
                            "user_id": identified_user_id,
                            "tool_calls": final_response # Send the new tool calls to client/MCP
                        }
                         await self.websocket.send_json(response_payload)
                    else:
                         logger.warning(f"Received unexpected final response type from Gemini: {type(final_response)}")
                         response_text = "I apologize, but I received an unexpected final response after executing tools."
                         response_payload = {
                             "type": "error",
                             "user_id": identified_user_id,
                             "message": response_text
                         }
                         await self.websocket.send_json(response_payload)

                else:
                    # This case should ideally not be reached if gemini_response was a list (tool calls)
                    logger.error("Tool calls were indicated, but the tool_results list is empty after execution attempt.")
                    response_text = "An internal error occurred during tool execution."
                    response_payload = {
                        "type": "error",
                        "user_id": identified_user_id,
                        "message": response_text
                    }
                    await self.websocket.send_json(response_payload)

            elif isinstance(gemini_response, str):
                # Gemini returned a text response directly (no tool calls needed)
                response_text = gemini_response
                logger.info(f"Received Gemini text response (first 50 chars): {response_text[:50]}...")

                # Send the Gemini response back to the WebSocket client
                response_payload = {
                    "type": "gemini_response",
                    "user_id": identified_user_id,
                    "text": response_text,
                }
                await self.websocket.send_json(response_payload)

            else:
                # Handle unexpected initial response types
                logger.warning(f"Received unexpected initial response type from Gemini: {type(gemini_response)}")
                response_text = "I apologize, but I received an unexpected initial response."
                response_payload = {
                    "type": "error",
                    "user_id": identified_user_id,
                    "message": response_text
                }
                await self.websocket.send_json(response_payload)

        except Exception as e:
            logger.error(f"Error in _process_transcript: {str(e)}")
            # Send an error message back to the client
            error_message = f"Error processing your request: {str(e)}"
            # Ensure websocket is still open before sending
            if self.websocket:
                await self.websocket.send_json({"type": "error", "message": error_message})

    async def process_audio(self, audio_data: bytes, user_id: str) -> None:
        """
        Process incoming audio data for a specific user.
        This is a simplified example, in a real app this would involve:
        1. User/session management
        2. Passing user context to Deepgram or other ASR
        3. Sending data in appropriate chunks
        """
        logger.debug(f"Processing {len(audio_data)} bytes of audio for user {user_id}")

        # TODO: Implement actual audio processing and sending to Deepgram/ASR
        # For demonstration, simply acknowledge receipt
        # await self.deepgram_client.send_audio(audio_data)

        # TODO: Handle potential exceptions during send
        try:
            if self.websocket:
                await self.websocket.send(audio_data)
                logger.debug("Sent audio data to Deepgram WebSocket.")
            else:
                logger.warning("Deepgram WebSocket is not connected. Cannot send audio.")
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {str(e)}")
        # TODO: Implement reconnection logic or error handling

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