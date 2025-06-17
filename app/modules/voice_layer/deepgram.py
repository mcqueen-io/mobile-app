import asyncio
import json
import os
import tempfile
import wave
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
from app.core.config import settings
from app.modules.voice_layer.voice_processor import get_voice_processor
import logging

logger = logging.getLogger(__name__)

class DeepgramTranscriber:
    def __init__(self):
        self.deepgram = None
        self.live_transcription = None
        self.is_connected = False
        self._on_transcript_callback = None
        self._event_loop = None
        
        # Audio file management
        self._temp_dir = tempfile.mkdtemp()
        self._audio_file = None
        self._audio_writer = None
        self._session_id = None
        self._speaker_map = {}  # Cache for speaker identification
        
        # Conversation tracking
        self._conversation_segments = []  # Store all segments for reconstruction
        self._current_utterance_segments = []  # Store segments for current utterance
        
        # Voice processor for speaker identification
        self._voice_processor = None
        
        self._initialize_deepgram()

    def _initialize_deepgram(self):
        """Initialize the Deepgram client with appropriate options."""
        try:
            if not hasattr(settings, 'DEEPGRAM_API_KEY') or not settings.DEEPGRAM_API_KEY:
                raise ValueError("Deepgram API key not found in settings")

            # Use default config as shown in app.py
            self.deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY)
            logger.info("Deepgram client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Deepgram client: {str(e)}")
            raise

    def _initialize_audio_file(self, session_id: str):
        """Initialize audio file for a new session."""
        try:
            self._session_id = session_id
            file_path = os.path.join(self._temp_dir, f"{session_id}.wav")
            
            # Create WAV file with proper parameters
            self._audio_writer = wave.open(file_path, 'wb')
            self._audio_writer.setnchannels(1)  # Mono
            self._audio_writer.setsampwidth(2)  # 16-bit
            self._audio_writer.setframerate(16000)  # 16kHz
            
            self._audio_file = file_path
            logger.info(f"Initialized audio file for session {session_id}")
            print(f"📁 Audio file being saved to: {file_path}")
            print(f"📁 Temp directory: {self._temp_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize audio file: {str(e)}")
            raise

    def _write_audio_chunk(self, audio_data: bytes):
        """Write audio chunk to the WAV file."""
        try:
            if self._audio_writer:
                self._audio_writer.writeframes(audio_data)
            else:
                logger.warning("Audio writer is None, cannot write audio chunk")
        except Exception as e:
            logger.error(f"Failed to write audio chunk: {str(e)}")
            # Try to reinitialize the audio writer if it failed
            if self._session_id and self._audio_file:
                try:
                    logger.info("Attempting to reinitialize audio writer")
                    self._audio_writer = wave.open(self._audio_file, 'wb')
                    self._audio_writer.setnchannels(1)
                    self._audio_writer.setsampwidth(2)
                    self._audio_writer.setframerate(16000)
                    self._audio_writer.writeframes(audio_data)
                    logger.info("Audio writer reinitialized successfully")
                except Exception as reinit_error:
                    logger.error(f"Failed to reinitialize audio writer: {str(reinit_error)}")
                    raise
            else:
                raise

    def _close_audio_file(self):
        """Close the audio file and clean up."""
        try:
            if self._audio_writer:
                self._audio_writer.close()
                self._audio_writer = None
            if self._audio_file and os.path.exists(self._audio_file):
                # Note: We'll implement S3 upload later
                # For now, just log the file path
                logger.info(f"Audio file ready for processing: {self._audio_file}")
        except Exception as e:
            logger.error(f"Failed to close audio file: {str(e)}")
            raise

    def _on_open(self, ws_client, open_event, **kwargs):
        """Handle connection open event."""
        logger.info("Deepgram connection opened")
        self.is_connected = True

    def _on_message(self, ws_client, result, **kwargs):
        print(" _on_message called!")
        """Handle incoming transcript messages."""
        try:
            transcript = result.channel.alternatives[0].transcript
            print(f"Received transcript: '{transcript}' (is_final: {result.is_final}) with words: {result.channel.alternatives[0].words}")
            
            # Always send transcript data, even if empty
            transcript_data = {
                "type": "transcript",
                "text": transcript,
                "is_final": result.is_final,
                "speakers": []
            }
            
            # Extract speaker information if available and transcript is not empty
            if len(transcript) > 0 and hasattr(result.channel.alternatives[0], 'words'):
                words = result.channel.alternatives[0].words
                
                # Extract speaker segments with timestamps
                speaker_segments = self._extract_speaker_segments(words)
                
                # Process segments for voice embedding if this is final
                if result.is_final:
                    print(f"🎯 Processing {len(speaker_segments)} speaker segments for voice embedding")
                    # Schedule async processing on the event loop
                    if self._event_loop:
                        for segment in speaker_segments:
                            asyncio.run_coroutine_threadsafe(
                                self._process_speaker_segment(segment),
                                self._event_loop
                            )
                
                # Format speaker data for response (existing logic)
                current_speaker = None
                current_text = []
                
                for word in words:
                    speaker = getattr(word, 'speaker', 'unknown')
                    if speaker != current_speaker:
                        if current_text:
                            transcript_data["speakers"].append({
                                "speaker": current_speaker,
                                "text": " ".join(current_text)
                            })
                        current_speaker = speaker
                        current_text = []
                    current_text.append(word.word)
                
                # Add the last speaker's text
                if current_text:
                    transcript_data["speakers"].append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text)
                    })
            
            # Schedule the async callback on the event loop
            if self._on_transcript_callback and self._event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._on_transcript_callback(transcript_data),
                    self._event_loop
                )
                
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            logger.exception("Full traceback:")

    async def _process_speaker_segment(self, segment: Dict[str, Any]):
        """Process a speaker segment for voice embedding and identification."""
        try:
            speaker_num = segment["speaker"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            text = " ".join(segment["text"])
            
            print(f"🔍 Processing segment - Speaker {speaker_num}: '{text}' ({start_time:.2f}s - {end_time:.2f}s)")
            
            # Check if we already have this speaker identified
            if speaker_num in self._speaker_map:
                identified_user = self._speaker_map[speaker_num]
                print(f"✅ Speaker {speaker_num} already identified as: {identified_user}")
                segment["identified_user"] = identified_user
            else:
                print(f"🔍 Speaker {speaker_num} not in cache, extracting voice embedding...")
                
                # Extract audio segment for this speaker
                audio_segment = self._extract_audio_segment_with_speaker_info(start_time, end_time, speaker_num)
                
                if audio_segment is not None and len(audio_segment) > 0:
                    print(f"📊 Extracted {len(audio_segment)} audio samples for Speaker {speaker_num}")
                    
                    # Use voice processor to identify speaker
                    identified_user = await self._identify_speaker_from_audio(audio_segment)
                    
                    if identified_user:
                        # Cache the identification for future segments
                        self._speaker_map[speaker_num] = identified_user
                        segment["identified_user"] = identified_user
                        print(f"✅ Speaker {speaker_num} identified as: {identified_user}")
                    else:
                        print(f"❓ Speaker {speaker_num} could not be identified")
                        segment["identified_user"] = f"Unknown_Speaker_{speaker_num}"
                    
                    segment["audio_length"] = len(audio_segment)
                else:
                    print(f"❌ Failed to extract audio for Speaker {speaker_num}")
                    segment["identified_user"] = f"Unknown_Speaker_{speaker_num}"
            
            # Store segment for conversation reconstruction
            self._current_utterance_segments.append(segment)
            
        except Exception as e:
            logger.error(f"Error processing speaker segment: {str(e)}")
            logger.exception("Full traceback:")

    async def _identify_speaker_from_audio(self, audio_segment: np.ndarray) -> Optional[str]:
        """Identify speaker using voice embedding extraction and comparison."""
        try:
            if self._voice_processor is None:
                logger.warning("Voice processor not available for speaker identification")
                return None
            
            # Apply VAD to clean the audio segment
            processed_audio = self._voice_processor._process_with_vad(audio_segment)
            
            if len(processed_audio) == 0:
                print("⚠️ No speech detected after VAD processing")
                return None
            
            print(f"🎙️ Audio segment processed with VAD: {len(processed_audio)} samples")
            
            # Extract voice embedding
            embedding = self._voice_processor.extract_voice_embedding(processed_audio)
            
            if embedding is None:
                print("❌ Failed to extract voice embedding")
                return None
            
            print(f"🧠 Voice embedding extracted: shape {embedding.shape}")
            
            # Identify speaker using the voice processor
            identified_user_id = self._voice_processor.identify_speaker(embedding)
            
            if identified_user_id:
                print(f"🎯 Speaker identified: {identified_user_id}")
                return identified_user_id
            else:
                print("❓ No matching speaker found in database")
                return None
                
        except Exception as e:
            logger.error(f"Error identifying speaker from audio: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def _on_metadata(self, ws_client, metadata, **kwargs):
        """Handle metadata events."""
        logger.info(f"Metadata received: {metadata}")

    def _on_speech_started(self, ws_client, speech_started, **kwargs):
        print(" _on_speech_started called!")
        """Handle speech started events."""
        logger.info("Speech Started")

    def _on_utterance_end(self, ws_client, utterance_end, **kwargs):
        print(" _on_utterance_end called!", utterance_end)
        """Handle utterance end events."""
        logger.info("Utterance End")
        
        # Schedule async processing for utterance end
        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._process_utterance_end(utterance_end),
                self._event_loop
            )

    async def _process_utterance_end(self, utterance_end):
        """Process utterance end and reconstruct conversation."""
        try:
            print(f"🏁 Utterance ended. Processing {len(self._current_utterance_segments)} segments")
            
            if not self._current_utterance_segments:
                print("⚠️ No segments to process for this utterance")
                return
            
            # Reconstruct the conversation for this utterance
            conversation_text = self._reconstruct_conversation(self._current_utterance_segments)
            print(f"📝 Reconstructed conversation: {conversation_text}")
            
            # Add segments to conversation history
            self._conversation_segments.extend(self._current_utterance_segments)
            
            # Show speaker statistics
            stats = self.get_speaker_statistics()
            print(f"📊 Speaker Statistics:")
            print(f"   Total speakers: {stats['total_speakers']}")
            print(f"   Identified: {len(stats['identified_speakers'])}")
            print(f"   Unknown: {len(stats['unknown_speakers'])}")
            
            if stats['identified_speakers']:
                print("   Identified speakers:")
                for speaker in stats['identified_speakers']:
                    print(f"     Speaker {speaker['speaker_number']}: {speaker['display_name']}")
            
            # TODO: Send to LLM here
            # llm_response = await self._send_to_llm(conversation_text)
            # await self._send_llm_response_to_client(llm_response)
            
            # Clear current utterance segments
            self._current_utterance_segments = []
            
        except Exception as e:
            logger.error(f"Error processing utterance end: {str(e)}")
            logger.exception("Full traceback:")

    def _reconstruct_conversation(self, segments: List[Dict[str, Any]]) -> str:
        """Reconstruct conversation text with speaker attribution."""
        try:
            conversation_lines = []
            
            for segment in segments:
                speaker_num = segment["speaker"]
                text = " ".join(segment["text"])
                
                # Use identified user if available, otherwise use speaker number
                if "identified_user" in segment:
                    identified_user_id = segment["identified_user"]
                    
                    # If it's a real user ID (not Unknown_Speaker_X), try to get user details
                    if not identified_user_id.startswith("Unknown_Speaker_"):
                        speaker_name = self._get_speaker_display_name(identified_user_id)
                    else:
                        speaker_name = identified_user_id
                else:
                    speaker_name = f"Speaker_{speaker_num}"
                
                conversation_lines.append(f"{speaker_name}: {text}")
            
            return "\n".join(conversation_lines)
            
        except Exception as e:
            logger.error(f"Error reconstructing conversation: {str(e)}")
            return "Error reconstructing conversation"

    def _get_speaker_display_name(self, user_id: str) -> str:
        """Get display name for a user ID. For now, return the user_id itself."""
        try:
            # TODO: In the future, we could fetch user details from the database
            # For now, just return the user_id
            return f"User_{user_id}"
        except Exception as e:
            logger.error(f"Error getting speaker display name: {str(e)}")
            return f"User_{user_id}"

    def _on_close(self, ws_client, close, **kwargs):
        """Handle connection close events."""
        logger.info(f"Deepgram connection closed: {close}")
        self.is_connected = False
        # Clean up resources
        if self._audio_writer:
            try:
                self._close_audio_file()
            except Exception as e:
                logger.error(f"Error closing audio file: {str(e)}")

    def _on_error(self, ws_client, error, **kwargs):
        """Handle error events."""
        logger.error(f"Deepgram error: {error}")
        self.is_connected = False

    def _on_unhandled(self, ws_client, unhandled, **kwargs):
        """Handle unhandled events."""
        logger.warning(f"Unhandled Websocket Message: {unhandled}")

    async def start_transcription(self, on_transcript: Callable[[Dict[str, Any]], None], session_id: str):
        print("start_transcription called!", session_id)
        """Start a live transcription session."""
        try:
            if not self.deepgram:
                raise RuntimeError("Deepgram client not initialized")

            # Reset connection state
            self.is_connected = False
            self._on_transcript_callback = on_transcript
            self._event_loop = asyncio.get_event_loop()

            # Initialize audio file for this session
            self._initialize_audio_file(session_id)
            
            # Initialize voice processor for speaker identification
            await self._initialize_voice_processor()

            # Create new connection using websocket.v("1") as shown in app.py
            self.live_transcription = self.deepgram.listen.websocket.v("1")
            logger.info("Created Deepgram WebSocket connection")

            # Register all event handlers as shown in app.py
            logger.info("Registering event handlers...")
            
            self.live_transcription.on(LiveTranscriptionEvents.Open, self._on_open)
            self.live_transcription.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self.live_transcription.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            self.live_transcription.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self.live_transcription.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self.live_transcription.on(LiveTranscriptionEvents.Close, self._on_close)
            self.live_transcription.on(LiveTranscriptionEvents.Error, self._on_error)
            self.live_transcription.on(LiveTranscriptionEvents.Unhandled, self._on_unhandled)
            
            logger.info("Event handlers registered")

            # Define options following app.py pattern
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                interim_results=True,
                utterance_end_ms="5000",
                vad_events=True,
                endpointing=1000,
                diarize=True,
                punctuate=True,
            )

            logger.info("Starting Deepgram connection...")
            success = self.live_transcription.start(options)
            if not success:
                logger.error("Failed to start Deepgram connection")
                raise RuntimeError("Failed to start Deepgram connection")
            
            logger.info("Deepgram connection started successfully")

            await asyncio.sleep(1)
            if not self.is_connected:
                logger.warning("Connection not established after 1 second, but continuing...")

            return True

        except Exception as e:
            logger.error(f"Failed to start transcription: {str(e)}")
            if self.live_transcription:
                self.live_transcription.finish()
            return False

    async def send_audio(self, audio_data: bytes):
        """Send audio data to Deepgram for transcription and save to file."""
        try:
            if not self.live_transcription:
                raise RuntimeError("Transcription not started")

            # Write audio chunk to file
            self._write_audio_chunk(audio_data)

            # Send to Deepgram
            logger.debug(f"Sending audio chunk of size: {len(audio_data)} bytes")
            self.live_transcription.send(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio data: {str(e)}")
            raise

    async def stop_transcription(self):
        print("stop_transcription called!")
        """Stop the live transcription session."""
        try:
            # Give a small delay to allow any pending events to complete
            await asyncio.sleep(0.1)
            
            if self.live_transcription and self.is_connected:
                self.live_transcription.finish()
                self.is_connected = False
                logger.info("Transcription stopped")
            
            # Close the audio file
            if self._audio_writer:
                self._close_audio_file()
            
            # Clear conversation state
            self._clear_conversation_state()
            
        except Exception as e:
            logger.error(f"Error stopping transcription: {str(e)}")
            # Don't re-raise the exception as this is cleanup code

    def _clear_conversation_state(self):
        """Clear conversation state and speaker mappings."""
        try:
            print("🧹 Clearing conversation state")
            self._speaker_map.clear()
            self._conversation_segments.clear()
            self._current_utterance_segments.clear()
            logger.info("Conversation state cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation state: {str(e)}")

    def _extract_speaker_segments(self, words: List[Any]) -> List[Dict[str, Any]]:
        """Extract speaker segments from words array with timestamps."""
        segments = []
        current_speaker = None
        current_segment = {
            "speaker": None,
            "text": [],
            "start_time": None,
            "end_time": None,
            "words": []
        }
        
        for word in words:
            speaker = getattr(word, 'speaker', 'unknown')
            start_time = getattr(word, 'start', 0)
            end_time = getattr(word, 'end', 0)
            word_text = getattr(word, 'word', '')
            
            if speaker != current_speaker:
                # Save previous segment if it exists
                if current_segment["speaker"] is not None and current_segment["text"]:
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    "speaker": speaker,
                    "text": [word_text],
                    "start_time": start_time,
                    "end_time": end_time,
                    "words": [word]
                }
                current_speaker = speaker
            else:
                # Continue current segment
                current_segment["text"].append(word_text)
                current_segment["end_time"] = end_time
                current_segment["words"].append(word)
        
        # Add the last segment
        if current_segment["speaker"] is not None and current_segment["text"]:
            segments.append(current_segment)
        
        return segments

    def _extract_audio_segment(self, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Extract audio segment from the saved audio file using timestamps."""
        try:
            if not self._audio_file or not os.path.exists(self._audio_file):
                logger.error("Audio file not found for segment extraction")
                return None
            
            # Flush the current writer to ensure data is written
            if self._audio_writer:
                self._audio_writer._file.flush()
            
            # Use a separate file handle for reading (don't close the writer)
            with wave.open(self._audio_file, 'rb') as reader:
                frame_rate = reader.getframerate()
                n_frames = reader.getnframes()
                
                # Convert time to frame indices
                start_frame = int(start_time * frame_rate)
                end_frame = int(end_time * frame_rate)
                
                # Ensure we don't go beyond file bounds
                start_frame = max(0, start_frame)
                end_frame = min(n_frames, end_frame)
                
                if start_frame >= end_frame:
                    logger.warning(f"Invalid segment: start_frame={start_frame}, end_frame={end_frame}")
                    return None
                
                # Read the segment
                reader.setpos(start_frame)
                frames_to_read = end_frame - start_frame
                audio_data = reader.readframes(frames_to_read)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Convert to float32 and normalize
                audio_array = audio_array.astype(np.float32) / 32767.0
                
                logger.info(f"Extracted audio segment: {start_time:.2f}s to {end_time:.2f}s, {len(audio_array)} samples")
                
                # Save the extracted audio segment for debugging
                self._save_audio_segment_for_debug(audio_array, start_time, end_time, frame_rate)
                
            return audio_array
            
        except Exception as e:
            logger.error(f"Error extracting audio segment: {str(e)}")
            return None

    def _save_audio_segment_for_debug(self, audio_array: np.ndarray, start_time: float, end_time: float, sample_rate: int, speaker_info: str = None):
        """Save extracted audio segment to audio_dumps directory for debugging."""
        try:
            # Create audio_dumps directory if it doesn't exist
            dumps_dir = os.path.join(os.getcwd(), "audio_dumps")
            os.makedirs(dumps_dir, exist_ok=True)
            
            # Generate filename with timestamp, session info, and speaker info
            session_id = self._session_id or "unknown_session"
            speaker_part = f"_speaker_{speaker_info}" if speaker_info else ""
            filename = f"segment_{session_id}{speaker_part}_{start_time:.2f}s_to_{end_time:.2f}s.wav"
            filepath = os.path.join(dumps_dir, filename)
            
            # Convert float32 back to int16 for WAV file
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            speaker_info_str = f" (Speaker: {speaker_info})" if speaker_info else ""
            print(f"🎵 Saved audio segment to: {filepath}")
            print(f"   Duration: {end_time - start_time:.2f}s, Samples: {len(audio_array)}{speaker_info_str}")
            logger.debug(f"Saved extracted audio segment to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving audio segment for debug: {str(e)}")
            # Don't raise exception since this is just for debugging

    async def _initialize_voice_processor(self):
        """Initialize the voice processor for speaker identification."""
        try:
            if self._voice_processor is None:
                # Initialize voice processor
                self._voice_processor = await get_voice_processor()
                logger.info("Voice processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice processor: {str(e)}")
            # Continue without voice processor for basic transcription
            self._voice_processor = None

    def get_speaker_statistics(self) -> Dict[str, Any]:
        """Get statistics about speakers in the current session."""
        try:
            stats = {
                "total_speakers": len(self._speaker_map),
                "identified_speakers": [],
                "unknown_speakers": [],
                "speaker_map": self._speaker_map.copy(),
                "total_segments": len(self._conversation_segments)
            }
            
            for speaker_num, user_id in self._speaker_map.items():
                if user_id.startswith("Unknown_Speaker_"):
                    stats["unknown_speakers"].append(speaker_num)
                else:
                    stats["identified_speakers"].append({
                        "speaker_number": speaker_num,
                        "user_id": user_id,
                        "display_name": self._get_speaker_display_name(user_id)
                    })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting speaker statistics: {str(e)}")
            return {"error": str(e)}

    def get_full_conversation(self) -> str:
        """Get the full conversation transcript with speaker attribution."""
        try:
            if not self._conversation_segments:
                return "No conversation recorded yet"
            
            return self._reconstruct_conversation(self._conversation_segments)
            
        except Exception as e:
            logger.error(f"Error getting full conversation: {str(e)}")
            return "Error retrieving conversation"

    def _extract_audio_segment_with_speaker_info(self, start_time: float, end_time: float, speaker_num: int) -> Optional[np.ndarray]:
        """Extract audio segment from the saved audio file using timestamps and speaker information."""
        try:
            if not self._audio_file or not os.path.exists(self._audio_file):
                logger.error("Audio file not found for segment extraction")
                return None
            
            # Flush the current writer to ensure data is written
            if self._audio_writer:
                self._audio_writer._file.flush()
            
            # Use a separate file handle for reading (don't close the writer)
            with wave.open(self._audio_file, 'rb') as reader:
                frame_rate = reader.getframerate()
                n_frames = reader.getnframes()
                
                # Convert time to frame indices
                start_frame = int(start_time * frame_rate)
                end_frame = int(end_time * frame_rate)
                
                # Ensure we don't go beyond file bounds
                start_frame = max(0, start_frame)
                end_frame = min(n_frames, end_frame)
                
                if start_frame >= end_frame:
                    logger.warning(f"Invalid segment: start_frame={start_frame}, end_frame={end_frame}")
                    return None
                
                # Read the segment
                reader.setpos(start_frame)
                frames_to_read = end_frame - start_frame
                audio_data = reader.readframes(frames_to_read)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Convert to float32 and normalize
                audio_array = audio_array.astype(np.float32) / 32767.0
                
                logger.info(f"Extracted audio segment: {start_time:.2f}s to {end_time:.2f}s, {len(audio_array)} samples")
                
                # Save the extracted audio segment for debugging
                self._save_audio_segment_for_debug(audio_array, start_time, end_time, frame_rate, speaker_info=f"Speaker_{speaker_num}")
                
            return audio_array
            
        except Exception as e:
            logger.error(f"Error extracting audio segment: {str(e)}")
            return None

async def get_deepgram_transcriber() -> DeepgramTranscriber:
    """Factory function to get a DeepgramTranscriber instance."""
    return DeepgramTranscriber()
