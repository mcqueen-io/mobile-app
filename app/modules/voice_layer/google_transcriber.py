import asyncio
import json
import os
import tempfile
import time
import wave
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
import queue
import threading
from google.cloud import speech
import logging
from app.modules.voice_layer.voice_processor import get_voice_processor
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.modules.voice_layer.deepgram_tts import get_deepgram_tts

logger = logging.getLogger(__name__)

class GoogleTranscriber:
    def __init__(self):
        self.client = None
        self.streaming_config = None
        self.is_connected = False
        self._on_transcript_callback = None
        self._event_loop = None
        
        # Audio management
        self._temp_dir = tempfile.mkdtemp()
        self._audio_file = None
        self._audio_writer = None
        self._session_id = None
        self._speaker_map = {}  # Cache for speaker identification
        
        # Streaming management
        self._audio_queue = queue.Queue()
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
        
        # Conversation tracking
        self._conversation_segments = []
        self._current_utterance_segments = []
        
        # Custom silence detection for reliable conversation end detection
        self._silence_timer_task = None
        self._last_transcript_time = None
        self._conversation_buffer = []
        self._silence_threshold = 2.0  # seconds - reduced for testing
        
        # Text deduplication tracking
        self._last_final_text = ""  # Track last final transcript to extract deltas
        self._conversation_text_buffer = []  # Clean segments only
        
        # Voice processor for speaker identification
        self._voice_processor = None
        
        # TTS for responses
        self._tts_engine = None
        self._audio_output_callback = None
        
        self._initialize_google_client()

    def _initialize_google_client(self):
        """Initialize the Google Cloud Speech client with appropriate credentials."""
        try:
            # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS environment variable)
            self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Speech client: {str(e)}")
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
        except Exception as e:
            logger.error(f"Failed to initialize audio file: {str(e)}")
            raise

    def _write_audio_chunk(self, audio_data: bytes):
        """Write audio chunk to the WAV file."""
        try:
            if self._audio_writer:
                self._audio_writer.writeframes(audio_data)
        except Exception as e:
            logger.error(f"Failed to write audio chunk: {str(e)}")

    def _close_audio_file(self):
        """Close the audio file and clean up."""
        try:
            if self._audio_writer:
                self._audio_writer.close()
                self._audio_writer = None
            if self._audio_file and os.path.exists(self._audio_file):
                logger.info(f"Audio file ready for processing: {self._audio_file}")
        except Exception as e:
            logger.error(f"Failed to close audio file: {str(e)}")

    def _create_streaming_config(self):
        """Create the streaming recognition configuration."""
        try:
            # Speaker diarization configuration
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=2,  # Adjust based on expected speakers
            )
            
            # Audio configuration
            audio_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='en-US',
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                diarization_config=diarization_config,
                max_alternatives=1,
                model='latest_long',
                use_enhanced=True
            )
            
            # Streaming configuration
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=audio_config,
                interim_results=True,
                single_utterance=False
            )
            
            logger.info("Google Cloud Speech streaming configuration created")
        except Exception as e:
            logger.error(f"Failed to create streaming configuration: {str(e)}")
            raise

    def _audio_generator(self):
        """Generate audio chunks for streaming."""
        while not self._stop_streaming.is_set():
            try:
                # Get audio data from queue with timeout
                audio_data = self._audio_queue.get(timeout=0.1)
                if audio_data is not None:
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                    # Write to file for processing
                    self._write_audio_chunk(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio generator: {str(e)}")
                break

    def _process_streaming_response(self, response):
        """Process streaming recognition response."""
        try:
            if not response.results:
                return
            
            result = response.results[0]
            if not result.alternatives:
                return
            
            alternative = result.alternatives[0]
            transcript = alternative.transcript
            is_final = result.is_final
            
            print(f"Google transcript: '{transcript}' (is_final: {is_final})")
            
            # Create transcript data
            transcript_data = {
                "type": "transcript",
                "text": transcript,
                "is_final": is_final,
                "speakers": []
            }
            
            # Process speaker information if available
            if len(transcript) > 0 and hasattr(alternative, 'words') and alternative.words:
                words = alternative.words
                
                # For final results, extract only NEW text to avoid duplication
                if is_final:
                    new_text = self._extract_new_text(transcript)
                    print(f"🔄 Extracted new text: '{new_text}' (from full: '{transcript}')")
                    
                    if new_text.strip():  # Only process if there's actually new content
                        # Extract words that correspond to the new text
                        new_words = self._extract_words_for_new_text(words, new_text, transcript)
                        
                        if new_words:
                            speaker_segments = self._extract_speaker_segments(new_words)
                            
                            print(f"🎯 Processing {len(speaker_segments)} NEW speaker segments for voice embedding")
                            if self._event_loop:
                                for segment in speaker_segments:
                                    asyncio.run_coroutine_threadsafe(
                                        self._process_speaker_segment(segment),
                                        self._event_loop
                                    )
                            
                            # Add ONLY the new segments to conversation buffer
                            self._add_to_conversation_buffer(speaker_segments)
                            self._reset_silence_timer()
                    
                    # Update last final text for next comparison
                    self._last_final_text = transcript
                
                # Format speaker data for response (use all words for real-time display)
                speaker_segments = self._extract_speaker_segments(words)
                current_speaker = None
                current_text = []
                
                for word in words:
                    speaker = getattr(word, 'speaker_tag', 0)
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
            
            # Send transcript to callback
            if self._on_transcript_callback and self._event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._on_transcript_callback(transcript_data),
                    self._event_loop
                )
                
        except Exception as e:
            logger.error(f"Error processing streaming response: {str(e)}")
            logger.exception("Full traceback:")

    def _streaming_recognition_thread(self):
        """Thread function for handling streaming recognition."""
        try:
            logger.info("Starting Google Cloud Speech streaming recognition")
            
            # Create audio generator
            audio_generator = self._audio_generator()
            
            # Start streaming recognition
            responses = self.client.streaming_recognize(
                self.streaming_config, 
                audio_generator
            )
            
            
            # Process responses
            for response in responses:
                if self._stop_streaming.is_set():
                    break
                self._process_streaming_response(response)
                
        except Exception as e:
            logger.error(f"Error in streaming recognition thread: {str(e)}")
            self.is_connected = False

    def _extract_new_text(self, current_transcript: str) -> str:
        """Extract only the new text from current transcript compared to last final."""
        if not self._last_final_text:
            return current_transcript
        
        # Handle case where current transcript doesn't start with last transcript
        if not current_transcript.startswith(self._last_final_text):
            # This is normal behavior - Google sends independent segments instead of cumulative ones
            logger.debug(f"Independent segment detected: '{current_transcript}' (previous: '{self._last_final_text}')")
            return current_transcript
        
        # Extract the delta - everything after the last final text
        new_text = current_transcript[len(self._last_final_text):].strip()
        return new_text

    def _extract_words_for_new_text(self, all_words: List[Any], new_text: str, full_transcript: str) -> List[Any]:
        """Extract only the words that correspond to the new text portion."""
        if not new_text.strip():
            return []
        
        # Find where the new text starts in the full transcript
        new_text_start_pos = full_transcript.find(new_text)
        if new_text_start_pos == -1:
            logger.warning(f"Could not find new text '{new_text}' in full transcript '{full_transcript}'")
            return all_words  # Fallback to all words
        
        # Calculate approximate word boundary
        # Count words in text before the new text starts
        text_before_new = full_transcript[:new_text_start_pos]
        words_before_count = len(text_before_new.split()) if text_before_new.strip() else 0
        
        # Return words from the calculated position onwards
        new_words = all_words[words_before_count:] if words_before_count < len(all_words) else all_words
        
        logger.debug(f"Extracted {len(new_words)} words for new text from position {words_before_count}")
        return new_words

    def _add_to_conversation_buffer(self, segments: List[Dict[str, Any]]):
        """Add segments to buffer with smart deduplication."""
        for segment in segments:
            # Check if this segment overlaps with existing ones based on text content
            if not self._is_duplicate_segment(segment):
                self._conversation_buffer.append(segment)
                logger.debug(f"Added new segment to buffer: {' '.join(segment['text'])}")
            else:
                logger.debug(f"Skipped duplicate segment: {' '.join(segment['text'])}")

    def _is_duplicate_segment(self, new_segment: Dict[str, Any]) -> bool:
        """Check if segment overlaps with existing segments based on text content."""
        new_text = " ".join(new_segment["text"]).strip().lower()
        
        if not new_text:
            return True  # Empty segments are considered duplicates
        
        for existing_segment in self._conversation_buffer:
            existing_text = " ".join(existing_segment["text"]).strip().lower()
            
            # Check for exact match or significant overlap
            if new_text == existing_text:
                return True
            
            # Check if new text is completely contained in existing text
            if new_text in existing_text and len(new_text) > 5:  # Avoid matching very short words
                return True
                
            # Check if existing text is completely contained in new text (replace scenario)
            if existing_text in new_text and len(existing_text) > 5:
                # Remove the existing segment as the new one is more complete
                self._conversation_buffer.remove(existing_segment)
                logger.debug(f"Replaced shorter segment '{existing_text}' with longer '{new_text}'")
                return False
        
        return False

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
            speaker = getattr(word, 'speaker_tag', 0)
            start_time = getattr(word, 'start_time', None)
            end_time = getattr(word, 'end_time', None)
            word_text = getattr(word, 'word', '')
            
            # Convert Google's timestamp to seconds
            start_seconds = 0
            end_seconds = 0
            if start_time:
                # Handle both timedelta and protobuf Duration objects
                if hasattr(start_time, 'total_seconds'):
                    # datetime.timedelta object
                    start_seconds = start_time.total_seconds()
                elif hasattr(start_time, 'seconds') and hasattr(start_time, 'nanos'):
                    # protobuf Duration object
                    start_seconds = start_time.seconds + start_time.nanos / 1e9
                else:
                    # Fallback: assume it's already a number
                    start_seconds = float(start_time)
                    
            if end_time:
                # Handle both timedelta and protobuf Duration objects
                if hasattr(end_time, 'total_seconds'):
                    # datetime.timedelta object
                    end_seconds = end_time.total_seconds()
                elif hasattr(end_time, 'seconds') and hasattr(end_time, 'nanos'):
                    # protobuf Duration object  
                    end_seconds = end_time.seconds + end_time.nanos / 1e9
                else:
                    # Fallback: assume it's already a number
                    end_seconds = float(end_time)
            
            if speaker != current_speaker:
                # Save previous segment if it exists
                if current_segment["speaker"] is not None and current_segment["text"]:
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    "speaker": speaker,
                    "text": [word_text],
                    "start_time": start_seconds,
                    "end_time": end_seconds,
                    "words": [word]
                }
                current_speaker = speaker
            else:
                # Continue current segment
                current_segment["text"].append(word_text)
                current_segment["end_time"] = end_seconds
                current_segment["words"].append(word)
        
        # Add the last segment
        if current_segment["speaker"] is not None and current_segment["text"]:
            segments.append(current_segment)
        
        return segments

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

    def _extract_audio_segment_with_speaker_info(self, start_time: float, end_time: float, speaker_num: int) -> Optional[np.ndarray]:
        """Extract audio segment from the saved audio file using timestamps and speaker information."""
        try:
            if not self._audio_file or not os.path.exists(self._audio_file):
                logger.error("Audio file not found for segment extraction")
                return None
            
            # Flush the current writer to ensure data is written
            if self._audio_writer:
                self._audio_writer._file.flush()
            
            # Use a separate file handle for reading
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

    def _save_audio_segment_for_debug(self, audio_array: np.ndarray, start_time: float, end_time: float, sample_rate: int, speaker_info: str = None):
        """Save extracted audio segment to audio_dumps directory for debugging."""
        try:
            # Create audio_dumps directory if it doesn't exist
            dumps_dir = os.path.join(os.getcwd(), "audio_dumps")
            os.makedirs(dumps_dir, exist_ok=True)
            
            # Generate filename with timestamp, session info, and speaker info
            session_id = self._session_id or "unknown_session"
            speaker_part = f"_speaker_{speaker_info}" if speaker_info else ""
            filename = f"google_segment_{session_id}{speaker_part}_{start_time:.2f}s_to_{end_time:.2f}s.wav"
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
            print(f"🎵 Saved Google audio segment to: {filepath}")
            print(f"   Duration: {end_time - start_time:.2f}s, Samples: {len(audio_array)}{speaker_info_str}")
            
        except Exception as e:
            logger.error(f"Error saving audio segment for debug: {str(e)}")

    async def _initialize_voice_processor(self):
        """Initialize the voice processor for speaker identification."""
        try:
            if self._voice_processor is None:
                self._voice_processor = await get_voice_processor()
                logger.info("Voice processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice processor: {str(e)}")
            self._voice_processor = None

    def _reset_silence_timer(self):
        """Reset the silence detection timer for custom conversation end detection."""
        try:
            # Cancel existing timer
            if self._silence_timer_task:
                self._silence_timer_task.cancel()
            
            # Set new timer
            if self._event_loop:
                self._silence_timer_task = asyncio.run_coroutine_threadsafe(
                    self._silence_timeout(),
                    self._event_loop
                )
                
            # Use regular time instead of event loop time to avoid thread issues
            self._last_transcript_time = time.time()
            
        except Exception as e:
            logger.error(f"Error resetting silence timer: {str(e)}")

    async def _silence_timeout(self):
        """Handle silence timeout for conversation end detection."""
        try:
            await asyncio.sleep(self._silence_threshold)
            
            # Check if we have segments to process
            if self._conversation_buffer:
                print(f"🔕 Silence timeout reached. Processing {len(self._conversation_buffer)} segments")
                await self._process_conversation_end()
            
        except asyncio.CancelledError:
            # Timer was cancelled due to new speech
            pass
        except Exception as e:
            logger.error(f"Error in silence timeout: {str(e)}")

    async def _process_conversation_end(self):
        """Process conversation end and reconstruct conversation."""
        try:
            if not self._conversation_buffer:
                print("⚠️ No segments to process for conversation")
                return
            
            # Reconstruct the conversation
            conversation_text = self._reconstruct_conversation(self._conversation_buffer)
            print(f"📝 Reconstructed Google conversation: {conversation_text}")
            
            # Add segments to conversation history
            self._conversation_segments.extend(self._conversation_buffer)
            
            # Show speaker statistics
            stats = self.get_speaker_statistics()
            print(f"📊 Google Speaker Statistics:")
            print(f"   Total speakers: {stats['total_speakers']}")
            print(f"   Identified: {len(stats['identified_speakers'])}")
            print(f"   Unknown: {len(stats['unknown_speakers'])}")
            
            if stats['identified_speakers']:
                print("   Identified speakers:")
                for speaker in stats['identified_speakers']:
                    print(f"     Speaker {speaker['speaker_number']}: {speaker['display_name']}")
            
            # Get LLM response
            llm_wrapper = await get_gemini_wrapper()
            llm_response = await llm_wrapper.chat("test_user_123", conversation_text)
            print(f"🤖 LLM response: {llm_response}")
            
            # Convert LLM response to speech using streaming TTS
            await self._speak_response(llm_response)
            
            # Clear conversation buffer
            self._conversation_buffer = []
            
        except Exception as e:
            logger.error(f"Error processing conversation end: {str(e)}")
            logger.exception("Full traceback:")

    def _reconstruct_conversation(self, segments: List[Dict[str, Any]]) -> str:
        """Reconstruct conversation text with speaker attribution and deduplication."""
        try:
            if not segments:
                return "No conversation segments available"
            
            # Sort segments by start time to ensure chronological order
            sorted_segments = sorted(segments, key=lambda x: x.get("start_time", 0))
            
            # Group consecutive segments by same speaker to create clean conversation flow
            conversation_lines = []
            current_speaker = None
            current_text_parts = []
            
            for segment in sorted_segments:
                speaker_num = segment["speaker"]
                text = " ".join(segment["text"]).strip()
                
                if not text:  # Skip empty segments
                    continue
                
                # Determine speaker name
                if "identified_user" in segment:
                    identified_user_id = segment["identified_user"]
                    if not identified_user_id.startswith("Unknown_Speaker_"):
                        speaker_name = self._get_speaker_display_name(identified_user_id)
                    else:
                        speaker_name = identified_user_id
                else:
                    speaker_name = f"Speaker_{speaker_num}"
                
                if speaker_name == current_speaker:
                    # Same speaker, append text (handling potential continuations)
                    current_text_parts.append(text)
                else:
                    # New speaker, finalize previous speaker's text
                    if current_speaker and current_text_parts:
                        # Join text parts and clean up
                        full_text = " ".join(current_text_parts).strip()
                        if full_text:
                            conversation_lines.append(f"{current_speaker}: {full_text}")
                    
                    # Start new speaker section
                    current_speaker = speaker_name
                    current_text_parts = [text]
            
            # Add the last speaker's text
            if current_speaker and current_text_parts:
                full_text = " ".join(current_text_parts).strip()
                if full_text:
                    conversation_lines.append(f"{current_speaker}: {full_text}")
            
            return "\n".join(conversation_lines)
            
        except Exception as e:
            logger.error(f"Error reconstructing conversation: {str(e)}")
            logger.exception("Full traceback:")
            return "Error reconstructing conversation"

    def _get_speaker_display_name(self, user_id: str) -> str:
        """Get display name for a user ID."""
        # TODO: uncomment this when we have a way to fetch user details from the database
        # try:
        #     # TODO: In the future, we could fetch user details from the database
        #     return f"User_{user_id}"
        # except Exception as e:
        #     logger.error(f"Error getting speaker display name: {str(e)}")
        #     return f"User_{user_id}"
        return user_id

    async def start_transcription(self, on_transcript: Callable[[Dict[str, Any]], None], session_id: str, on_audio_output: Optional[Callable[[bytes], None]] = None):
        """Start a Google Cloud Speech transcription session."""
        try:
            if not self.client:
                raise RuntimeError("Google Cloud Speech client not initialized")

            print(f"Starting Google transcription for session: {session_id}")
            
            # Reset connection state
            self.is_connected = False
            self._on_transcript_callback = on_transcript
            self._event_loop = asyncio.get_event_loop()

            # Set audio output callback for TTS
            if on_audio_output:
                self.set_audio_output_callback(on_audio_output)

            # Initialize audio file for this session
            self._initialize_audio_file(session_id)
            
            # Initialize voice processor for speaker identification
            await self._initialize_voice_processor()

            # Create streaming configuration
            self._create_streaming_config()

            # Reset streaming state
            self._stop_streaming.clear()
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reset deduplication state for new session
            self._last_final_text = ""
            self._conversation_text_buffer.clear()

            # Start streaming recognition thread
            self._streaming_thread = threading.Thread(
                target=self._streaming_recognition_thread,
                daemon=True
            )
            self._streaming_thread.start()

            # Mark as connected
            self.is_connected = True
            logger.info("Google Cloud Speech transcription started successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to start Google transcription: {str(e)}")
            return False

    async def send_audio(self, audio_data: bytes):
        """Send audio data to Google Cloud Speech for transcription."""
        try:
            if not self.is_connected:
                raise RuntimeError("Transcription not started")

            # Add audio to queue for streaming
            self._audio_queue.put(audio_data)
            
            logger.debug(f"Sending audio chunk of size: {len(audio_data)} bytes to Google")
            
        except Exception as e:
            logger.error(f"Error sending audio data to Google: {str(e)}")
            raise

    async def stop_transcription(self):
        """Stop the Google Cloud Speech transcription session."""
        try:
            print("Stopping Google transcription...")
            
            # Stop streaming
            self._stop_streaming.set()
            
            # Wait for streaming thread to finish
            if self._streaming_thread and self._streaming_thread.is_alive():
                self._streaming_thread.join(timeout=5.0)
            
            # Process any remaining conversation
            if self._conversation_buffer:
                await self._process_conversation_end()
            
            # Cancel silence timer
            if self._silence_timer_task:
                self._silence_timer_task.cancel()
            
            # Close the audio file
            if self._audio_writer:
                self._close_audio_file()
            
            # Clear conversation state
            self._clear_conversation_state()
            
            self.is_connected = False
            logger.info("Google transcription stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Google transcription: {str(e)}")

    def _clear_conversation_state(self):
        """Clear conversation state and speaker mappings."""
        try:
            print("🧹 Clearing Google conversation state")
            self._speaker_map.clear()
            self._conversation_segments.clear()
            self._current_utterance_segments.clear()
            self._conversation_buffer.clear()
            
            # Clear deduplication state
            self._last_final_text = ""
            self._conversation_text_buffer.clear()
            
            logger.info("Google conversation state cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation state: {str(e)}")

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

    async def _speak_response(self, text: str):
        """Convert text response to speech using streaming TTS"""
        try:
            if not self._tts_engine:
                # Initialize TTS engine if not already done
                from app.core.config import settings
                self._tts_engine = await get_deepgram_tts(settings.DEEPGRAM_API_KEY)
            
            print(f"🗣️ Speaking response: {text[:50]}...")
            
            # Helper function to safely call audio output callback
            def safe_audio_callback(message_bytes: bytes):
                """Safely call the audio output callback, handling both sync and async callbacks"""
                if self._audio_output_callback:
                    try:
                        if self._event_loop and self._event_loop.is_running():
                            # Schedule the coroutine on the event loop
                            asyncio.run_coroutine_threadsafe(
                                self._audio_output_callback(message_bytes),
                                self._event_loop
                            )
                        else:
                            logger.warning("Event loop not available for audio callback")
                    except Exception as e:
                        logger.error(f"Error in audio output callback: {e}")
            
            # First, send a message indicating TTS start
            if self._audio_output_callback:
                import json
                tts_start_message = {
                    "type": "tts_start",
                    "message": "Starting text-to-speech conversion",
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                }
                try:
                    # Send as JSON string bytes for WebSocket
                    message_bytes = json.dumps(tts_start_message).encode('utf-8')
                    safe_audio_callback(message_bytes)
                except Exception as e:
                    logger.error(f"Error sending TTS start message: {e}")
            
            def on_audio_chunk(audio_chunk: bytes):
                """Handle streaming audio chunks from TTS"""
                try:
                    # Send audio chunk with type indicator to the audio output callback
                    if self._audio_output_callback:
                        import json
                        import base64
                        
                        # Create audio message
                        audio_message = {
                            "type": "tts_audio",
                            "audio_data": base64.b64encode(audio_chunk).decode('utf-8'),
                            "size": len(audio_chunk)
                        }
                        
                        # Send as JSON string bytes for WebSocket
                        message_bytes = json.dumps(audio_message).encode('utf-8')
                        safe_audio_callback(message_bytes)
                    else:
                        # Default: save chunks to file for debugging
                        self._save_tts_chunk_for_debug(audio_chunk)
                except Exception as e:
                    logger.error(f"Error handling TTS audio chunk: {str(e)}")
            
            # Use streaming TTS
            success = await self._tts_engine.speak_streaming(text, on_audio_chunk)
            
            # Send completion message
            if self._audio_output_callback:
                import json
                tts_end_message = {
                    "type": "tts_end",
                    "message": "Text-to-speech conversion completed",
                    "success": success
                }
                try:
                    message_bytes = json.dumps(tts_end_message).encode('utf-8')
                    safe_audio_callback(message_bytes)
                except Exception as e:
                    logger.error(f"Error sending TTS end message: {e}")
            
            if success:
                print("✅ TTS streaming completed successfully")
            else:
                print("❌ TTS streaming failed")
                
        except Exception as e:
            logger.error(f"Error in speech response: {str(e)}")
            logger.exception("TTS error:")

    def _save_tts_chunk_for_debug(self, audio_chunk: bytes):
        """Save TTS audio chunks for debugging"""
        try:
            # Create audio_dumps directory if it doesn't exist
            dumps_dir = os.path.join(os.getcwd(), "audio_dumps")
            os.makedirs(dumps_dir, exist_ok=True)
            
            # Generate filename with timestamp
            session_id = self._session_id or "unknown_session"
            timestamp = int(time.time() * 1000)  # milliseconds
            filename = f"tts_chunk_{session_id}_{timestamp}.wav"
            filepath = os.path.join(dumps_dir, filename)
            
            # Save chunk to file
            with open(filepath, 'ab') as f:  # Append mode for multiple chunks
                f.write(audio_chunk)
            
            logger.debug(f"Saved TTS chunk to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving TTS chunk: {str(e)}")

    def set_audio_output_callback(self, callback: Callable[[bytes], None]):
        """Set callback for handling TTS audio output"""
        self._audio_output_callback = callback

async def get_google_transcriber() -> GoogleTranscriber:
    """Factory function to get a GoogleTranscriber instance."""
    return GoogleTranscriber()