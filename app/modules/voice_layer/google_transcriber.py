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
        
        # TTS state management to prevent Google timeout
        self._tts_active = False
        self._tts_pause_transcription = True  # Option to pause vs send silence
        self._silence_audio_data = None  # Pre-generated silence
        self._last_audio_time = None
        
        # Configuration options for TTS timeout prevention
        self._tts_timeout_prevention_enabled = True  # Enable/disable TTS timeout prevention
        self._tts_silence_threshold = 5.0  # Seconds without audio before starting prevention
        self._tts_max_silence_duration = 50.0  # Maximum duration to send silence
        self._tts_silence_interval = 0.5  # Interval between silence chunks
        
        # Configuration options for general keep-alive (prevent normal operation timeouts)
        self._keepalive_enabled = True  # Enable/disable general keep-alive
        self._keepalive_interval = 5.0  # EXTREME AGGRESSIVE: Send keep-alive every 5 seconds (reduced from 10)
        self._keepalive_max_silence = 1800.0  # PERSISTENT: Max 30 minutes of keep-alive (increased from 5 minutes)
        
        # PERSISTENT SESSION: Additional ultra-aggressive settings for TTS periods
        self._tts_keepalive_interval = 3.0  # EVEN MORE AGGRESSIVE: Every 3 seconds during TTS
        self._persistent_session_mode = True  # Enable persistent session mode
        self._max_session_duration = 3600.0  # Allow 1 hour sessions (prevent indefinite connections)
        
        # IMMEDIATE RESTART: Flag to trigger restart without breaking audio flow
        self._needs_restart = False
        self._restart_in_progress = False
        
        # INTERRUPT FUNCTIONALITY: Prioritize user over AI
        self._interrupt_enabled = True
        self._interrupt_threshold = 0.01  # Audio amplitude threshold to detect user speech
        self._interrupt_duration_threshold = 0.3  # 300ms of speech to trigger interrupt
        
        # SPEED OPTIMIZATION: Auto-switch to faster config after TTS
        self._speed_optimization_enabled = True  # Enable automatic speed optimization
        self._speed_config_active = False  # Track if speed config is currently active
        self._speed_config_switch_time = None  # When speed config was activated
        self._speed_config_duration = 10.0  # How long to use speed config (10 seconds)
        self._needs_config_switch = False  # Flag to trigger config switch
        
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
        
        # CRITICAL FIX: Interim result accumulation to capture full sentences
        self._current_interim_transcript = ""  # Track the most complete interim result
        self._interim_result_counter = 0  # Track interim result frequency
        self._last_complete_interim = ""  # Store the most complete interim before final
        
        # Voice processor for speaker identification
        self._voice_processor = None
        
        # TTS for responses
        self._tts_engine = None
        self._audio_output_callback = None
        
        self._initialize_google_client()
        self._generate_silence_audio()

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

    def _create_streaming_config(self, speed_optimized=False):
        """Create the streaming recognition configuration with optional speed optimization."""
        try:
            if speed_optimized:
                # SPEED-OPTIMIZED CONFIG: For immediate post-TTS responses
                logger.info("Creating SPEED-OPTIMIZED Google Speech config for immediate post-TTS responses")
                
                # Minimal speaker diarization for speed
                diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=False  # Disable for faster response
                )
                
                # Speed-optimized audio configuration
                audio_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code='en-US',
                    enable_word_time_offsets=False,  # Disable for speed
                    enable_automatic_punctuation=False,  # Disable for speed
                    diarization_config=diarization_config,
                    max_alternatives=1,
                    model='latest_short',  # SHORT model for faster responses
                    use_enhanced=False  # Disable enhancement for speed
                )
                
                # Speed-optimized streaming configuration
                self.streaming_config = speech.StreamingRecognitionConfig(
                    config=audio_config,
                    interim_results=True,
                    single_utterance=False
                )
                
                logger.info("SPEED-OPTIMIZED Google Speech config created - faster responses expected")
            else:
                # STANDARD CONFIG: For normal high-quality processing
                logger.info("Creating STANDARD Google Speech config for high-quality processing")
                
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
                
                logger.info("STANDARD Google Speech config created - high quality processing")
            
        except Exception as e:
            logger.error(f"Failed to create streaming configuration: {str(e)}")
            raise

    def _audio_generator(self):
        """Generate audio chunks for streaming with ULTRA-AGGRESSIVE timeout prevention for persistent sessions."""
        silence_counter = 0
        last_keepalive_time = time.time()
        keepalive_start_time = None  # Track when keep-alive period started
        session_start_time = time.time()  # Track total session duration
        
        # TTS TRANSITION: Simplified tracking for more reliable operation
        tts_just_ended = False
        tts_end_time = None
        previous_tts_state = False
        
        while not self._stop_streaming.is_set():
            try:
                # Check if session has exceeded maximum duration (safety limit)
                current_time = time.time()
                session_duration = current_time - session_start_time
                if self._persistent_session_mode and session_duration > self._max_session_duration:
                    logger.warning(f"Session exceeded maximum duration ({self._max_session_duration}s) - allowing natural timeout")
                    break
                
                # SIMPLIFIED TTS TRANSITION DETECTION: Less aggressive, more reliable
                current_tts_state = self._tts_active
                if previous_tts_state and not current_tts_state:
                    # TTS just ended - enter immediate audio priority mode
                    tts_just_ended = True
                    tts_end_time = current_time
                    logger.info("POST-TTS: TTS just ended - prioritizing user audio")
                elif tts_just_ended and (current_time - tts_end_time) > 10.0:
                    # Exit immediate mode after 10 seconds (reduced from 15)
                    tts_just_ended = False
                    logger.info("POST-TTS: Exiting immediate audio priority mode")
                
                previous_tts_state = current_tts_state
                
                # SIMPLIFIED AUDIO PROCESSING: More reliable queue checking
                audio_data = None
                
                # Get audio with appropriate timeout
                try:
                    # Simplified timeout logic - just use slightly longer timeout after TTS
                    timeout = 0.05 if tts_just_ended else 0.1
                    audio_data = self._audio_queue.get(timeout=timeout)
                        
                except queue.Empty:
                    audio_data = None
                
                # REAL AUDIO PROCESSING: Always prioritize real audio over keep-alive
                if audio_data is not None:
                    self._last_audio_time = current_time
                    last_keepalive_time = current_time  # Reset keep-alive timer on real audio
                    keepalive_start_time = None  # Reset keep-alive period
                    
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                    # Write to file for processing
                    self._write_audio_chunk(audio_data)
                    
                    # Reset silence counter when we get real audio
                    silence_counter = 0
                    
                    # Enhanced logging when TTS just ended
                    if tts_just_ended:
                        logger.info(f"POST-TTS: Processing user audio immediately")
                    
                    continue  # Process next audio immediately
                
                # KEEP-ALIVE LOGIC: Only run if no real audio was found
                time_since_last_audio = current_time - (self._last_audio_time or current_time)
                time_since_keepalive = current_time - last_keepalive_time
                
                # Track keep-alive period duration
                if keepalive_start_time is None:
                    keepalive_start_time = current_time
                keepalive_duration = current_time - keepalive_start_time
                
                # EXTREME TTS Keep-Alive: Prevent any timeouts during TTS
                if self._tts_active and self._persistent_session_mode:
                    if time_since_keepalive > self._tts_keepalive_interval:
                        if self._silence_audio_data is not None:
                            silence_bytes = self._silence_audio_data.tobytes()
                            yield speech.StreamingRecognizeRequest(audio_content=silence_bytes)
                            last_keepalive_time = current_time
                            
                            if silence_counter % 15 == 0:  # Log every 15 chunks during TTS
                                logger.info(f"EXTREME TTS keep-alive #{silence_counter} (every {self._tts_keepalive_interval}s) - session: {session_duration:.0f}s")
                            silence_counter += 1
                        continue
                
                # Handle TTS-specific timeout prevention (legacy mode if not in persistent mode)
                elif self._tts_timeout_prevention_enabled and self._tts_active and time_since_last_audio > self._tts_silence_threshold:
                    if self._tts_pause_transcription:
                        # Option 1: Pause transcription during TTS (stop sending audio)
                        if silence_counter == 0:  # Log only once
                            logger.info("TTS active - pausing audio stream to prevent Google timeout")
                        silence_counter += 1
                        continue
                    else:
                        # Option 2: Send silence to keep connection alive during TTS
                        max_silence_chunks = int(self._tts_max_silence_duration / self._tts_silence_interval)
                        if silence_counter < max_silence_chunks:
                            if self._silence_audio_data is not None:
                                silence_bytes = self._silence_audio_data.tobytes()
                                yield speech.StreamingRecognizeRequest(audio_content=silence_bytes)
                                
                                if silence_counter % 10 == 0:  # Log every 10 silence chunks
                                    logger.debug(f"Sent TTS silence to prevent Google timeout (#{silence_counter})")
                                
                                silence_counter += 1
                                time.sleep(self._tts_silence_interval)
                                continue
                        else:
                            if silence_counter == max_silence_chunks:  # Log only once
                                logger.warning(f"Maximum TTS silence duration ({self._tts_max_silence_duration}s) reached")
                            silence_counter += 1
                
                # SIMPLIFIED POST-TTS MODE: Less aggressive, more reliable
                elif tts_just_ended:
                    # Reduced keep-alive frequency right after TTS to allow more audio processing
                    if time_since_keepalive > 3.0:  # Increased from 2.0 to 3.0 seconds
                        if self._silence_audio_data is not None:
                            silence_bytes = self._silence_audio_data.tobytes()
                            yield speech.StreamingRecognizeRequest(audio_content=silence_bytes)
                            last_keepalive_time = current_time
                            logger.info(f"POST-TTS: Keep-alive after {time_since_keepalive:.1f}s")
                            silence_counter += 1
                        continue
                    else:
                        # Shorter delay to check for audio more frequently
                        time.sleep(0.02)  # Slightly longer delay
                        continue
                
                # EXTREME KEEP-ALIVE: Prevent timeout during normal operation
                elif self._keepalive_enabled and time_since_keepalive > self._keepalive_interval:
                    # In persistent mode, be much more lenient about maximum duration
                    max_keepalive_duration = self._keepalive_max_silence
                    if self._persistent_session_mode:
                        # Only stop if we've hit the absolute session limit
                        if keepalive_duration > max_keepalive_duration:
                            logger.info(f"Keep-alive duration ({keepalive_duration:.0f}s) reached limit ({max_keepalive_duration:.0f}s) - but continuing in persistent mode")
                    
                    # Always send keep-alive in persistent mode, regardless of duration
                    if self._silence_audio_data is not None:
                        silence_bytes = self._silence_audio_data.tobytes()
                        yield speech.StreamingRecognizeRequest(audio_content=silence_bytes)
                        last_keepalive_time = current_time
                        
                        # More detailed logging for persistent sessions
                        if self._persistent_session_mode:
                            if silence_counter % 20 == 0:  # Log every 20 keep-alives (every 1.7 minutes at 5s intervals)
                                logger.info(f"EXTREME SESSION keep-alive #{silence_counter} - no audio: {time_since_last_audio:.0f}s, session: {session_duration:.0f}s")
                        else:
                            logger.info(f"Sent keep-alive silence to prevent Google timeout (no audio for {time_since_last_audio:.1f}s, keep-alive duration: {keepalive_duration:.1f}s)")
                        
                        silence_counter += 1
                    continue
                
                # Normal timeout without issues - just continue with small delay
                time.sleep(0.02)  # Slightly longer delay for more stable operation
                continue
                    
            except Exception as e:
                logger.error(f"Error in audio generator: {str(e)}")
                break

    def _process_streaming_response(self, response):
        """Process streaming recognition response with interim result accumulation."""
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
            
            # CRITICAL FIX: Track interim results to capture complete sentences
            if not is_final:
                # Update interim tracking
                self._interim_result_counter += 1
                
                # Store the most complete interim result (longest one)
                if len(transcript) > len(self._current_interim_transcript):
                    self._current_interim_transcript = transcript
                    print(f"📋 Updated most complete interim: '{transcript}' (length: {len(transcript)})")
                
                # Store the last significant interim (for comparison)
                if len(transcript) > 10:  # Only store substantial interim results
                    self._last_complete_interim = transcript
            
            # Create transcript data
            transcript_data = {
                "type": "transcript",
                "text": transcript,
                "is_final": is_final,
                "speakers": []
            }
            
            # CRITICAL FIX: Enhanced final transcript processing with interim fallback
            if is_final and len(transcript) > 0:
                # Check if final result is much shorter than the best interim result
                best_interim = self._current_interim_transcript or self._last_complete_interim
                
                # If final result is significantly shorter, use the complete interim instead
                if (best_interim and 
                    len(best_interim) > len(transcript) * 2 and  # Interim is much longer
                    len(best_interim) > 20 and  # Interim is substantial
                    (transcript.lower() in best_interim.lower() or  # Final is contained in interim
                     len(transcript.split()) <= 3)):  # OR final is very short (likely truncated)
                    
                    print(f"🔄 INTERIM FALLBACK: Final '{transcript}' too short, using complete interim: '{best_interim}'")
                    working_transcript = best_interim
                    transcript_data["text"] = best_interim  # Update the response data too
                else:
                    working_transcript = transcript
                
                new_text = self._extract_new_text(working_transcript)
                print(f"🔄 Extracted new text: '{new_text}' (from full: '{working_transcript}')")
                
                # Update last final text for next comparison
                self._last_final_text = working_transcript
                
                # Reset interim tracking after processing final result
                self._current_interim_transcript = ""
                self._last_complete_interim = ""
                self._interim_result_counter = 0
                
                # Process speaker information if available
                speaker_segments = []
                has_speaker_data = hasattr(alternative, 'words') and alternative.words
                
                # CRITICAL FIX: Always create segments for new text, regardless of speaker data availability
                if new_text.strip():
                    if has_speaker_data and working_transcript == transcript:
                        # Normal case with speaker data and no interim fallback
                        words = alternative.words
                        new_words = self._extract_words_for_new_text(words, new_text, working_transcript)
                        
                        if new_words:
                            speaker_segments = self._extract_speaker_segments(new_words)
                            
                            print(f"🎯 Processing {len(speaker_segments)} NEW speaker segments for voice embedding")
                            if self._event_loop:
                                for segment in speaker_segments:
                                    asyncio.run_coroutine_threadsafe(
                                        self._process_speaker_segment(segment),
                                        self._event_loop
                                    )
                    else:
                        # Interim fallback used OR no speaker data - create complete text segment
                        if working_transcript != transcript:
                            print(f"📝 Interim fallback used - creating segments for complete text: '{new_text}'")
                        else:
                            print(f"📝 No speaker data available - creating default segment for: '{new_text}'")
                        
                        default_segment = {
                            "speaker": 0,
                            "text": new_text.split(),
                            "start_time": 0,
                            "end_time": len(new_text.split()),
                            "words": [],
                            "identified_user": "default_user"
                        }
                        speaker_segments = [default_segment]
                    
                    # Add segments to conversation buffer
                    if speaker_segments:
                        self._add_to_conversation_buffer(speaker_segments)
                        self._reset_silence_timer()
            
            # Process speaker information for display (if available)
            if len(transcript) > 0 and hasattr(alternative, 'words') and alternative.words:
                words = alternative.words
                
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
        """Thread function for handling streaming recognition with non-blocking error recovery."""
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
                    logger.info("Streaming stopped by request")
                    break
                self._process_streaming_response(response)
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in streaming recognition thread: {error_message}")
            
            # Handle specific Google Cloud Speech timeout errors
            if "Audio Timeout Error" in error_message or "Long duration elapsed without audio" in error_message:
                if self._tts_active:
                    logger.info("Audio timeout during TTS - will restart immediately after TTS completes")
                else:
                    logger.warning("Audio timeout during normal operation - triggering immediate restart")
                
                # CRITICAL FIX: Don't mark as disconnected immediately - this breaks audio flow
                # Instead, trigger immediate restart while keeping audio flowing
                self._needs_restart = True
                logger.info("Timeout detected - immediate restart will be triggered")
                
                # Schedule immediate restart in background
                if self._event_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._immediate_restart_after_timeout(),
                        self._event_loop
                    )
                
            elif "400" in error_message and "INVALID_ARGUMENT" in error_message:
                logger.error("Invalid argument error - check audio configuration")
                self.is_connected = False
            elif "401" in error_message or "403" in error_message:
                logger.error("Authentication error - check Google Cloud credentials")
                self.is_connected = False
            else:
                # Other errors
                logger.error(f"Unexpected error in streaming recognition: {error_message}")
                self.is_connected = False
                
        logger.info("Streaming recognition thread ended")

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

            logger.info(f"Starting Google transcription for session: {session_id}")
            
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
            
            # IMPORTANT: Clear the audio queue to prevent old audio from interfering
            queue_size = 0
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                    queue_size += 1
                except queue.Empty:
                    break
            
            if queue_size > 0:
                logger.info(f"Cleared {queue_size} old audio chunks from queue during restart")
            
            # Reset deduplication state for new session
            self._last_final_text = ""
            self._conversation_text_buffer.clear()
            
            # CRITICAL FIX: Reset interim tracking for new session
            self._current_interim_transcript = ""
            self._last_complete_interim = ""
            self._interim_result_counter = 0
            
            # Reset timeout prevention state
            self._last_audio_time = time.time()  # Reset audio time to now

            # Start streaming recognition thread
            self._streaming_thread = threading.Thread(
                target=self._streaming_recognition_thread,
                daemon=True
            )
            self._streaming_thread.start()
            
            # Give the thread a moment to start
            await asyncio.sleep(0.5)

            # Mark as connected
            self.is_connected = True
            logger.info("Google Cloud Speech transcription started successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to start Google transcription: {str(e)}")
            return False

    async def send_audio(self, audio_data: bytes):
        """Send audio data to Google Cloud Speech for transcription - RESILIENT to connection issues with INTERRUPT support."""
        try:
            # RESILIENT AUDIO FLOW: Don't reject audio during restart - queue it instead
            if not self.is_connected and not self._restart_in_progress:
                logger.debug("Attempted to send audio but transcription not connected")
                raise RuntimeError("Transcription not started")
            
            # TTS TRANSITION DEBUGGING: Track when audio arrives after TTS
            current_time = time.time()
            if hasattr(self, '_tts_end_debug_time') and (current_time - self._tts_end_debug_time) < 20.0:
                time_since_tts_end = current_time - self._tts_end_debug_time
                logger.info(f"POST-TTS AUDIO: Audio received {time_since_tts_end:.2f}s after TTS ended (queue size: {self._audio_queue.qsize()})")
            
            # SPEED OPTIMIZATION TIMEOUT CHECK: Removed - this was causing connection issues
            # Previous code was switching configs which broke the transcription connection
            
            # CRITICAL FIX: Remove speed optimization logic that was breaking connection
            # The config switching was too disruptive and caused transcription failures
            
            # INTERRUPT DETECTION: Check if user is speaking during TTS
            if self._interrupt_enabled and self._tts_active:
                # Convert audio bytes to numpy array for analysis
                try:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    if len(audio_array) > 0:
                        # Calculate audio amplitude (volume)
                        audio_float = audio_array.astype(np.float32) / 32767.0
                        max_amplitude = np.max(np.abs(audio_float))
                        
                        # Check if user is speaking (above threshold)
                        if max_amplitude > self._interrupt_threshold:
                            # Track speech duration for interrupt decision
                            if not hasattr(self, '_interrupt_start_time'):
                                self._interrupt_start_time = time.time()
                            
                            speech_duration = time.time() - self._interrupt_start_time
                            
                            # If user has been speaking for threshold duration, interrupt TTS
                            if speech_duration > self._interrupt_duration_threshold:
                                logger.info(f"USER INTERRUPT: Detected speech for {speech_duration:.2f}s during TTS - interrupting AI audio")
                                await self._interrupt_tts()
                                
                                # Reset interrupt tracking
                                if hasattr(self, '_interrupt_start_time'):
                                    delattr(self, '_interrupt_start_time')
                        else:
                            # Reset interrupt tracking if audio is quiet
                            if hasattr(self, '_interrupt_start_time'):
                                delattr(self, '_interrupt_start_time')
                except Exception as interrupt_error:
                    logger.debug(f"Error in interrupt detection: {interrupt_error}")
                    # Don't let interrupt detection break audio flow
                    pass
            
            # If restart is needed or in progress, trigger it but keep audio flowing
            if self._needs_restart and not self._restart_in_progress:
                logger.info("Restart needed - triggering immediate restart while maintaining audio flow")
                if self._event_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._immediate_restart_after_timeout(),
                        self._event_loop
                    )

            # Always add audio to queue - even during restarts
            self._audio_queue.put(audio_data)
            
            # Track queue size for debugging (reduced frequency)
            queue_size = self._audio_queue.qsize()
            
            # During restart, allow larger queue to buffer audio
            restart_queue_limit = 200 if self._restart_in_progress else 50
            
            # Enhanced post-TTS monitoring
            if hasattr(self, '_tts_end_debug_time') and (time.time() - self._tts_end_debug_time) < 10.0:
                # We're in the critical post-TTS period - monitor more closely
                if queue_size > 10:
                    logger.info(f"POST-TTS MONITORING: Queue size {queue_size} items, connection: {self.is_connected}, restart: {self._restart_in_progress}")
            
            # Only log if there are issues
            if queue_size > restart_queue_limit:  # Higher limit during restart
                if self._restart_in_progress:
                    logger.info(f"Audio buffering during restart: {queue_size} items (restart in progress)")
                else:
                    logger.warning(f"Audio queue backing up: {queue_size} items (sending {len(audio_data)} bytes)")
            elif queue_size > 20:  # Debug level for moderate backup
                logger.debug(f"Audio queue size: {queue_size} items")
            
            # Periodic verbose logging (every 100 calls) for debugging
            if not hasattr(self, '_send_audio_call_count'):
                self._send_audio_call_count = 0
            self._send_audio_call_count += 1
            
            if self._send_audio_call_count % 100 == 0:
                status = "restarting" if self._restart_in_progress else "connected" if self.is_connected else "disconnected"
                logger.debug(f"Audio processing stats: {self._send_audio_call_count} chunks sent, queue: {queue_size}, status: {status}")
            
        except Exception as e:
            logger.error(f"Error sending audio data to Google: {str(e)}")
            raise

    async def stop_transcription(self):
        """Stop the Google Cloud Speech transcription session."""
        try:
            logger.info("Stopping Google transcription...")
            
            # Mark as disconnected first to stop health checks
            self.is_connected = False
            
            # Stop streaming
            self._stop_streaming.set()
            
            # Wait for streaming thread to finish with timeout
            if self._streaming_thread and self._streaming_thread.is_alive():
                logger.debug("Waiting for streaming thread to finish...")
                self._streaming_thread.join(timeout=3.0)  # Reduced timeout
                
                if self._streaming_thread.is_alive():
                    logger.warning("Streaming thread did not stop gracefully within timeout")
                else:
                    logger.debug("Streaming thread stopped gracefully")
            
            # Process any remaining conversation only if we have segments
            if self._conversation_buffer:
                logger.debug(f"Processing {len(self._conversation_buffer)} remaining conversation segments")
                await self._process_conversation_end()
            
            # Cancel silence timer
            if self._silence_timer_task:
                self._silence_timer_task.cancel()
                self._silence_timer_task = None
            
            # Close the audio file
            if self._audio_writer:
                self._close_audio_file()
            
            # Clear conversation state
            self._clear_conversation_state()
            
            logger.info("Google transcription stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Google transcription: {str(e)}")
            # Don't re-raise - this is cleanup code

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
            
            # Reset TTS state
            self._tts_active = False
            self._last_audio_time = None
            
            logger.info("Google conversation state cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation state: {str(e)}")

    def is_transcription_healthy(self) -> bool:
        """Check if transcription connection is healthy and recovery is needed - PERSISTENT SESSION mode with relaxed checks."""
        try:
            # Connection is unhealthy if explicitly marked as disconnected
            if not self.is_connected:
                logger.debug("Health check: is_connected = False")
                return False
            
            # Connection is unhealthy if streaming thread is not alive
            if not self._streaming_thread or not self._streaming_thread.is_alive():
                logger.debug("Health check: streaming thread not alive")
                return False
            
            # SPEED OPTIMIZATION: Check if we need to switch back to standard config
            if self._speed_config_active and self._speed_config_switch_time:
                elapsed = time.time() - self._speed_config_switch_time
                if elapsed > self._speed_config_duration:
                    logger.info(f"SPEED CONFIG TIMEOUT: {elapsed:.1f}s elapsed (>{self._speed_config_duration}s) - need to switch back")
                    # Mark for config switch but don't do it here (health check should be fast)
                    self._needs_config_switch = True
                    # Trigger the switch in the event loop
                    if self._event_loop:
                        asyncio.run_coroutine_threadsafe(
                            self._switch_to_speed_config_if_needed(),
                            self._event_loop
                        )
            
            # PERSISTENT SESSION: Much more lenient timeout checking
            if self._persistent_session_mode:
                # Only consider unhealthy if no audio for extremely long periods (20+ minutes)
                if not self._tts_active and self._last_audio_time:
                    time_since_audio = time.time() - self._last_audio_time
                    # RELAXED: 20 minutes timeout instead of 90 seconds
                    persistent_timeout = 1200  # 20 minutes
                    if time_since_audio > persistent_timeout:
                        logger.warning(f"PERSISTENT SESSION: No audio for {time_since_audio:.1f}s (>{persistent_timeout}s) during non-TTS - considering restart")
                        return False
                    elif time_since_audio > 600:  # Log at 10 minutes but don't restart
                        logger.info(f"PERSISTENT SESSION: Long silence {time_since_audio:.1f}s - but keeping connection alive")
                
                # RELAXED: Much higher queue threshold for persistent sessions
                queue_size = self._audio_queue.qsize()
                if queue_size > 500:  # 5x higher threshold for persistent sessions
                    logger.warning(f"PERSISTENT SESSION: Audio queue backing up significantly ({queue_size} items)")
                    return False
                
                return True
            
            # Legacy health checking for non-persistent mode
            else:
                # Check if we've been without audio for too long during non-TTS periods
                if not self._tts_active and self._last_audio_time:
                    time_since_audio = time.time() - self._last_audio_time
                    # More aggressive timeout - 90 seconds instead of 120
                    if time_since_audio > 90:
                        logger.warning(f"Health check: No audio for {time_since_audio:.1f}s during non-TTS")
                        return False
                
                # Additional check: if we have an audio queue that's backing up, 
                # it might indicate the streaming thread is stuck
                queue_size = self._audio_queue.qsize()
                if queue_size > 100:  # If queue is backing up significantly
                    logger.warning(f"Health check: Audio queue backing up ({queue_size} items)")
                    return False
                
                return True
            
        except Exception as e:
            logger.error(f"Error checking transcription health: {str(e)}")
            return False

    async def restart_transcription_if_needed(self):
        """Restart transcription if the connection is unhealthy with retry logic."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.is_transcription_healthy():
                    if retry_count == 0:
                        logger.info("Transcription connection unhealthy - attempting restart")
                    else:
                        logger.info(f"Transcription restart attempt #{retry_count + 1}/{max_retries}")
                    
                    # Store current state
                    session_id = self._session_id
                    callback = self._on_transcript_callback
                    audio_output_callback = self._audio_output_callback
                    
                    # Stop current transcription
                    await self.stop_transcription()
                    
                    # Wait before retry with exponential backoff
                    if retry_count > 0:
                        wait_time = min(retry_count * 2, 10)  # Max 10 seconds
                        logger.info(f"Waiting {wait_time}s before retry #{retry_count + 1}")
                        await asyncio.sleep(wait_time)
                    else:
                        # Small delay for first restart
                        await asyncio.sleep(1.0)
                    
                    # Restart transcription
                    if session_id and callback:
                        logger.info(f"Restarting transcription (attempt #{retry_count + 1})")
                        success = await self.start_transcription(callback, session_id, audio_output_callback)
                        
                        if success:
                            # Give it a moment to establish
                            await asyncio.sleep(2.0)
                            
                            # Verify the restart was successful
                            if self.is_transcription_healthy():
                                logger.info(f"Transcription restarted successfully after {retry_count + 1} attempts")
                                return True
                            else:
                                logger.warning(f"Transcription restart attempt #{retry_count + 1} failed health check")
                                retry_count += 1
                                continue
                        else:
                            logger.error(f"Failed to start transcription on attempt #{retry_count + 1}")
                            retry_count += 1
                            continue
                    else:
                        logger.error("Missing session_id or callback for restart")
                        return False
                else:
                    # Connection is healthy
                    return True
                    
            except Exception as e:
                logger.error(f"Error during transcription restart attempt #{retry_count + 1}: {str(e)}")
                retry_count += 1
                continue
        
        logger.error(f"Failed to restart transcription after {max_retries} attempts")
        return False

    async def _immediate_restart_after_timeout(self):
        """Immediate restart after timeout while maintaining audio flow - CRITICAL for preventing crashes."""
        try:
            if self._restart_in_progress:
                logger.debug("Restart already in progress - skipping duplicate restart")
                return
            
            self._restart_in_progress = True
            logger.info("IMMEDIATE RESTART: Starting timeout recovery while maintaining audio flow")
            
            # Store current state for restart
            session_id = self._session_id
            callback = self._on_transcript_callback
            audio_output_callback = self._audio_output_callback
            
            if not session_id or not callback:
                logger.error("IMMEDIATE RESTART: Missing session data - cannot restart")
                return
            
            # Wait briefly if TTS is active to complete current response
            if self._tts_active:
                logger.info("IMMEDIATE RESTART: Waiting for TTS to complete before restart")
                max_wait = 5  # Reduced wait time for faster recovery
                wait_count = 0
                while self._tts_active and wait_count < max_wait:
                    await asyncio.sleep(1.0)
                    wait_count += 1
                
                if self._tts_active:
                    logger.warning("IMMEDIATE RESTART: TTS still active - interrupting for urgent restart")
                    self._tts_active = False  # Force TTS off for restart
            
            # CRITICAL FIX: Don't just restart thread - do full transcription restart
            logger.info("IMMEDIATE RESTART: Performing full transcription restart")
            
            # Stop current streaming cleanly
            self._stop_streaming.set()
            
            # Wait for streaming thread to finish
            if self._streaming_thread and self._streaming_thread.is_alive():
                await asyncio.sleep(0.5)  # Brief wait
            
            # Clear the queue partially to prevent overwhelming new connection
            queue_size = self._audio_queue.qsize()
            if queue_size > 50:
                # Keep only recent 20 audio chunks for faster restart
                recent_chunks = []
                while not self._audio_queue.empty() and len(recent_chunks) < 20:
                    try:
                        recent_chunks.append(self._audio_queue.get_nowait())
                    except queue.Empty:
                        break
                
                # Clear remaining old chunks
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Put back recent chunks
                for chunk in recent_chunks:
                    self._audio_queue.put(chunk)
                
                logger.info(f"IMMEDIATE RESTART: Cleared queue from {queue_size} to {len(recent_chunks)} items")
            
            # FULL RESTART: Create new streaming configuration and connection
            logger.info("IMMEDIATE RESTART: Creating new streaming configuration")
            self._create_streaming_config()
            
            # Reset streaming state completely
            self._stop_streaming.clear()
            self._needs_restart = False
            self._last_audio_time = time.time()
            
            # Start new streaming recognition thread with fresh connection
            logger.info("IMMEDIATE RESTART: Starting fresh streaming recognition thread")
            self._streaming_thread = threading.Thread(
                target=self._streaming_recognition_thread,
                daemon=True
            )
            self._streaming_thread.start()
            
            # Give the thread a moment to establish connection
            await asyncio.sleep(1.5)  # Slightly longer for connection establishment
            
            # Mark as connected again
            self.is_connected = True
            
            logger.info("IMMEDIATE RESTART: Full transcription restart completed successfully - audio flow maintained")
            
        except Exception as e:
            logger.error(f"IMMEDIATE RESTART: Error during restart: {str(e)}")
            logger.exception("Full restart error:")
            # Mark connection as truly broken if restart fails
            self.is_connected = False
        finally:
            self._restart_in_progress = False
            self._needs_restart = False

    async def _interrupt_tts(self):
        """Interrupt TTS when user starts speaking - PRIORITIZE USER OVER AI."""
        try:
            if not self._tts_active:
                return  # Nothing to interrupt
            
            logger.info("INTERRUPT: User speech detected - stopping TTS immediately")
            
            # Force stop TTS
            self._tts_active = False
            
            # Send interrupt notification to client
            if self._audio_output_callback:
                import json
                interrupt_message = {
                    "type": "tts_interrupted",
                    "message": "TTS interrupted by user speech",
                    "reason": "user_priority",
                    "timestamp": time.time()
                }
                try:
                    message_bytes = json.dumps(interrupt_message).encode('utf-8')
                    if self._event_loop and self._event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._audio_output_callback(message_bytes),
                            self._event_loop
                        )
                except Exception as e:
                    logger.error(f"Error sending interrupt notification: {e}")
            
            # TODO: If we have access to the TTS engine, we could stop it more gracefully
            # For now, just setting _tts_active = False will stop the keep-alive behavior
            
            logger.info("INTERRUPT: TTS successfully interrupted - prioritizing user transcription")
            
        except Exception as e:
            logger.error(f"Error during TTS interrupt: {str(e)}")
            # Always ensure TTS is stopped even if notification fails
            self._tts_active = False

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
        """Convert text response to speech using streaming TTS with persistent session coordination"""
        try:
            if not self._tts_engine:
                # Initialize TTS engine if not already done
                from app.core.config import settings
                self._tts_engine = await get_deepgram_tts(settings.DEEPGRAM_API_KEY)
            
            print(f"🗣️ Speaking response: {text[:50]}...")
            
            # Set TTS active state to enable ultra-aggressive keep-alive
            self._tts_active = True
            
            if self._persistent_session_mode:
                logger.info(f"TTS started in PERSISTENT SESSION mode - ultra-aggressive keep-alive every {self._tts_keepalive_interval}s")
            else:
                logger.info("TTS started - preventing Google Cloud Speech timeout")
            
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
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "persistent_session": self._persistent_session_mode
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
            
            # CRITICAL FIX: Remove problematic immediate speed optimization that breaks connection
            # The speed optimization config switching is too disruptive and causes transcription failures
            logger.info("TTS ending - maintaining stable connection without disruptive config changes")
            
            # Send completion message
            if self._audio_output_callback:
                import json
                session_age = time.time() - (self._last_audio_time or time.time()) if hasattr(self, '_session_start_time') else None
                tts_end_message = {
                    "type": "tts_end",
                    "message": "Text-to-speech conversion completed",
                    "success": success,
                    "persistent_session": self._persistent_session_mode,
                    "session_age": session_age,
                    "connection_stable": True
                }
                try:
                    message_bytes = json.dumps(tts_end_message).encode('utf-8')
                    safe_audio_callback(message_bytes)
                except Exception as e:
                    logger.error(f"Error sending TTS end message: {e}")
            
            if success:
                if self._persistent_session_mode:
                    print("✅ TTS streaming completed successfully - persistent session maintained")
                    logger.info("POST-TTS: Connection stable - ready for immediate audio processing")
                else:
                    print("✅ TTS streaming completed successfully")
                    
        except Exception as e:
            logger.error(f"Error in speech response: {str(e)}")
            logger.exception("TTS error:")
        finally:
            # Always clear TTS active state
            self._tts_active = False
            
            # TTS TRANSITION DEBUG: Set timestamp for transition debugging
            self._tts_end_debug_time = time.time()
            
            # CRITICAL FIX: Remove speed optimization flag to prevent disruptive config changes
            self._needs_config_switch = False  # Explicitly disable to prevent connection issues
            
            if self._persistent_session_mode:
                logger.info("POST-TTS: Resuming normal persistent session - connection maintained")
                # Add enhanced monitoring for post-TTS audio processing
                logger.info("POST-TTS: Audio processing monitoring enabled - debugging any issues")
            else:
                logger.info("TTS completed - resuming normal Google Cloud Speech operation")

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

    def configure_tts_timeout_prevention(self, 
                                       enabled: bool = True,
                                       pause_transcription: bool = True,
                                       silence_threshold: float = 5.0,
                                       max_silence_duration: float = 50.0,
                                       silence_interval: float = 0.5):
        """
        Configure TTS timeout prevention settings.
        
        Args:
            enabled: Enable/disable TTS timeout prevention
            pause_transcription: If True, pause transcription during TTS. If False, send silence.
            silence_threshold: Seconds without audio before starting prevention
            max_silence_duration: Maximum duration to send silence (when not pausing)
            silence_interval: Interval between silence chunks (when not pausing)
        """
        self._tts_timeout_prevention_enabled = enabled
        self._tts_pause_transcription = pause_transcription
        self._tts_silence_threshold = silence_threshold
        self._tts_max_silence_duration = max_silence_duration
        self._tts_silence_interval = silence_interval
        
        strategy = "pause transcription" if pause_transcription else "send silence"
        logger.info(f"TTS timeout prevention configured: {'enabled' if enabled else 'disabled'}, strategy: {strategy}")

    def configure_keepalive(self,
                           enabled: bool = True,
                           interval: float = 10.0,  # Default to ultra-aggressive for persistent sessions
                           max_silence_duration: float = 1800.0):  # Default to 30 minutes
        """
        Configure general keep-alive settings to prevent timeouts during normal operation.
        
        Args:
            enabled: Enable/disable keep-alive mechanism
            interval: Seconds between keep-alive silence packets
            max_silence_duration: Maximum duration to send keep-alive before giving up
        """
        self._keepalive_enabled = enabled
        self._keepalive_interval = interval
        self._keepalive_max_silence = max_silence_duration
        
        logger.info(f"Keep-alive configured: {'enabled' if enabled else 'disabled'}, interval: {interval}s, max: {max_silence_duration}s")

    def configure_persistent_session(self,
                                   enabled: bool = True,
                                   max_session_duration: float = 3600.0,
                                   tts_keepalive_interval: float = 3.0):
        """
        Configure persistent session mode for maintaining long-running conversations.
        
        Args:
            enabled: Enable/disable persistent session mode
            max_session_duration: Maximum session duration in seconds (safety limit)
            tts_keepalive_interval: Seconds between keep-alive during TTS (more aggressive)
        """
        self._persistent_session_mode = enabled
        self._max_session_duration = max_session_duration
        self._tts_keepalive_interval = tts_keepalive_interval
        
        if enabled:
            # Auto-configure ULTRA-aggressive settings for persistent sessions
            self.configure_keepalive(
                enabled=True,
                interval=5.0,  # Every 5 seconds (more aggressive)
                max_silence_duration=1800.0  # 30 minutes
            )
            self.configure_tts_timeout_prevention(
                enabled=True,
                pause_transcription=False,  # Send silence instead of pausing
                silence_threshold=3.0,
                max_silence_duration=300.0,  # 5 minutes during TTS
                silence_interval=1.0
            )
            self.configure_interrupt_detection(
                enabled=True,
                amplitude_threshold=0.01,  # Sensitive to user speech
                duration_threshold=0.3     # 300ms to trigger interrupt
            )
            self.configure_speed_optimization(
                enabled=True,
                duration=10.0  # 10 seconds of faster responses after TTS
            )
        
        logger.info(f"Persistent session mode: {'enabled' if enabled else 'disabled'}, max duration: {max_session_duration}s, TTS interval: {tts_keepalive_interval}s")

    def configure_interrupt_detection(self,
                                    enabled: bool = True,
                                    amplitude_threshold: float = 0.01,
                                    duration_threshold: float = 0.3):
        """
        Configure interrupt detection for prioritizing user speech over TTS.
        
        Args:
            enabled: Enable/disable interrupt detection
            amplitude_threshold: Audio amplitude threshold to detect speech (0.0-1.0)
            duration_threshold: Minimum speech duration to trigger interrupt (seconds)
        """
        self._interrupt_enabled = enabled
        self._interrupt_threshold = amplitude_threshold
        self._interrupt_duration_threshold = duration_threshold
        
        logger.info(f"Interrupt detection: {'enabled' if enabled else 'disabled'}, amplitude: {amplitude_threshold}, duration: {duration_threshold}s")

    def configure_speed_optimization(self,
                                   enabled: bool = True,
                                   duration: float = 10.0):
        """
        Configure automatic speed optimization after TTS completion.
        
        Args:
            enabled: Enable/disable automatic speed optimization
            duration: How long to use speed-optimized config (seconds)
        """
        self._speed_optimization_enabled = enabled
        self._speed_config_duration = duration
        
        if enabled:
            logger.info(f"Speed optimization enabled: {duration}s of faster config after TTS")
        else:
            logger.info("Speed optimization disabled")

    def get_tts_state(self) -> Dict[str, Any]:
        """Get current TTS and timeout prevention state for debugging."""
        current_time = time.time()
        return {
            "tts_active": self._tts_active,
            "timeout_prevention_enabled": self._tts_timeout_prevention_enabled,
            "pause_transcription": self._tts_pause_transcription,
            "silence_threshold": self._tts_silence_threshold,
            "max_silence_duration": self._tts_max_silence_duration,
            "silence_interval": self._tts_silence_interval,
            "keepalive_enabled": self._keepalive_enabled,
            "keepalive_interval": self._keepalive_interval,
            "keepalive_max_silence": self._keepalive_max_silence,
            "persistent_session_mode": self._persistent_session_mode,
            "tts_keepalive_interval": self._tts_keepalive_interval,
            "max_session_duration": self._max_session_duration,
            "interrupt_enabled": self._interrupt_enabled,
            "interrupt_threshold": self._interrupt_threshold,
            "interrupt_duration_threshold": self._interrupt_duration_threshold,
            "restart_in_progress": self._restart_in_progress,
            "needs_restart": self._needs_restart,
            "last_audio_time": self._last_audio_time,
            "time_since_last_audio": current_time - (self._last_audio_time or current_time) if self._last_audio_time else None,
            "session_age": current_time - (self._last_audio_time or current_time) if hasattr(self, '_session_start_time') else None,
            "speed_optimization_enabled": self._speed_optimization_enabled,
            "speed_config_duration": self._speed_config_duration
        }

    def _generate_silence_audio(self):
        """Generate a pre-recorded silence audio segment for keep-alive and TTS prevention."""
        try:
            # Generate 1 second of 16kHz, 1-channel, 16-bit silence
            sample_rate = 16000
            num_channels = 1
            sample_width = 2
            num_samples = sample_rate * 1  # 1 second
            
            # Create a numpy array filled with zeros
            silence_data = np.zeros(num_samples, dtype=np.int16)
            
            # Store the silence data
            self._silence_audio_data = silence_data
            self._last_audio_time = time.time()
            
            # Also save to a temporary file for debugging if needed
            try:
                silence_file_path = os.path.join(self._temp_dir, "silence.wav")
                with wave.open(silence_file_path, 'wb') as wf:
                    wf.setnchannels(num_channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(silence_data.tobytes())
                logger.debug(f"Generated silence audio file: {silence_file_path}")
            except Exception as file_error:
                logger.warning(f"Could not save silence audio file: {file_error}")
                # Don't fail initialization if file saving fails
            
            logger.info("Silence audio data generated successfully for timeout prevention")
            
        except Exception as e:
            logger.error(f"Failed to generate silence audio: {str(e)}")
            # Set to None so audio generator knows to skip silence sending
            self._silence_audio_data = None

    async def _switch_to_speed_config_if_needed(self):
        """Switch to speed-optimized configuration after TTS for faster responses."""
        try:
            if not self._speed_optimization_enabled:
                return False
            
            # Check if we should switch back to standard config
            if self._speed_config_active and self._speed_config_switch_time:
                elapsed = time.time() - self._speed_config_switch_time
                if elapsed > self._speed_config_duration:
                    # Switch back to standard config
                    logger.info("SPEED CONFIG TIMEOUT: Switching back to STANDARD config for quality")
                    await self._perform_config_switch(speed_optimized=False)
                    return True
                else:
                    # Still in speed config period, no switch needed
                    return False
            
            # Check if we need to switch TO speed config
            if self._needs_config_switch and not self._speed_config_active:
                # Switch to speed config
                logger.info("SPEED OPTIMIZATION: Switching to FASTER config for immediate post-TTS responses")
                await self._perform_config_switch(speed_optimized=True)
                return True
            
            # No switch needed
            return False
            
        except Exception as e:
            logger.error(f"Error in speed config switch: {e}")
            return False
    
    async def _perform_config_switch(self, speed_optimized=False):
        """Perform the actual configuration switch by restarting streaming with new config."""
        try:
            # Store current state
            session_id = self._session_id
            callback = self._on_transcript_callback
            audio_output_callback = self._audio_output_callback
            
            if not session_id or not callback:
                logger.error("Config switch failed: Missing session data")
                return False
            
            config_type = "SPEED-OPTIMIZED" if speed_optimized else "STANDARD"
            logger.info(f"SWITCHING to {config_type} Google Speech configuration")
            
            # Stop current streaming
            self._stop_streaming.set()
            
            # Wait briefly for streaming to stop
            if self._streaming_thread and self._streaming_thread.is_alive():
                await asyncio.sleep(0.5)
            
            # Create new configuration
            self._create_streaming_config(speed_optimized=speed_optimized)
            
            # Update state tracking
            if speed_optimized:
                self._speed_config_active = True
                self._speed_config_switch_time = time.time()
                self._needs_config_switch = False
            else:
                self._speed_config_active = False
                self._speed_config_switch_time = None
                self._needs_config_switch = False
            
            # Clear queue partially to prevent overwhelming new connection
            queue_size = self._audio_queue.qsize()
            if queue_size > 20:
                # Keep only recent audio chunks
                recent_chunks = []
                while not self._audio_queue.empty() and len(recent_chunks) < 10:
                    try:
                        recent_chunks.append(self._audio_queue.get_nowait())
                    except queue.Empty:
                        break
                
                # Clear remaining old chunks
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Put back recent chunks
                for chunk in recent_chunks:
                    self._audio_queue.put(chunk)
                
                logger.info(f"CONFIG SWITCH: Cleared queue from {queue_size} to {len(recent_chunks)} items")
            
            # Reset streaming state
            self._stop_streaming.clear()
            self._last_audio_time = time.time()
            
            # Start new streaming thread with new configuration
            self._streaming_thread = threading.Thread(
                target=self._streaming_recognition_thread,
                daemon=True
            )
            self._streaming_thread.start()
            
            # Give thread time to establish connection
            await asyncio.sleep(1.0)
            
            self.is_connected = True
            logger.info(f"CONFIG SWITCH: Successfully switched to {config_type} configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error performing config switch: {e}")
            self.is_connected = False
            return False

async def get_google_transcriber() -> GoogleTranscriber:
    """Factory function to get a GoogleTranscriber instance."""
    return GoogleTranscriber()