from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from app.modules.voice_layer.voice_processor import get_voice_processor, VoiceProcessor
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper, GeminiWrapper
from app.services.user_service import UserService, get_user_service
from app.services.context_service import ContextService, get_context_service
import json
import numpy as np
import wave
import io
import logging
from typing import Optional, Deque
from collections import deque
import asyncio
import time

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants for audio processing
AUDIO_BUFFER_SIZE = 16000 * 2  # 2 seconds of audio at 16kHz
VAD_FRAME_SIZE = 480  # 30ms at 16kHz
MIN_SPEECH_FRAMES = 10  # Minimum number of speech frames to process

# Get VoiceProcessor and other dependencies
voice_processor: Optional[VoiceProcessor] = None

async def get_async_voice_processor() -> VoiceProcessor:
    """Async getter for VoiceProcessor"""
    global voice_processor
    if voice_processor is None:
        # Ensure dependencies are available before initializing VoiceProcessor
        gemini_wrapper = await get_gemini_wrapper()
        user_service = await get_user_service()
        context_service = await get_context_service()
        # Pass all dependencies to get_voice_processor
        voice_processor = await get_voice_processor(gemini_wrapper, user_service, context_service) 
    return voice_processor

class AudioBuffer:
    """Class to handle audio buffering and processing"""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.buffer: Deque[float] = deque(maxlen=AUDIO_BUFFER_SIZE)
        self.speech_frames: Deque[float] = deque()
        self.is_speaking = False
        self.last_processing_time = 0
        self.processing_interval = 1.0  # Process every 1 second

    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        # Convert to float32 and normalize if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32) / 32767.0
        
        # Add to buffer
        self.buffer.extend(audio_chunk)

    def process_buffer(self, voice_processor: VoiceProcessor) -> Optional[np.ndarray]:
        """Process buffer and return embedding if speech is detected"""
        current_time = time.time()
        if current_time - self.last_processing_time < self.processing_interval:
            return None

        self.last_processing_time = current_time
        
        if len(self.buffer) < VAD_FRAME_SIZE:
            return None

        # Convert buffer to numpy array
        audio_data = np.array(self.buffer)
        
        # Process with VAD
        processed_audio = voice_processor._process_with_vad(audio_data)
        
        if len(processed_audio) > 0:
            # Extract embedding
            embedding = voice_processor.extract_voice_embedding(processed_audio)
            # Clear buffer after processing
            self.buffer.clear()
            return embedding
        
        return None

@router.post("/register-speaker")
async def register_speaker(
    user_id: str = Form(...), # Changed from name to user_id
    audio_file: UploadFile = File(...)
):
    """
    Register a new speaker using an audio file and their MongoDB user ID.
    
    Args:
        user_id: The MongoDB User ID of the speaker.
        audio_file: Audio file in WAV format (16kHz, mono).
    
    Returns:
        dict: Registration status and details.
    """
    try:
        logger.info(f"Received registration request for user ID: {user_id}")
        logger.info(f"Audio file: {audio_file.filename}, content_type: {audio_file.content_type}")
        
        # Read the audio file
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
            
            # Read audio data
            audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            # Convert stereo to mono if needed
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # Get voice processor instance
        # Use the async getter for VoiceProcessor
        processor = await get_async_voice_processor()
        
        # Process audio with VAD
        processed_audio = processor._process_with_vad(audio_data)
        logger.info(f"Processed audio length: {len(processed_audio)}")
        
        if len(processed_audio) == 0:
            logger.warning("No speech detected in the audio")
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the audio. Please ensure the audio contains clear speech."
            )
        
        # Extract voice embedding
        embedding = processor.extract_voice_embedding(processed_audio)
        
        if embedding is None:
            logger.warning("Failed to extract voice features")
            raise HTTPException(
                status_code=400,
                detail="Failed to extract voice features. Please ensure the audio is clear and contains speech."
            )
        
        # Register the speaker using the user_id
        success = processor.register_new_speaker(user_id, embedding)
        
        if success:
            logger.info(f"Speaker {user_id} registered successfully")
            return {
                "status": "success",
                "message": f"Speaker {user_id} registered successfully",
                "details": {
                    "embedding_shape": embedding.shape,
                    "sample_rate": frame_rate,
                    "processed_audio_length": len(processed_audio)
                }
            }
        else:
             logger.error(f"Failed to register speaker {user_id}")
             raise HTTPException(
                 status_code=500,
                 detail=f"Failed to register speaker {user_id}"
             )
        
    except wave.Error as e:
        logger.error(f"Invalid WAV file format: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Invalid WAV file format. Please ensure the file is a valid WAV file."
        )
    except Exception as e:
        logger.error(f"Error registering speaker: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error registering speaker: {str(e)}"
        )

@router.post("/identify-speaker")
async def identify_speaker(
    audio_file: UploadFile = File(...)
):
    """
    Identify a speaker from an audio file.
    
    Args:
        audio_file: Audio file in WAV format (16kHz, mono)
    
    Returns:
        dict: Identification results.
    """
    try:
        # Read the audio file
        contents = await audio_file.read()
        
        # Convert to numpy array using wave
        with wave.open(io.BytesIO(contents), 'rb') as wf:
            # Get audio parameters
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read audio data
            audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            # Convert stereo to mono if needed
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # Get voice processor instance
        processor = await get_async_voice_processor()
        
        # Process audio with VAD
        processed_audio = processor._process_with_vad(audio_data)
        
        if len(processed_audio) == 0:
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the audio. Please ensure the audio contains clear speech."
            )
        
        # Extract voice embedding
        embedding = processor.extract_voice_embedding(processed_audio)
        
        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract voice features. Please ensure the audio is clear and contains speech."
            )
        
        # Identify the speaker
        identified_user_id = processor.identify_speaker(embedding)
        
        if identified_user_id:
            # Get UserService instance
            user_service = await get_user_service() # Use async getter
            
            # Fetch user details from MongoDB
            user = await user_service.get_user(identified_user_id)
            
            if user:
                logger.info(f"Identified user {user.username} (ID: {identified_user_id})")
                return {
                    "status": "success",
                    "identified_user": user.model_dump(), # Return user model (Pydantic v2)
                    "details": {
                        "embedding_shape": embedding.shape,
                        "sample_rate": frame_rate,
                        "processed_audio_length": len(processed_audio)
                    }
                }
            else:
                logger.warning(f"Identified embedding for unknown user ID: {identified_user_id}")
                return {
                     "status": "not_identified",
                     "message": f"Identified embedding for unknown user ID: {identified_user_id}"
                 }
        else:
             return {
                 "status": "not_identified",
                 "message": "No speaker identified above threshold"
             }
        
    except wave.Error:
        raise HTTPException(
            status_code=400,
            detail="Invalid WAV file format. Please ensure the file is a valid WAV file."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error identifying speaker: {str(e)}"
        )

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Get voice processor instance and set the websocket
    voice_processor = await get_async_voice_processor()
    voice_processor.set_websocket(websocket)
    
    # Audio buffer for potential local processing (e.g., VAD, speaker identification)
    # Note: This buffer is separate from the audio sent directly to Deepgram.
    audio_buffer = AudioBuffer()
    
    # Placeholder for Deepgram stream
    deepgram_stream = None

    try:
        # Start the Deepgram stream. The processor handles the transcript callback internally.
        deepgram_stream = await voice_processor.start_stream(None) # Callback is now handled internally

        if deepgram_stream is None:
             logger.error("Failed to start Deepgram stream.")
             await websocket.close(code=1011) # Internal Error
             return

        # Handle incoming audio data from the WebSocket client
        while True:
            try:
                data = await websocket.receive_bytes()
                
                # Send audio data to the VoiceProcessor for Deepgram and potential local processing
                await voice_processor.process_audio(data)

                # Optional: process audio locally for immediate feedback or speaker ID hint
                # This part remains for potential local processing if needed.
                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                # Add to buffer
                audio_buffer.add_audio(audio_chunk)
                # Process buffer periodically for speaker identification hint
                embedding = audio_buffer.process_buffer(voice_processor)
                if embedding is not None:
                    # Identify speaker and update the VoiceProcessor's current_speaker hint
                    identified_user_id = voice_processor.identify_speaker(embedding)
                    if identified_user_id:
                        voice_processor.current_speaker = identified_user_id
                        # Optional: Send speaker identification update to client
                        # user_service = await get_user_service()
                        # user = await user_service.get_user(identified_user_id)
                        # if user:
                        #     await websocket.send_json({
                        #         "type": "speaker_identified",
                        #         "user": user.model_dump()
                        #     })
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected.")
                break
            except Exception as e:
                logger.error(f"Error processing incoming audio data: {str(e)}")
                await websocket.close(code=1011) # Internal Error
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        await websocket.close(code=1011) # Internal Error
    finally:
        # Stop the Deepgram stream when the WebSocket closes
        if deepgram_stream:
             await voice_processor.stop_stream()
        await websocket.close() 