from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from app.modules.voice_layer.voice_processor import get_voice_processor, VoiceProcessor
from app.modules.voice_layer.deepgram import get_deepgram_transcriber
import numpy as np
import wave
import io
import logging
import time
from typing import Optional, Deque, Dict, Any
from collections import deque
import asyncio

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
        voice_processor = await get_voice_processor() 
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
        identified_user_id = processor.identify_speaker(embedding, 0.4)  # Lower threshold for cosine distance
        print("identified_user_id: ", identified_user_id)
        
        if identified_user_id:
            
            # Fetch user details from MongoDB
            # TODO: Uncomment this when we have a way to get user details from MongoDB
            # user = await user_service.get_user(identified_user_id)

            user = identified_user_id
            
            if user:
                # TODO: Uncomment this when we have a way to get user details from MongoDB
                # logger.info(f"Identified user {user.username} (ID: {identified_user_id})")
                return {
                    "status": "success",
                    # TODO: Uncomment this when we have a way to get user details from MongoDB
                    #"identified_user": user.model_dump(), # Return user model (Pydantic v2)
                    "identified_user": user,
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

@router.websocket("/ws/stream")
async def websocket_transcribe(websocket: WebSocket, provider: str = "deepgram"):
    """
    WebSocket endpoint for real-time audio transcription with diarization.
    Accepts audio data in chunks and returns transcription results.
    
    Query Parameters:
        provider: Transcription provider to use ("deepgram" or "google")
    """
    transcriber = None
    websocket_active = True
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for transcription with provider: {provider}")

        # Generate a unique session ID
        session_id = f"session_{int(time.time())}_{id(websocket)}"
        logger.info(f"Created new session: {session_id}")

        # Initialize transcriber based on provider
        from app.modules.voice_layer.transcriber_factory import get_transcriber
        transcriber = await get_transcriber(provider)
        logger.info(f"{provider.capitalize()} transcriber initialized")
        
        # Callback function to handle transcription results
        async def handle_transcript(transcript_data: Dict[str, Any]):
            nonlocal websocket_active
            if not websocket_active:
                return
            
            try:
                await websocket.send_json(transcript_data)
            except WebSocketDisconnect:
                logger.info("Client disconnected while sending response")
                websocket_active = False
            except Exception as e:
                logger.error(f"Error sending transcript to client: {str(e)}")
                websocket_active = False

        # Start transcription with session_id
        success = await transcriber.start_transcription(handle_transcript, session_id)
        if not success:
            logger.error("Failed to start transcription")
            return

        try:
            while websocket_active:
                # Receive audio data from client
                audio_data = await websocket.receive_bytes()
                
                # Send audio data to Deepgram
                await transcriber.send_audio(audio_data)
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
            websocket_active = False
        except Exception as e:
            logger.error(f"Error in transcription websocket: {str(e)}")
            websocket_active = False
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error in transcription: {str(e)}"
                })
            except:
                # If we can't send the error message, the connection is likely already closed
                logger.info("Could not send error message, connection likely closed")

    except Exception as e:
        logger.error(f"Error in transcription websocket: {str(e)}")
        websocket_active = False
    finally:
        # Mark connection as inactive first
        websocket_active = False
        
        # Clean up resources
        if transcriber:
            try:
                await transcriber.stop_transcription()
            except Exception as e:
                logger.error(f"Error stopping transcription: {str(e)}")
        
        logger.info("WebSocket endpoint cleanup completed") 
