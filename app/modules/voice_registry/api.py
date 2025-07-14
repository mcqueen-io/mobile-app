from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from app.modules.voice_registry.voice_processor import get_voice_processor, VoiceProcessor
import logging
import time
from typing import Optional, Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

# Get VoiceProcessor and other dependencies
voice_processor: Optional[VoiceProcessor] = None

async def get_async_voice_processor() -> VoiceProcessor:
    """Async getter for VoiceProcessor"""
    global voice_processor
    if voice_processor is None:
        voice_processor = await get_voice_processor() 
    return voice_processor

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

        voice_processor = await get_async_voice_processor()

        embedding_data = await voice_processor.process_audio(audio_file)
        
        if not embedding_data["success"]:
            raise HTTPException(
                status_code=embedding_data["status_code"],
                detail=embedding_data["message"]
            )
        
        embedding = embedding_data["embedding"]
        
        # Register the speaker using the user_id
        success = voice_processor.register_new_speaker(user_id, embedding)
        
        if success:
            logger.info(f"Speaker {user_id} registered successfully")
            return {
                "status": "success",
                "detail": f"Speaker {user_id} registered successfully"
            }
        else:
             logger.error(f"Failed to register speaker {user_id}")
             raise HTTPException(
                 status_code=500,
                 detail=f"Failed to register speaker {user_id}"
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
        voice_processor = await get_async_voice_processor()

        embedding_data = await voice_processor.process_audio(audio_file)
        
        if not embedding_data["success"]:
            raise HTTPException(
                status_code=embedding_data["status_code"],
                detail=embedding_data["message"]
            )
        
        embedding = embedding_data["embedding"]
        
        # Identify the speaker
        identified_user_id = voice_processor.identify_speaker(embedding, 0.4)  # Lower threshold for cosine distance
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
                    "identified_user": user
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

        # Callback function to handle audio output from TTS
        async def handle_audio_output(audio_data: bytes):
            nonlocal websocket_active
            if not websocket_active:
                return
            
            try:
                # Try to decode as JSON message first (TTS status messages)
                try:
                    import json
                    message = json.loads(audio_data.decode('utf-8'))
                    await websocket.send_json(message)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If not JSON, treat as raw audio data (shouldn't happen with current implementation)
                    await websocket.send_bytes(audio_data)
            except WebSocketDisconnect:
                logger.info("Client disconnected while sending audio")
                websocket_active = False
            except Exception as e:
                logger.error(f"Error sending audio to client: {str(e)}")
                websocket_active = False

        # Start transcription with session_id and audio output callback
        success = await transcriber.start_transcription(handle_transcript, session_id, handle_audio_output)
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
