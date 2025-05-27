from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from app.modules.voice_layer.voice_processor import get_voice_processor
from app.modules.ai_wrapper.llm_wrapper import get_llm_wrapper
import json
import numpy as np
import wave
import io
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register-speaker")
async def register_speaker(
    name: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Register a new speaker using an audio file.
    
    Args:
        name: Name/ID of the speaker
        audio_file: Audio file in WAV format (16kHz, mono)
    
    Returns:
        dict: Registration status and details
    """
    try:
        logger.info(f"Received registration request for speaker: {name}")
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
        processor = get_voice_processor()
        
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
        
        # Register the speaker
        processor.register_new_speaker(name, embedding)
        logger.info(f"Speaker {name} registered successfully")
        
        return {
            "status": "success",
            "message": f"Speaker {name} registered successfully",
            "details": {
                "embedding_shape": embedding.shape,
                "sample_rate": frame_rate,
                "processed_audio_length": len(processed_audio)
            }
        }
        
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
        dict: Identification results
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
        processor = get_voice_processor()
        
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
        speaker = processor.identify_speaker(embedding)
        
        return {
            "status": "success",
            "identified_speaker": speaker,
            "details": {
                "embedding_shape": embedding.shape,
                "sample_rate": frame_rate,
                "processed_audio_length": len(processed_audio)
            }
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
    voice_processor = get_voice_processor()
    llm_wrapper = get_llm_wrapper()
    
    try:
        # Start Deepgram stream
        async def handle_transcript(data):
            if 'channel' in data and 'alternatives' in data['channel']:
                transcript = data['channel']['alternatives'][0]['transcript']
                if transcript.strip():
                    # Get AI response
                    response = await llm_wrapper.generate_response(
                        user_id="current_user",  # TODO: Get actual user ID
                        user_input=transcript
                    )
                    # Send response back to client
                    await websocket.send_json({
                        "type": "response",
                        "text": response
                    })
        
        stream = await voice_processor.start_stream(handle_transcript)
        
        # Handle incoming audio data
        while True:
            try:
                data = await websocket.receive_bytes()
                await voice_processor.process_audio(data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                break
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await voice_processor.stop_stream()
        await websocket.close() 