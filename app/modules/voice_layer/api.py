from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.modules.voice_layer.voice_processor import get_voice_processor
from app.modules.ai_wrapper.llm_wrapper import get_llm_wrapper
import json

router = APIRouter()

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