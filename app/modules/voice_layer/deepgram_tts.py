import asyncio
import logging
from typing import Optional, Callable, Any
from deepgram import DeepgramClient, SpeakOptions
try:
    from deepgram import SpeakWebSocketEvents
except ImportError:
    # Fallback for older versions
    SpeakWebSocketEvents = None
import queue
import threading
import time

logger = logging.getLogger(__name__)

class DeepgramTTS:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = DeepgramClient(api_key)
        self.is_streaming = False
        self._audio_callback = None
        self._audio_queue = queue.Queue()
        self._playback_thread = None
        self._stop_playback = threading.Event()
        
    async def speak_streaming(self, text: str, on_audio_chunk: Callable[[bytes], None]) -> bool:
        """
        Convert text to speech using Deepgram's streaming TTS via WebSocket
        
        Args:
            text: Text to convert to speech
            on_audio_chunk: Callback function to handle audio chunks as they arrive
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Starting streaming TTS for text: {text[:50]}...")
            
            # Create WebSocket connection for streaming TTS
            connection = self.client.speak.websocket.v("1")
            
            # Track audio chunks received
            audio_chunks_received = []
            connection_completed = asyncio.Event()
            text_sent = False
            
            # Set up event handlers with proper signatures
            def on_open(self, open_event, **kwargs):
                nonlocal text_sent
                logger.info("TTS WebSocket connection opened")
                try:
                    # Send text for conversion after connection opens
                    connection.send_text(text)
                    text_sent = True
                    logger.info(f"Text sent to TTS: {text[:50]}...")
                    # Flush to ensure processing
                    connection.flush()
                    logger.info("TTS flush sent")
                except Exception as e:
                    logger.error(f"Error sending text to TTS: {e}")
            
            def on_close(self, close_event, **kwargs):
                logger.info("TTS WebSocket connection closed")
                connection_completed.set()
            
            def on_error(self, error, **kwargs):
                logger.error(f"TTS WebSocket error: {error}")
                # Don't set completion on error - let it timeout gracefully
            
            def on_audio_data(self, data, **kwargs):
                logger.debug(f"Received TTS audio chunk: {len(data)} bytes")
                audio_chunks_received.append(data)
                if on_audio_chunk:
                    try:
                        on_audio_chunk(data)
                    except Exception as e:
                        logger.error(f"Error in audio chunk callback: {e}")
            
            def on_flushed(self, **kwargs):
                logger.info("TTS flush completed - all text processed")
                # Set completion when flush is done
                connection_completed.set()
            
            # Register event handlers
            if SpeakWebSocketEvents:
                # Use event constants if available
                connection.on(SpeakWebSocketEvents.Open, on_open)
                connection.on(SpeakWebSocketEvents.Close, on_close)
                connection.on(SpeakWebSocketEvents.Error, on_error)
                connection.on(SpeakWebSocketEvents.AudioData, on_audio_data)
                connection.on(SpeakWebSocketEvents.Flushed, on_flushed)
            else:
                # Fallback to string events
                connection.on("open", on_open)
                connection.on("close", on_close)
                connection.on("error", on_error)
                connection.on("audio", on_audio_data)
                connection.on("flushed", on_flushed)
            
            # Configure TTS options
            options = {
                "model": "aura-asteria-en",  # Natural female voice
                "encoding": "linear16",
                "sample_rate": 16000
            }
            
            # Start the connection
            success = connection.start(options)
            if not success:
                logger.error("Failed to start TTS WebSocket connection")
                return False
            
            # Wait for connection to complete with longer timeout for longer text
            text_length_factor = len(text) / 100  # Rough estimate: 100 chars per second
            timeout_duration = max(15.0, text_length_factor * 2)  # At least 15 seconds
            
            try:
                await asyncio.wait_for(connection_completed.wait(), timeout=timeout_duration)
                logger.info("TTS connection completed normally")
            except asyncio.TimeoutError:
                logger.info(f"TTS connection timeout after {timeout_duration}s, but received {len(audio_chunks_received)} chunks")
                # This is not necessarily an error if we got audio chunks
            
            # Clean up connection
            try:
                connection.finish()
                logger.debug("TTS connection finished cleanly")
            except Exception as e:
                logger.debug(f"TTS connection cleanup: {e}")
            
            success = len(audio_chunks_received) > 0
            if success:
                logger.info(f"Streaming TTS completed successfully - received {len(audio_chunks_received)} chunks")
            else:
                logger.warning("Streaming TTS completed but no audio chunks received")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in streaming TTS: {str(e)}")
            logger.exception("Streaming TTS error:")
            return False

async def get_deepgram_tts(api_key: str) -> DeepgramTTS:
    """Factory function to get a DeepgramTTS instance"""
    return DeepgramTTS(api_key) 