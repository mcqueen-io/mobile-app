from typing import Union
from enum import Enum
import logging

from app.modules.voice_layer.deepgram import DeepgramTranscriber, get_deepgram_transcriber
from app.modules.voice_layer.google_transcriber import GoogleTranscriber, get_google_transcriber

logger = logging.getLogger(__name__)

class TranscriberProvider(str, Enum):
    DEEPGRAM = "deepgram"
    GOOGLE = "google"

class TranscriberFactory:
    """Factory class for creating different transcription providers."""
    
    @staticmethod
    async def create_transcriber(provider: TranscriberProvider) -> Union[DeepgramTranscriber, GoogleTranscriber]:
        """Create a transcriber instance based on the provider."""
        try:
            if provider == TranscriberProvider.DEEPGRAM:
                logger.info("Creating Deepgram transcriber")
                return await get_deepgram_transcriber()
            
            elif provider == TranscriberProvider.GOOGLE:
                logger.info("Creating Google Cloud Speech transcriber")
                return await get_google_transcriber()
            
            else:
                raise ValueError(f"Unknown transcriber provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to create transcriber for provider {provider}: {str(e)}")
            raise

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available transcription providers."""
        return [provider.value for provider in TranscriberProvider]

async def get_transcriber(provider: str = "deepgram") -> Union[DeepgramTranscriber, GoogleTranscriber]:
    """
    Convenience function to get a transcriber instance.
    
    Args:
        provider: The transcription provider to use ("deepgram" or "google")
        
    Returns:
        A transcriber instance
    """
    try:
        provider_enum = TranscriberProvider(provider.lower())
        return await TranscriberFactory.create_transcriber(provider_enum)
    except ValueError:
        logger.warning(f"Unknown provider '{provider}', falling back to Deepgram")
        return await TranscriberFactory.create_transcriber(TranscriberProvider.DEEPGRAM) 