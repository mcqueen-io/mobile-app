import chromadb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ChromaManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaManager, cls).__new__(cls)
            # Initialize client here for singleton
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize ChromaDB client (in-memory for now)"""
        try:
            # Use an in-memory client for prototyping
            # For persistent storage, you would configure this differently,
            # e.g., chromadb.PersistentClient(path="./chroma_data")
            self.client = chromadb.Client()
            logger.info("Successfully initialized ChromaDB in-memory client")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            # Check if collection exists
            existing_collections = [col.name for col in self.client.list_collections()]
            if collection_name in existing_collections:
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"Retrieved existing ChromaDB collection: {collection_name}")
            else:
                collection = self.client.create_collection(name=collection_name)
                logger.info(f"Created new ChromaDB collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error getting or creating collection {collection_name}: {str(e)}")
            raise

    def add_embedding(self, collection_name: str, embedding: List[float], metadata: Dict, id: str):
        """Add an embedding to a collection"""
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[id]
            )
            logger.debug(f"Added embedding with ID {id} to {collection_name}")
        except Exception as e:
            logger.error(f"Error adding embedding to {collection_name}: {str(e)}")
            raise

    def query_embeddings(self, collection_name: str, query_embedding: List[float], n_results: int = 1):
        """Query a collection for similar embeddings"""
        try:
            collection = self.client.get_collection(name=collection_name) # Assume collection exists after adding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            logger.debug(f"Queried {collection_name} with {n_results} results")
            return results
        except Exception as e:
            logger.error(f"Error querying embeddings from {collection_name}: {str(e)}")
            raise
            
    def delete_embedding(self, collection_name: str, id: str):
        """Delete an embedding by ID"""
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=[id])
            logger.debug(f"Deleted embedding with ID {id} from {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting embedding from {collection_name}: {str(e)}")
            raise

# Singleton instance and getter function
_chroma_manager = None

def get_chroma_manager() -> ChromaManager:
    """Get singleton instance of ChromaManager"""
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager 