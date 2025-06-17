import chromadb
import logging
import os
import shutil
import numpy as np
from typing import List, Dict, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class ChromaManager:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaManager, cls).__new__(cls)
        return cls._instance

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length (L2 normalization)"""
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            logger.warning("Zero norm embedding encountered, returning original")
            return embedding
        normalized = embedding_array / norm
        return normalized.tolist()

    def _initialize_client(self):
        """Initialize ChromaDB client with persistent storage"""
        if self._client is not None:
            logger.info("ChromaDB client already initialized")
            return

        try:
            # Use persistent storage instead of in-memory to avoid conflicts
            persist_directory = getattr(settings, 'CHROMA_PERSIST_DIRECTORY', './data/chroma')
            
            # Ensure the directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize persistent client
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Successfully initialized ChromaDB persistent client at {persist_directory}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            # Try to clean up and reinitialize if there's a conflict
            try:
                logger.info("Attempting to clean up and reinitialize ChromaDB...")
                self._cleanup_and_reinitialize()
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up and reinitialize: {str(cleanup_error)}")
                raise e

    def _cleanup_and_reinitialize(self):
        """Clean up existing ChromaDB instance and reinitialize"""
        persist_directory = getattr(settings, 'CHROMA_PERSIST_DIRECTORY', './data/chroma')
        
        # Close existing client if any
        if self._client is not None:
            try:
                # ChromaDB doesn't have an explicit close method, just clear the reference
                self._client = None
            except Exception as e:
                logger.warning(f"Error during client cleanup: {e}")
        
        # For development only: remove the directory to start fresh
        # In production, you might want to handle this differently
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
                logger.info(f"Removed existing ChromaDB directory: {persist_directory}")
            except Exception as e:
                logger.warning(f"Could not remove directory: {e}")
        
        # Recreate directory and client
        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        logger.info("Successfully reinitialized ChromaDB client")

    @property
    def client(self):
        """Get the ChromaDB client, initializing if necessary"""
        if self._client is None:
            self._initialize_client()
        return self._client

    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection with cosine similarity"""
        try:
            # Use the client property to ensure initialization
            client = self.client
            
            # Try to get existing collection first
            try:
                collection = client.get_collection(name=collection_name)
                logger.info(f"Retrieved existing ChromaDB collection: {collection_name}")
                return collection
            except Exception:
                # Collection doesn't exist, create it with cosine distance
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine distance
                )
                logger.info(f"Created new ChromaDB collection: {collection_name} with cosine distance")
                return collection
                
        except Exception as e:
            logger.error(f"Error getting or creating collection {collection_name}: {str(e)}")
            raise

    def add_embedding(self, collection_name: str, embedding: List[float], metadata: Dict, id: str):
        """Add an embedding to a collection with normalization"""
        try:
            # Normalize the embedding before storing
            normalized_embedding = self._normalize_embedding(embedding)
            logger.debug(f"Normalized embedding from norm {np.linalg.norm(embedding):.4f} to {np.linalg.norm(normalized_embedding):.4f}")
            
            collection = self.get_or_create_collection(collection_name)
            
            # Check if embedding already exists and update or add accordingly
            try:
                existing = collection.get(ids=[id])
                if existing['ids']:
                    # Update existing embedding
                    collection.update(
                        ids=[id],
                        embeddings=[normalized_embedding],
                        metadatas=[metadata]
                    )
                    logger.debug(f"Updated normalized embedding with ID {id} in {collection_name}")
                else:
                    # Add new embedding
                    collection.add(
                        embeddings=[normalized_embedding],
                        metadatas=[metadata],
                        ids=[id]
                    )
                    logger.debug(f"Added new normalized embedding with ID {id} to {collection_name}")
            except Exception:
                # If get fails, assume it doesn't exist and add
                collection.add(
                    embeddings=[normalized_embedding],
                    metadatas=[metadata],
                    ids=[id]
                )
                logger.debug(f"Added normalized embedding with ID {id} to {collection_name}")
                
        except Exception as e:
            logger.error(f"Error adding embedding to {collection_name}: {str(e)}")
            raise

    def query_embeddings(self, collection_name: str, query_embedding: List[float], n_results: int = 1):
        """Query a collection for similar embeddings with normalization"""
        try:
            # Normalize the query embedding
            normalized_query = self._normalize_embedding(query_embedding)
            logger.debug(f"Normalized query embedding from norm {np.linalg.norm(query_embedding):.4f} to {np.linalg.norm(normalized_query):.4f}")
            
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_embeddings=[normalized_query],
                n_results=n_results
            )
            logger.debug(f"Queried {collection_name} with {n_results} results using normalized embedding")
            return results
        except Exception as e:
            logger.error(f"Error querying embeddings from {collection_name}: {str(e)}")
            raise
            
    def delete_embedding(self, collection_name: str, id: str):
        """Delete an embedding by ID"""
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(ids=[id])
            logger.debug(f"Deleted embedding with ID {id} from {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting embedding from {collection_name}: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup ChromaDB resources"""
        try:
            if self._client is not None:
                self._client = None
                logger.info("ChromaDB client cleaned up")
        except Exception as e:
            logger.error(f"Error during ChromaDB cleanup: {str(e)}")

# Singleton instance and getter function
_chroma_manager = None

def get_chroma_manager() -> ChromaManager:
    """Get singleton instance of ChromaManager"""
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager 