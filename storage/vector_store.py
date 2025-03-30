"""
Vector store implementation using Qdrant in embedded mode.
"""
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    raise ImportError("Qdrant client not installed. Install with 'pip install qdrant-client'")

from processing.chunker import TextChunk


class VectorStore:
    """Vector store for document chunks using Qdrant in embedded mode."""
    
    def __init__(self, path: str, collection_name: str = "documents", vector_size: int = 768):
        """
        Initialize the vector store.
        
        Args:
            path: Path to store the vector database
            collection_name: Name of the collection to use
            vector_size: Dimension of the embedding vectors
        """
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client in embedded mode
        self.client = QdrantClient(path=str(self.path))
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """
        Add text chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors corresponding to the chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
            
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Convert metadata to payload (ensure all values are JSON serializable)
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
                "parent_id": chunk.parent_id
            }
            
            # Add all metadata as separate fields
            for key, value in chunk.metadata.items():
                payload[key] = value
                
            points.append(models.PointStruct(
                id=i,
                vector=embedding,
                payload=payload
            ))
            
        if points:
            # Add points in batches (or all at once for small sets)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Embedding of the query
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of chunks with similarity scores
        """
        filter_param = None
        if filter_conditions:
            # Convert filter conditions to Qdrant filter format
            filter_param = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )
            
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_param
        )
        
        # Convert search results to a more usable format
        results = []
        for hit in search_result:
            result = hit.payload
            result["score"] = hit.score
            results.append(result)
            
        return results