"""
Processing pipeline for digital humanities texts.
"""
from typing import List, Dict, Any, Optional, Callable
import time
from pathlib import Path

from connectors.obsidian import ObsidianConnector, ObsidianNote
from processing.chunker import SemanticChunker, TextChunk
from models.llm import OllamaClient, DigitalHumanitiesLLM
from storage.vector_store import VectorStore
import config


class Pipeline:
    """Sequential processing pipeline for digital humanities texts."""
    
    def __init__(
        self,
        vault_path: str = config.DEFAULT_VAULT_PATH,
        vector_store_path: str = config.VECTOR_STORE_PATH,
        ollama_base_url: str = config.OLLAMA_BASE_URL
    ):
        """
        Initialize the pipeline.
        
        Args:
            vault_path: Path to the Obsidian vault
            vector_store_path: Path to store vectors
            ollama_base_url: URL for Ollama API
        """
        # Initialize components
        self.obsidian = ObsidianConnector(vault_path)
        
        self.chunker = SemanticChunker(
            target_chunk_size=config.TARGET_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            min_chunk_size=config.MIN_CHUNK_SIZE,
            max_chunk_size=config.MAX_CHUNK_SIZE
        )
        
        self.ollama = OllamaClient(
            base_url=ollama_base_url,
            generation_model=config.GENERATION_MODEL,
            embedding_model=config.EMBEDDING_MODEL
        )
        
        self.llm = DigitalHumanitiesLLM(client=self.ollama)
        
        self.vector_store = VectorStore(
            path=vector_store_path,
            vector_size=config.VECTOR_DIMENSION
        )
        
        # Processing stats
        self.stats = {
            "processed_notes": 0,
            "total_chunks": 0,
            "start_time": None,
            "end_time": None
        }
    
    def process_vault(self, callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        """
        Process all notes in the vault.
        
        Args:
            callback: Optional callback function for progress updates
            
        Returns:
            Processing statistics
        """
        self.stats["start_time"] = time.time()
        
        all_notes = self.obsidian.get_all_notes()
        total_notes = len(all_notes)
        
        for i, note in enumerate(all_notes):
            self.process_note(note)
            
            # Call progress callback if provided
            if callback:
                callback(note.title, i + 1, total_notes)
                
        self.stats["end_time"] = time.time()
        self.stats["processing_time"] = self.stats["end_time"] - self.stats["start_time"]
        
        return self.stats
    
    def process_note(self, note: ObsidianNote) -> List[TextChunk]:
        """
        Process a single note.
        
        Args:
            note: The note to process
            
        Returns:
            List of chunks created from the note
        """
        # Create metadata for the note
        metadata = {
            "doc_id": note.path,
            "title": note.title,
            "tags": note.tags,
            "links": note.links,
            **note.metadata  # Include original YAML frontmatter
        }
        
        # Chunk the note
        chunks = self.chunker.chunk_text(note.content, metadata)
        
        # Create embeddings for chunks (process in batches to avoid overwhelming the API)
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.ollama.batch_get_embeddings(chunk_texts, batch_size=config.BATCH_SIZE)
        
        # Store chunks in vector database
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Update stats
        self.stats["processed_notes"] += 1
        self.stats["total_chunks"] += len(chunks)
        
        return chunks
    
    def query(self, query: str, limit: int = 5, filter_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the vault with a natural language query.
        
        Args:
            query: The query string
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            Query results with LLM analysis
        """
        # Get embedding for the query
        query_embedding = self.ollama.get_embeddings(query)
        
        # Search for relevant chunks
        chunks = self.vector_store.search(query_embedding, limit=limit, filter_conditions=filter_conditions)
        
        # Get context texts from chunks
        context_texts = [chunk["text"] for chunk in chunks]
        
        # Generate a response using the LLM
        response = self.llm.analyze_text(
            text=query,
            instruction="Answer the query based on the provided context. If the information needed is not in the context, indicate this clearly.",
            context=context_texts
        )
        
        # Return the results
        return {
            "query": query,
            "chunks": chunks,
            "response": response
        }