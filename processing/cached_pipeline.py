"""
Cached processing pipeline for digital humanities texts.

This extends the basic pipeline with caching capabilities to enable
incremental processing of documents.
"""
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
import time
import os
from pathlib import Path
import hashlib

from connectors.obsidian import ObsidianConnector, ObsidianNote
from processing.markdown_chunker import MarkdownChunker, TextChunk
from models.llm import RobustOllamaClient, RobustDigitalHumanitiesLLM
from storage.vector_store import VectorStore
from processing.debug_cache import DebugCache as ProcessingCache
import config


class CachedPipeline:
    """
    Processing pipeline for digital humanities texts with caching support.
    
    This pipeline only processes new or changed documents, improving
    performance for subsequent runs.
    """
    
    def __init__(
        self,
        vault_path: str = config.DEFAULT_VAULT_PATH,
        vector_store_path: str = config.VECTOR_STORE_PATH,
        cache_dir: str = None,
        ollama_base_url: str = config.OLLAMA_BASE_URL
    ):
        """
        Initialize the cached pipeline.
        
        Args:
            vault_path: Path to the Obsidian vault
            vector_store_path: Path to store vectors
            cache_dir: Directory to store cache files (defaults to vector_store_path/cache)
            ollama_base_url: URL for Ollama API
        """
        # Initialize components
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.obsidian = ObsidianConnector(str(self.vault_path))
        
        self.chunker = MarkdownChunker(
            target_chunk_size=config.TARGET_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            min_chunk_size=100,  # Reduced for markdown notes
            max_chunk_size=config.MAX_CHUNK_SIZE,
            combine_small_sections=True
        )
        
        self.ollama = RobustOllamaClient(
            base_url=ollama_base_url,
            generation_model=config.GENERATION_MODEL,
            embedding_model=config.EMBEDDING_MODEL,
            timeout=300  # 5 minute timeout
        )

        self.llm = RobustDigitalHumanitiesLLM(client=self.ollama)

        
        self.vector_store_path = Path(vector_store_path).expanduser().resolve()
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = VectorStore(
            path=str(self.vector_store_path),
            vector_size=config.VECTOR_DIMENSION
        )
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = os.path.join(str(self.vector_store_path), "cache")
        self.cache = ProcessingCache(cache_dir)
        
        # Update cache metadata with current configuration
        self.cache.update_metadata({
            "vault_path": str(self.vault_path),
            "config": {
                "target_chunk_size": config.TARGET_CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "min_chunk_size": 100,
                "max_chunk_size": config.MAX_CHUNK_SIZE,
                "generation_model": config.GENERATION_MODEL,
                "embedding_model": config.EMBEDDING_MODEL
            }
        })
        
        # Processing stats
        self.stats = {
            "processed_notes": 0,
            "new_notes": 0,
            "updated_notes": 0,
            "unchanged_notes": 0,
            "deleted_notes": 0,
            "total_chunks": 0,
            "start_time": None,
            "end_time": None
        }
    
    def process_vault(
        self, 
        callback: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process all notes in the vault, using cache for unchanged files.
        
        Args:
            callback: Optional callback function for progress updates
                    Signature: callback(note_title, current_idx, total_notes, stats)
            force_reprocess: If True, reprocess all files regardless of cache
            
        Returns:
            Processing statistics
        """
        self.stats["start_time"] = time.time()
        
        # Get all current notes in the vault
        all_notes = self.obsidian.get_all_notes()
        total_notes = len(all_notes)
        
        # Get all cached document paths
        cached_paths = self.cache.get_all_cached_paths()
        current_paths = {note.path for note in all_notes}
        
        # Identify deleted documents
        deleted_paths = cached_paths - current_paths
        self.stats["deleted_notes"] = len(deleted_paths)
        
        # Process each note
        for i, note in enumerate(all_notes):
            # Check if this document is in the cache with the same hash
            if not force_reprocess and self.cache.is_document_cached(note.path, note.file_hash):
                # Document is unchanged, skip processing
                self.stats["unchanged_notes"] += 1
                
                # Call progress callback if provided
                if callback:
                    callback(note.title, i + 1, total_notes, self.stats)
                    
                continue
            
            # Document is new or changed, process it
            if note.path in cached_paths:
                self.stats["updated_notes"] += 1
            else:
                self.stats["new_notes"] += 1
                
            # Process the note
            chunks, embeddings = self.process_note(note)
            
            # Update the vector store
            # First, remove any existing entries for this document
            # (This would require a method to delete by document ID in VectorStore)
            # For now, we simply add the new chunks
            
            # Store chunks in the vector store
            if chunks and embeddings:
                self.vector_store.add_chunks(chunks, embeddings)
                
                # Add to cache
                chunk_dicts = []
                for chunk in chunks:
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "text_length": len(chunk.text),
                        "metadata": chunk.metadata
                    }
                    chunk_dicts.append(chunk_dict)
                    
                self.cache.add_document(
                    note.path, 
                    note.file_hash,
                    chunk_dicts,
                    {
                        "title": note.title,
                        "tags": note.tags,
                        "links": note.links
                    }
                )
            
            # Update stats
            self.stats["processed_notes"] += 1
            self.stats["total_chunks"] += len(chunks) if chunks else 0
            
            # Call progress callback if provided
            if callback:
                callback(note.title, i + 1, total_notes, self.stats)
        
        # Handle deleted documents (would need a method to remove from vector store)
        for doc_path in deleted_paths:
            self.cache.remove_document(doc_path)
        
        # Save the cache
        self.cache.update_metadata({
            "processing_count": self.cache.metadata["processing_count"] + 1,
            "total_documents": len(current_paths),
            "total_chunks": self.stats["total_chunks"]
        })
        self.cache.save()
        
        # Update final stats
        self.stats["end_time"] = time.time()
        self.stats["processing_time"] = self.stats["end_time"] - self.stats["start_time"]
        self.stats["cached_document_count"] = len(current_paths)
        
        return self.stats
    
    def process_note(self, note: ObsidianNote) -> Tuple[List[TextChunk], List[List[float]]]:
        """
        Process a single note.
        
        Args:
            note: The note to process
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        # Create metadata for the note
        metadata = {
            "doc_id": note.path,
            "title": note.title,
            "tags": note.tags,
            "links": note.links,
            **{k: v for k, v in note.metadata.items() if isinstance(v, (str, int, float, bool, list))}
        }
        
        # Chunk the note
        try:
            chunks = self.chunker.chunk_text(note.content, metadata)
            if not chunks:
                print(f"Warning: No chunks created for note: {note.path}")
                return [], []
                
            # Create embeddings for chunks (process in batches to avoid overwhelming the API)
            chunk_texts = [chunk.text for chunk in chunks]
            
            try:
                embeddings = self.ollama.batch_get_embeddings(chunk_texts, batch_size=config.BATCH_SIZE)
                if len(embeddings) != len(chunks):
                    print(f"Warning: Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
                    return [], []
                    
                return chunks, embeddings
                
            except Exception as e:
                print(f"Error generating embeddings for note {note.path}: {str(e)}")
                return [], []
                
        except Exception as e:
            print(f"Error chunking note {note.path}: {str(e)}")
            return [], []
    
    def query(
        self, 
        query: str, 
        limit: int = 8,  # Increased from 5
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vault with a natural language query.
        
        Args:
            query: The query string
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            Query results with LLM analysis
        """
        try:
            # Get embedding for the query
            query_embedding = self.ollama.get_embeddings(query)
            
            # Search for relevant chunks
            chunks = self.vector_store.search(query_embedding, limit=limit, filter_conditions=filter_conditions)
            
            if not chunks:
                return {
                    "query": query,
                    "chunks": [],
                    "response": "I couldn't find any relevant information in your vault to answer this question."
                }
            
            # Get context texts from chunks
            context_texts = [chunk["text"] for chunk in chunks]
            
            # Generate a response using the LLM
            instruction = "Answer the query based on the provided context. If the information needed is not in the context, indicate this clearly."
            
            response = self.llm.analyze_text(
                text=query,
                instruction=instruction,
                context=context_texts
            )
            
            # Return the results
            return {
                "query": query,
                "chunks": chunks,
                "response": response
            }
        except Exception as e:
            import logging
            logging.error(f"Error in query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "chunks": [],
                "response": f"I encountered an error while processing your query: {str(e)}. Please try rephrasing or simplifying your question."
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = self.cache.get_stats()
        stats["cached_document_count"] = stats["document_count"]
        return stats
    
    def clear_cache(self) -> None:
        """Clear the cache and force reprocessing of all documents."""
        self.cache.clear()