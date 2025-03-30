"""
Cache management for document processing state.

This module provides functionality to track processed files
and their state between application runs.
"""
import os
import json
import time
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import hashlib


class ProcessingCache:
    """Manages caching of processed documents to enable incremental processing."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize the processing cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache state
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.document_cache_file = self.cache_dir / "document_cache.json"
        
        # Load cache if exists
        self.metadata = self._load_metadata()
        self.document_cache = self._load_document_cache()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache metadata: {str(e)}")
                return self._create_default_metadata()
        else:
            return self._create_default_metadata()
    
    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default metadata structure."""
        return {
            "creation_time": time.time(),
            "last_update": time.time(),
            "processing_count": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "vault_path": "",
            "config": {}
        }
    
    def _load_document_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load document cache from file."""
        if self.document_cache_file.exists():
            try:
                with open(self.document_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load document cache: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _save_document_cache(self) -> None:
        """Save document cache to file."""
        with open(self.document_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_cache, f, indent=2)
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Update cache metadata.
        
        Args:
            updates: Dictionary of metadata fields to update
        """
        self.metadata.update(updates)
        self.metadata["last_update"] = time.time()
        self._save_metadata()
    
    def is_document_cached(self, doc_path: str, file_hash: str) -> bool:
        """
        Check if a document is in the cache with the same hash.
        
        Args:
            doc_path: Path to the document (relative to vault)
            file_hash: Hash of the document content
            
        Returns:
            True if the document is cached and unchanged
        """
        if doc_path in self.document_cache:
            return self.document_cache[doc_path].get("file_hash") == file_hash
        return False
    
    def get_document_chunks(self, doc_path: str) -> List[Dict[str, Any]]:
        """
        Get cached chunk information for a document.
        
        Args:
            doc_path: Path to the document (relative to vault)
            
        Returns:
            List of chunk metadata dictionaries
        """
        if doc_path in self.document_cache:
            return self.document_cache[doc_path].get("chunks", [])
        return []
    
    def add_document(
        self, 
        doc_path: str, 
        file_hash: str, 
        chunks: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add or update a document in the cache.
        
        Args:
            doc_path: Path to the document (relative to vault)
            file_hash: Hash of the document content
            chunks: List of chunk information
            metadata: Optional additional document metadata
        """
        self.document_cache[doc_path] = {
            "file_hash": file_hash,
            "last_processed": time.time(),
            "chunks": chunks,
            "metadata": metadata or {}
        }
        
        # Update periodically to avoid losing all work if interrupted
        if len(self.document_cache) % 10 == 0:
            self._save_document_cache()
    
    def remove_document(self, doc_path: str) -> None:
        """
        Remove a document from the cache.
        
        Args:
            doc_path: Path to the document (relative to vault)
        """
        if doc_path in self.document_cache:
            del self.document_cache[doc_path]
            # No need to save immediately, will be saved on next periodic save
    
    def get_all_cached_paths(self) -> Set[str]:
        """
        Get all document paths in the cache.
        
        Returns:
            Set of document paths
        """
        return set(self.document_cache.keys())
    
    def save(self) -> None:
        """Save all cache data to disk."""
        self._save_metadata()
        self._save_document_cache()
    
    def clear(self) -> None:
        """Clear all cache data."""
        self.document_cache = {}
        self.metadata = self._create_default_metadata()
        self.save()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        doc_count = len(self.document_cache)
        chunk_count = sum(len(doc.get("chunks", [])) for doc in self.document_cache.values())
        
        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "creation_time": self.metadata["creation_time"],
            "last_update": self.metadata["last_update"],
            "processing_count": self.metadata["processing_count"]
        }
    
    def get_document_info(self, doc_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached document.
        
        Args:
            doc_path: Path to the document (relative to vault)
        
        Returns:
            Document information or None if not cached
        """
        return self.document_cache.get(doc_path)