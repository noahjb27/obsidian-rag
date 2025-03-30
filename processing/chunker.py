"""
Semantic Chunking for Digital Humanities Texts

This module provides a semantic chunking approach that respects the 
natural structure of humanities texts, preserving context and relationships.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A semantic chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    # Store relationships to other chunks
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_id: Optional[str] = None


class SemanticChunker:
    """Chunks text based on semantic boundaries like sections and paragraphs.
    Better for long form writing."""
    
    def __init__(
        self, 
        target_chunk_size: int = 1500,
        chunk_overlap: int = 150,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2000
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            target_chunk_size: Target size of chunks in tokens (approximate)
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to maintain context
            max_chunk_size: Maximum chunk size (should be below model's context window)
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Common section header patterns in humanities texts
        self.section_patterns = [
            r"^#+\s+.+$",  # Markdown headers
            r"^[A-Z\s]+$",  # ALL CAPS HEADERS
            r"^\d+\.\s+.+$",  # Numbered sections
            r"^[IVX]+\.\s+.+$",  # Roman numeral sections
        ]
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        This is a simple approximation. For more accurate token counting,
        you might want to use the tokenizer from your LLM.
        """
        return len(text.split())
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header."""
        line = line.strip()
        if not line:
            return False
            
        # Check against common section header patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
                
        return False
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections based on headers."""
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if self._is_section_header(line) and current_section:
                # Found a new section header and current section has content
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
                
        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on blank lines."""
        # Split on double newlines (typical paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs and strip whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Create semantic chunks from a text document.
        
        Args:
            text: The text to chunk
            doc_metadata: Metadata about the document
            
        Returns:
            List of TextChunk objects
        """
        # Skip empty documents
        if not text or len(text.strip()) == 0:
            print(f"Warning: Skipping empty document: {doc_metadata.get('doc_id', 'unknown')}")
            return []
            
        # First split by sections
        sections = self._split_into_sections(text)
        chunks = []
        chunk_id_counter = 0
        
        for section_idx, section in enumerate(sections):
            # Skip empty sections
            if not section or len(section.strip()) == 0:
                continue
                
            # For short sections, use them as is
            if self._estimate_tokens(section) <= self.max_chunk_size:
                # Ensure section meets minimum size requirement
                if self._estimate_tokens(section) < self.min_chunk_size and len(sections) > 1:
                    # If this is a very small section and we have other sections,
                    # consider appending it to previous or next section
                    print(f"Warning: Section {section_idx} in {doc_metadata.get('doc_id', 'doc')} is below min size")
                    # Still process it if it's the only section
                
                chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                chunk_id_counter += 1
                
                # Create chunk metadata
                chunk_metadata = {
                    **doc_metadata,
                    "section_idx": section_idx,
                    "is_section_start": True,
                    "section_length": self._estimate_tokens(section)
                }
                
                # Add relationships to previous chunks
                prev_id = chunks[-1].chunk_id if chunks else None
                
                chunks.append(TextChunk(
                    text=section,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    prev_chunk_id=prev_id,
                ))
                
                # Update previous chunk's next_chunk_id
                if prev_id and chunks:
                    chunks[-2].next_chunk_id = chunk_id
                
                continue
            
            # For longer sections, split into paragraphs
            paragraphs = self._split_into_paragraphs(section)
            current_chunk_text = []
            current_chunk_size = 0
            is_section_start = True
            
            for i, paragraph in enumerate(paragraphs):
                para_size = self._estimate_tokens(paragraph)
                
                # If adding this paragraph exceeds the target size
                # and we already have content, create a new chunk
                if (current_chunk_size + para_size > self.target_chunk_size and 
                    current_chunk_size >= self.min_chunk_size):
                    
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                    chunk_id_counter += 1
                    
                    # Create chunk metadata
                    chunk_metadata = {
                        **doc_metadata,
                        "section_idx": section_idx,
                        "paragraph_range": f"{i-len(current_chunk_text)}-{i-1}",
                        "is_section_start": is_section_start,
                    }
                    
                    # Add relationships to previous chunks
                    prev_id = chunks[-1].chunk_id if chunks else None
                    
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=chunk_id,
                        prev_chunk_id=prev_id,
                    ))
                    
                    # Update previous chunk's next_chunk_id
                    if prev_id and chunks:
                        chunks[-2].next_chunk_id = chunk_id
                    
                    # Prepare for next chunk, with overlap
                    overlap_paragraphs = current_chunk_text[-1:]  # Simple overlap
                    current_chunk_text = overlap_paragraphs
                    current_chunk_size = self._estimate_tokens('\n\n'.join(overlap_paragraphs))
                    is_section_start = False
                
                # Add paragraph to current chunk
                current_chunk_text.append(paragraph)
                current_chunk_size += para_size
            
            # Add the last chunk from this section
            if current_chunk_text:
                chunk_text = '\n\n'.join(current_chunk_text)
                chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                chunk_id_counter += 1
                
                # Create chunk metadata
                chunk_metadata = {
                    **doc_metadata,
                    "section_idx": section_idx,
                    "paragraph_range": f"{len(paragraphs)-len(current_chunk_text)}-{len(paragraphs)-1}",
                    "is_section_start": is_section_start,
                    "is_section_end": True
                }
                
                # Add relationships to previous chunks
                prev_id = chunks[-1].chunk_id if chunks else None
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    prev_chunk_id=prev_id,
                ))
                
                # Update previous chunk's next_chunk_id
                if prev_id and chunks:
                    chunks[-2].next_chunk_id = chunk_id
        
        return chunks
