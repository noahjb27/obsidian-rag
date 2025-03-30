"""
Enhanced Markdown-aware Semantic Chunking

This module provides an improved chunking approach that's optimized for
markdown-formatted notes, especially those with lists, short sections,
and other common markdown structures.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
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


class MarkdownChunker:
    """
    Chunks markdown text with awareness of markdown structures like lists,
    headers, and code blocks.
    """
    
    def __init__(
        self, 
        target_chunk_size: int = 1500,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,  # Lower minimum for markdown notes
        max_chunk_size: int = 2000,
        combine_small_sections: bool = True
    ):
        """
        Initialize the markdown chunker.
        
        Args:
            target_chunk_size: Target size of chunks in tokens (approximate)
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to maintain context
            max_chunk_size: Maximum chunk size
            combine_small_sections: Whether to combine small adjacent sections
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.combine_small_sections = combine_small_sections
        
        # Markdown patterns
        self.header_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
        self.list_item_pattern = re.compile(r'^(\s*)([*+-]|\d+\.)\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```.*?```', re.DOTALL)
        self.table_pattern = re.compile(r'\|.*\|.*\n\|[-:| ]+\|.*(\n\|.*\|.*)*', re.MULTILINE)
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        A slightly more sophisticated approach that handles markdown better.
        """
        # Roughly 1.3 tokens per word for markdown with formatting
        words = len(re.findall(r'\S+', text))
        formatting_chars = len(re.findall(r'[*_`#>|-]', text))
        return int(words * 1.3) + int(formatting_chars * 0.5)
    
    def _split_by_headers(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Split text by headers, returning header level, header text, and content.
        
        Returns:
            List of tuples (header_level, header_text, content)
        """
        # Find all headers and their positions
        headers = [(len(m.group(1)), m.group(2), m.start()) 
                   for m in self.header_pattern.finditer(text)]
        
        # Add a sentinel for the end of the document
        headers.append((0, "", len(text)))
        
        # Create sections based on headers
        sections = []
        for i in range(len(headers) - 1):
            level, title, start = headers[i]
            next_start = headers[i + 1][2]
            
            # Find the end of the header line
            header_end = text.find('\n', start)
            if header_end == -1:
                header_end = len(text)
            
            # Extract content (without the header itself)
            content = text[header_end:next_start].strip()
            
            sections.append((level, title, content))
            
        # If the document doesn't start with a header, add the content before the first header
        if headers[0][2] > 0:
            first_header_pos = headers[0][2]
            intro_content = text[:first_header_pos].strip()
            if intro_content:
                sections.insert(0, (0, "", intro_content))
                
        return sections
    
    def _is_list_continuation(self, text: str) -> bool:
        """Check if text is likely part of a markdown list."""
        list_markers = ['-', '*', '+', '1.', '2.', '3.']
        indented_line = text.startswith('    ') or text.startswith('\t')
        starts_with_marker = any(text.lstrip().startswith(marker + ' ') for marker in list_markers)
        return indented_line or starts_with_marker
    
    def _extract_markdown_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract blocks of related markdown content, preserving structures.
        
        Returns:
            List of tuples (block_type, content)
        """
        blocks = []
        
        # Extract code blocks first (to protect them from further splitting)
        code_blocks = self.code_block_pattern.finditer(text)
        positions = []
        for m in code_blocks:
            positions.append((m.start(), m.end(), 'code_block', m.group(0)))
        
        # Extract tables
        tables = self.table_pattern.finditer(text)
        for m in tables:
            positions.append((m.start(), m.end(), 'table', m.group(0)))
        
        # Sort by position
        positions.sort()
        
        # Process the text with protected regions
        last_end = 0
        for start, end, block_type, content in positions:
            # Add text before this block
            if start > last_end:
                pre_text = text[last_end:start].strip()
                if pre_text:
                    blocks.append(('text', pre_text))
            
            # Add the protected block
            blocks.append((block_type, content))
            last_end = end
        
        # Add any remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                blocks.append(('text', remaining))
                
        # Further process text blocks to extract lists and paragraphs
        processed_blocks = []
        for block_type, content in blocks:
            if block_type == 'text':
                # Split text into paragraphs and list items
                lines = content.split('\n')
                current_block = []
                current_type = 'paragraph'
                
                for line in lines:
                    # Check if this line is part of a list
                    is_list_item = bool(self.list_item_pattern.match(line))
                    is_continuation = self._is_list_continuation(line)
                    
                    if (is_list_item or is_continuation) and current_type != 'list':
                        # Starting a new list
                        if current_block:
                            processed_blocks.append(('paragraph', '\n'.join(current_block)))
                            current_block = []
                        current_type = 'list'
                        current_block.append(line)
                    elif not (is_list_item or is_continuation) and current_type == 'list':
                        # Ending a list
                        if current_block:
                            processed_blocks.append(('list', '\n'.join(current_block)))
                            current_block = []
                        current_type = 'paragraph'
                        if line.strip():
                            current_block.append(line)
                    else:
                        # Continuing current block type
                        if line.strip() or current_block:  # Don't add empty lines to empty blocks
                            current_block.append(line)
                
                # Add the last block
                if current_block:
                    processed_blocks.append((current_type, '\n'.join(current_block)))
            else:
                # Keep non-text blocks as they are
                processed_blocks.append((block_type, content))
                
        return processed_blocks
    
    def _maybe_combine_small_blocks(self, blocks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Optionally combine small adjacent blocks of the same type.
        """
        if not self.combine_small_sections or len(blocks) <= 1:
            return blocks
            
        combined = []
        current_type = blocks[0][0]
        current_content = blocks[0][1]
        
        for i in range(1, len(blocks)):
            block_type, content = blocks[i]
            
            # If same type and combined size is within limits, combine
            if (block_type == current_type and 
                self._estimate_tokens(current_content) + self._estimate_tokens(content) <= self.target_chunk_size):
                if current_type in ['paragraph', 'text']:
                    current_content += '\n\n' + content
                else:
                    current_content += '\n' + content
            else:
                # Add current combined block and start a new one
                combined.append((current_type, current_content))
                current_type = block_type
                current_content = content
                
        # Add the last block
        combined.append((current_type, current_content))
        return combined
    
    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Create semantic chunks from a markdown document.
        
        Args:
            text: The markdown text to chunk
            doc_metadata: Metadata about the document
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
            
        # First split by headers
        sections = self._split_by_headers(text)
        chunks = []
        chunk_id_counter = 0
        
        for section_idx, (level, title, section_content) in enumerate(sections):
            if not section_content.strip():
                continue
                
            section_size = self._estimate_tokens(section_content)
            section_metadata = {
                **doc_metadata,
                "section_idx": section_idx,
                "section_level": level,
                "section_title": title
            }
            
            # For small sections, keep them intact
            if section_size <= self.max_chunk_size:
                chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                chunk_id_counter += 1
                
                # Create chunk metadata
                chunk_metadata = {
                    **section_metadata,
                    "is_section_start": True,
                    "is_section_end": True,
                    "section_length": section_size
                }
                
                # Add relationships to previous chunks
                prev_id = chunks[-1].chunk_id if chunks else None
                
                chunks.append(TextChunk(
                    text=section_content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    prev_chunk_id=prev_id,
                ))
                
                # Update previous chunk's next_chunk_id
                if prev_id and chunks:
                    chunks[-2].next_chunk_id = chunk_id
                
                continue
                
            # For larger sections, process block by block
            blocks = self._extract_markdown_blocks(section_content)
            
            # Maybe combine small adjacent blocks
            if self.combine_small_sections:
                blocks = self._maybe_combine_small_blocks(blocks)
                
            # Process blocks into chunks
            current_chunk_blocks = []
            current_chunk_size = 0
            is_section_start = True
            
            for block_idx, (block_type, block_content) in enumerate(blocks):
                block_size = self._estimate_tokens(block_content)
                
                # Skip empty blocks
                if not block_content.strip():
                    continue
                    
                # If adding this block exceeds the target size and we have content,
                # finalize the current chunk
                if (current_chunk_size + block_size > self.target_chunk_size and 
                    current_chunk_size >= self.min_chunk_size):
                    
                    chunk_text = '\n\n'.join(current_chunk_blocks)
                    chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                    chunk_id_counter += 1
                    
                    # Create chunk metadata
                    chunk_metadata = {
                        **section_metadata,
                        "block_range": f"{block_idx-len(current_chunk_blocks)}-{block_idx-1}",
                        "is_section_start": is_section_start,
                        "is_section_end": False
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
                    
                    # Reset for next chunk
                    current_chunk_blocks = []
                    current_chunk_size = 0
                    is_section_start = False
                
                # For very large individual blocks that exceed max size,
                # we need to split them (preserving structure where possible)
                if block_size > self.max_chunk_size:
                    # Special handling based on block type
                    if block_type == 'list':
                        # Split list by items but keep related items together
                        list_items = re.split(r'\n(?=\s*[*+-]|\s*\d+\.)', block_content)
                        sub_list = []
                        sub_size = 0
                        
                        for item in list_items:
                            item_size = self._estimate_tokens(item)
                            if sub_size + item_size > self.target_chunk_size and sub_list:
                                # Create a chunk from current sub-list
                                sub_text = '\n'.join(sub_list)
                                chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                                chunk_id_counter += 1
                                
                                chunk_metadata = {
                                    **section_metadata,
                                    "content_type": "list",
                                    "is_section_start": is_section_start,
                                    "is_section_end": False
                                }
                                
                                prev_id = chunks[-1].chunk_id if chunks else None
                                
                                chunks.append(TextChunk(
                                    text=sub_text,
                                    metadata=chunk_metadata,
                                    chunk_id=chunk_id,
                                    prev_chunk_id=prev_id,
                                ))
                                
                                if prev_id and chunks:
                                    chunks[-2].next_chunk_id = chunk_id
                                
                                # Reset
                                sub_list = [item]
                                sub_size = item_size
                                is_section_start = False
                            else:
                                sub_list.append(item)
                                sub_size += item_size
                        
                        # Add remaining items
                        if sub_list:
                            sub_text = '\n'.join(sub_list)
                            current_chunk_blocks.append(sub_text)
                            current_chunk_size += self._estimate_tokens(sub_text)
                    
                    elif block_type in ['code_block', 'table']:
                        # These should be kept intact if possible, but if too large,
                        # we'll split with a warning
                        print(f"Warning: Large {block_type} ({block_size} tokens) in {doc_metadata.get('doc_id', 'unknown')}")
                        
                        # Still add it as a single chunk to avoid breaking structure
                        chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                        chunk_id_counter += 1
                        
                        chunk_metadata = {
                            **section_metadata,
                            "content_type": block_type,
                            "is_section_start": is_section_start,
                            "is_section_end": False,
                            "warning": f"Oversized {block_type}"
                        }
                        
                        prev_id = chunks[-1].chunk_id if chunks else None
                        
                        chunks.append(TextChunk(
                            text=block_content,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                            prev_chunk_id=prev_id,
                        ))
                        
                        if prev_id and chunks:
                            chunks[-2].next_chunk_id = chunk_id
                        
                        is_section_start = False
                    
                    else:  # paragraph or text
                        # Split by sentences for paragraphs
                        sentences = re.split(r'(?<=[.!?])\s+', block_content)
                        sub_text = ''
                        sub_size = 0
                        
                        for sentence in sentences:
                            sentence_size = self._estimate_tokens(sentence)
                            if sub_size + sentence_size > self.target_chunk_size and sub_text:
                                # Finalize current sub-chunk
                                current_chunk_blocks.append(sub_text)
                                current_chunk_size += sub_size
                                
                                # If overall chunk is now too large, create a chunk
                                if current_chunk_size >= self.target_chunk_size:
                                    chunk_text = '\n\n'.join(current_chunk_blocks)
                                    chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                                    chunk_id_counter += 1
                                    
                                    chunk_metadata = {
                                        **section_metadata,
                                        "is_section_start": is_section_start,
                                        "is_section_end": False
                                    }
                                    
                                    prev_id = chunks[-1].chunk_id if chunks else None
                                    
                                    chunks.append(TextChunk(
                                        text=chunk_text,
                                        metadata=chunk_metadata,
                                        chunk_id=chunk_id,
                                        prev_chunk_id=prev_id,
                                    ))
                                    
                                    if prev_id and chunks:
                                        chunks[-2].next_chunk_id = chunk_id
                                    
                                    current_chunk_blocks = []
                                    current_chunk_size = 0
                                    is_section_start = False
                                
                                sub_text = sentence
                                sub_size = sentence_size
                            else:
                                if sub_text:
                                    sub_text += ' ' + sentence
                                else:
                                    sub_text = sentence
                                sub_size += sentence_size
                        
                        # Add remaining content
                        if sub_text:
                            current_chunk_blocks.append(sub_text)
                            current_chunk_size += sub_size
                
                else:  # Block fits within maximum size
                    current_chunk_blocks.append(block_content)
                    current_chunk_size += block_size
            
            # Create final chunk for this section if there's content left
            if current_chunk_blocks:
                chunk_text = '\n\n'.join(current_chunk_blocks)
                chunk_id = f"{doc_metadata.get('doc_id', 'doc')}_{chunk_id_counter}"
                chunk_id_counter += 1
                
                chunk_metadata = {
                    **section_metadata,
                    "is_section_start": is_section_start,
                    "is_section_end": True
                }
                
                prev_id = chunks[-1].chunk_id if chunks else None
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    prev_chunk_id=prev_id,
                ))
                
                if prev_id and chunks:
                    chunks[-2].next_chunk_id = chunk_id
        
        return chunks