"""
Obsidian Vault Connector

This module provides functionality to connect to and extract content from an Obsidian vault,
respecting the structure and links between notes.
"""
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Set
from dataclasses import dataclass
import hashlib


@dataclass
class ObsidianNote:
    """Represents a note from an Obsidian vault."""
    path: str  # Full path to the note
    content: str  # Note content
    title: str  # Note title
    metadata: Dict[str, Any]  # YAML frontmatter
    links: List[str]  # Links to other notes
    tags: List[str]  # Tags in the note
    last_modified: float  # Last modified timestamp
    file_hash: str  # Hash of the file content for change detection
    

class ObsidianConnector:
    """Connector for reading and monitoring an Obsidian vault."""
    
    def __init__(self, vault_path: str):
        """
        Initialize the Obsidian connector.
        
        Args:
            vault_path: Path to the Obsidian vault
        """
        self.vault_path = Path(vault_path).expanduser().resolve()
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Obsidian vault not found at {vault_path}")
            
        # Store the processed notes to avoid re-processing unchanged files
        self.note_cache: Dict[str, ObsidianNote] = {}
        
        # Regular expressions for parsing
        self.yaml_pattern = re.compile(r'^---\n(.*?)\n---', re.DOTALL)
        self.tag_pattern = re.compile(r'#([a-zA-Z0-9_-]+)')
        self.link_pattern = re.compile(r'\[\[(.*?)(?:\|.*?)?\]\]')
        
    def _extract_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from note content."""
        metadata = {}
        yaml_match = self.yaml_pattern.search(content)
        
        if yaml_match:
            yaml_text = yaml_match.group(1)
            
            # Basic YAML parsing without external dependencies
            for line in yaml_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle lists (denoted by "- " at start of value)
                    if value.startswith('- '):
                        value = [item.strip()[2:] for item in value.split('\n') if item.strip().startswith('- ')]
                        
                    metadata[key] = value
                    
        return metadata
    
    def _extract_content_without_yaml(self, content: str) -> str:
        """Extract note content without the YAML frontmatter."""
        yaml_match = self.yaml_pattern.search(content)
        
        if yaml_match:
            # Return content after the YAML block
            return content[yaml_match.end():].strip()
        
        return content.strip()
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from note content."""
        return list(set(self.tag_pattern.findall(content)))
    
    def _extract_links(self, content: str) -> List[str]:
        """Extract links to other notes."""
        return list(set(self.link_pattern.findall(content)))
    
    def _get_note_title(self, file_path: Path, content: str) -> str:
        """
        Get the title of a note.
        
        This uses the following priority:
        1. title from YAML frontmatter
        2. First heading in the document
        3. Filename without extension
        """
        # Check YAML frontmatter
        yaml_match = self.yaml_pattern.search(content)
        if yaml_match:
            yaml_text = yaml_match.group(1)
            for line in yaml_text.strip().split('\n'):
                if line.strip().startswith('title:'):
                    title = line.split(':', 1)[1].strip()
                    if title:
                        return title
        
        # Check first heading
        content_without_yaml = self._extract_content_without_yaml(content)
        for line in content_without_yaml.split('\n'):
            if line.strip().startswith('#'):
                # Remove heading markers and get text
                title = line.strip().lstrip('#').strip()
                if title:
                    return title
        
        # Fall back to filename
        return file_path.stem
    
    def _calculate_file_hash(self, content: str) -> str:
        """Calculate a hash of file content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _process_note(self, file_path: Path) -> ObsidianNote:
        """Process a single note file into an ObsidianNote object."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        file_hash = self._calculate_file_hash(content)
        
        # Check if file is unchanged since last processing
        rel_path = str(file_path.relative_to(self.vault_path))
        if rel_path in self.note_cache and self.note_cache[rel_path].file_hash == file_hash:
            return self.note_cache[rel_path]
            
        # Process the note
        metadata = self._extract_yaml_frontmatter(content)
        content_without_yaml = self._extract_content_without_yaml(content)
        title = self._get_note_title(file_path, content)
        tags = self._extract_tags(content)
        links = self._extract_links(content)
        last_modified = file_path.stat().st_mtime
        
        note = ObsidianNote(
            path=rel_path,
            content=content_without_yaml,
            title=title,
            metadata=metadata,
            links=links,
            tags=tags,
            last_modified=last_modified,
            file_hash=file_hash
        )
        
        # Update cache
        self.note_cache[rel_path] = note
        
        return note
    
    def get_all_notes(self) -> List[ObsidianNote]:
        """
        Get all notes from the vault.
        
        Returns:
            List of ObsidianNote objects
        """
        notes = []
        
        for file_path in self.vault_path.glob('**/*.md'):
            # Skip files in hidden directories (start with .)
            parts = file_path.parts
            if any(part.startswith('.') for part in parts):
                continue
                
            try:
                note = self._process_note(file_path)
                notes.append(note)
            except Exception as e:
                print(f"Error processing note {file_path}: {str(e)}")
                
        return notes
    
    def get_notes_by_tag(self, tag: str) -> List[ObsidianNote]:
        """
        Get all notes with a specific tag.
        
        Args:
            tag: Tag to filter by (without the # symbol)
            
        Returns:
            List of ObsidianNote objects with the tag
        """
        all_notes = self.get_all_notes()
        return [note for note in all_notes if tag in note.tags]
    
    def get_notes_by_links(self, target_note_path: str) -> List[ObsidianNote]:
        """
        Get all notes that link to a specific note.
        
        Args:
            target_note_path: Path to the target note (relative to vault)
            
        Returns:
            List of ObsidianNote objects that link to the target
        """
        target_name = Path(target_note_path).stem
        all_notes = self.get_all_notes()
        return [note for note in all_notes if target_name in note.links]
    
    def monitor_changes(self, callback, interval_seconds: int = 60) -> None:
        """
        Monitor the vault for changes and call the callback when changes are detected.
        
        Args:
            callback: Function to call with list of changed notes
            interval_seconds: How often to check for changes
        """
        print(f"Monitoring Obsidian vault at {self.vault_path} for changes...")
        
        while True:
            changed_notes = []
            
            for file_path in self.vault_path.glob('**/*.md'):
                # Skip files in hidden directories
                parts = file_path.parts
                if any(part.startswith('.') for part in parts):
                    continue
                
                # Get last modified time
                last_modified = file_path.stat().st_mtime
                rel_path = str(file_path.relative_to(self.vault_path))
                
                # Check if file is new or modified
                if (rel_path not in self.note_cache or 
                    last_modified > self.note_cache[rel_path].last_modified):
                    
                    try:
                        note = self._process_note(file_path)
                        changed_notes.append(note)
                    except Exception as e:
                        print(f"Error processing changed note {file_path}: {str(e)}")
            
            # Check for deleted files
            cached_paths = set(self.note_cache.keys())
            current_paths = {str(f.relative_to(self.vault_path)) 
                            for f in self.vault_path.glob('**/*.md')
                            if not any(part.startswith('.') for part in f.parts)}
            
            deleted_paths = cached_paths - current_paths
            for path in deleted_paths:
                del self.note_cache[path]
            
            # Call callback if changes detected
            if changed_notes:
                callback(changed_notes)
            
            # Wait before checking again
            time.sleep(interval_seconds)
    
    def get_note_network(self) -> Dict[str, Set[str]]:
        """
        Build a network of note relationships based on links.
        
        Returns:
            Dictionary mapping note paths to sets of linked note paths
        """
        all_notes = self.get_all_notes()
        note_dict = {note.path: note for note in all_notes}
        
        # Build the network
        network = {}
        for path, note in note_dict.items():
            network[path] = set()
            
            for link in note.links:
                # Try to find the linked note
                for other_path, other_note in note_dict.items():
                    if (other_note.title == link or 
                        Path(other_path).stem == link):
                        network[path].add(other_path)
                        break
        
        return network
