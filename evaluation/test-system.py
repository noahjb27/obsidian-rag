#!/usr/bin/env python3
"""
Comprehensive test script for Digital Humanities Assistant.
Tests each component in sequence and reports results.
"""
import os
import sys
import time
from pathlib import Path
import requests
import json
from colorama import init, Fore, Style  # For colored terminal output

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project components
import config
from connectors.obsidian import ObsidianConnector
from processing.chunker import SemanticChunker
from models.llm import OllamaClient
from storage.vector_store import VectorStore

# Initialize colorama for colored terminal output
init()

def print_header(message):
    """Print a highlighted header message."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f" {message}")
    print(f"{'='*80}{Style.RESET_ALL}")

def print_success(message):
    """Print a success message."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print an error message."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Fore.YELLOW}! {message}{Style.RESET_ALL}")

def print_info(message):
    """Print an info message."""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def test_obsidian_access():
    """Test access to Obsidian vault."""
    print_header("TESTING OBSIDIAN VAULT ACCESS")
    
    # Get vault path from config
    vault_path = Path(config.DEFAULT_VAULT_PATH).expanduser().resolve()
    print_info(f"Obsidian vault path: {vault_path}")
    
    # Check if path exists
    if not vault_path.exists():
        print_error(f"Vault directory not found at {vault_path}")
        return False
    
    print_success(f"Vault directory exists at {vault_path}")
    
    # Check if there are markdown files
    md_files = list(vault_path.glob('**/*.md'))
    if not md_files:
        print_error(f"No markdown files found in {vault_path}")
        return False
    
    print_success(f"Found {len(md_files)} markdown files")
    
    # Try to initialize ObsidianConnector
    try:
        connector = ObsidianConnector(str(vault_path))
        note_count = len(connector.get_all_notes())
        print_success(f"Successfully processed {note_count} notes with ObsidianConnector")
        
        # Show sample note
        if note_count > 0:
            sample_note = connector.get_all_notes()[0]
            print_info(f"Sample note: '{sample_note.title}' with {len(sample_note.content)} characters")
            print_info(f"Tags: {sample_note.tags}")
            return connector, note_count
        
    except Exception as e:
        print_error(f"Error initializing ObsidianConnector: {str(e)}")
        return False
    
    return False

def test_chunking(connector, max_notes=3):
    """Test semantic chunking on sample notes."""
    print_header("TESTING SEMANTIC CHUNKING")
    
    chunker = SemanticChunker(
        target_chunk_size=config.TARGET_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        min_chunk_size=config.MIN_CHUNK_SIZE,
        max_chunk_size=config.MAX_CHUNK_SIZE
    )
    
    notes = connector.get_all_notes()[:max_notes]  # Test with a few notes
    
    chunk_results = []
    total_chunks = 0
    
    for note in notes:
        print_info(f"Chunking note: {note.title}")
        
        metadata = {
            "doc_id": note.path,
            "title": note.title,
            "tags": note.tags,
            "links": note.links
        }
        
        try:
            chunks = chunker.chunk_text(note.content, metadata)
            print_success(f"Created {len(chunks)} chunks")
            total_chunks += len(chunks)
            
            # Sample chunk info
            if chunks:
                sample_chunk = chunks[0]
                print_info(f"Sample chunk: {len(sample_chunk.text)} chars, ID: {sample_chunk.chunk_id}")
                chunk_results.append(chunks)
                
        except Exception as e:
            print_error(f"Error chunking note: {str(e)}")
    
    print_success(f"Successfully chunked {len(notes)} notes into {total_chunks} total chunks")
    return chunk_results if chunk_results else False

def test_ollama_connection():
    """Test connection to Ollama and model availability."""
    print_header("TESTING OLLAMA CONNECTION")
    
    print_info(f"Ollama base URL: {config.OLLAMA_BASE_URL}")
    
    # Test basic connection
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        print_success(f"Connected to Ollama server at {config.OLLAMA_BASE_URL}")
        print_info(f"Available models: {', '.join(model_names)}")
        
        # Check for required models
        generation_model = config.GENERATION_MODEL.split(':')[0]
        embedding_model = config.EMBEDDING_MODEL.split(':')[0]
        
        if any(m.startswith(generation_model) for m in model_names):
            print_success(f"Generation model '{generation_model}' is available")
        else:
            print_warning(f"Generation model '{generation_model}' not found, may need to be pulled")
            
        if any(m.startswith(embedding_model) for m in model_names):
            print_success(f"Embedding model '{embedding_model}' is available")
        else:
            print_warning(f"Embedding model '{embedding_model}' not found, may need to be pulled")
        
        # Initialize Ollama client
        try:
            client = OllamaClient(
                base_url=config.OLLAMA_BASE_URL,
                generation_model=config.GENERATION_MODEL,
                embedding_model=config.EMBEDDING_MODEL
            )
            print_success("Successfully initialized OllamaClient")
            
            # Test simple generation
            try:
                start_time = time.time()
                response = client.generate("Hello, can you hear me?", params={"temperature": 0.1})
                end_time = time.time()
                
                print_success(f"Successfully generated text ({end_time - start_time:.2f}s)")
                print_info(f"Sample response: {response[:100]}...")
                
                return client
                
            except Exception as e:
                print_error(f"Error generating text: {str(e)}")
                
        except Exception as e:
            print_error(f"Error initializing OllamaClient: {str(e)}")
        
    except requests.exceptions.ConnectionError:
        print_error(f"Could not connect to Ollama at {config.OLLAMA_BASE_URL}")
        print_info("Ensure Ollama is running and accessible")
    except requests.exceptions.RequestException as e:
        print_error(f"Error connecting to Ollama: {str(e)}")
    
    return False

def test_embeddings(client, chunks):
    """Test embedding generation with chunk samples."""
    print_header("TESTING EMBEDDINGS GENERATION")
    
    if not client or not chunks:
        print_error("Skipping embedding test due to previous failures")
        return False
    
    # Select a sample of chunks to embed
    sample_chunks = chunks[0][:2]  # First 2 chunks from first note
    
    try:
        chunk_texts = [chunk.text for chunk in sample_chunks]
        
        print_info(f"Generating embeddings for {len(chunk_texts)} chunks")
        start_time = time.time()
        embeddings = client.batch_get_embeddings(chunk_texts, batch_size=1)
        end_time = time.time()
        
        print_success(f"Generated {len(embeddings)} embeddings ({end_time - start_time:.2f}s)")
        print_info(f"Embedding dimension: {len(embeddings[0])}")
        
        return embeddings
        
    except Exception as e:
        print_error(f"Error generating embeddings: {str(e)}")
    
    return False

def test_vector_store(chunks, embeddings):
    """Test vector store creation and search."""
    print_header("TESTING VECTOR STORE")
    
    if not chunks or not embeddings:
        print_error("Skipping vector store test due to previous failures")
        return False
    
    # Create a temporary vector store path for testing
    temp_store_path = os.path.join(project_root, "temp_test_vector_store")
    print_info(f"Creating temporary vector store at {temp_store_path}")
    
    try:
        # Initialize vector store
        vector_store = VectorStore(
            path=temp_store_path,
            vector_size=config.VECTOR_DIMENSION
        )
        print_success(f"Created vector store")
        
        # Add sample chunks
        sample_chunks = chunks[0][:2]  # First 2 chunks from first note
        vector_store.add_chunks(sample_chunks, embeddings)
        print_success(f"Added {len(sample_chunks)} chunks to vector store")
        
        # Test search
        test_query = embeddings[0]  # Use first embedding as test query
        results = vector_store.search(test_query, limit=5)
        print_success(f"Successfully searched vector store")
        print_info(f"Found {len(results)} results")
        
        # Clean up test store
        import shutil
        shutil.rmtree(temp_store_path, ignore_errors=True)
        print_info(f"Cleaned up temporary vector store")
        
        return True
        
    except Exception as e:
        print_error(f"Error testing vector store: {str(e)}")
        # Try to clean up
        import shutil
        shutil.rmtree(temp_store_path, ignore_errors=True)
    
    return False

def run_all_tests():
    """Run all tests in sequence."""
    print_header("RUNNING DIGITAL HUMANITIES ASSISTANT SYSTEM TESTS")
    print_info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Project root: {project_root}")
    
    # Test components in sequence
    connector_result = test_obsidian_access()
    
    if connector_result:
        connector, note_count = connector_result
        chunks = test_chunking(connector)
    else:
        print_error("Skipping chunking test due to Obsidian access failure")
        chunks = False
    
    client = test_ollama_connection()
    
    if client and chunks:
        embeddings = test_embeddings(client, chunks)
    else:
        embeddings = False
    
    if chunks and embeddings:
        vector_store_result = test_vector_store(chunks, embeddings)
    else:
        vector_store_result = False
    
    # Print summary
    print_header("TEST RESULTS SUMMARY")
    
    if connector_result:
        print_success("✓ Obsidian access: PASSED")
    else:
        print_error("✗ Obsidian access: FAILED")
        
    if chunks:
        print_success("✓ Semantic chunking: PASSED")
    else:
        print_error("✗ Semantic chunking: FAILED")
        
    if client:
        print_success("✓ Ollama connection: PASSED")
    else:
        print_error("✗ Ollama connection: FAILED")
        
    if embeddings:
        print_success("✓ Embeddings generation: PASSED")
    else:
        print_error("✗ Embeddings generation: FAILED")
        
    if vector_store_result:
        print_success("✓ Vector store: PASSED")
    else:
        print_error("✗ Vector store: FAILED")
    
    # Overall result
    all_passed = all([connector_result, chunks, client, embeddings, vector_store_result])
    
    if all_passed:
        print_success("\nALL TESTS PASSED - System ready for use")
    else:
        print_error("\nSOME TESTS FAILED - Review issues before using system")

if __name__ == "__main__":
    run_all_tests()