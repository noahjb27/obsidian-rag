"""
Configuration settings for the Digital Humanities application.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT", "/mnt/c/users/noahb/PhD")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "mistral:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Vector store settings
VECTOR_STORE_PATH = os.path.join(DEFAULT_DATA_DIR, "vector_store")
VECTOR_DIMENSION = 768  # For nomic-embed-text

# Chunking settings
TARGET_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000

# Processing settings
BATCH_SIZE = 10  # Number of documents to process in a batch
