# Digital Humanities Assistant

A local LLM-powered assistant for working with humanities texts stored in Obsidian vaults.

## Features

- **Obsidian Integration**: Connects to your Obsidian vault to process notes, respecting structure and relationships
- **Local LLM Processing**: Uses Ollama for local LLM inference and embeddings generation
- **Semantic Chunking**: Intelligently splits texts along semantic boundaries
- **Vector Search**: Enables natural language querying of your knowledge base
- **Privacy-Focused**: All processing happens locally, no data leaves your machine

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Required models pulled in Ollama:
  - `mistral:7b-instruct-v0.2` (or your choice of generation model)
  - `nomic-embed-text` (for embeddings)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/digital-humanities-assistant.git
   cd digital-humanities-assistant
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure your settings:
   - Open `config.py` and set your Obsidian vault path

## Usage

1. Start the application:
   ```bash
   python ui/app.py
   ```

2. In the UI:
   - Go to the "Initialize" tab and click "Initialize System" to process your vault
   - Switch to the "Query" tab to ask questions about your notes

## Project Structure

- `connectors/`: Integration with data sources (Obsidian)
- `models/`: LLM and embeddings integration
- `processing/`: Text processing and chunking
- `storage/`: Vector database integration
- `ui/`: User interface

## Dependencies

- `qdrant-client`: Vector database
- `gradio`: Web UI
- `requests`: API communication with Ollama

## Configuration

Key settings in `config.py`:
- `DEFAULT_VAULT_PATH`: Path to your Obsidian vault
- `GENERATION_MODEL`: Ollama model for text generation
- `EMBEDDING_MODEL`: Ollama model for embeddings
- `TARGET_CHUNK_SIZE`: Target size of text chunks in tokens