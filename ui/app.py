"""
Simple Gradio UI for the Digital Humanities application.
"""
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio not installed. Install with 'pip install gradio'")

from processing.pipeline import Pipeline
import config


class DigitalHumanitiesUI:
    """Gradio UI for the Digital Humanities application."""
    
    def __init__(self):
        """Initialize the UI."""
        self.pipeline = None
        self.is_initialized = False
        self.initialization_message = "System not initialized. Click 'Initialize System' to start."
    
    def initialize_system(self, vault_path: str) -> str:
        """
        Initialize the pipeline and process the vault.
        
        Args:
            vault_path: Path to the Obsidian vault
            
        Returns:
            Initialization status message
        """
        try:
            start_time = time.time()
            
            # Initialize pipeline
            self.pipeline = Pipeline(vault_path=vault_path)
            
            # Process the vault (this might take a while)
            stats = self.pipeline.process_vault()
            
            # Format the results
            end_time = time.time()
            total_time = end_time - start_time
            
            self.is_initialized = True
            
            return (f"System initialized successfully!\n"
                   f"Processed {stats['processed_notes']} notes into {stats['total_chunks']} chunks.\n"
                   f"Total processing time: {total_time:.2f} seconds.")
            
        except Exception as e:
            return f"Error initializing system: {str(e)}"
    
    def query_system(self, query: str, tag_filter: str = "") -> str:
        """
        Query the system with a natural language query.
        
        Args:
            query: The query string
            tag_filter: Optional tag to filter results
            
        Returns:
            Query response
        """
        if not self.is_initialized or not self.pipeline:
            return self.initialization_message
            
        try:
            # Apply tag filter if provided
            filter_conditions = None
            if tag_filter:
                filter_conditions = {"tags": tag_filter}
                
            # Query the system
            result = self.pipeline.query(query, filter_conditions=filter_conditions)
            
            # Format the response
            response = f"Query: {query}\n\n"
            response += f"Response:\n{result['response']}\n\n"
            
            response += "Sources:\n"
            for i, chunk in enumerate(result['chunks'], 1):
                response += f"{i}. {chunk.get('title', 'Unnamed')} (Score: {chunk.get('score', 0):.2f})\n"
                
            return response
            
        except Exception as e:
            return f"Error querying system: {str(e)}"
    
    def create_ui(self) -> gr.Blocks:
        """
        Create the Gradio UI.
        
        Returns:
            Gradio Blocks instance
        """
        with gr.Blocks(title="Digital Humanities Assistant") as ui:
            gr.Markdown("# Digital Humanities Assistant")
            gr.Markdown("Query your Obsidian vault with natural language")
            
            with gr.Tab("Initialize"):
                vault_path_input = gr.Textbox(
                    value=config.DEFAULT_VAULT_PATH,
                    label="Obsidian Vault Path"
                )
                init_button = gr.Button("Initialize System")
                init_output = gr.Textbox(
                    value=self.initialization_message,
                    label="Initialization Status",
                    lines=5
                )
                
                init_button.click(
                    fn=self.initialize_system,
                    inputs=[vault_path_input],
                    outputs=[init_output]
                )
            
            with gr.Tab("Query"):
                query_input = gr.Textbox(
                    placeholder="Enter your query here...",
                    label="Query"
                )
                tag_filter = gr.Textbox(
                    placeholder="Optional: Filter by tag",
                    label="Tag Filter"
                )
                query_button = gr.Button("Submit Query")
                query_output = gr.Textbox(
                    label="Response",
                    lines=10
                )
                
                query_button.click(
                    fn=self.query_system,
                    inputs=[query_input, tag_filter],
                    outputs=[query_output]
                )
            
        return ui
    
    def launch(self, share: bool = False) -> None:
        """
        Launch the UI.
        
        Args:
            share: Whether to create a public link
        """
        ui = self.create_ui()
        ui.launch(share=share)


if __name__ == "__main__":
    app = DigitalHumanitiesUI()
    app.launch()