"""
Gradio UI for the Digital Humanities application with caching support.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio not installed. Install with 'pip install gradio'")

from processing.cached_pipeline import CachedPipeline
import config


class CachedDigitalHumanitiesUI:
    """Gradio UI for the Digital Humanities application with caching support."""
    
    def __init__(self):
        """Initialize the UI."""
        self.pipeline = None
        self.is_initialized = False
        self.initialization_message = "System not initialized. Click 'Initialize System' to start."
        self.processing_logs = []
    
    def initialize_system(
        self, 
        vault_path: str, 
        force_reprocess: bool
    ) -> tuple:
        """
        Initialize the pipeline and process the vault.
        
        Args:
            vault_path: Path to the Obsidian vault
            force_reprocess: Whether to force reprocessing of all files
            
        Returns:
            Tuple of (status_message, progress_html, log_text)
        """
        try:
            self.processing_logs = []
            start_time = time.time()
            
            # Initialize pipeline if needed
            if self.pipeline is None:
                self.pipeline = CachedPipeline(vault_path=vault_path)
                cache_stats = self.pipeline.get_cache_stats()
                self.add_log(f"Cache initialized. Found {cache_stats['document_count']} cached documents with {cache_stats['chunk_count']} chunks.")
            
            # Process the vault with progress updates
            self.add_log(f"Starting vault processing at {datetime.now().strftime('%H:%M:%S')}...")
            self.add_log(f"Force reprocessing: {force_reprocess}")
            
            stats = self.pipeline.process_vault(
                callback=self.process_callback,
                force_reprocess=force_reprocess
            )
            
            # Format the results
            end_time = time.time()
            total_time = end_time - start_time
            
            status_message = (
                f"System initialized successfully!\n"
                f"Processed {stats['processed_notes']} notes into {stats['total_chunks']} chunks.\n"
                f"New: {stats['new_notes']}, Updated: {stats['updated_notes']}, Unchanged: {stats['unchanged_notes']}, Deleted: {stats['deleted_notes']}\n"
                f"Total processing time: {total_time:.2f} seconds."
            )
            
            # Build HTML progress bar - shows 100% complete
            progress_html = self.build_progress_html(1, 1, stats)
            
            # Final log message
            self.add_log(f"Processing complete at {datetime.now().strftime('%H:%M:%S')}.")
            self.add_log(f"Total time: {total_time:.2f} seconds")
            
            self.is_initialized = True
            
            return status_message, progress_html, "\n".join(self.processing_logs)
            
        except Exception as e:
            error_message = f"Error initializing system: {str(e)}"
            self.add_log(f"ERROR: {error_message}")
            return error_message, "", "\n".join(self.processing_logs)
    
    def process_callback(self, note_title: str, current: int, total: int, stats: dict) -> None:
        """
        Callback function for processing progress updates.
        
        Args:
            note_title: Title of the current note
            current: Current note index
            total: Total number of notes
            stats: Processing statistics
        """
        percent = int((current / total) * 100)
        
        # Only log every 5% or for the first few files to avoid excessive logging
        if current <= 5 or current == total or current % max(1, int(total / 20)) == 0:
            self.add_log(f"Processing {current}/{total} ({percent}%): {note_title}")
    
    def add_log(self, message: str) -> None:
        """Add a message to the processing log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.processing_logs.append(f"[{timestamp}] {message}")
    
    def build_progress_html(self, current: int, total: int, stats: dict) -> str:
        """
        Build HTML progress bar.
        
        Args:
            current: Current progress
            total: Total items
            stats: Processing statistics
            
        Returns:
            HTML string for progress display
        """
        percent = int((current / total) * 100)
        
        # Create a progress bar
        html = f"""
        <div style="width: 100%; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
            <div style="width: {percent}%; background-color: #4CAF50; height: 20px; border-radius: 5px; text-align: center; line-height: 20px; color: white;">
                {percent}%
            </div>
        </div>
        """
        
        # Add statistics
        html += f"""
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="margin: 5px; padding: 10px; background-color: #007BFF; color: white; border-radius: 5px; min-width: 120px; text-align: center;">
            <strong>New:</strong> {stats.get('new_notes', 0)}
            </div>
            <div style="margin: 5px; padding: 10px; background-color: #28A745; color: white; border-radius: 5px; min-width: 120px; text-align: center;">
            <strong>Updated:</strong> {stats.get('updated_notes', 0)}
            </div>
            <div style="margin: 5px; padding: 10px; background-color: #FF6600; color: black; border-radius: 5px; min-width: 120px; text-align: center;">
            <strong>Unchanged:</strong> {stats.get('unchanged_notes', 0)}
            </div>
            <div style="margin: 5px; padding: 10px; background-color: #DC3545; color: white; border-radius: 5px; min-width: 120px; text-align: center;">
            <strong>Deleted:</strong> {stats.get('deleted_notes', 0)}
            </div>
        </div>
        """
        
        return html
    
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
                response += f"{i}. {chunk.get('title', 'Unnamed')} "
                if 'section_title' in chunk and chunk['section_title']:
                    response += f"- {chunk['section_title']} "
                response += f"(Score: {chunk.get('score', 0):.2f})\n"
                
            return response
            
        except Exception as e:
            return f"Error querying system: {str(e)}"
    
    def clear_cache(self) -> str:
        """
        Clear the processing cache.
        
        Returns:
            Status message
        """
        if not self.pipeline:
            return "System not initialized. Cannot clear cache."
            
        try:
            self.pipeline.clear_cache()
            return "Cache cleared successfully. Next initialization will reprocess all documents."
        except Exception as e:
            return f"Error clearing cache: {str(e)}"
    
    def get_cache_stats(self) -> str:
        """
        Get cache statistics.
        
        Returns:
            Formatted cache statistics
        """
        if not self.pipeline:
            return "System not initialized. No cache statistics available."
            
        try:
            stats = self.pipeline.get_cache_stats()
            
            # Format the statistics
            creation_time = datetime.fromtimestamp(stats["creation_time"]).strftime("%Y-%m-%d %H:%M:%S")
            last_update = datetime.fromtimestamp(stats["last_update"]).strftime("%Y-%m-%d %H:%M:%S")
            
            message = f"Cache Statistics:\n\n"
            message += f"Documents: {stats['document_count']}\n"
            message += f"Chunks: {stats['chunk_count']}\n"
            message += f"Processing Count: {stats['processing_count']}\n"
            message += f"Created: {creation_time}\n"
            message += f"Last Updated: {last_update}\n"
            
            return message
            
        except Exception as e:
            return f"Error getting cache statistics: {str(e)}"
    
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
                with gr.Row():
                    vault_path_input = gr.Textbox(
                        value=config.DEFAULT_VAULT_PATH,
                        label="Obsidian Vault Path"
                    )
                    force_reprocess = gr.Checkbox(
                        value=False,
                        label="Force Reprocessing",
                        info="Process all documents, ignoring cache"
                    )
                    
                init_button = gr.Button("Initialize System")
                
                with gr.Row():
                    init_output = gr.Textbox(
                        value=self.initialization_message,
                        label="Initialization Status",
                        lines=5
                    )
                
                progress_display = gr.HTML(
                    value="<div>Progress will be displayed here during processing.</div>",
                    label="Processing Progress"
                )
                
                process_log = gr.Textbox(
                    value="",
                    label="Processing Log",
                    lines=10
                )
                
                init_button.click(
                    fn=self.initialize_system,
                    inputs=[vault_path_input, force_reprocess],
                    outputs=[init_output, progress_display, process_log]
                )
                
                with gr.Row():
                    cache_stats_button = gr.Button("View Cache Statistics")
                    clear_cache_button = gr.Button("Clear Cache")
                
                cache_output = gr.Textbox(
                    value="",
                    label="Cache Information",
                    lines=5
                )
                
                cache_stats_button.click(
                    fn=self.get_cache_stats,
                    inputs=[],
                    outputs=[cache_output]
                )
                
                clear_cache_button.click(
                    fn=self.clear_cache,
                    inputs=[],
                    outputs=[cache_output]
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
                    lines=15
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
    app = CachedDigitalHumanitiesUI()
    app.launch()