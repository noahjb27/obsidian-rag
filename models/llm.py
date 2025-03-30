"""
Ollama LLM integration for local model inference.

This module provides a wrapper around Ollama's API for both 
LLM text generation and embeddings generation.
"""
import json
import time
from typing import List, Dict, Any, Optional, Union
import requests
from functools import lru_cache


class OllamaClient:
    """Client for interacting with Ollama's REST API."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        generation_model: str = "mistral:7b-instruct-v0.2",
        embedding_model: str = "nomic-embed-text",
        timeout: int = 120
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: URL where Ollama is running
            generation_model: Model to use for text generation
            embedding_model: Model to use for embeddings
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        
        # Default generation parameters
        self.default_gen_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 4096,  # Maximum tokens to generate
            "stop": ["</answer>", "Human:", "human:"],  # Default stop sequences
        }
        
        # Test connection to Ollama
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test the connection to Ollama and ensure models are available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            available_models = [model["name"] for model in response.json()["models"]]
            
            # Check if our models are available
            for model in [self.generation_model, self.embedding_model]:
                model_name = model.split(':')[0]  # Handle model:tag format
                
                if model not in available_models and model_name not in available_models:
                    print(f"Warning: Model {model} not found. Available models: {available_models}")
                    print(f"Pulling model {model}...")
                    self._pull_model(model)
                    
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
    
    def _pull_model(self, model: str) -> None:
        """Pull a model from Ollama if it's not available."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600  # Longer timeout for model pulling
            )
            response.raise_for_status()
            print(f"Successfully pulled model {model}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to pull model {model}: {str(e)}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, List[str]]:
        """
        Generate text using the specified LLM.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to control model behavior
            params: Generation parameters to override defaults
            stream: Whether to stream the response or return all at once
            
        Returns:
            Generated text or list of text chunks if streaming
        """
        # Merge default parameters with any provided parameters
        request_params = {**self.default_gen_params}
        if params:
            request_params.update(params)
            
        # Prepare the request
        request_data = {
            "model": self.generation_model,
            "prompt": prompt,
            "options": request_params,
            "stream": stream
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
            
        try:
            # Make the request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                # Return a generator of text chunks
                return self._process_streaming_response(response)
            else:
                # Return the complete response
                return response.json()["response"]
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to generate text: {str(e)}")
            
    def _process_streaming_response(self, response) -> List[str]:
        """Process a streaming response from Ollama."""
        chunks = []
        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                if "response" in chunk_data:
                    chunks.append(chunk_data["response"])
                if chunk_data.get("done", False):
                    break
        return chunks
    
    @lru_cache(maxsize=1024)
    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()["embedding"]
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to generate embeddings: {str(e)}")
            
    def batch_get_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in a single batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches to avoid overloading Ollama
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.get_embeddings(text)
                batch_embeddings.append(embedding)
                
                # Small delay to avoid overwhelming the service
                time.sleep(0.1)
                
            all_embeddings.extend(batch_embeddings)
            
            # Larger delay between batches
            if i + batch_size < len(texts):
                time.sleep(1)
                
        return all_embeddings


class DigitalHumanitiesLLM:
    """
    Specialized LLM interface for digital humanities applications.
    
    This class wraps the OllamaClient with domain-specific prompting and
    functionality tailored to humanities research.
    """
    
    def __init__(self, client: Optional[OllamaClient] = None):
        """
        Initialize the digital humanities LLM interface.
        
        Args:
            client: Optional OllamaClient instance (creates a new one if not provided)
        """
        self.client = client or OllamaClient()
        
        # Default system prompt tailored to humanities research
        self.default_system_prompt = """You are a digital humanities assistant specializing 
        in analyzing and contextualizing texts, historical documents, and cultural artifacts. 
        Think carefully about the historical, cultural, and social contexts of the material 
        you're analyzing. Balance close reading with broader interpretive frameworks, and 
        consider multiple perspectives when appropriate. Support claims with textual evidence, 
        and acknowledge limitations of your analysis. When relevant, note connections to 
        other texts, historical events, or cultural phenomena."""
    
    def analyze_text(
        self, 
        text: str, 
        instruction: str,
        context: Optional[List[str]] = None
    ) -> str:
        """
        Analyze a humanities text based on an instruction.
        
        Args:
            text: The primary text to analyze
            instruction: What type of analysis to perform
            context: Optional additional context texts
            
        Returns:
            Analysis result
        """
        # Construct a prompt that clearly separates the components
        prompt_parts = ["# Text to Analyze", text]
        
        if context:
            prompt_parts.append("# Additional Context")
            for i, ctx in enumerate(context, 1):
                prompt_parts.append(f"Context {i}:\n{ctx}")
                
        prompt_parts.append("# Instruction")
        prompt_parts.append(instruction)
        
        prompt_parts.append("# Analysis")
        prompt = "\n\n".join(prompt_parts)
        
        # Generate the analysis
        result = self.client.generate(prompt, system_prompt=self.default_system_prompt)
        return result
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a text."""
        return self.client.get_embeddings(text)
        
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.client.batch_get_embeddings(texts)
