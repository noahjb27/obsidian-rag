"""
Robust LLM client with improved error handling, retries, and resource management.
"""
import json
import time
import sys
import psutil
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import requests
from functools import lru_cache
import backoff

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("robust_llm")


class RobustOllamaClient:
    """Enhanced client for interacting with Ollama with improved error handling."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        generation_model: str = "mistral:7b-instruct-v0.2",
        embedding_model: str = "nomic-embed-text",
        timeout: int = 180,  # Increased timeout
        max_retries: int = 3,
        max_context_length: int = 4000
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: URL where Ollama is running
            generation_model: Model to use for text generation
            embedding_model: Model to use for embeddings
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            max_context_length: Maximum context length to send to the model
        """
        self.base_url = base_url.rstrip('/')
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_context_length = max_context_length
        
        # Default generation parameters
        self.default_gen_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 4096,  # Maximum tokens to generate
            "stop": ["</answer>", "Human:", "human:"],  # Default stop sequences
        }
        
        # Check connection and warm up models
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """
        Test the connection to Ollama and ensure models are available.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            available_models = [model["name"] for model in response.json().get("models", [])]
            logger.info(f"Connected to Ollama. Available models: {available_models}")
            
            # Check if our models are available
            for model in [self.generation_model, self.embedding_model]:
                model_name = model.split(':')[0]  # Handle model:tag format
                
                if not any(m == model or m.startswith(f"{model_name}:") for m in available_models):
                    logger.warning(f"Model {model} not found. Available models: {available_models}")
                    logger.info(f"Pulling model {model}...")
                    self._pull_model(model)
            
            # Warm up models with a small request
            logger.info("Warming up models...")
            self._warm_up_models()
            
            return True
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            return False
    
    def _pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama if it's not available.
        
        Args:
            model: Model name to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model {model}. This may take several minutes...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=1200  # Long timeout for model pulling (20 minutes)
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model {model}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model}: {str(e)}")
            return False
    
    def _warm_up_models(self) -> None:
        """Warm up models with a simple request to keep them loaded in memory."""
        try:
            # Warm up generation model
            logger.info(f"Warming up generation model {self.generation_model}...")
            self.generate(
                "Hello, this is a warm-up message to ensure the model is loaded.",
                params={"temperature": 0.1, "num_predict": 10}
            )
            
            # Warm up embedding model
            logger.info(f"Warming up embedding model {self.embedding_model}...")
            self.get_embeddings("This is a warm-up message for the embedding model.")
            
            logger.info("Models warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {str(e)}")
    
    def _check_system_resources(self) -> Dict[str, float]:
        """
        Check system resources before making a request.
        
        Returns:
            Dictionary with resource usage information
        """
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        resources = {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024 * 1024 * 1024),
            "cpu_percent": cpu_percent
        }
        
        # Log warning if resources are low
        if memory.percent > 90:
            logger.warning(f"System memory usage is high: {memory.percent}%")
        if cpu_percent > 90:
            logger.warning(f"CPU usage is high: {cpu_percent}%")
            
        return resources
    
    def _truncate_context(self, context: List[str], max_tokens: int) -> List[str]:
        """
        Intelligently truncate context to fit within token limit.
        
        Args:
            context: List of context strings
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated context list
        """
        if not context:
            return []
            
        # Simple token estimation
        def estimate_tokens(text):
            return len(text.split())
        
        # Sort context by relevance (assuming first items are most relevant)
        # This preserves the original order which should be by relevance score
        
        total_tokens = 0
        truncated_context = []
        
        for i, text in enumerate(context):
            tokens = estimate_tokens(text)
            if total_tokens + tokens <= max_tokens:
                truncated_context.append(text)
                total_tokens += tokens
            else:
                # For the last item, truncate it to fit if possible
                if i == len(context) - 1:
                    words = text.split()
                    remaining = max_tokens - total_tokens
                    if remaining > 50:  # Only include if we can get a meaningful chunk
                        truncated_text = " ".join(words[:remaining])
                        truncated_context.append(truncated_text)
                        logger.info(f"Truncated last context item from {len(words)} to {remaining} tokens")
                break
                
        if len(truncated_context) < len(context):
            logger.info(f"Context truncated from {len(context)} to {len(truncated_context)} items to fit token limit")
            
        return truncated_context
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        on_backoff=lambda details: logger.info(f"Retrying request after error. Attempt {details['tries']}/3")
    )
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        context: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        safe_mode: bool = True
    ) -> Union[str, List[str]]:
        """
        Generate text using the specified LLM with robust error handling.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to control model behavior
            context: Optional list of context passages to include
            params: Generation parameters to override defaults
            stream: Whether to stream the response or return all at once
            safe_mode: Whether to enable safety measures like context truncation
            
        Returns:
            Generated text or list of text chunks if streaming
        """
        # Check system resources
        resources = self._check_system_resources()
        
        # Merge default parameters with any provided parameters
        request_params = {**self.default_gen_params}
        if params:
            request_params.update(params)
        
        # Process context if provided
        if context and safe_mode:
            # Limit context size to avoid overloading the model
            context = self._truncate_context(context, self.max_context_length)
            
            # Prepare the full prompt with context
            full_prompt = f"Here is some relevant context information:\n\n"
            for i, ctx in enumerate(context, 1):
                full_prompt += f"Context {i}:\n{ctx}\n\n"
            
            full_prompt += f"Based on the above context, please respond to this query:\n{prompt}"
            
            # Log the context size
            logger.info(f"Using {len(context)} context passages for generation")
        else:
            full_prompt = prompt
            
        # Prepare the request
        request_data = {
            "model": self.generation_model,
            "prompt": full_prompt,
            "options": request_params,
            "stream": stream
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
            
        try:
            logger.info(f"Sending generation request to {self.base_url}/api/generate")
            start_time = time.time()
            
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
                result = response.json().get("response", "")
                end_time = time.time()
                logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
                return result
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return "I'm sorry, but the request timed out. The model might be overloaded. Try asking a simpler question or try again later."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate text: {str(e)}")
            raise ConnectionError(f"Failed to generate text: {str(e)}")
            
    def _process_streaming_response(self, response) -> List[str]:
        """Process a streaming response from Ollama."""
        chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = json.loads(line)
                    if "response" in chunk_data:
                        chunks.append(chunk_data["response"])
                    if chunk_data.get("done", False):
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode streaming response: {line}")
        return chunks
    
    @lru_cache(maxsize=1024)
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=3,
        on_backoff=lambda details: logger.info(f"Retrying embedding request. Attempt {details['tries']}/3")
    )
    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text with error handling.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        # Handle empty text input
        if not text or len(text.strip()) < 3:
            logger.warning(f"Empty or very short text provided for embedding: '{text}'")
            # Return a zero vector of the expected dimension
            return [0.0] * 768
            
        try:
            logger.debug(f"Generating embedding for text (length {len(text)})")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30  # Shorter timeout for embeddings
            )
            response.raise_for_status()
            
            embedding = response.json().get("embedding", [])
            
            end_time = time.time()
            logger.debug(f"Embedding generated in {end_time - start_time:.2f} seconds")
            
            # Verify we got a valid embedding
            if not embedding or len(embedding) == 0:
                logger.warning(f"Received empty embedding from server")
                return [0.0] * 768
                
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise ConnectionError(f"Failed to generate embedding: {str(e)}")
            
    def batch_get_embeddings(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching and error handling.
        
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
            
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            for text in batch:
                try:
                    embedding = self.get_embeddings(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error embedding text: {str(e)}")
                    # Return a zero vector for failed embeddings
                    batch_embeddings.append([0.0] * 768)
                
                # Small delay to avoid overwhelming the service
                time.sleep(0.05)
                
            all_embeddings.extend(batch_embeddings)
            
            # Larger delay between batches
            if i + batch_size < len(texts):
                time.sleep(0.5)
                
        return all_embeddings


class RobustDigitalHumanitiesLLM:
    """
    Robust LLM interface for digital humanities applications.
    
    This class wraps the RobustOllamaClient with domain-specific prompting and
    better error handling.
    """
    
    def __init__(self, client: Optional[RobustOllamaClient] = None):
        """
        Initialize the robust digital humanities LLM interface.
        
        Args:
            client: Optional RobustOllamaClient instance (creates a new one if not provided)
        """
        self.client = client or RobustOllamaClient()
        
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
            text: The primary text to analyze (query)
            instruction: What type of analysis to perform
            context: Optional additional context texts
            
        Returns:
            Analysis result
        """
        try:
            # For very complex queries with long contexts, use a specialized approach
            if context and len(context) > 10:
                logger.info(f"Large context provided ({len(context)} passages). Using optimized approach.")
                return self._analyze_with_large_context(text, instruction, context)
            
            # Standard approach for normal queries
            prompt = instruction
            
            # Generate the analysis with context
            result = self.client.generate(
                prompt=text, 
                system_prompt=self.default_system_prompt,
                context=context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_text: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try a simpler query or try again later."
    
    def _analyze_with_large_context(self, query: str, instruction: str, context: List[str]) -> str:
        """
        Handle analysis with very large context by breaking it into stages.
        
        Args:
            query: The query text
            instruction: Analysis instruction
            context: List of context passages
            
        Returns:
            Analysis result
        """
        logger.info("Using multi-stage processing for large context")
        
        try:
            # Step 1: Extract key information from each context passage
            extraction_prompt = f"""
            Extract the key facts and information relevant to this query: "{query}"
            
            Summarize only the most important points, focusing on information that directly answers the query.
            """
            
            # Process context in smaller batches
            batch_size = 5
            extracted_info = []
            
            for i in range(0, len(context), batch_size):
                batch = context[i:i+batch_size]
                logger.info(f"Processing context batch {i//batch_size + 1}/{(len(context) + batch_size - 1)//batch_size}")
                
                for ctx in batch:
                    try:
                        extraction = self.client.generate(
                            prompt=extraction_prompt,
                            context=[ctx],
                            params={"temperature": 0.1}  # Low temperature for factual extraction
                        )
                        extracted_info.append(extraction)
                    except Exception as e:
                        logger.warning(f"Error extracting from context: {str(e)}")
            
            # Step 2: Synthesize the extracted information
            synthesis_prompt = f"""
            Based on the extracted information, provide a comprehensive answer to this query: "{query}"
            
            {instruction}
            """
            
            # Generate final response
            final_response = self.client.generate(
                prompt=synthesis_prompt,
                system_prompt=self.default_system_prompt,
                context=extracted_info,
                params={"temperature": 0.7}  # Regular temperature for final synthesis
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in multi-stage processing: {str(e)}")
            return f"I apologize, but I encountered an error while processing your large query: {str(e)}. Please try simplifying your query or reducing the context size."
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a text with error handling."""
        try:
            return self.client.get_embeddings(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 768
        
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with error handling."""
        try:
            return self.client.batch_get_embeddings(texts)
        except Exception as e:
            logger.error(f"Error batch generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in range(len(texts))]