"""LLM operations business logic."""
import logging
import os
from typing import List, Dict, Optional
import asyncio
from openai import AzureOpenAI
import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM API operations.
    
    Manages interactions with Azure OpenAI API including rate limiting,
    token estimation, and various LLM operations such as extraction,
    summarization, and query answering.
    """
    
    def __init__(self):
        """Initialize LLM service with configuration from environment variables.
        
        Raises:
            ValueError: If required environment variables are not set.
        """
        self.client: Optional[AzureOpenAI] = None
        
        # Required environment variables - no defaults
        self.rate_limiter_url: str = os.getenv("RATE_LIMITER_URL")
        if not self.rate_limiter_url:
            raise ValueError("RATE_LIMITER_URL environment variable is required")
        
        rate_limit_timeout_str = os.getenv("RATE_LIMIT_TIMEOUT")
        if not rate_limit_timeout_str:
            raise ValueError("RATE_LIMIT_TIMEOUT environment variable is required")
        self.rate_limit_timeout: float = float(rate_limit_timeout_str)
        
        self.rate_limit_bucket_id: str = os.getenv("RATE_LIMIT_BUCKET_ID")
        if not self.rate_limit_bucket_id:
            raise ValueError("RATE_LIMIT_BUCKET_ID environment variable is required")
    
    def initialize(self):
        """Initialize Azure OpenAI client.
        
        Sets up the Azure OpenAI client with credentials and configuration
        from environment variables.
        
        Raises:
            ValueError: If required environment variables are not set.
        """
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            # Get API version from query params - required, no default
            query_params = os.getenv("OPENAI_QUERY_PARAMS")
            if not query_params:
                raise ValueError("OPENAI_QUERY_PARAMS environment variable is required")
            api_version = query_params.replace("api-version=", "")
            
            # Get Azure endpoint (remove deployment path if present) - required, no default
            base_url = os.getenv("OPENAI_BASE_URL")
            if not base_url:
                raise ValueError("OPENAI_BASE_URL environment variable is required")
            if "/openai/deployments" in base_url:
                azure_endpoint = base_url.split("/openai/deployments")[0] + "/"
            else:
                azure_endpoint = base_url if base_url.endswith("/") else base_url + "/"
            
            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            )
            logger.info(f"Azure OpenAI client initialized with endpoint: {azure_endpoint}, api_version: {api_version}")
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (rough approximation).
        
        Args:
            text: Input text.
            
        Returns:
            int: Estimated token count (minimum of 1).
        """
        return max(1, len(text) // 4)
    
    def estimate_messages_tokens(self, messages: List[Dict]) -> int:
        """Estimate total tokens for messages.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            int: Estimated total token count including overhead.
        """
        total = 0
        for message in messages:
            content = message.get('content', '')
            role = message.get('role', '')
            total += self.estimate_tokens(content)
            total += self.estimate_tokens(role)
            total += 4  # Overhead
        return total
    
    async def consume_rate_limit(
        self,
        tokens: int,
        bucket_id: str = None
    ) -> Dict:
        """Consume tokens from rate limiter service.
        
        Args:
            tokens: Number of tokens to consume.
            bucket_id: Identifier for the token bucket (uses default if not provided).
            
        Returns:
            Dict: Rate limiter response containing status and token information.
            
        Raises:
            Exception: If rate limit is exceeded.
        """
        if bucket_id is None:
            bucket_id = self.rate_limit_bucket_id
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.rate_limiter_url}/tokens/consume",
                    json={
                        "tokens": tokens,
                        "bucket_id": bucket_id,
                        "timeout": self.rate_limit_timeout
                    },
                    timeout=self.rate_limit_timeout + 5.0
                )
                if response.status_code == 429:
                    raise Exception("Rate limit exceeded")
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.exception("Rate limiter request error")
                logger.warning(
                    "Rate limiter unavailable, proceeding without rate limiting"
                )
                return {"status": "bypassed"}
    
    async def call_llm_api(
        self,
        messages: List[Dict[str, str]],
        request_type: str = "",
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Make a rate-limited LLM API call.
        
        Args:
            messages: List of message dictionaries for the API.
            request_type: Description of the request type for logging.
            temperature: Sampling temperature (uses env var if not provided).
            max_tokens: Maximum tokens to generate (uses env var if not provided).
            
        Returns:
            str: LLM response text.
            
        Raises:
            ValueError: If required environment variables are not set.
            Exception: If LLM API call fails.
        """
        if self.client is None:
            self.initialize()
        
        model_name = os.getenv("OPENAI_INFERENCE_MODEL_NAME")
        if not model_name:
            raise ValueError("OPENAI_INFERENCE_MODEL_NAME environment variable is required")
        
        # Use provided values or fall back to environment variables
        if temperature is None:
            temperature_str = os.getenv("OPENAI_TEMPERATURE")
            temperature = float(temperature_str) if temperature_str else None
        
        if max_tokens is None:
            max_tokens_str = os.getenv("OPENAI_MAX_TOKENS")
            max_tokens = int(max_tokens_str) if max_tokens_str else None
        
        # Estimate tokens for rate limiting
        estimated_tokens = self.estimate_messages_tokens(messages)
        logger.debug(
            f"API Request [{request_type}] - Estimated tokens: {estimated_tokens}"
        )
        
        # Wait for rate limit capacity
        await self.consume_rate_limit(estimated_tokens)
        
        try:
            # Build API call parameters
            api_params = {
                "model": model_name,
                "messages": messages
            }
            
            # Add optional parameters if provided
            if temperature is not None:
                api_params["temperature"] = temperature
            if max_tokens is not None:
                api_params["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**api_params)
            
            logger.info(f"API Response [{request_type}] - Status: Success")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.exception(f"API Request [{request_type}] failed")
            raise Exception(f"LLM API error: {str(e)}")
    
    async def extract_elements(
        self,
        chunks: List[str],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> List[str]:
        """Extract entities and relationships from text chunks.
        
        Args:
            chunks: List of text chunks to process.
            system_prompt: Custom system prompt (uses default if not provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            List[str]: List of extracted elements, one per chunk.
        """
        logger.info(f"Extracting elements from {len(chunks)} chunks")
        
        if system_prompt is None:
            system_prompt = os.getenv(
                "EXTRACTION_SYSTEM_PROMPT",
                "Extract entities and relationships from the following text."
            )
        
        # Process chunks in parallel
        
        async def process_chunk(i: int, chunk: str) -> str:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ]
            return await self.call_llm_api(
                messages,
                f"extract_chunk_{i}",
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        elements = await asyncio.gather(*tasks)
        
        return list(elements)
    
    async def summarize_elements(
        self,
        elements: List[str],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> List[str]:
        """Summarize extracted elements.
        
        Args:
            elements: List of extracted elements to summarize.
            system_prompt: Custom system prompt (uses default if not provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            List[str]: List of summaries, one per element.
        """
        logger.info(f"Summarizing {len(elements)} elements")
        
        if system_prompt is None:
            system_prompt = os.getenv(
                "SUMMARIZE_ELEMENTS_SYSTEM_PROMPT",
                """Summarize the extracted entities and relationships in a clear, structured format.

Output format:
Entities:
- Entity1
- Entity2
- Entity3

Relationships:
Entity1 -> relationship_type -> Entity2
Entity3 -> relationship_type -> Entity1

Guidelines:
- List each entity on a separate line with a bullet point or number
- Use arrow notation (A -> relation -> B) for relationships
- Keep entity names consistent throughout
- Include only factual information from the input"""
            )
        
        # Process elements in parallel
        import asyncio
        
        async def process_element(i: int, element: str) -> str:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": element
                }
            ]
            return await self.call_llm_api(
                messages,
                f"summarize_element_{i}",
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        tasks = [process_element(i, element) for i, element in enumerate(elements)]
        summaries = await asyncio.gather(*tasks)
        
        return list(summaries)
    
    async def summarize_communities(
        self,
        descriptions: List[Dict],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> List[str]:
        """Summarize communities in the knowledge graph.
        
        Args:
            descriptions: List of community description dictionaries.
            system_prompt: Custom system prompt (uses default if not provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            List[str]: List of community summaries.
        """
        logger.info(f"Summarizing {len(descriptions)} communities")
        
        if system_prompt is None:
            system_prompt = os.getenv(
                "SUMMARIZE_COMMUNITIES_SYSTEM_PROMPT",
                "Summarize the following community of entities and relationships."
            )
        
        # Process communities in parallel
        import asyncio
        
        async def process_community(i: int, description: Dict) -> str:
            entities = description.get('entities', [])
            relationships = description.get('relationships', [])
            
            content = (
                f"Entities: {', '.join(entities)}\n"
                f"Relationships: {', '.join(relationships)}"
            )
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
            return await self.call_llm_api(
                messages,
                f"summarize_community_{i}",
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        tasks = [process_community(i, description) for i, description in enumerate(descriptions)]
        summaries = await asyncio.gather(*tasks)
        
        return list(summaries)
    
    async def answer_query(
        self,
        summaries: List[str],
        query: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> List[str]:
        """Generate answers from community summaries.
        
        Args:
            summaries: List of community summaries.
            query: User query.
            system_prompt: Custom system prompt (uses default if not provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            List[str]: List of intermediate answers, one per summary.
        """
        logger.info(f"Generating answers from {len(summaries)} summaries")
        
        if system_prompt is None:
            system_prompt = os.getenv(
                "ANSWER_QUERY_SYSTEM_PROMPT",
                "Answer the following query based on the provided summary."
            )
        
        # Process summaries in parallel
        import asyncio
        
        async def process_summary(i: int, summary: str) -> str:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nSummary: {summary}"
                }
            ]
            return await self.call_llm_api(
                messages,
                f"answer_community_{i}",
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        tasks = [process_summary(i, summary) for i, summary in enumerate(summaries)]
        answers = await asyncio.gather(*tasks)
        
        return list(answers)
    
    async def combine_answers(
        self,
        intermediate_answers: List[str],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Combine intermediate answers into final answer.
        
        Args:
            intermediate_answers: List of intermediate answers.
            system_prompt: Custom system prompt (uses default if not provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            str: Final combined answer.
        """
        logger.info("Generating final answer from intermediate answers")
        
        if system_prompt is None:
            system_prompt = os.getenv(
                "COMBINE_ANSWERS_SYSTEM_PROMPT",
                "Combine these answers into a final, concise response."
            )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Intermediate answers: {intermediate_answers}"
            }
        ]
        
        final_answer = await self.call_llm_api(
            messages,
            "final_answer",
            temperature=temperature,
            max_tokens=max_tokens
        )
        return final_answer
