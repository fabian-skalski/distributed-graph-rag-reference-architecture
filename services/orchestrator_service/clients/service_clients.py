"""HTTP clients for microservices."""
import logging
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class CacheServiceClient:
    """Client for cache microservice."""
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.timeout = 30.0
        self._client = None
    
    async def _get_client(self):
        """Get or create persistent HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def check_document_cached(self, file_path: str, content: str) -> Dict[str, Any]:
        """Check if document is cached."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/documents/check",
            json={"file_path": file_path, "content": content}
        )
        response.raise_for_status()
        return response.json()
    
    async def save_document(self, document_id: str, file_path: str, content: str, metadata: Optional[Dict] = None):
        """Save document."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/documents",
            json={
                "document_id": document_id,
                "file_path": file_path,
                "content": content,
                "metadata": metadata
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def save_chunks(self, document_id: str, chunks: List[Dict]):
        """Save chunks."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/chunks",
            json={"document_id": document_id, "chunks": chunks}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_chunks(self, document_id: str) -> List[Dict]:
        """Get chunks."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/chunks/{document_id}")
        response.raise_for_status()
        return response.json()["chunks"]
    
    async def save_elements(self, document_id: str, elements: List[Dict]):
        """Save elements."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/elements",
            json={"document_id": document_id, "elements": elements}
        )
        response.raise_for_status()
        return response.json()
    
    async def save_summaries(self, document_id: str, summaries: List[Dict]):
        """Save summaries."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/summaries",
            json={"document_id": document_id, "summaries": summaries}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_summaries(self, document_id: str) -> List[Dict]:
        """Get summaries."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/summaries/{document_id}")
        response.raise_for_status()
        return response.json()["summaries"]
    
    async def save_graph(self, summaries_hash: str, nodes: int, edges: int, communities: List[Dict]):
        """Save graph structure and communities."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/graph",
            json={
                "summaries_hash": summaries_hash,
                "nodes": nodes,
                "edges": edges,
                "communities": communities
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def get_graph(self, summaries_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached graph structure."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/graph/{summaries_hash}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    async def save_community_descriptions(self, summaries_hash: str, descriptions: List[Dict]):
        """Save community descriptions."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/community-descriptions",
            json={
                "summaries_hash": summaries_hash,
                "descriptions": descriptions
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def get_community_descriptions(self, summaries_hash: str) -> Optional[List[Dict]]:
        """Get cached community descriptions."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/community-descriptions/{summaries_hash}")
            response.raise_for_status()
            return response.json()["descriptions"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    async def save_community_summaries(self, summaries_hash: str, summaries: List[str]):
        """Save community summaries."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/community-summaries",
            json={
                "summaries_hash": summaries_hash,
                "summaries": summaries
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def get_community_summaries(self, summaries_hash: str) -> Optional[List[str]]:
        """Get cached community summaries."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/community-summaries/{summaries_hash}")
            response.raise_for_status()
            return response.json()["summaries"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    async def save_query_answer(self, query_hash: str, answer: str):
        """Save query answer to cache."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/query-answers",
            json={
                "query_hash": query_hash,
                "answer": answer
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def get_query_answer(self, query_hash: str) -> Optional[str]:
        """Get cached query answer."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/query-answers/{query_hash}")
            response.raise_for_status()
            return response.json()["answer"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise



class RateLimiterClient:
    """Client for rate limiter microservice."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 70.0
        self._client = None
    
    async def _get_client(self):
        """Get or create persistent HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def initialize_bucket(self, bucket_id: str, capacity: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            bucket_id: Identifier for the token bucket
            capacity: Maximum number of tokens in the bucket
            refill_rate: Rate at which tokens are added per minute
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/buckets/init",
            json={
                "bucket_id": bucket_id,
                "capacity": capacity,
                "refill_rate": refill_rate
            }
        )
        response.raise_for_status()
        return response.json()


class DocumentProcessorClient:
    """Client for document processor microservice."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 30.0
        self._client = None
    
    async def _get_client(self):
        """Get or create persistent HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def generate_document_id(self, file_path: str, content: str) -> str:
        """Generate document ID."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/generate-id",
            json={"file_path": file_path, "content": content}
        )
        response.raise_for_status()
        return response.json()["document_id"]
    
    async def chunk_document(self, document_id: str, content: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[Dict]:
        """Chunk document."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/chunk",
            json={
                "document_id": document_id,
                "content": content,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        response.raise_for_status()
        return response.json()["chunks"]


class LLMServiceClient:
    """Client for LLM operations microservice."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 300.0  # Longer timeout for LLM operations
        self._client = None
    
    async def _get_client(self):
        """Get or create persistent HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def extract_elements(self, chunks: List[str]) -> List[str]:
        """Extract elements from chunks."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/extract",
            json={"chunks": chunks}
        )
        response.raise_for_status()
        return response.json()["elements"]
    
    async def summarize_elements(self, elements: List[str]) -> List[str]:
        """Summarize elements."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/summarize/elements",
            json={"elements": elements}
        )
        response.raise_for_status()
        return response.json()["summaries"]
    
    async def summarize_communities(self, descriptions: List[Dict]) -> List[str]:
        """Summarize communities."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/summarize/communities",
            json={"descriptions": descriptions}
        )
        response.raise_for_status()
        return response.json()["summaries"]
    
    async def answer_query(self, summaries: List[str], query: str) -> List[str]:
        """Generate answers from summaries."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/query/answer",
            json={"summaries": summaries, "query": query}
        )
        response.raise_for_status()
        return response.json()["answers"]
    
    async def combine_answers(self, intermediate_answers: List[str]) -> str:
        """Combine intermediate answers."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/query/combine",
            json={"intermediate_answers": intermediate_answers}
        )
        response.raise_for_status()
        return response.json()["answer"]


class GraphProcessorClient:
    """Client for graph processor microservice."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 60.0
        self._client = None
    
    async def _get_client(self):
        """Get or create persistent HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def build_graph(self, summaries: List[str]) -> Dict[str, Any]:
        """Build graph and detect communities."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/graph/build",
            json={"summaries": summaries}
        )
        response.raise_for_status()
        return response.json()
    
    async def describe_community(self, community_members: List[str]) -> Dict[str, Any]:
        """Get community description."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/community/describe",
            json={"community_members": community_members}
        )
        response.raise_for_status()
        return response.json()
