"""Distributed orchestrator for Graph RAG with microservices."""
import logging
import os
import asyncio
from typing import List, Dict, Any

from clients.service_clients import (
    CacheServiceClient,
    RateLimiterClient,
    DocumentProcessorClient,
    LLMServiceClient,
    GraphProcessorClient
)

logger = logging.getLogger(__name__)


class DistributedGraphRAGOrchestrator:
    """Orchestrates Graph RAG pipeline using microservices."""
    
    def __init__(
        self,
        cache_client: CacheServiceClient,
        rate_limiter_client: RateLimiterClient,
        document_processor_client: DocumentProcessorClient,
        llm_service_client: LLMServiceClient,
        graph_processor_client: GraphProcessorClient
    ):
        """Initialize distributed orchestrator.
        
        Args:
            cache_client: Cache service client
            rate_limiter_client: Rate limiter client
            document_processor_client: Document processor client
            llm_service_client: LLM service client
            graph_processor_client: Graph processor client
        """
        self.cache = cache_client
        self.rate_limiter = rate_limiter_client
        self.doc_processor = document_processor_client
        self.llm_service = llm_service_client
        self.graph_processor = graph_processor_client
        
        # Get config from environment - no defaults, require explicit configuration
        chunk_size_str = os.getenv("CHUNK_SIZE")
        chunk_overlap_str = os.getenv("CHUNK_OVERLAP")
        
        if not chunk_size_str:
            raise ValueError("CHUNK_SIZE environment variable is required")
        if not chunk_overlap_str:
            raise ValueError("CHUNK_OVERLAP environment variable is required")
            
        self.chunk_size = int(chunk_size_str)
        self.chunk_overlap = int(chunk_overlap_str)
        
        # Get rate limiter config from environment - no defaults
        bucket_id = os.getenv("RATE_LIMIT_BUCKET_ID")
        capacity_str = os.getenv("RATE_LIMIT_CAPACITY")
        refill_rate_str = os.getenv("RATE_LIMIT_REFILL_RATE")
        
        if not bucket_id:
            raise ValueError("RATE_LIMIT_BUCKET_ID environment variable is required")
        if not capacity_str:
            raise ValueError("RATE_LIMIT_CAPACITY environment variable is required")
        if not refill_rate_str:
            raise ValueError("RATE_LIMIT_REFILL_RATE environment variable is required")
            
        self.rate_limit_bucket_id = bucket_id
        self.rate_limit_capacity = int(capacity_str)
        self.rate_limit_refill_rate = float(refill_rate_str)
    
    def load_documents(self, folder: str) -> List[tuple]:
        """Load documents from folder (sync operation)."""
        documents = []
        
        try:
            if not os.path.exists(folder):
                logger.error(f"Folder not found: {folder}")
                return documents
            
            txt_files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
            
            if not txt_files:
                logger.warning(f"No .txt files found in {folder}")
                return documents
            
            for filename in txt_files:
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append((filepath, content))
                        logger.info(f"Loaded document from {filename}: {len(content)} chars")
            
            logger.info(f"Loaded {len(documents)} documents from {folder}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}", exc_info=True)
            raise
    
    async def process_document(self, file_path: str, content: str) -> Dict[str, Any]:
        """Process a single document."""
        # Generate document ID
        doc_id = await self.doc_processor.generate_document_id(file_path, content)
        
        # Check cache
        cache_check = await self.cache.check_document_cached(file_path, content)
        
        if cache_check["cached"]:
            logger.info(f"Document {doc_id} found in cache")
            
            # Get cached data
            chunks = await self.cache.get_chunks(doc_id)
            summaries = await self.cache.get_summaries(doc_id)
            
            return {
                "document_id": doc_id,
                "chunks": chunks,
                "summaries": [s["summary"] for s in summaries]
            }
        else:
            logger.info(f"Processing new document {doc_id}")
            
            # Save document
            await self.cache.save_document(doc_id, file_path, content)
            
            # Chunk document
            chunks = await self.doc_processor.chunk_document(
                doc_id, content, self.chunk_size, self.chunk_overlap
            )
            
            # Save chunks
            await self.cache.save_chunks(doc_id, chunks)
            
            # Extract elements
            chunk_contents = [c["content"] for c in chunks]
            elements = await self.llm_service.extract_elements(chunk_contents)
            
            # Save elements
            await self.cache.save_elements(doc_id, [
                {"chunk_index": i, "content": elem}
                for i, elem in enumerate(elements)
            ])
            
            # Summarize elements
            summaries = await self.llm_service.summarize_elements(elements)
            
            # Save summaries
            await self.cache.save_summaries(doc_id, [
                {"element_id": i, "summary": summ}
                for i, summ in enumerate(summaries)
            ])
            
            return {
                "document_id": doc_id,
                "chunks": chunks,
                "summaries": summaries
            }
    
    async def run_pipeline_async(self, query: str, documents_folder: str) -> str:
        """Execute the complete Graph RAG pipeline asynchronously."""
        logger.info("=" * 80)
        logger.info("Starting Distributed Graph RAG Pipeline")
        logger.info(f"Query: {query}")
        logger.info(f"Documents folder: {documents_folder}")
        logger.info("=" * 80)
        
        # Initialize rate limiter with config from environment
        try:
            await self.rate_limiter.initialize_bucket(
                bucket_id=self.rate_limit_bucket_id,
                capacity=self.rate_limit_capacity,
                refill_rate=self.rate_limit_refill_rate
            )
            logger.info(f"Rate limiter initialized: bucket_id={self.rate_limit_bucket_id}, capacity={self.rate_limit_capacity}, refill_rate={self.rate_limit_refill_rate}")
        except Exception as e:
            logger.warning(f"Could not initialize rate limiter: {e}")
        
        # Load documents (sync)
        documents = self.load_documents(documents_folder)
        
        # Process all documents in parallel
        logger.info(f"Processing {len(documents)} documents in parallel...")
        tasks = [self.process_document(file_path, content) for file_path, content in documents]
        results = await asyncio.gather(*tasks)
        
        # Collect all summaries
        all_summaries = []
        for result in results:
            all_summaries.extend(result["summaries"])
        
        logger.info(f"Collected {len(all_summaries)} summaries from {len(documents)} documents")
        
        # Create cache key based on summaries AND query
        import hashlib
        summaries_hash = hashlib.md5(str(sorted(all_summaries)).encode()).hexdigest()
        query_hash = hashlib.md5(f"{summaries_hash}:{query}".encode()).hexdigest()
        
        # **NEW: Check for cached final answer for this exact query+documents combination**
        cached_answer = await self.cache.get_query_answer(query_hash)
        if cached_answer:
            logger.info("✨ Using cached answer - instant response!")
            logger.info("=" * 80)
            return cached_answer
        
        # Check if graph is cached in Neo4j
        cached_graph = await self.cache.get_graph(summaries_hash)
        
        if cached_graph:
            logger.info("Using cached graph from Neo4j")
            graph_result = cached_graph
            communities = graph_result["communities"]
        else:
            logger.info("Building new graph")
            # Build graph and detect communities
            graph_result = await self.graph_processor.build_graph(all_summaries)
            communities = graph_result["communities"]
            
            # Save to Neo4j
            await self.cache.save_graph(
                summaries_hash,
                graph_result["nodes"],
                graph_result["edges"],
                communities
            )
            logger.info("Graph saved to Neo4j")
        
        logger.info(f"Graph: {graph_result['nodes']} nodes, {graph_result['edges']} edges")
        logger.info(f"Communities: {len(communities)}")
        
        if not communities:
            logger.warning("No communities detected")
            return "The answer cannot be determined from the provided documents at this time, please try again later."
        
        # Check if community summaries are cached in Neo4j (FIXED: cache summaries not just descriptions)
        cached_summaries = await self.cache.get_community_summaries(summaries_hash)
        
        if cached_summaries:
            logger.info("Using cached community summaries from Neo4j")
            community_summaries = cached_summaries
        else:
            # Check if community descriptions are cached in Neo4j
            cached_descriptions = await self.cache.get_community_descriptions(summaries_hash)
            
            if cached_descriptions:
                logger.info("Using cached community descriptions from Neo4j")
                community_descriptions = cached_descriptions
            else:
                logger.info(f"Describing {len(communities)} communities in parallel...")
                # Get community descriptions in parallel (FIXED: removed unused all_summaries parameter)
                community_tasks = [
                    self.graph_processor.describe_community(community["members"])
                    for community in communities
                ]
                community_descriptions = await asyncio.gather(*community_tasks)
                
                # Save to Neo4j
                await self.cache.save_community_descriptions(summaries_hash, community_descriptions)
                logger.info("Community descriptions saved to Neo4j")
            
            # Summarize communities (only if not cached)
            logger.info(f"Summarizing {len(communities)} communities...")
            community_summaries = await self.llm_service.summarize_communities(community_descriptions)
            
            # Save community summaries to cache
            await self.cache.save_community_summaries(summaries_hash, community_summaries)
            logger.info("Community summaries saved to Neo4j")
        
        # Generate answers
        intermediate_answers = await self.llm_service.answer_query(community_summaries, query)
        
        # Combine answers
        final_answer = await self.llm_service.combine_answers(intermediate_answers)
        
        # **NEW: Cache the final answer for instant future responses**
        await self.cache.save_query_answer(query_hash, final_answer)
        logger.info("Query answer cached for future requests")
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 80)
        
        return final_answer
    
    def run_pipeline(self, query: str, documents_folder: str) -> str:
        """Execute the pipeline (sync wrapper for async implementation)."""
        return asyncio.run(self.run_pipeline_async(query, documents_folder))
