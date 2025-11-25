"""Orchestrator service implementation."""
import logging
import os
import asyncio
import uuid
from typing import Optional, Dict, Any
from fastapi import BackgroundTasks

# Import from local modules
from clients.service_clients import (
    CacheServiceClient,
    RateLimiterClient,
    DocumentProcessorClient,
    LLMServiceClient,
    GraphProcessorClient
)
from distributed_orchestrator import DistributedGraphRAGOrchestrator

logger = logging.getLogger(__name__)


class OrchestratorService:
    """Service for managing Graph RAG orchestration."""
    
    def __init__(self):
        """Initialize the orchestrator service."""
        self.orchestrator: Optional[DistributedGraphRAGOrchestrator] = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self):
        """Initialize the distributed orchestrator with service clients."""
        try:
            # Get service URLs from environment - no defaults, must be provided
            cache_url = os.getenv("CACHE_SERVICE_URL")
            rate_limiter_url = os.getenv("RATE_LIMITER_URL")
            llm_url = os.getenv("LLM_SERVICE_URL")
            doc_processor_url = os.getenv("DOCUMENT_PROCESSOR_URL")
            graph_processor_url = os.getenv("GRAPH_PROCESSOR_URL")
            
            if not cache_url:
                raise ValueError("CACHE_SERVICE_URL environment variable is required")
            if not rate_limiter_url:
                raise ValueError("RATE_LIMITER_URL environment variable is required")
            if not llm_url:
                raise ValueError("LLM_SERVICE_URL environment variable is required")
            if not doc_processor_url:
                raise ValueError("DOCUMENT_PROCESSOR_URL environment variable is required")
            if not graph_processor_url:
                raise ValueError("GRAPH_PROCESSOR_URL environment variable is required")
            
            # Create service clients
            cache_client = CacheServiceClient(cache_url)
            rate_limiter_client = RateLimiterClient(rate_limiter_url)
            doc_processor_client = DocumentProcessorClient(doc_processor_url)
            llm_client = LLMServiceClient(llm_url)
            graph_processor_client = GraphProcessorClient(graph_processor_url)
            
            # Create orchestrator
            self.orchestrator = DistributedGraphRAGOrchestrator(
                cache_client=cache_client,
                rate_limiter_client=rate_limiter_client,
                document_processor_client=doc_processor_client,
                llm_service_client=llm_client,
                graph_processor_client=graph_processor_client
            )
            
            logger.info("Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
            raise
    
    async def _run_indexing_job(self, job_id: str, documents_folder: str):
        """Run indexing job in background."""
        try:
            self.jobs[job_id]["status"] = "running"
            logger.info(f"Starting indexing job {job_id} for folder: {documents_folder}")
            
            # Load and process all documents
            documents = self.orchestrator.load_documents(documents_folder)
            
            if not documents:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = "No documents found"
                return
            
            # Process documents
            tasks = [
                self.orchestrator.process_document(file_path, content)
                for file_path, content in documents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(errors[0])
                logger.error(f"Indexing job {job_id} failed: {errors[0]}")
            else:
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["documents_processed"] = len(documents)
                logger.info(f"Indexing job {job_id} completed successfully")
        
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            logger.error(f"Indexing job {job_id} failed with exception: {e}", exc_info=True)
    
    async def start_indexing(
        self,
        documents_folder: str,
        background_tasks: BackgroundTasks
    ) -> str:
        """Start indexing documents in background."""
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        self.jobs[job_id] = {
            "status": "pending",
            "documents_folder": documents_folder,
            "created_at": asyncio.get_event_loop().time()
        }
        
        # Add to background tasks
        background_tasks.add_task(self._run_indexing_job, job_id, documents_folder)
        
        return job_id
    
    async def query(self, query: str, documents_folder: str = "test_docs") -> str:
        """Query the indexed documents."""
        try:
            logger.info(f"Processing query: {query}")
            
            # Run the full pipeline
            answer = await self.orchestrator.run_pipeline_async(
                query=query,
                documents_folder=documents_folder
            )
            
            return answer
        
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an indexing job."""
        return self.jobs.get(job_id)
