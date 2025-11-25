"""Orchestrator routes for Graph RAG pipeline."""
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from services.orchestrator_service import OrchestratorService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["orchestrator"]
)

# Create global service instance
orchestrator_service = OrchestratorService()


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    documents_folder: str = Field(
        default="test_docs",
        description="Path to folder containing documents to index"
    )


class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str = Field(..., description="The query/question to answer")
    documents_folder: str = Field(
        default="test_docs",
        description="Path to folder containing documents"
    )


class IndexResponse(BaseModel):
    """Response model for indexing."""
    status: str
    message: str
    job_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str
    answer: str


@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index documents from the specified folder.
    This runs in the background and returns immediately.
    """
    try:
        # Start indexing in background
        job_id = await orchestrator_service.start_indexing(
            request.documents_folder,
            background_tasks
        )
        
        return IndexResponse(
            status="started",
            message=f"Indexing started for folder: {request.documents_folder}",
            job_id=job_id
        )
    except Exception as e:
        logger.error(f"Error starting indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the indexed documents.
    Returns the answer to the query.
    """
    try:
        answer = await orchestrator_service.query(
            request.query,
            request.documents_folder
        )
        
        return QueryResponse(
            query=request.query,
            answer=answer
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_indexing_status(job_id: str):
    """Get the status of an indexing job."""
    status = orchestrator_service.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status
