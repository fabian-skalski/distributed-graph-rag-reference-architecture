"""Document-related routes."""
from fastapi import APIRouter, Depends, HTTPException
from cache_service.models import (
    DocumentCheckRequest,
    DocumentCheckResponse,
    DocumentCreateRequest
)
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(prefix="/documents", tags=["documents"])


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


@router.post("/check", response_model=DocumentCheckResponse)
async def check_document(
    request: DocumentCheckRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Check if a document is already cached.
    
    Args:
        request: Document check request.
        neo4j: Neo4j service instance.
        
    Returns:
        DocumentCheckResponse: Document check response with cached status and document_id.
        
    Example:
        Request:
        ```json
        {
            "file_path": "/path/to/document.txt",
            "content": "Sample content..."
        }
        ```
        
        Response:
        ```json
        {
            "cached": true,
            "document_id": "document.txt_a1b2c3d4"
        }
        ```
    """
    try:
        cached, document_id = neo4j.check_document_cached(
            request.file_path,
            request.content
        )
        return DocumentCheckResponse(cached=cached, document_id=document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def save_document(
    request: DocumentCreateRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save a new document.
    
    Args:
        request: Document creation request.
        neo4j: Neo4j service instance.
        
    Returns:
        dict: Success status with document_id.
        
    Example:
        Request:
        ```json
        {
            "document_id": "document.txt_a1b2c3d4",
            "file_path": "/path/to/document.txt",
            "content": "Sample content...",
            "metadata": {"author": "John Doe"}
        }
        ```
        
        Response:
        ```json
        {
            "status": "success",
            "document_id": "document.txt_a1b2c3d4"
        }
        ```
    """
    try:
        neo4j.save_document(
            request.document_id,
            request.file_path,
            request.content,
            request.metadata
        )
        return {"status": "success", "document_id": request.document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
