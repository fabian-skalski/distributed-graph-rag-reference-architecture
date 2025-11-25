"""Chunk-related routes."""
from fastapi import APIRouter, Depends, HTTPException
from cache_service.models import ChunksSaveRequest, ChunksResponse
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(prefix="/chunks", tags=["chunks"])


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


@router.post("")
async def save_chunks(
    request: ChunksSaveRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save chunks for a document.
    
    Args:
        request: Chunks save request
        neo4j: Neo4j service instance
        
    Returns:
        Success status with chunk count
    """
    try:
        neo4j.save_chunks(request.document_id, request.chunks)
        return {"status": "success", "count": len(request.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=ChunksResponse)
async def get_chunks(
    document_id: str,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Get chunks for a document.
    
    Args:
        document_id: Document identifier
        neo4j: Neo4j service instance
        
    Returns:
        Chunks response
    """
    try:
        chunks = neo4j.get_chunks(document_id)
        return ChunksResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
