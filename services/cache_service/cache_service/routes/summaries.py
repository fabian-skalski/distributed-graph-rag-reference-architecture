"""Summary-related routes."""
from fastapi import APIRouter, Depends, HTTPException
from cache_service.models import SummariesSaveRequest, SummariesResponse, ElementsSaveRequest
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(tags=["summaries"])


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


@router.post("/elements")
async def save_elements(
    request: ElementsSaveRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save extracted elements for a document.
    
    Args:
        request: Elements save request
        neo4j: Neo4j service instance
        
    Returns:
        Success status with element count
    """
    try:
        neo4j.save_elements(request.document_id, request.elements)
        return {"status": "success", "count": len(request.elements)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summaries")
async def save_summaries(
    request: SummariesSaveRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save summaries for a document.
    
    Args:
        request: Summaries save request
        neo4j: Neo4j service instance
        
    Returns:
        Success status with summary count
    """
    try:
        neo4j.save_summaries(request.document_id, request.summaries)
        return {"status": "success", "count": len(request.summaries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summaries/{document_id}", response_model=SummariesResponse)
async def get_summaries(
    document_id: str,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Get summaries for a document.
    
    Args:
        document_id: Document identifier
        neo4j: Neo4j service instance
        
    Returns:
        Summaries response
    """
    try:
        summaries = neo4j.get_summaries(document_id)
        return SummariesResponse(summaries=summaries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
