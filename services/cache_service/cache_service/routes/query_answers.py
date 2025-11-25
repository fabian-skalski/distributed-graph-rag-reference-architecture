"""Query answer caching routes."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(prefix="/query-answers", tags=["query-answers"])


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


class QueryAnswerRequest(BaseModel):
    """Request model for saving query answer."""
    query_hash: str
    answer: str


@router.post("")
async def save_query_answer(
    request: QueryAnswerRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save a query answer to cache.
    
    Args:
        request: Query answer data
        neo4j: Neo4j service instance
        
    Returns:
        Success status
    """
    try:
        neo4j.save_query_answer(request.query_hash, request.answer)
        return {"status": "success", "message": "Query answer saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{query_hash}")
async def get_query_answer(
    query_hash: str,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Get cached query answer.
    
    Args:
        query_hash: Hash of the query+documents
        neo4j: Neo4j service instance
        
    Returns:
        Cached answer if found
        
    Raises:
        404: If answer not found in cache
    """
    try:
        answer = neo4j.get_query_answer(query_hash)
        if answer is None:
            raise HTTPException(status_code=404, detail="Query answer not found in cache")
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
