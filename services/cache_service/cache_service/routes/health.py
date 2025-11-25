"""Health check routes."""
from fastapi import APIRouter, Depends
from cache_service.models import HealthResponse
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(tags=["health"])


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


@router.get("/health", response_model=HealthResponse)
async def health_check(neo4j: Neo4jService = Depends(get_neo4j_service)):
    """Health check endpoint.
    
    Args:
        neo4j: Neo4j service instance
        
    Returns:
        Health status response
    """
    db_status = "connected" if neo4j.health_check() else "disconnected"
    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        service="cache",
        database=db_status
    )
