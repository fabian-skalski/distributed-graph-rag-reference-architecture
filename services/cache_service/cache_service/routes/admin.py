"""Administrative routes."""
from fastapi import APIRouter, Depends, HTTPException
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(prefix="/admin", tags=["admin"])


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


@router.delete("/clear-all")
async def clear_all(
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Clear all data from Neo4j database.
    
    WARNING: This will delete ALL documents, chunks, summaries, 
    graph data, and communities from the database.
    
    Args:
        neo4j: Neo4j service instance
        
    Returns:
        Success status
    """
    try:
        neo4j.clear_all()
        return {
            "status": "success", 
            "message": "All data cleared from Neo4j database"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
