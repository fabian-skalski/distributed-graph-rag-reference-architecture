"""Graph caching routes."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from cache_service.models import (
    GraphSaveRequest, 
    GraphResponse,
    CommunityDescriptionsSaveRequest,
    CommunityDescriptionsResponse
)
from cache_service.services.neo4j_service import Neo4jService

router = APIRouter(tags=["graph"])


# Additional models for community summaries
class CommunitySummariesSaveRequest(BaseModel):
    summaries_hash: str
    summaries: List[str]


class CommunitySummariesResponse(BaseModel):
    summaries: List[str]


def get_neo4j_service() -> Neo4jService:
    """Dependency to get Neo4j service."""
    from cache_service.app import neo4j_service
    return neo4j_service


@router.post("/graph")
async def save_graph(
    request: GraphSaveRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save graph structure and communities.
    
    Args:
        request: Graph save request
        neo4j: Neo4j service instance
        
    Returns:
        Success status with graph info
    """
    try:
        neo4j.save_graph(
            request.summaries_hash,
            request.nodes,
            request.edges,
            request.communities
        )
        return {
            "status": "success",
            "nodes": request.nodes,
            "edges": request.edges,
            "communities": len(request.communities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{summaries_hash}", response_model=GraphResponse)
async def get_graph(
    summaries_hash: str,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Get cached graph structure.
    
    Args:
        summaries_hash: Hash of summaries used to build graph
        neo4j: Neo4j service instance
        
    Returns:
        Graph structure if cached, 404 otherwise
    """
    try:
        graph_data = neo4j.get_graph(summaries_hash)
        if graph_data is None:
            raise HTTPException(status_code=404, detail="Graph not found in cache")
        return GraphResponse(**graph_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/community-descriptions")
async def save_community_descriptions(
    request: CommunityDescriptionsSaveRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save community descriptions.
    
    Args:
        request: Community descriptions save request
        neo4j: Neo4j service instance
        
    Returns:
        Success status with count
    """
    try:
        neo4j.save_community_descriptions(
            request.summaries_hash,
            request.descriptions
        )
        return {
            "status": "success",
            "count": len(request.descriptions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community-descriptions/{summaries_hash}", response_model=CommunityDescriptionsResponse)
async def get_community_descriptions(
    summaries_hash: str,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Get cached community descriptions.
    
    Args:
        summaries_hash: Hash of summaries
        neo4j: Neo4j service instance
        
    Returns:
        Community descriptions if cached, 404 otherwise
    """
    try:
        descriptions = neo4j.get_community_descriptions(summaries_hash)
        if descriptions is None:
            raise HTTPException(status_code=404, detail="Community descriptions not found in cache")
        return CommunityDescriptionsResponse(descriptions=descriptions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/community-summaries")
async def save_community_summaries(
    request: CommunitySummariesSaveRequest,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Save community summaries.
    
    Args:
        request: Community summaries save request
        neo4j: Neo4j service instance
        
    Returns:
        Success status with count
    """
    try:
        neo4j.save_community_summaries(
            request.summaries_hash,
            request.summaries
        )
        return {
            "status": "success",
            "count": len(request.summaries)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community-summaries/{summaries_hash}", response_model=CommunitySummariesResponse)
async def get_community_summaries(
    summaries_hash: str,
    neo4j: Neo4jService = Depends(get_neo4j_service)
):
    """Get cached community summaries.
    
    Args:
        summaries_hash: Hash of summaries
        neo4j: Neo4j service instance
        
    Returns:
        Community summaries if cached, 404 otherwise
    """
    try:
        summaries = neo4j.get_community_summaries(summaries_hash)
        if summaries is None:
            raise HTTPException(status_code=404, detail="Community summaries not found in cache")
        return CommunitySummariesResponse(summaries=summaries)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
