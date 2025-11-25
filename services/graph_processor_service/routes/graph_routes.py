"""Graph processing routes."""
import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.graph_service import GraphService

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class BuildGraphRequest(BaseModel):
    summaries: List[str]


class Community(BaseModel):
    community_id: int
    members: List[str]
    size: int


class GraphResponse(BaseModel):
    nodes: int
    edges: int
    communities: List[Community]


class CommunityDescriptionRequest(BaseModel):
    community_members: List[str]


# Initialize service
graph_service = GraphService()


@router.post("/graph/build", response_model=GraphResponse)
async def build_and_analyze_graph(request: BuildGraphRequest):
    """Build graph and detect communities.
    
    Args:
        request: Request containing summaries to build graph from
        
    Returns:
        Response with graph statistics and detected communities
    """
    try:
        # Build graph
        graph_stats = graph_service.build_graph(request.summaries)
        
        # Detect communities
        communities_data = graph_service.detect_communities()
        
        communities = [
            Community(**community) for community in communities_data
        ]
        
        return GraphResponse(
            nodes=graph_stats["nodes"],
            edges=graph_stats["edges"],
            communities=communities
        )
        
    except Exception as e:
        logger.exception("Error building graph")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/community/describe")
async def describe_community(request: CommunityDescriptionRequest):
    """Get detailed description of a community.
    
    Args:
        request: Request containing community members
        
    Returns:
        Dictionary with entities and relationships
    """
    try:
        description = graph_service.get_community_description(
            request.community_members
        )
        return description
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error describing community")
        raise HTTPException(status_code=500, detail=str(e))
