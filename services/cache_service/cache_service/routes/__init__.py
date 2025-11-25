"""API routes for cache service."""
from fastapi import APIRouter

router = APIRouter()

# Re-export sub-routers
from .documents import router as documents_router
from .chunks import router as chunks_router  
from .summaries import router as summaries_router
from .health import router as health_router
from .graph import router as graph_router
from .admin import router as admin_router
from .query_answers import router as query_answers_router

__all__ = ["router", "documents_router", "chunks_router", "summaries_router", "health_router", "graph_router", "admin_router", "query_answers_router"]
