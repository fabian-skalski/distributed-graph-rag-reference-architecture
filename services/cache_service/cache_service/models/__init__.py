"""Pydantic models for cache service."""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


class DocumentCheckRequest(BaseModel):
    """Request to check if document is cached."""
    file_path: str
    content: str


class DocumentCheckResponse(BaseModel):
    """Response for document check."""
    cached: bool
    document_id: Optional[str] = None


class DocumentCreateRequest(BaseModel):
    """Request to create/save a document."""
    document_id: str
    file_path: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ChunksSaveRequest(BaseModel):
    """Request to save chunks."""
    document_id: str
    chunks: List[Dict[str, Any]]


class ChunksResponse(BaseModel):
    """Response with chunks."""
    chunks: List[Dict[str, Any]]


class ElementsSaveRequest(BaseModel):
    """Request to save elements."""
    document_id: str
    elements: List[Dict[str, Any]]


class SummariesSaveRequest(BaseModel):
    """Request to save summaries."""
    document_id: str
    summaries: List[Dict[str, Any]]


class SummariesResponse(BaseModel):
    """Response with summaries."""
    summaries: List[Dict[str, Any]]


class GraphSaveRequest(BaseModel):
    """Request to save graph structure."""
    summaries_hash: str
    nodes: int
    edges: int
    communities: List[Dict[str, Any]]


class GraphResponse(BaseModel):
    """Response with graph structure."""
    nodes: int
    edges: int
    communities: List[Dict[str, Any]]


class CommunityDescriptionsSaveRequest(BaseModel):
    """Request to save community descriptions."""
    summaries_hash: str
    descriptions: List[Dict[str, Any]]


class CommunityDescriptionsResponse(BaseModel):
    """Response with community descriptions."""
    descriptions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    database: str = "connected"


__all__ = [
    "DocumentCheckRequest",
    "DocumentCheckResponse",
    "DocumentCreateRequest",
    "ChunksSaveRequest",
    "ChunksResponse",
    "ElementsSaveRequest",
    "SummariesSaveRequest",
    "SummariesResponse",
    "GraphSaveRequest",
    "GraphResponse",
    "CommunityDescriptionsSaveRequest",
    "CommunityDescriptionsResponse",
    "HealthResponse",
]
