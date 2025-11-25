"""LLM operations routes."""
import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ExtractRequest(BaseModel):
    chunks: List[str]
    system_prompt: str = None
    temperature: float = None
    max_tokens: int = None


class SummarizeElementsRequest(BaseModel):
    elements: List[str]
    system_prompt: str = None
    temperature: float = None
    max_tokens: int = None


class SummarizeCommunitiesRequest(BaseModel):
    descriptions: List[Dict[str, Any]]
    system_prompt: str = None
    temperature: float = None
    max_tokens: int = None


class AnswerQueryRequest(BaseModel):
    summaries: List[str]
    query: str
    system_prompt: str = None
    temperature: float = None
    max_tokens: int = None


class CombineAnswersRequest(BaseModel):
    intermediate_answers: List[str]
    system_prompt: str = None
    temperature: float = None
    max_tokens: int = None


# Initialize service
llm_service = LLMService()


@router.post("/extract")
async def extract_elements(request: ExtractRequest):
    """Extract entities and relationships from text chunks.
    
    Args:
        request: Request containing chunks to process
        
    Returns:
        Dictionary with extracted elements
    """
    try:
        elements = await llm_service.extract_elements(
            request.chunks,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"elements": elements}
        
    except Exception as e:
        logger.exception("Error extracting elements")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize/elements")
async def summarize_elements(request: SummarizeElementsRequest):
    """Summarize extracted elements.
    
    Args:
        request: Request containing elements to summarize
        
    Returns:
        Dictionary with summaries
    """
    try:
        summaries = await llm_service.summarize_elements(
            request.elements,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"summaries": summaries}
        
    except Exception as e:
        logger.exception("Error summarizing elements")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize/communities")
async def summarize_communities(request: SummarizeCommunitiesRequest):
    """Summarize communities in the knowledge graph.
    
    Args:
        request: Request containing community descriptions
        
    Returns:
        Dictionary with community summaries
    """
    try:
        summaries = await llm_service.summarize_communities(
            request.descriptions,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"summaries": summaries}
        
    except Exception as e:
        logger.exception("Error summarizing communities")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/answer")
async def answer_query(request: AnswerQueryRequest):
    """Generate answers from community summaries.
    
    Args:
        request: Request containing summaries and query
        
    Returns:
        Dictionary with intermediate answers
    """
    try:
        answers = await llm_service.answer_query(
            request.summaries,
            request.query,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"answers": answers}
        
    except Exception as e:
        logger.exception("Error generating answers")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/combine")
async def combine_answers(request: CombineAnswersRequest):
    """Combine intermediate answers into final answer.
    
    Args:
        request: Request containing intermediate answers
        
    Returns:
        Dictionary with final answer
    """
    try:
        answer = await llm_service.combine_answers(
            request.intermediate_answers,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"answer": answer}
        
    except Exception as e:
        logger.exception("Error combining answers")
        raise HTTPException(status_code=500, detail=str(e))
