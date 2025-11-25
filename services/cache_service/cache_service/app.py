"""Main FastAPI application for cache service."""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from cache_service.services.neo4j_service import Neo4jService, Neo4jSettings
from cache_service.middleware import SecurityHeadersMiddleware
from cache_service.routes import (
    health_router,
    documents_router,
    chunks_router,
    summaries_router,
    graph_router,
    admin_router,
    query_answers_router
)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Neo4j service instance
neo4j_service: Neo4jService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.
    
    Manages startup and shutdown of the cache service, including
    Neo4j database connection initialization and cleanup.
    
    Args:
        app: FastAPI application instance.
        
    Yields:
        None: Control to the application during its lifetime.
    """
    global neo4j_service
    
    # Startup
    logger.info("Starting cache service...")
    settings = Neo4jSettings()
    neo4j_service = Neo4jService(settings)
    neo4j_service.connect()
    logger.info(f"Connected to Neo4j at {settings.uri}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down cache service...")
    neo4j_service.close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance with all routers registered.
    """
    app = FastAPI(
        title="Cache Service",
        description="Document caching and storage service with Neo4j",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Register routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chunks_router)
    app.include_router(summaries_router)
    app.include_router(graph_router)
    app.include_router(admin_router)
    app.include_router(query_answers_router)
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT"))
    uvicorn.run(app, host="0.0.0.0", port=port)
