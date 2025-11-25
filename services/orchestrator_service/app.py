"""Orchestrator FastAPI microservice."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.orchestrator_routes import router as orchestrator_router
from routes.health_routes import router as health_router
from middleware import SecurityHeadersMiddleware

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan.
    
    Handles initialization and shutdown of the orchestrator service,
    logging startup and shutdown events.
    
    Args:
        app: FastAPI application instance.
        
    Yields:
        None: Control to the application during its lifetime.
    """
    # Startup
    logger.info("Orchestrator service started and ready to accept requests")
    yield
    # Shutdown (if needed)
    logger.info("Orchestrator service shutting down")


# Create FastAPI app
app = FastAPI(
    title="Graph RAG Orchestrator Service",
    description="Orchestrates the Graph RAG pipeline for document indexing and querying",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(orchestrator_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
