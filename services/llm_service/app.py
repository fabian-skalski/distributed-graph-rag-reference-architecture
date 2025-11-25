"""LLM Operations FastAPI microservice."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes.llm_routes import router as llm_router, llm_service
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
    
    Handles initialization and shutdown of the LLM operations service,
    including setting up the LLM service client.
    
    Args:
        app: FastAPI application instance.
        
    Yields:
        None: Control to the application during its lifetime.
    """
    # Startup
    llm_service.initialize()
    logger.info("LLM operations service started")
    yield
    # Shutdown (if needed)
    logger.info("LLM operations service shutting down")


# Create FastAPI app
app = FastAPI(
    title="LLM Operations Service",
    description="LLM-based extraction, summarization, and query answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(llm_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
