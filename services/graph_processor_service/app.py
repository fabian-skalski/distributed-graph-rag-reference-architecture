"""Graph Processing FastAPI microservice."""
import logging
import os

from fastapi import FastAPI

from routes.graph_routes import router as graph_router
from routes.health_routes import router as health_router
from middleware import SecurityHeadersMiddleware

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Graph Processing Service",
    description="Knowledge graph building and community detection",
    version="1.0.0"
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(graph_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
