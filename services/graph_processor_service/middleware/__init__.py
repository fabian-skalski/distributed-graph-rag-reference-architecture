"""Middleware package for graph processor service."""
from .security import SecurityHeadersMiddleware

__all__ = ["SecurityHeadersMiddleware"]
