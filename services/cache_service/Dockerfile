FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy service code
COPY cache_service/ ./cache_service/

EXPOSE 8001

CMD ["poetry", "run", "uvicorn", "cache_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
