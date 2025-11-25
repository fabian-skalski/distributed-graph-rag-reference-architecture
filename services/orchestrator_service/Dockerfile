FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy orchestrator service files
COPY services/orchestrator_service/pyproject.toml ./
COPY services/orchestrator_service/clients/ ./clients/
COPY services/orchestrator_service/routes/ ./routes/
COPY services/orchestrator_service/services/ ./services/
COPY services/orchestrator_service/distributed_orchestrator.py ./
COPY services/orchestrator_service/app.py ./

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Expose port
EXPOSE 8000

CMD ["python", "app.py"]
