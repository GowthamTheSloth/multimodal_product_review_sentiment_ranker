# Multi-stage / optimized slim Python runtime for DS37 inference service
FROM python:3.11-slim

# Set environment variables for Python and Hugging Face cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface

# Install system dependencies required by XGBoost and health check tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# CPU-only torch from the PyTorch CPU index. Default PyPI torch Linux wheels
# pull NVIDIA/CUDA packages (nvidia-cublas, nvidia-cudnn, etc.).
# Strip torch from requirements-prod.txt and constrain recursive deps (accelerate,
# transformers, …) so pip cannot replace 2.13.0+cpu with PyPI CUDA torch.
COPY requirements-prod.txt /app/requirements-prod.txt
RUN pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^[[:space:]]*torch==' /app/requirements-prod.txt > /tmp/requirements.notorch.txt && \
    printf '%s\n' 'torch==2.13.0+cpu' > /tmp/torch-constraint.txt && \
    pip install --no-cache-dir -c /tmp/torch-constraint.txt -r /tmp/requirements.notorch.txt

# Create a non-root user (appuser) and configure cache permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app

# Copy production model artifacts and source code into container
COPY models /app/models
COPY src /app/src

# Set ownership to appuser
RUN chown -R appuser:appuser /app/models /app/src /app/.cache

# Switch to non-root user for security
USER appuser

# Expose API port
EXPOSE 8000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
