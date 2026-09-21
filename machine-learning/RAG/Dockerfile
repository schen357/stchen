# Use the official uv image for caching and fast installation
FROM astralsh/uv:latest AS uv

# Use a stable, lightweight Python slim image
FROM python:3.11-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for FAISS and PDF/document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition files first to leverage Docker layer caching
# (Assumes you are using a standard pyproject.toml or uv.lock setup)
COPY pyproject.toml uv.lock ./

# Install dependencies into the system environment using uv
RUN uv sync --frozen --no-cache

# # Expose the application port (assuming FastAPI/Streamlit on 8000)
# EXPOSE 8000

# # Start the application (Change main:app to match your entry point)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]