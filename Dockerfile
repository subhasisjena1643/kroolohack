# Multi-stage Docker build for Classroom Engagement Analyzer
# Industry-grade precision with continuous learning capabilities

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Audio dependencies
    libasound2-dev \
    portaudio19-dev \
    # System utilities
    wget \
    curl \
    git \
    # GUI support for display (optional)
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_minimal.txt requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements_minimal.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY templates/ ./templates/
COPY tests/ ./tests/
COPY *.py ./

# Create necessary directories
RUN mkdir -p data/models data/datasets data/feedback data/external logs

# Copy setup and configuration files
COPY setup_continuous_learning.py setup_venv_simple.py test_mediapipe.py test_packages.py ./

# Set permissions
RUN chmod +x *.py

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 5001 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/api/feedback_stats')" || exit 1

# Default command
CMD ["python", "src/main.py"]
