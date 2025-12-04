# ESP-CSI-Toolkit Docker Image
# Uses Python 3.11 for TensorFlow/PyTorch compatibility

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy project files needed for installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with all extras
RUN pip install ".[ml,advanced]"

# Create directories for data and models (to be mounted as volumes)
RUN mkdir -p /app/data /app/models /app/output /app/processed

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Default command
ENTRYPOINT ["python", "-m", "csi_toolkit"]
CMD ["--help"]
