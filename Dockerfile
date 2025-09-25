# Multi-stage build for Edge Vision AI Platform
# Optimized for both x86_64 and ARM64 (Jetson, RPi)

FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/config /app/logs /app/models /app/recordings

# Download YOLOv8 model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Default command
CMD ["python", "src/main.py"]

# ===== GPU Support Stage (NVIDIA) =====
FROM base AS gpu

# Install CUDA runtime (for NVIDIA GPUs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-8 \
    cuda-cudart-11-8 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ===== Jetson-specific Stage =====
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS jetson

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies (Jetson-optimized)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    opencv-python \
    ultralytics \
    flask \
    numpy \
    pyyaml

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/config /app/logs /app/models /app/recordings

# Download model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8080

CMD ["python", "src/main.py"]

# ===== Development Stage =====
FROM base AS development

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    ipython \
    jupyter

# Enable hot reload
ENV FLASK_ENV=development \
    FLASK_DEBUG=1

CMD ["python", "src/main.py", "--debug"]

# ===== Production Stage (default) =====
FROM base AS production

# Create non-root user
RUN groupadd -r edgevision && useradd -r -g edgevision edgevision

# Change ownership
RUN chown -R edgevision:edgevision /app

# Switch to non-root user
USER edgevision

CMD ["python", "src/main.py"]