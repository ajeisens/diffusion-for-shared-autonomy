# Dockerfile for KTO Lunar Lander Demo
# Python 3.10 + Drake + Pygame
#
# Build: docker build -t lunar-lander-kto .
# Run (Linux/macOS): docker-compose up
# Run (Windows): docker run -e DISPLAY=host.docker.internal:0 lunar-lander-kto

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    # Graphics and display
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # X11 for pygame
    libx11-6 \
    libxrandr2 \
    libxi6 \
    libxcursor1 \
    libxinerama1 \
    libxxf86vm1 \
    # Audio (pygame)
    libasound2-dev \
    libpulse-dev \
    # SDL for pygame
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Drake first (large download, ~800MB)
RUN pip install --no-cache-dir drake==1.34.0

# Install other Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pygame==2.5.2 \
    gym==0.26.2 \
    torch==2.2.0

# Copy project files
COPY . /app

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Configure pygame for headless rendering (fallback if no display)
ENV SDL_VIDEODRIVER=x11
ENV SDL_AUDIODRIVER=dummy

# Create directory for recorded episodes
RUN mkdir -p /app/recorded_episodes

# Default command
CMD ["python", "play.py"]
