#!/bin/bash
# Quick start script for KTO Lunar Lander Demo (Linux/macOS)
# Requires: Docker and X Server

set -e

echo "============================================================"
echo "KTO Lunar Lander Demo - Docker Setup"
echo "============================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

echo "[1/4] Docker is running..."

# Platform-specific X11 setup
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "[2/4] Configuring X11 for macOS..."

    # Check if XQuartz is installed
    if ! command -v xquartz &> /dev/null; then
        echo "WARNING: XQuartz not found!"
        echo "Install with: brew install --cask xquartz"
        echo "Then restart this script."
        exit 1
    fi

    # Allow localhost connections
    xhost +localhost > /dev/null 2>&1 || true
    export DISPLAY=:0

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "[2/4] Configuring X11 for Linux..."

    # Allow Docker to access display
    xhost +local:docker > /dev/null 2>&1 || true

else
    echo "[2/4] Unknown OS type, skipping X11 check..."
fi

# Build Docker image
echo "[3/4] Building Docker image (may take 5-10 minutes first time)..."
docker-compose build

# Run container
echo "[4/4] Starting container..."
echo ""
echo "============================================================"
echo "Controls:"
echo "  Arrow Up: Main engine (Teleop mode)"
echo "  Left/Right: Rotate"
echo "  R: Reset"
echo "  Q/Escape: Quit"
echo "  1: Teleop  2: Heuristic  3: KTO mode"
echo "============================================================"
echo ""

docker-compose up

echo ""
echo "Demo stopped."
