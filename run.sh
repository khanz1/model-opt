#!/bin/bash

# Build and run the model-opt Docker container

echo "Building Docker image..."
docker build -t model-opt .

echo "Stopping existing container if running..."
docker stop model-opt-container 2>/dev/null || true
docker rm model-opt-container 2>/dev/null || true

echo "Creating directories for volumes..."
mkdir -p ./static/uploads
mkdir -p ./models

echo "Starting model-opt container..."
docker run \
  --name model-opt-container \
  --restart unless-stopped \
  -e PORT=8000 \
  -p 8000:8000 \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -v $(pwd)/models:/app/models \
  model-opt

echo "Container started successfully!"
echo "Application available at: http://localhost:8000"
echo ""
echo "To view logs: docker logs model-opt-container"
echo "To stop: docker stop model-opt-container" 