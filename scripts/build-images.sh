#!/bin/bash

echo "Building Docker images..."

# Copy your existing modules and datasets
mkdir -p modules datasets
cp -r ../modules/* ./modules/ 2>/dev/null || true
cp -r ../datasets/* ./datasets/ 2>/dev/null || true

# Build server image
docker build -f docker/server/Dockerfile -t fl-server:latest .

# Build client image
docker build -f docker/client/Dockerfile -t fl-client:latest .

echo "Images built successfully!"
docker images | grep fl-