#!/bin/bash
# build-docker.sh
# Helper script to build Docker image with proper tags

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Building Daemon RAG Agent Docker Image${NC}"
echo -e "${GREEN}=====================================${NC}"

# Get version from config or use default
VERSION="${VERSION:-v4}"
IMAGE_NAME="${IMAGE_NAME:-daemon-rag-agent}"

echo -e "${YELLOW}Image name: ${IMAGE_NAME}${NC}"
echo -e "${YELLOW}Version: ${VERSION}${NC}"
echo ""

# Detect container runtime (prefer docker, fallback to podman)
if command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
elif command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    echo -e "${YELLOW}Using Podman (Docker-compatible)${NC}"
else
    echo -e "${RED}Error: Neither docker nor podman found${NC}"
    exit 1
fi

# Build the image
echo -e "${GREEN}Building image...${NC}"
$CONTAINER_CMD build \
    --tag "${IMAGE_NAME}:${VERSION}" \
    --tag "${IMAGE_NAME}:latest" \
    --file Dockerfile \
    .

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "Tagged images:"
echo "  - ${IMAGE_NAME}:${VERSION}"
echo "  - ${IMAGE_NAME}:latest"
echo ""
echo "To run the image:"
echo "  $CONTAINER_CMD run -p 7860:7860 --env-file .env ${IMAGE_NAME}:latest"
echo ""
if [ "$CONTAINER_CMD" = "podman" ]; then
    echo "To run with podman-compose:"
    echo "  podman-compose up -d"
else
    echo "To run with docker-compose:"
    echo "  docker-compose up -d"
fi
echo ""
