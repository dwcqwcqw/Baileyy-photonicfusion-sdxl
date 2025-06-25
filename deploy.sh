#!/bin/bash

# PhotonicFusion SDXL RunPod Deployment Script

set -e

# Configuration
IMAGE_NAME="photonicfusion-sdxl"
REGISTRY_URL="your-registry.com"  # Update this with your registry
VERSION="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 PhotonicFusion SDXL RunPod Deployment Script${NC}"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}📦 Building Docker image...${NC}"
    docker build -t $IMAGE_NAME:$VERSION .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Image built successfully: $IMAGE_NAME:$VERSION${NC}"
    else
        echo -e "${RED}❌ Failed to build image${NC}"
        exit 1
    fi
}

# Main execution
case "${1:-help}" in
    "build")
        build_image
        ;;
    "help")
        echo "Usage: $0 [build|help]"
        echo ""
        echo "Commands:"
        echo "  build  - Build Docker image"
        echo "  help   - Show this help message"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Deployment script completed!${NC}"
