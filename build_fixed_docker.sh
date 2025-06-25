#!/bin/bash
# Build script for fixed PhotonicFusion SDXL Docker image

echo "🔧 PhotonicFusion SDXL - Building Fixed Docker Image"
echo "=================================================="

# Set script to exit on any error
set -e

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Tag for the image
IMAGE_TAG="photonicfusion-sdxl:fixed"

echo "📋 Building with fixes for:"
echo "  - device_map='auto' error"
echo "  - protobuf dependency missing"
echo "  - improved error handling"
echo "  - multi-image support"

echo ""
echo "🔨 Building Docker image..."
docker build -t $IMAGE_TAG .

echo ""
echo "✅ Build complete!"
echo "   Image: $IMAGE_TAG"

echo ""
echo "🧪 Testing the image..."
echo "   You can test the image with:"
echo "   docker run --rm -it $IMAGE_TAG python test_protobuf_fix.py"

echo ""
echo "🚀 Deployment instructions:"
echo "   1. Tag the image for your registry:"
echo "      docker tag $IMAGE_TAG your-registry/photonicfusion-sdxl:latest"
echo ""
echo "   2. Push to your registry:"
echo "      docker push your-registry/photonicfusion-sdxl:latest"
echo ""
echo "   3. Update your RunPod endpoint with the new image"
echo ""
echo "   4. Test with a simple prompt to verify fixes" 