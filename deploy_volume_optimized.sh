#!/bin/bash

# PhotonicFusion SDXL RunPod Deployment Script (Volume Optimized)
# This version is optimized for Volume usage to avoid disk space issues

set -e

echo "🚀 PhotonicFusion SDXL RunPod Deployment (Volume Optimized)"
echo "============================================================"

# Build Volume optimized Docker image
echo "📦 Building Volume optimized Docker image..."
docker build -f Dockerfile.volume_optimized -t baileyy/photonicfusion-sdxl:volume-optimized .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Push to registry
echo "📤 Pushing to Docker registry..."
docker push baileyy/photonicfusion-sdxl:volume-optimized

if [ $? -eq 0 ]; then
    echo "✅ Docker image pushed successfully"
else
    echo "❌ Docker push failed"
    exit 1
fi

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📋 RunPod Configuration (Volume Optimized):"
echo "   Docker Image: baileyy/photonicfusion-sdxl:volume-optimized"
echo "   Volume Mount: /runpod-volume"
echo "   Required Volume: photonicfusion-models"
echo ""
echo "🔧 Key Optimizations:"
echo "   ✅ No fallback downloads (Volume only)"
echo "   ✅ FP16 variant support with graceful fallback"
echo "   ✅ Enhanced error handling and logging"
echo "   ✅ Memory optimization enabled"
echo ""
echo "⚠️  Important Notes:"
echo "   • Ensure Volume 'photonicfusion-models' is properly mounted"
echo "   • Model files must be in diffusers format at /runpod-volume/photonicfusion-sdxl"
echo "   • This version will NOT download fallback models (saves disk space)"
echo ""
echo "🧪 Test your deployment:"
echo "   curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -H 'Authorization: Bearer YOUR_API_KEY' \\"
echo "        -d '{\"input\": {\"prompt\": \"a beautiful sunset over mountains\"}}'" 