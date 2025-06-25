#!/bin/bash
# Quick fix deployment script for PhotonicFusion SDXL RunPod handler

echo "🔧 PhotonicFusion SDXL - Quick Fix Deployment"
echo "============================================="

# Set script to exit on any error
set -e

echo "📦 Building updated Docker image..."
docker build -t photonicfusion-sdxl:latest .

echo "🚀 Pushing to registry..."
# Replace with your actual registry
# docker tag photonicfusion-sdxl:latest your-registry/photonicfusion-sdxl:latest
# docker push your-registry/photonicfusion-sdxl:latest

echo "✅ Fix applied! Key changes:"
echo "  - Removed device_map='auto' (not supported)"
echo "  - Added support for multiple images per prompt"
echo "  - Fixed model loading for volume and HF Hub"

echo ""
echo "📋 To deploy to RunPod:"
echo "  1. Update your endpoint with the new image"
echo "  2. Test with a simple prompt"
echo "  3. Check logs for successful model loading"

echo ""
echo "🧪 Test the fix locally:"
echo "  python test_local.py"

echo ""
echo "�� Deployment ready!" 