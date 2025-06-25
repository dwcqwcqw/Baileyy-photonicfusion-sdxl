# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Ensure protobuf is installed correctly
RUN pip install --no-cache-dir protobuf==3.20.3

# --- Model Conversion Step ---
# Copy the model and the conversion script into the image
COPY ./model/PhotonicFusionSDXL_V3.safetensors /app/model/
COPY convert_sdxl_simple.py /app/

# Run the conversion script to create the diffusers-compatible model directory
# This step uses the pre-installed dependencies in the container to avoid local environment issues.
# The converted model will be baked directly into the Docker image.
RUN python /app/convert_sdxl_simple.py \
    /app/model/PhotonicFusionSDXL_V3.safetensors \
    /app/PhotonicFusionSDXL_V3-diffusers-fixed \
    --fp16

# Clean up the original large model file to reduce image size
RUN rm /app/model/PhotonicFusionSDXL_V3.safetensors
RUN rm /app/convert_sdxl_simple.py

# Copy the rest of the application code
COPY . .

# Create volume mount point for model files (still useful for other potential files)
RUN mkdir -p /runpod-volume

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (not really needed for RunPod serverless)
EXPOSE 8000

# Run the handler
CMD ["python", "-u", "handler.py"] 