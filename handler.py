"""
RunPod Serverless Handler for PhotonicFusion SDXL
"""

import runpod
import torch
from diffusers import StableDiffusionXLPipeline
import base64
from io import BytesIO
import os
import time
from typing import Dict, Any, Optional

# Global variables for model caching
pipeline = None
device = None

def load_model():
    """Load the PhotonicFusion SDXL model"""
    global pipeline, device
    
    if pipeline is not None:
        return pipeline
    
    print("Loading PhotonicFusion SDXL model...")
    start_time = time.time()
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    
    print(f"Using device: {device}")
    
    try:
        # Load from Hugging Face Hub
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "Baileyy/photonicfusion-sdxl",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            device_map="auto" if device == "cuda" else None
        )
        
        # Move to device if not using device_map
        if device == "cuda" and pipeline.device != torch.device("cuda"):
            pipeline = pipeline.to(device)
        
        # Enable memory optimizations for GPU
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_model_cpu_offload()
            
            # Try to enable xformers if available
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ XFormers enabled")
            except ImportError:
                print("⚠️ XFormers not available, using default attention")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e

def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> str:
    """
    Generate image and return base64 encoded string
    """
    global pipeline
    
    if pipeline is None:
        pipeline = load_model()
    
    print(f"Generating image with prompt: '{prompt[:50]}...'")
    start_time = time.time()
    
    # Set random seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image
    with torch.no_grad():
        try:
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            
            image = result.images[0]
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️ CUDA OOM, trying with CPU offload")
            torch.cuda.empty_cache()
            
            # Retry with more aggressive memory management
            pipeline.enable_model_cpu_offload()
            
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=min(width, 768),  # Reduce resolution if OOM
                height=min(height, 768),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            
            image = result.images[0]
    
    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    generation_time = time.time() - start_time
    print(f"✅ Image generated in {generation_time:.2f} seconds")
    
    return img_base64

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    """
    try:
        # Extract input parameters
        input_data = event.get("input", {})
        
        # Required parameters
        prompt = input_data.get("prompt")
        if not prompt:
            return {
                "error": "Missing required parameter: prompt"
            }
        
        # Optional parameters with defaults
        negative_prompt = input_data.get("negative_prompt", "blurry, low quality, distorted, ugly")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        seed = input_data.get("seed")
        
        # Validate parameters
        width = max(512, min(width, 1536))  # Clamp to reasonable range
        height = max(512, min(height, 1536))
        num_inference_steps = max(10, min(num_inference_steps, 100))
        guidance_scale = max(1.0, min(guidance_scale, 20.0))
        
        print(f"Request parameters:")
        print(f"  Prompt: {prompt}")
        print(f"  Negative: {negative_prompt}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Seed: {seed}")
        
        # Generate image
        image_base64 = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        return {
            "image": image_base64,
            "prompt": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        }
        
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    # For local testing
    print("Starting PhotonicFusion SDXL RunPod Handler...")
    
    # Load model once
    load_model()
    
    # Start RunPod serverless
    runpod.serverless.start({"handler": handler}) 