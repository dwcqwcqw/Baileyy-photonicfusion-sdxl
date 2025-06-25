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
    
    # Define model paths (local volume path first, then fallback to HF Hub)
    local_model_path = "/runpod-volume/photonicfusion-sdxl"
    hf_model_name = "Baileyy/photonicfusion-sdxl"
    
    try:
        # Try loading from local volume first
        try:
            if os.path.exists(local_model_path):
                print(f"📁 Loading model from local volume: {local_model_path}")
                
                # Check if tokenizer files exist
                tokenizer_files_exist = os.path.exists(os.path.join(local_model_path, "tokenizer"))
                
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    # Skip missing files
                    ignore_mismatched_sizes=True
                )
                print("✅ Model loaded from local volume")
            else:
                print(f"📦 Local volume not found, loading from Hugging Face Hub: {hf_model_name}")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    hf_model_name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                print("✅ Model loaded from Hugging Face Hub")
        except Exception as e:
            print(f"❌ Error loading model from primary source: {str(e)}")
            print("⚠️ Attempting to load from Hugging Face Hub as fallback...")
            
            # Fallback to official SDXL model if custom model fails
            try:
                print("📦 Loading official SDXL model as fallback")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    variant="fp16"
                )
                print("✅ Fallback model loaded successfully")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {str(fallback_error)}")
                raise fallback_error
        
        # Move to device
        if device == "cuda":
            pipeline = pipeline.to(device)
            print(f"✅ Pipeline moved to {device}")
        
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
    num_images_per_prompt: int = 1,
    seed: Optional[int] = None
) -> list:
    """
    Generate images and return list of base64 encoded strings
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
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            )
            
            images = result.images
            
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
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            )
            
            images = result.images
    
    # Convert to base64
    images_base64 = []
    for i, image in enumerate(images):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        images_base64.append(img_base64)
    
    generation_time = time.time() - start_time
    print(f"✅ {len(images)} image(s) generated in {generation_time:.2f} seconds")
    
    return images_base64

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
        num_images_per_prompt = input_data.get("num_images_per_prompt", 1)
        seed = input_data.get("seed")
        
        # Validate parameters
        width = max(512, min(width, 1536))  # Clamp to reasonable range
        height = max(512, min(height, 1536))
        num_inference_steps = max(10, min(num_inference_steps, 100))
        guidance_scale = max(1.0, min(guidance_scale, 20.0))
        num_images_per_prompt = max(1, min(num_images_per_prompt, 4))  # Limit to 4 images max
        
        print(f"Request parameters:")
        print(f"  Prompt: {prompt}")
        print(f"  Negative: {negative_prompt}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Images: {num_images_per_prompt}")
        print(f"  Seed: {seed}")
        
        # Generate images
        images_base64 = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed
        )
        
        return {
            "images": images_base64,
            "prompt": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images_per_prompt,
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