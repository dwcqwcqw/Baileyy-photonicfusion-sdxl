"""
RunPod Serverless Handler for PhotonicFusion SDXL
"""

import runpod
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import base64
from io import BytesIO
from PIL import Image
import os
import gc
import logging
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and pipeline
pipeline = None
device = None

def load_model():
    """Load the PhotonicFusion SDXL model with intelligent path detection"""
    global pipeline, device
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Define potential model paths in order of preference
    model_paths = [
        "/runpod-volume/photonicfusion-sdxl",  # RunPod volume - primary
        "Baileyy/photonicfusion-sdxl",  # Hugging Face Hub - fallback
        "stabilityai/stable-diffusion-xl-base-1.0"  # Official SDXL - last resort
    ]
    
    pipeline = None
    last_error = None
    
    for i, model_path in enumerate(model_paths):
        try:
            logger.info(f"📁 Attempting to load model from: {model_path}")
            
            # For local paths, verify the correct diffusers structure
            if model_path.startswith("/"):
                if not os.path.exists(model_path):
                    logger.warning(f"⚠️ Local path {model_path} does not exist, skipping...")
                    continue
                
                # Check for diffusers model structure (not single model.safetensors)
                required_components = {
                    "model_index.json": os.path.join(model_path, "model_index.json"),
                    "unet": os.path.join(model_path, "unet"),
                    "vae": os.path.join(model_path, "vae"),
                    "text_encoder": os.path.join(model_path, "text_encoder"),
                    "text_encoder_2": os.path.join(model_path, "text_encoder_2"),
                    "scheduler": os.path.join(model_path, "scheduler")
                }
                
                missing_components = []
                for component_name, component_path in required_components.items():
                    if not os.path.exists(component_path):
                        missing_components.append(component_name)
                
                if missing_components:
                    logger.warning(f"⚠️ Missing required components in {model_path}: {missing_components}")
                    continue
                
                # Check for either standard or fp16 model files in text encoders
                text_encoder_standard = os.path.join(model_path, "text_encoder", "model.safetensors")
                text_encoder_fp16 = os.path.join(model_path, "text_encoder", "model.fp16.safetensors")
                text_encoder_2_standard = os.path.join(model_path, "text_encoder_2", "model.safetensors")
                text_encoder_2_fp16 = os.path.join(model_path, "text_encoder_2", "model.fp16.safetensors")
                
                # Check if either standard or fp16 versions exist
                if not (os.path.exists(text_encoder_standard) or os.path.exists(text_encoder_fp16)):
                    logger.warning(f"⚠️ Missing text_encoder model files (both standard and fp16) in {model_path}")
                    continue
                    
                if not (os.path.exists(text_encoder_2_standard) or os.path.exists(text_encoder_2_fp16)):
                    logger.warning(f"⚠️ Missing text_encoder_2 model files (both standard and fp16) in {model_path}")
                    continue
                
                # Log which version we found
                te1_version = "fp16" if os.path.exists(text_encoder_fp16) else "standard"
                te2_version = "fp16" if os.path.exists(text_encoder_2_fp16) else "standard"
                logger.info(f"✅ Found text_encoder ({te1_version}) and text_encoder_2 ({te2_version}) in {model_path}")                
                logger.info(f"✅ Verified complete diffusers model structure at {model_path}")
            
            # Load the pipeline with proper error handling
            logger.info(f"🔄 Loading StableDiffusionXLPipeline from {model_path}...")
            
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None,
                local_files_only=model_path.startswith("/")  # Only use local files for volume paths
            )
            
            # Set scheduler
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            
            # Move to device and optimize
            pipeline = pipeline.to(device)
            
            if device == "cuda":
                # Enable memory optimizations
                pipeline.enable_attention_slicing()
                pipeline.enable_model_cpu_offload()
                
                # Try to enable xformers if available
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ XFormers memory efficient attention enabled")
                except Exception as e:
                    logger.info(f"ℹ️ XFormers not available: {e}")
            
            logger.info(f"✅ Successfully loaded model from: {model_path}")
            return pipeline
            
        except Exception as e:
            last_error = e
            logger.error(f"❌ Failed to load from {model_path}: {str(e)}")
            
            # Clean up any partially loaded model
            if pipeline is not None:
                del pipeline
                pipeline = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue
    
    # If we get here, all attempts failed
    raise RuntimeError(f"❌ Failed to load model from all sources. Last error: {last_error}")

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