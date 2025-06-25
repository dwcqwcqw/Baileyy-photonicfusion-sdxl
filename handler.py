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

# --- Configuration ---
# The model path in the RunPod volume.
# This should point to the correctly converted diffusers directory.
MODEL_PATH = "/runpod-volume/photonicfusion-sdxl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global pipeline variable
pipeline = None

def load_model() -> StableDiffusionXLPipeline:
    """
    Loads the StableDiffusionXLPipeline from the fixed path on the RunPod volume.
    """
    global pipeline
    if pipeline is not None:
        logger.info("✅ Model pipeline already loaded.")
        return pipeline

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Critical Error: Model not found at the specified volume path {MODEL_PATH}")
        # Add a check for the old directory to guide the user
        if os.path.exists("/runpod-volume/photonicfusion-sdxl"):
            logger.error("👉 An old directory '/runpod-volume/photonicfusion-sdxl' was found.")
            logger.error("👉 Please upload the new 'photonicfusion-sdxl' directory and remove the old one.")
        raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")

    logger.info(f"🚀 Loading model from volume: {MODEL_PATH}")
    logger.info(f"⚙️ Using device: {DEVICE}")

    try:
        loaded_pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True
        )
        
        loaded_pipeline.scheduler = EulerDiscreteScheduler.from_config(loaded_pipeline.scheduler.config)
        loaded_pipeline.to(DEVICE)
        
        # Memory optimizations
        if DEVICE == "cuda":
            loaded_pipeline.enable_attention_slicing()
            try:
                loaded_pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers memory efficient attention enabled.")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable XFormers. Reason: {e}")

        pipeline = loaded_pipeline
        logger.info("✅ Model pipeline loaded successfully from volume.")
        return pipeline

    except Exception as e:
        logger.exception(f"❌ Failed to load the model pipeline from volume. Error: {e}")
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        raise

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
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
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
    The main handler function for the RunPod serverless worker.
    """
    global pipeline
    
    try:
        if pipeline is None:
            load_model()
    except Exception as e:
        logger.exception("❌ Model loading failed during handler initialization.")
        return {"error": f"Failed to load model: {e}"}

    job_input = event.get("input", {})
    
    # Input validation
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A 'prompt' is required in the input."}

    # Set generation parameters
    params = {
        "prompt": prompt,
        "negative_prompt": job_input.get("negative_prompt"),
        "width": job_input.get("width", 1024),
        "height": job_input.get("height", 1024),
        "num_inference_steps": job_input.get("num_inference_steps", 30),
        "guidance_scale": job_input.get("guidance_scale", 7.5),
        "seed": job_input.get("seed")
    }
    
    logger.info(f"Processing job with params: {params}")
    start_time = time.time()

    generator = None
    if params["seed"] is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(params["seed"])

    # Image Generation
    try:
        with torch.no_grad():
            images = pipeline(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                width=params["width"],
                height=params["height"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                generator=generator
            ).images
    
    except Exception as e:
        logger.exception(f"❌ Image generation failed. Error: {e}")
        return {"error": f"An error occurred during image generation: {e}"}

    # Process and Return Images
    image_urls = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_urls.append(f"data:image/jpeg;base64,{img_str}")

    end_time = time.time()
    logger.info(f"✅ Job completed in {end_time - start_time:.2f} seconds.")
    
    # Clean up memory
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {"images": image_urls}

# RunPod Entrypoint
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler}) 