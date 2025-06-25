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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# The model is now baked into the Docker image at this specific path.
MODEL_PATH = "/app/PhotonicFusionSDXL_V3-diffusers-fixed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global pipeline variable
pipeline = None

def load_model() -> StableDiffusionXLPipeline:
    """
    Loads the StableDiffusionXLPipeline from the fixed path inside the Docker image.
    This function is called once when the handler is initialized.
    """
    global pipeline
    if pipeline is not None:
        logger.info("✅ Model pipeline already loaded.")
        return pipeline

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Critical Error: Baked-in model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model directory not found inside the image at {MODEL_PATH}")

    logger.info(f"🚀 Loading model from baked-in path: {MODEL_PATH}")
    logger.info(f"⚙️ Using device: {DEVICE}")

    try:
        loaded_pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True  # Ensure it only loads from the local, baked-in path
        )
        
        loaded_pipeline.scheduler = EulerDiscreteScheduler.from_config(loaded_pipeline.scheduler.config)
        loaded_pipeline.to(DEVICE)
        
        # Memory optimizations for CUDA
        if DEVICE == "cuda":
            loaded_pipeline.enable_attention_slicing()
            try:
                loaded_pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers memory efficient attention enabled.")
            except ImportError:
                logger.warning("⚠️ XFormers is not available. For optimal performance, consider installing it.")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable XFormers. Reason: {e}")

        pipeline = loaded_pipeline
        logger.info("✅ Model pipeline loaded successfully.")
        return pipeline

    except Exception as e:
        logger.exception(f"❌ Failed to load the model pipeline. Error: {e}")
        # Clear any partially loaded state
        if 'loaded_pipeline' in locals():
            del loaded_pipeline
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        raise  # Re-raise the exception to fail the pod initialization

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    The main handler function for the RunPod serverless worker.
    """
    global pipeline
    
    # Initialize the model on the first call
    if pipeline is None:
        load_model()

    job_input = event.get("input", {})
    
    # --- Input Validation ---
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A 'prompt' is required in the input."}

    # --- Set Generation Parameters ---
    params = {
        "prompt": prompt,
        "negative_prompt": job_input.get("negative_prompt"),
        "width": job_input.get("width", 1024),
        "height": job_input.get("height", 1024),
        "num_inference_steps": job_input.get("num_inference_steps", 30),
        "guidance_scale": job_input.get("guidance_scale", 7.5),
        "num_images_per_prompt": job_input.get("num_images_per_prompt", 1),
        "seed": job_input.get("seed")
    }
    
    logger.info(f"Processing job with params: {params}")
    start_time = time.time()

    generator = None
    if params["seed"] is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(params["seed"])

    # --- Image Generation ---
    try:
        with torch.no_grad():
            images = pipeline(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                width=params["width"],
                height=params["height"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["num_images_per_prompt"],
                generator=generator
            ).images
    
    except Exception as e:
        logger.exception(f"❌ Image generation failed. Error: {e}")
        return {"error": f"An error occurred during image generation: {e}"}

    # --- Process and Return Images ---
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

# --- RunPod Entrypoint ---
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless worker...")
    load_model()  # Pre-load the model on start
    runpod.serverless.start({"handler": handler}) 