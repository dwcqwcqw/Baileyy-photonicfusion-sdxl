import runpod
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import base64
from io import BytesIO
from PIL import Image
import os
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and pipeline
pipeline = None
device = None

def load_model():
    """Load the PhotonicFusion SDXL model with Volume optimization"""
    global pipeline, device
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Primary Volume path - this should always work in production
    volume_path = "/runpod-volume/photonicfusion-sdxl"
    
    # Check if Volume is available
    if not os.path.exists(volume_path):
        raise RuntimeError(f"❌ Volume not found at {volume_path}. Please ensure Volume is properly mounted.")
    
    # Verify diffusers model structure
    required_components = {
        "model_index.json": os.path.join(volume_path, "model_index.json"),
        "unet": os.path.join(volume_path, "unet"),
        "vae": os.path.join(volume_path, "vae"),
        "text_encoder": os.path.join(volume_path, "text_encoder"),
        "text_encoder_2": os.path.join(volume_path, "text_encoder_2"),
        "scheduler": os.path.join(volume_path, "scheduler")
    }
    
    missing_components = []
    for component_name, component_path in required_components.items():
        if not os.path.exists(component_path):
            missing_components.append(component_name)
    
    if missing_components:
        raise RuntimeError(f"❌ Missing required components in {volume_path}: {missing_components}")
    
    # Additional check for model.safetensors in text encoders
    text_encoder_model = os.path.join(volume_path, "text_encoder", "model.safetensors")
    text_encoder_2_model = os.path.join(volume_path, "text_encoder_2", "model.safetensors")
    
    if not os.path.exists(text_encoder_model):
        raise RuntimeError(f"❌ Missing text_encoder model.safetensors in {volume_path}")
        
    if not os.path.exists(text_encoder_2_model):
        raise RuntimeError(f"❌ Missing text_encoder_2 model.safetensors in {volume_path}")
    
    logger.info(f"✅ Verified complete diffusers model structure at {volume_path}")
    
    # Load the pipeline with fp16 fallback
    pipeline = None
    
    try:
        # First try with fp16 variant for CUDA (if available)
        if device == "cuda":
            try:
                logger.info(f"🔄 Trying fp16 variant for {volume_path}...")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    volume_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    local_files_only=True
                )
                logger.info(f"✅ fp16 variant loaded successfully")
            except (OSError, ValueError, RuntimeError) as variant_error:
                logger.warning(f"⚠️ fp16 variant failed: {str(variant_error)}")
                pipeline = None
        
        # If fp16 failed or not CUDA, try without variant
        if pipeline is None:
            logger.info(f"🔄 Trying standard model loading...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                volume_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                local_files_only=True
            )
            logger.info(f"✅ Standard model loaded successfully")
        
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
        
        logger.info(f"✅ Successfully loaded model from Volume: {volume_path}")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to load from Volume: {str(e)}")
        raise RuntimeError(f"❌ Volume model loading failed. Error: {str(e)}")

def generate_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=7.0, 
                  width=1024, height=1024, seed=None):
    """Generate an image using the loaded pipeline"""
    global pipeline, device
    
    if pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    try:
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
        
        # Convert to base64
        image = result.images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "status": "success",
            "image": img_base64,
            "seed": seed
        }
        
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA OOM by clearing cache and retrying with CPU offload
        logger.warning("⚠️ CUDA OOM detected, clearing cache and retrying...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Force CPU offload
        if hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
        
        # Retry with lower precision or smaller size
        try:
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=max(10, num_inference_steps // 2),
                    guidance_scale=guidance_scale,
                    width=min(width, 768),
                    height=min(height, 768),
                    generator=generator
                )
            
            image = result.images[0]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "status": "success",
                "image": img_base64,
                "seed": seed,
                "note": "Reduced quality due to memory constraints"
            }
            
        except Exception as retry_error:
            logger.error(f"❌ Retry also failed: {retry_error}")
            return {
                "status": "error",
                "error": f"CUDA OOM and retry failed: {str(retry_error)}"
            }
    
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def handler(event):
    """RunPod handler function"""
    try:
        # Extract parameters from event
        input_data = event.get("input", {})
        
        prompt = input_data.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is required"}
        
        negative_prompt = input_data.get("negative_prompt", "")
        num_inference_steps = input_data.get("num_inference_steps", 20)
        guidance_scale = input_data.get("guidance_scale", 7.0)
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        seed = input_data.get("seed", None)
        
        # Validate parameters
        num_inference_steps = max(1, min(50, int(num_inference_steps)))
        guidance_scale = max(1.0, min(20.0, float(guidance_scale)))
        width = max(256, min(2048, int(width)))
        height = max(256, min(2048, int(height)))
        
        # Generate image
        result = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "status": "error", 
            "error": str(e)
        }

# Load model when module is imported
if __name__ == "__main__":
    print("Starting PhotonicFusion SDXL RunPod Handler (Volume Optimized)...")
    print("Loading PhotonicFusion SDXL model from Volume...")
    
    try:
        load_model()
        print("✅ Model loaded successfully from Volume!")
        
        # Test generation if running directly
        if len(os.sys.argv) > 1 and os.sys.argv[1] == "--test":
            print("🧪 Running test generation...")
            test_result = generate_image(
                prompt="a beautiful landscape with mountains and a lake, photorealistic",
                num_inference_steps=10,
                width=512,
                height=512
            )
            
            if test_result["status"] == "success":
                print("✅ Test generation successful!")
                # Save test image
                img_data = base64.b64decode(test_result["image"])
                with open("test_output.png", "wb") as f:
                    f.write(img_data)
                print("💾 Test image saved as test_output.png")
            else:
                print(f"❌ Test failed: {test_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("🔧 Please check the Volume configuration and model files")
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler}) 