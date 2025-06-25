#!/usr/bin/env python3
"""
PhotonicFusion SDXL RunPod Handler with compatibility fixes
This version applies fixes before importing problematic packages
"""

# Apply compatibility fixes BEFORE importing other packages
import os
import sys
import inspect

# Fix formatargspec for Python 3.12 compatibility
if not hasattr(inspect, 'formatargspec'):
    def formatargspec(args, varargs=None, varkw=None, defaults=None,
                    kwonlyargs=(), kwonlydefaults={}, annotations={},
                    formatarg=str, formatvarargs=lambda name: '*' + name,
                    formatvarkw=lambda name: '**' + name,
                    formatvalue=lambda value: '=' + repr(value)):
        """Format an argument spec from the values returned by inspect.getfullargspec."""
        def formatargandannotation(arg):
            if arg in annotations:
                return arg + ': ' + str(annotations[arg])
            return arg
        
        specs = []
        if defaults:
            firstdefault = len(args) - len(defaults)
        else:
            firstdefault = len(args)
        
        for i, arg in enumerate(args):
            spec = formatargandannotation(arg)
            if defaults and i >= firstdefault:
                spec += formatvalue(defaults[i - firstdefault])
            specs.append(spec)
        
        if varargs:
            specs.append(formatvarargs(formatargandannotation(varargs)))
        elif kwonlyargs:
            specs.append('*')
        
        if kwonlyargs:
            for kw in kwonlyargs:
                spec = formatargandannotation(kw)
                if kw in kwonlydefaults:
                    spec += formatvalue(kwonlydefaults[kw])
                specs.append(spec)
        
        if varkw:
            specs.append(formatvarkw(formatargandannotation(varkw)))
        
        return '(' + ', '.join(specs) + ')'
    
    # Monkey patch the missing function
    inspect.formatargspec = formatargspec
    print("✅ formatargspec compatibility fix applied")

# Set protobuf environment variables
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Now import the packages that had issues
try:
    import runpod
    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    import base64
    from io import BytesIO
    from PIL import Image
    import gc
    import logging
    print("✅ All imports successful after compatibility fixes")
except ImportError as e:
    print(f"❌ Import failed even after fixes: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and pipeline
pipeline = None
device = None

def load_model():
    """Load the PhotonicFusion SDXL model with fallback options"""
    global pipeline, device
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Define potential model paths in order of preference
    model_paths = [
        "/runpod-volume/photonicfusion-sdxl",  # RunPod volume
        "../PhotonicFusionSDXL_V3-diffusers-manual",  # Local converted model
        "Baileyy/photonicfusion-sdxl",  # Hugging Face Hub
        "stabilityai/stable-diffusion-xl-base-1.0"  # Official SDXL fallback
    ]
    
    pipeline = None
    last_error = None
    
    for i, model_path in enumerate(model_paths):
        try:
            logger.info(f"📁 Attempting to load model from: {model_path}")
            
            # Check if it's a local path and exists
            if not model_path.startswith(("http", "Baileyy/", "stabilityai/")):
                if not os.path.exists(model_path):
                    logger.warning(f"⚠️ Local path {model_path} does not exist, skipping...")
                    continue
                    
                # For local paths, verify required files exist
                required_files = ["model_index.json", "unet", "vae", "text_encoder", "text_encoder_2"]
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                if missing_files:
                    logger.warning(f"⚠️ Missing required files in {model_path}: {missing_files}")
                    continue
            
            # Load the pipeline
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
            
            # Set scheduler
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            
            # Move to device and optimize
            pipeline = pipeline.to(device)
            
            if device == "cuda":
                # Enable memory optimizations
                pipeline.enable_attention_slicing()
                try:
                    pipeline.enable_model_cpu_offload()
                except:
                    pass  # Some versions don't have this method
                
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
    print("Starting PhotonicFusion SDXL RunPod Handler (Fixed Version)...")
    print("Loading PhotonicFusion SDXL model...")
    
    try:
        load_model()
        print("✅ Model loaded successfully!")
        
        # Test generation if running directly
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
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
                with open("test_output_fixed.png", "wb") as f:
                    f.write(img_data)
                print("💾 Test image saved as test_output_fixed.png")
            else:
                print(f"❌ Test failed: {test_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("🔧 Please check the model path and requirements")
    
    # Start RunPod serverless worker
    try:
        runpod.serverless.start({"handler": handler})
    except NameError:
        print("ℹ️ RunPod not available in test environment") 