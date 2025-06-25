#!/usr/bin/env python3
"""
Test local volume model loading for PhotonicFusion SDXL RunPod handler
"""

import os
import sys
import shutil
from pathlib import Path

def setup_local_volume_test():
    """Setup local volume test by copying model to test location"""
    print("=== Setting up Local Volume Test ===")
    
    # Define paths
    source_model_path = "../PhotonicFusionSDXL_V3-diffusers-manual"
    test_volume_path = "./test-runpod-volume/photonicfusion-sdxl"
    
    # Check if source model exists
    if not os.path.exists(source_model_path):
        print(f"❌ Source model not found: {source_model_path}")
        return False
    
    # Create test volume directory
    os.makedirs(os.path.dirname(test_volume_path), exist_ok=True)
    
    # Copy model files
    if os.path.exists(test_volume_path):
        print(f"🗑️ Removing existing test volume: {test_volume_path}")
        shutil.rmtree(test_volume_path)
    
    print(f"📂 Copying model from {source_model_path} to {test_volume_path}")
    shutil.copytree(source_model_path, test_volume_path)
    
    print(f"✅ Test volume setup complete")
    return True

def test_volume_loading():
    """Test model loading with local volume path"""
    print("\n=== Testing Volume Model Loading ===")
    
    # Temporarily modify the handler to use test path
    import handler
    
    # Override the local model path for testing
    original_load_model = handler.load_model
    
    def test_load_model():
        """Modified load_model for testing"""
        global pipeline, device
        
        if handler.pipeline is not None:
            return handler.pipeline
        
        print("Loading PhotonicFusion SDXL model (TEST MODE)...")
        
        # Use test volume path
        test_local_path = "./test-runpod-volume/photonicfusion-sdxl"
        hf_model_name = "Baileyy/photonicfusion-sdxl"
        
        import torch
        from diffusers import StableDiffusionXLPipeline
        
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
        print(f"Using device: {device}")
        
        try:
            if os.path.exists(test_local_path):
                print(f"📁 Loading model from test volume: {test_local_path}")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    test_local_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    local_files_only=True
                )
                print("✅ Model loaded from test volume")
                handler.pipeline = pipeline
                handler.device = device
                return pipeline
            else:
                print(f"❌ Test volume path not found: {test_local_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading from test volume: {str(e)}")
            return None
    
    # Replace load_model temporarily
    handler.load_model = test_load_model
    
    try:
        # Test loading
        pipeline = handler.load_model()
        
        if pipeline is not None:
            print("✅ Volume model loading test passed")
            
            # Test a quick generation
            print("🎨 Testing quick generation...")
            
            if torch.cuda.is_available():
                pipeline.enable_attention_slicing()
            
            prompt = "a simple test image"
            result = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=5,  # Very fast test
                guidance_scale=7.5
            )
            
            result.images[0].save("volume_test_output.png")
            print("💾 Test image saved as 'volume_test_output.png'")
            print("🎉 Volume loading and generation test successful!")
            return True
        else:
            print("❌ Volume model loading test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False
    finally:
        # Restore original function
        handler.load_model = original_load_model

def cleanup_test():
    """Clean up test files"""
    print("\n=== Cleaning up test files ===")
    
    test_volume_path = "./test-runpod-volume"
    if os.path.exists(test_volume_path):
        shutil.rmtree(test_volume_path)
        print("🗑️ Test volume removed")
    
    test_output = "volume_test_output.png"
    if os.path.exists(test_output):
        os.remove(test_output)
        print("🗑️ Test output image removed")

def main():
    """Run volume loading test"""
    print("PhotonicFusion SDXL - Volume Loading Test")
    print("=" * 50)
    
    try:
        # Setup test environment
        if not setup_local_volume_test():
            return False
        
        # Test volume loading
        success = test_volume_loading()
        
        if success:
            print("\n🎉 All volume tests passed!")
            print("The handler is ready for RunPod deployment with volume mounting.")
        else:
            print("\n❌ Volume tests failed.")
        
        return success
        
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        return False
    finally:
        # Always cleanup
        cleanup_test()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 