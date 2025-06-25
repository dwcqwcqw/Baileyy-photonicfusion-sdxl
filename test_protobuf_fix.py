#!/usr/bin/env python3
"""
Test script to verify the protobuf fix
"""

import os
import sys
import importlib
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    print("=== Checking Dependencies ===")
    
    # Check protobuf
    try:
        import protobuf
        print(f"✅ protobuf is installed: {protobuf.__version__}")
    except ImportError:
        print("❌ protobuf is not installed")
        return False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ transformers is installed: {transformers.__version__}")
    except ImportError:
        print("❌ transformers is not installed")
        return False
    
    # Check diffusers
    try:
        import diffusers
        print(f"✅ diffusers is installed: {diffusers.__version__}")
    except ImportError:
        print("❌ diffusers is not installed")
        return False
    
    return True

def test_tokenizer_loading():
    """Test loading a CLIP tokenizer"""
    print("\n=== Testing CLIP Tokenizer Loading ===")
    
    try:
        from transformers import CLIPTokenizer
        
        print("🔄 Loading CLIP tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Test tokenization
        text = "A beautiful landscape"
        tokens = tokenizer(text)
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Input text: '{text}'")
        print(f"   Token IDs: {tokens['input_ids'][:5]}... (truncated)")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_sdxl_pipeline_loading():
    """Test loading the SDXL pipeline"""
    print("\n=== Testing SDXL Pipeline Loading ===")
    
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        
        print("🔄 Loading tiny SDXL test model...")
        # Use a tiny model for testing
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            torch_dtype=torch.float32
        )
        
        print(f"✅ Pipeline loaded successfully")
        print(f"   Pipeline type: {type(pipeline).__name__}")
        print(f"   Components loaded: {list(pipeline.components.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline loading failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_handler_import():
    """Test importing the handler module"""
    print("\n=== Testing Handler Import ===")
    
    try:
        print("🔄 Importing handler module...")
        import handler
        
        print(f"✅ Handler imported successfully")
        print(f"   Handler functions: {[f for f in dir(handler) if not f.startswith('_') and callable(getattr(handler, f))]}")
        
        return True
    except Exception as e:
        print(f"❌ Handler import failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False

def install_protobuf():
    """Install protobuf if needed"""
    print("\n=== Installing protobuf ===")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.3"])
        print("✅ protobuf 3.20.3 installed successfully")
        
        # Reload modules that might have been affected
        importlib.reload(__import__("transformers"))
        
        return True
    except Exception as e:
        print(f"❌ protobuf installation failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("PhotonicFusion SDXL - Protobuf Fix Verification")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️ Some dependencies are missing. Attempting to install protobuf...")
        install_protobuf()
    
    tests = [
        ("Tokenizer Loading", test_tokenizer_loading),
        ("SDXL Pipeline Loading", test_sdxl_pipeline_loading),
        ("Handler Import", test_handler_import)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"🎉 {test_name} PASSED")
        else:
            print(f"💥 {test_name} FAILED")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The protobuf fix is working correctly.")
        print("\n📋 Next steps:")
        print("  1. Update the Dockerfile to include protobuf==3.20.3")
        print("  2. Deploy the updated handler to RunPod")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 