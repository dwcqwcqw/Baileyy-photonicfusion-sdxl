#!/usr/bin/env python3
"""
Test the fixed handler to verify device_map issue is resolved
"""

import os
import sys
import time
from handler import load_model, generate_image, handler

def test_model_loading():
    """Test if model loads without device_map='auto' error"""
    print("=== Testing Model Loading Fix ===")
    
    try:
        print("🔄 Attempting to load model...")
        start_time = time.time()
        
        pipeline = load_model()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"   Pipeline device: {pipeline.device}")
        print(f"   UNet device: {pipeline.unet.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_single_image_generation():
    """Test single image generation"""
    print("\n=== Testing Single Image Generation ===")
    
    try:
        print("🎨 Generating single image...")
        
        images = generate_image(
            prompt="a simple test image, digital art",
            width=512,
            height=512,
            num_inference_steps=10,  # Fast test
            num_images_per_prompt=1
        )
        
        print(f"✅ Single image generation successful")
        print(f"   Generated {len(images)} image(s)")
        print(f"   Base64 length: {len(images[0])} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Single image generation failed: {str(e)}")
        return False

def test_multiple_image_generation():
    """Test multiple image generation"""
    print("\n=== Testing Multiple Image Generation ===")
    
    try:
        print("🎨 Generating multiple images...")
        
        images = generate_image(
            prompt="a colorful abstract art",
            width=512,
            height=512,
            num_inference_steps=10,  # Fast test
            num_images_per_prompt=2
        )
        
        print(f"✅ Multiple image generation successful")
        print(f"   Generated {len(images)} image(s)")
        
        for i, img in enumerate(images):
            print(f"   Image {i+1} base64 length: {len(img)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple image generation failed: {str(e)}")
        return False

def test_handler_api():
    """Test the handler API with various inputs"""
    print("\n=== Testing Handler API ===")
    
    test_cases = [
        {
            "name": "Basic single image",
            "input": {
                "prompt": "a beautiful sunset",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10,
                "num_images_per_prompt": 1
            }
        },
        {
            "name": "Multiple images",
            "input": {
                "prompt": "a modern cityscape",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10,
                "num_images_per_prompt": 2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        
        try:
            result = handler({"input": test_case["input"]})
            
            if "error" in result:
                print(f"❌ Handler returned error: {result['error']}")
                return False
            elif "images" in result:
                images = result["images"]
                print(f"✅ Handler test passed")
                print(f"   Generated {len(images)} image(s)")
                print(f"   Parameters: {result.get('parameters', {})}")
            else:
                print(f"❌ Unexpected response format: {result.keys()}")
                return False
                
        except Exception as e:
            print(f"❌ Handler test failed: {str(e)}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("PhotonicFusion SDXL - Fix Verification Tests")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Single Image Generation", test_single_image_generation),
        ("Multiple Image Generation", test_multiple_image_generation),
        ("Handler API", test_handler_api)
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
        print("🎉 All tests passed! The fix is working correctly.")
        print("\n📋 Next steps:")
        print("  1. Deploy the updated handler to RunPod")
        print("  2. Update your endpoint configuration")
        print("  3. Test with the web frontend")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 