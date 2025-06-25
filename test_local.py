#!/usr/bin/env python3
"""
Local testing script for PhotonicFusion SDXL RunPod handler
"""

import base64
import json
import time
from handler import handler
from PIL import Image
from io import BytesIO

def save_base64_image(base64_string: str, filename: str):
    """Save base64 encoded image to file"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image.save(filename)
    print(f"Image saved to: {filename}")

def test_basic_generation():
    """Test basic image generation"""
    print("=== Testing Basic Image Generation ===")
    
    test_event = {
        "input": {
            "prompt": "a beautiful sunset over mountains, photorealistic, high quality",
            "negative_prompt": "blurry, low quality, distorted",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "seed": 42
        }
    }
    
    start_time = time.time()
    result = handler(test_event)
    end_time = time.time()
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print(f"✅ Generation successful in {end_time - start_time:.2f} seconds")
    print(f"Parameters: {result['parameters']}")
    
    # Save the image
    save_base64_image(result["image"], "test_basic.png")
    
    return True

def test_different_styles():
    """Test different art styles"""
    print("\n=== Testing Different Styles ===")
    
    test_cases = [
        {
            "name": "portrait",
            "prompt": "portrait of a person, professional photography, studio lighting",
            "filename": "test_portrait.png"
        },
        {
            "name": "landscape",
            "prompt": "serene lake with mountains, golden hour, landscape photography",
            "filename": "test_landscape.png"
        },
        {
            "name": "digital_art",
            "prompt": "futuristic cityscape, cyberpunk style, neon lights, digital art",
            "filename": "test_digital_art.png"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        
        test_event = {
            "input": {
                "prompt": test_case["prompt"],
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }
        
        start_time = time.time()
        result = handler(test_event)
        end_time = time.time()
        
        if "error" in result:
            print(f"❌ Error in {test_case['name']}: {result['error']}")
            continue
        
        print(f"✅ {test_case['name']} generated in {end_time - start_time:.2f} seconds")
        save_base64_image(result["image"], test_case["filename"])

def test_parameter_validation():
    """Test parameter validation"""
    print("\n=== Testing Parameter Validation ===")
    
    # Test missing prompt
    test_event = {"input": {}}
    result = handler(test_event)
    
    if "error" in result:
        print(f"✅ Missing prompt validation works: {result['error']}")
    else:
        print("❌ Missing prompt validation failed")
    
    # Test extreme parameters
    test_event = {
        "input": {
            "prompt": "test",
            "width": 2048,  # Too large
            "height": 256,  # Too small
            "num_inference_steps": 200,  # Too many
            "guidance_scale": 50.0  # Too high
        }
    }
    
    result = handler(test_event)
    
    if "error" not in result:
        params = result["parameters"]
        print(f"✅ Parameter clamping works:")
        print(f"  Width: {params['width']} (clamped from 2048)")
        print(f"  Height: {params['height']} (clamped from 256)")
        print(f"  Steps: {params['num_inference_steps']} (clamped from 200)")
        print(f"  Guidance: {params['guidance_scale']} (clamped from 50.0)")
    else:
        print(f"❌ Parameter validation error: {result['error']}")

def test_memory_efficiency():
    """Test memory efficiency with smaller images"""
    print("\n=== Testing Memory Efficiency ===")
    
    test_event = {
        "input": {
            "prompt": "cute cat, high quality",
            "width": 768,
            "height": 768,
            "num_inference_steps": 15,
            "guidance_scale": 7.0
        }
    }
    
    start_time = time.time()
    result = handler(test_event)
    end_time = time.time()
    
    if "error" in result:
        print(f"❌ Memory efficiency test failed: {result['error']}")
        return False
    
    print(f"✅ Small image generated in {end_time - start_time:.2f} seconds")
    save_base64_image(result["image"], "test_small.png")
    
    return True

def main():
    """Run all tests"""
    print("Starting PhotonicFusion SDXL Local Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_generation,
        test_different_styles,
        test_parameter_validation,
        test_memory_efficiency
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for RunPod deployment.")
    else:
        print("⚠️ Some tests failed. Please check the issues before deployment.")

if __name__ == "__main__":
    main() 