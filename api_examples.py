#!/usr/bin/env python3
"""
API usage examples for PhotonicFusion SDXL RunPod endpoint
"""

import requests
import json
import base64
import time
from PIL import Image
from io import BytesIO

# Replace with your actual RunPod endpoint URL
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "YOUR_API_KEY"

def call_runpod_api(payload):
    """Call RunPod API with the given payload"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(RUNPOD_ENDPOINT, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

def save_image_from_base64(base64_string, filename):
    """Save base64 encoded image to file"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image.save(filename)
    print(f"Image saved to: {filename}")

def example_basic_generation():
    """Basic image generation example"""
    print("=== Basic Image Generation ===")
    
    payload = {
        "input": {
            "prompt": "a beautiful sunset over mountains, photorealistic, high quality",
            "negative_prompt": "blurry, low quality, distorted",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": 42
        }
    }
    
    print(f"Sending request...")
    start_time = time.time()
    
    try:
        result = call_runpod_api(payload)
        end_time = time.time()
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"✅ Generation completed in {end_time - start_time:.2f} seconds")
        
        # Extract the output
        output = result.get("output", {})
        if "image" in output:
            save_image_from_base64(output["image"], "api_basic_example.png")
            print(f"Parameters used: {output.get('parameters', {})}")
        else:
            print("❌ No image in response")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

def example_portrait():
    """Portrait generation example"""
    print("\n=== Portrait Generation ===")
    
    payload = {
        "input": {
            "prompt": "portrait of a person, professional photography, studio lighting, high resolution",
            "negative_prompt": "blurry, low quality, distorted, ugly, disfigured",
            "width": 768,
            "height": 1024,
            "num_inference_steps": 35,
            "guidance_scale": 8.0
        }
    }
    
    try:
        result = call_runpod_api(payload)
        
        if "error" not in result:
            output = result.get("output", {})
            if "image" in output:
                save_image_from_base64(output["image"], "api_portrait_example.png")
                print("✅ Portrait generated successfully")
            
    except Exception as e:
        print(f"❌ Portrait generation failed: {str(e)}")

def example_landscape():
    """Landscape generation example"""
    print("\n=== Landscape Generation ===")
    
    payload = {
        "input": {
            "prompt": "serene mountain lake at golden hour, landscape photography, ultra detailed",
            "negative_prompt": "people, buildings, cars, urban",
            "width": 1536,
            "height": 768,
            "num_inference_steps": 25,
            "guidance_scale": 7.0
        }
    }
    
    try:
        result = call_runpod_api(payload)
        
        if "error" not in result:
            output = result.get("output", {})
            if "image" in output:
                save_image_from_base64(output["image"], "api_landscape_example.png")
                print("✅ Landscape generated successfully")
            
    except Exception as e:
        print(f"❌ Landscape generation failed: {str(e)}")

def example_digital_art():
    """Digital art generation example"""
    print("\n=== Digital Art Generation ===")
    
    payload = {
        "input": {
            "prompt": "futuristic cityscape, cyberpunk style, neon lights, digital art, concept art",
            "negative_prompt": "realistic, photography, blurry",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 40,
            "guidance_scale": 9.0,
            "seed": 123456
        }
    }
    
    try:
        result = call_runpod_api(payload)
        
        if "error" not in result:
            output = result.get("output", {})
            if "image" in output:
                save_image_from_base64(output["image"], "api_digital_art_example.png")
                print("✅ Digital art generated successfully")
            
    except Exception as e:
        print(f"❌ Digital art generation failed: {str(e)}")

def example_batch_generation():
    """Generate multiple images with different seeds"""
    print("\n=== Batch Generation ===")
    
    base_prompt = "cute cat, high quality, detailed"
    
    for i in range(3):
        print(f"Generating image {i+1}/3...")
        
        payload = {
            "input": {
                "prompt": base_prompt,
                "negative_prompt": "blurry, low quality",
                "width": 768,
                "height": 768,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "seed": 1000 + i
            }
        }
        
        try:
            result = call_runpod_api(payload)
            
            if "error" not in result:
                output = result.get("output", {})
                if "image" in output:
                    save_image_from_base64(output["image"], f"api_batch_{i+1}.png")
                    print(f"✅ Batch image {i+1} generated")
                
        except Exception as e:
            print(f"❌ Batch image {i+1} failed: {str(e)}")
        
        # Small delay between requests
        time.sleep(1)

def example_curl_command():
    """Show equivalent curl command"""
    print("\n=== Equivalent cURL Command ===")
    
    curl_command = f'''
curl -X POST "{RUNPOD_ENDPOINT}" \\
  -H "Authorization: Bearer {API_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "input": {{
      "prompt": "a beautiful sunset over mountains, photorealistic, high quality",
      "negative_prompt": "blurry, low quality, distorted",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "seed": 42
    }}
  }}'
'''
    
    print(curl_command)

def main():
    """Run all examples"""
    print("PhotonicFusion SDXL RunPod API Examples")
    print("=" * 50)
    print("⚠️  Please update RUNPOD_ENDPOINT and API_KEY variables before running!")
    print("=" * 50)
    
    if "YOUR_ENDPOINT_ID" in RUNPOD_ENDPOINT or "YOUR_API_KEY" in API_KEY:
        print("❌ Please update the endpoint URL and API key before running examples")
        example_curl_command()
        return
    
    examples = [
        example_basic_generation,
        example_portrait,
        example_landscape,
        example_digital_art,
        example_batch_generation
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {example_func.__name__} failed: {str(e)}")
        
        time.sleep(2)  # Delay between examples
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    example_curl_command()

if __name__ == "__main__":
    main() 