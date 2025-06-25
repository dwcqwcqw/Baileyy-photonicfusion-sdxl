#!/usr/bin/env python3
"""Test YAML metadata fix"""

def test_metadata():
    try:
        from huggingface_hub import model_info
        
        print("Testing Hugging Face model metadata...")
        info = model_info("Baileyy/photonicfusion-sdxl")
        
        print(f"✅ Model info loaded successfully")
        print(f"Pipeline tag: {info.pipeline_tag}")
        print(f"Library: {info.library_name}")
        print(f"Tags: {info.tags[:5] if info.tags else 'None'}")
        
        if info.pipeline_tag and info.library_name:
            print("🎉 YAML metadata is working correctly!")
            return True
        else:
            print("⚠️ Some metadata still missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_metadata()
