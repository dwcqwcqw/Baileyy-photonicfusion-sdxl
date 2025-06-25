#!/usr/bin/env python3
"""
在 RunPod 上验证模型加载
"""

import torch
from diffusers import StableDiffusionXLPipeline
import traceback

def test_model_loading():
    """测试模型加载"""
    
    print("🧪 测试模型加载...")
    print("=" * 60)
    
    try:
        model_path = "/runpod-volume/photonicfusion-sdxl"
        
        print(f"📁 模型路径: {model_path}")
        print(f"💾 CUDA 可用: {torch.cuda.is_available()}")
        
        # 加载模型
        print(f"🔄 加载模型...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        print(f"✅ 模型加载成功!")
        
        # 验证组件
        print(f"\n🔍 验证组件:")
        print(f"   text_encoder: {type(pipeline.text_encoder)}")
        print(f"   text_encoder_2: {type(pipeline.text_encoder_2)}")
        print(f"   unet: {type(pipeline.unet)}")
        print(f"   vae: {type(pipeline.vae)}")
        print(f"   scheduler: {type(pipeline.scheduler)}")
        
        # 测试生成
        print(f"\n🎨 测试生成...")
        
        prompt = "a beautiful sunset over mountains"
        
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
        
        print(f"✅ 生成测试成功!")
        print(f"🖼️  图像尺寸: {image.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败:")
        print(f"📝 错误详情: {str(e)}")
        print(f"\n🔍 完整错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    
    if success:
        print(f"\n🎉 模型验证成功!")
        print(f"🚀 可以正常使用")
    else:
        print(f"\n❌ 模型验证失败")
        print(f"🔧 需要进一步调试")
