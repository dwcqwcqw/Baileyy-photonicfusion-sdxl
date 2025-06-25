#!/usr/bin/env python3
"""
测试 HuggingFace 上的 PhotonicFusion SDXL fp16 variant 模型
"""

import torch
import time
from datetime import datetime

def test_fp16_variant():
    """测试 FP16 variant 加载"""
    print("🧪 测试 HuggingFace PhotonicFusion SDXL (FP16 Variant)")
    print("=" * 60)
    
    # 检查环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  设备: {device}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
        
        # 显示显存信息
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"💾 显存: {memory_allocated:.1f}GB 已用 / {memory_cached:.1f}GB 缓存")
    
    try:
        from diffusers import StableDiffusionXLPipeline
        print("✅ diffusers 可用")
    except ImportError:
        print("❌ diffusers 未安装")
        return False
    
    repo_id = "Baileyy/photonicfusion-sdxl"
    
    # 测试 1: 加载 fp16 variant
    print(f"\n🔄 测试 1: 加载 fp16 variant from {repo_id}...")
    start_time = time.time()
    
    try:
        pipeline_fp16 = StableDiffusionXLPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            device_map="auto" if device == "cuda" else None
        )
        
        load_time = time.time() - start_time
        print(f"✅ FP16 variant 加载成功 ({load_time:.1f}s)")
        
        # 移动到设备
        if device == "cuda":
            pipeline_fp16 = pipeline_fp16.to(device)
        
        # 显示内存使用
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"💾 模型加载后显存使用: {memory_used:.2f}GB")
        
    except Exception as e:
        print(f"❌ FP16 variant 加载失败: {str(e)}")
        return False
    
    # 测试 2: 生成图像
    print(f"\n🔄 测试 2: 生成测试图像...")
    
    prompt = "a beautiful sunset over mountains, photorealistic, high quality"
    negative_prompt = "blurry, low quality, distorted"
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # 生成图像
            result = pipeline_fp16(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=1024,
                width=1024,
                num_inference_steps=20,
                guidance_scale=7.0
            )
        
        generation_time = time.time() - start_time
        print(f"✅ 图像生成成功 ({generation_time:.1f}s)")
        
        # 保存图像
        image = result.images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photonicfusion_fp16_test_{timestamp}.png"
        image.save(filename)
        print(f"💾 图像已保存: {filename}")
        
        # 性能统计
        pixels_per_second = (1024 * 1024) / generation_time
        print(f"📊 性能: {pixels_per_second:,.0f} 像素/秒")
        
        # 最终显存使用
        if device == "cuda":
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"💾 峰值显存使用: {peak_memory:.2f}GB")
            print(f"💾 当前显存使用: {final_memory:.2f}GB")
        
    except Exception as e:
        print(f"❌ 图像生成失败: {str(e)}")
        return False
    
    # 测试 3: 比较标准加载 vs fp16 variant
    print(f"\n🔄 测试 3: 比较标准加载...")
    
    try:
        start_time = time.time()
        pipeline_std = StableDiffusionXLPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto" if device == "cuda" else None
        )
        std_load_time = time.time() - start_time
        print(f"✅ 标准版本加载成功 ({std_load_time:.1f}s)")
        
        # 清理内存
        del pipeline_std
        if device == "cuda":
            torch.cuda.empty_cache()
        
        print(f"\n📊 加载时间比较:")
        print(f"   FP16 variant: {load_time:.1f}s")
        print(f"   标准版本: {std_load_time:.1f}s")
        print(f"   差异: {abs(load_time - std_load_time):.1f}s")
        
    except Exception as e:
        print(f"⚠️ 标准版本测试失败: {str(e)}")
    
    # 清理
    del pipeline_fp16
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\n🎉 测试完成!")
    print(f"\n📋 总结:")
    print(f"   ✅ FP16 variant 可用")
    print(f"   ✅ 加载时间: {load_time:.1f}s")
    print(f"   ✅ 生成时间: {generation_time:.1f}s")
    if device == "cuda":
        print(f"   ✅ 峰值显存: {peak_memory:.2f}GB")
    print(f"   ✅ 输出文件: {filename}")
    
    return True

def test_model_info():
    """获取模型信息"""
    print("\n🔍 获取模型信息...")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        model_info = api.model_info("Baileyy/photonicfusion-sdxl")
        
        print(f"📋 模型信息:")
        print(f"   仓库: {model_info.modelId}")
        print(f"   标签: {model_info.tags}")
        print(f"   下载量: {model_info.downloads}")
        print(f"   创建时间: {model_info.created_at}")
        print(f"   更新时间: {model_info.last_modified}")
        
        # 列出文件
        print(f"\n📁 模型文件:")
        for sibling in model_info.siblings:
            size_mb = sibling.size / (1024*1024) if sibling.size else 0
            file_type = "fp16" if "fp16" in sibling.rfilename else "standard"
            print(f"   📄 {sibling.rfilename} ({size_mb:.1f}MB) [{file_type}]")
        
    except Exception as e:
        print(f"⚠️ 无法获取模型信息: {str(e)}")

def main():
    """主函数"""
    print("🚀 PhotonicFusion SDXL FP16 Variant 测试器")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取模型信息
    test_model_info()
    
    # 测试 fp16 variant
    success = test_fp16_variant()
    
    print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n🎉 所有测试通过！FP16 variant 工作正常！")
        print("\n💡 推荐使用方式:")
        print("pipeline = StableDiffusionXLPipeline.from_pretrained(")
        print("    'Baileyy/photonicfusion-sdxl',")
        print("    torch_dtype=torch.float16,")
        print("    variant='fp16',")
        print("    use_safetensors=True")
        print(")")
    else:
        print("\n❌ 测试失败，请检查上述错误")
    
    return success

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 