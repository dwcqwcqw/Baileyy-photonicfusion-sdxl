#!/usr/bin/env python3
"""
删除标准文件，只保留FP16版本，然后重新上传到HuggingFace
这将节省约一半的存储空间
"""

import os
import shutil
from pathlib import Path

def delete_standard_files():
    """删除标准文件，只保留FP16版本"""
    
    model_path = "PhotonicFusionSDXL_V3-diffusers-manual"
    
    print("🗑️ PhotonicFusion SDXL 标准文件清理器")
    print("=" * 50)
    print(f"📁 模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        return False
    
    # 要删除的标准文件（保留FP16版本）
    files_to_delete = [
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors", 
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors"
    ]
    
    print("\n🔍 删除标准文件...")
    
    total_deleted_size = 0
    deleted_count = 0
    
    for file_path in files_to_delete:
        full_path = os.path.join(model_path, file_path)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            size_mb = file_size / (1024 * 1024)
            
            # 删除文件
            os.remove(full_path)
            
            total_deleted_size += file_size
            deleted_count += 1
            
            print(f"   ✅ 已删除: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚠️ 文件不存在: {file_path}")
    
    print(f"\n📊 删除统计:")
    print(f"   已删除文件数: {deleted_count}")
    print(f"   节省空间: {total_deleted_size / (1024**3):.2f} GB")
    
    # 验证剩余的FP16文件
    print("\n🔍 验证剩余的FP16文件...")
    
    remaining_files = [
        "model_index.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "scheduler/scheduler_config.json"
    ]
    
    remaining_size = 0
    missing_files = []
    
    for file_path in remaining_files:
        full_path = os.path.join(model_path, file_path)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            size_mb = file_size / (1024 * 1024)
            remaining_size += file_size
            
            file_type = "fp16" if "fp16" in file_path else "config"
            print(f"   ✅ {file_path} ({size_mb:.1f} MB) [{file_type}]")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    print(f"\n📊 剩余文件总大小: {remaining_size / (1024**3):.2f} GB")
    
    if missing_files:
        print(f"❌ 缺少必需文件: {missing_files}")
        return False
    
    print("✅ 标准文件删除完成，FP16版本完整保留")
    return True

def upload_to_huggingface_fp16_only():
    """上传只包含FP16版本的模型到HuggingFace"""
    
    model_path = "PhotonicFusionSDXL_V3-diffusers-manual"
    repo_id = "Baileyy/photonicfusion-sdxl"
    
    print("\n🚀 上传FP16优化版本到HuggingFace")
    print("=" * 40)
    
    # 检查 huggingface_hub
    try:
        from huggingface_hub import HfApi, upload_folder, login
        print("✅ huggingface_hub 可用")
    except ImportError:
        print("❌ huggingface_hub 未安装")
        print("请运行: pip install huggingface_hub")
        return False
    
    # 创建优化的README
    readme_content = """---
license: apache-2.0
language:
- en
library_name: diffusers
pipeline_tag: text-to-image
tags:
- stable-diffusion-xl
- sdxl
- text-to-image
- diffusers
- fp16
---

# PhotonicFusion SDXL V3 (FP16 Optimized)

PhotonicFusion SDXL V3 是一个基于 Stable Diffusion XL 的高质量图像生成模型。

## 模型特性

- **架构**: Stable Diffusion XL
- **分辨率**: 1024x1024 (原生)
- **精度**: FP16 优化版本 (节省50%存储空间)
- **优化**: 只包含 FP16 variant 文件，更快的推理速度

## 文件结构 (仅FP16版本)

此版本只包含FP16优化文件，节省存储空间：

- `text_encoder/model.fp16.safetensors` (235MB)
- `text_encoder_2/model.fp16.safetensors` (1.3GB)
- `unet/diffusion_pytorch_model.fp16.safetensors` (4.8GB) 
- `vae/diffusion_pytorch_model.fp16.safetensors` (160MB)

**总大小**: ~6.5GB (相比标准版本节省50%空间)

## 使用方法

### 自动FP16加载 (推荐)
```python
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "Baileyy/photonicfusion-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",  # 使用 fp16 variant
    use_safetensors=True
)
pipeline.to("cuda")

# 生成图像
image = pipeline(
    "a beautiful sunset over mountains, photorealistic", 
    height=1024, 
    width=1024,
    num_inference_steps=20
).images[0]
```

### 基础加载
```python
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "Baileyy/photonicfusion-sdxl",
    torch_dtype=torch.float16,
    use_safetensors=True
)
```

## 性能优势

- **推理速度**: 2-4 秒 (1024x1024, RTX 4090)
- **内存需求**: ~6-8GB VRAM (FP16 优化)
- **存储空间**: 节省50%磁盘空间
- **下载速度**: 更快的模型下载

## RunPod 部署

此模型已针对 RunPod Serverless 部署进行优化：
- Volume 挂载优化
- FP16 自动加载
- 内存效率优化
- 快速启动时间

部署仓库: [dwcqwcqw/Baileyy-photonicfusion-sdxl](https://github.com/dwcqwcqw/Baileyy-photonicfusion-sdxl)

## 更新历史

### v3.0 (FP16 Optimized) - 2025-06-25
- 🗑️ 移除标准safetensors文件
- ✅ 只保留FP16 variant文件
- 🚀 节省50%存储空间
- ⚡ 优化加载性能

### v2.0 (2025-06-25)
- ✅ 添加 FP16 variant 支持
- ✅ 优化推理性能

### v1.0 (2025-06-24)
- 初始发布

## 许可证

Apache 2.0 License
"""
    
    # 保存优化的README
    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ 创建了FP16优化版README.md")
    
    # 登录验证
    print("\n🔐 HuggingFace 身份验证...")
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ 已登录为: {user_info['name']}")
    except Exception as e:
        print("⚠️ 需要 HuggingFace token")
        print("请运行: huggingface-cli login")
        return False
    
    # 上传模型
    print(f"\n📤 开始上传到 {repo_id}...")
    try:
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="v3.0: FP16 Optimized (移除标准文件,节省50%空间)"
        )
        print("✅ 上传成功!")
        print(f"🌐 模型链接: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 PhotonicFusion SDXL FP16优化器")
    print("此工具将删除标准文件，只保留FP16版本，并重新上传")
    print("=" * 60)
    
    # 确认操作
    confirm = input("\n⚠️ 这将永久删除标准safetensors文件，只保留FP16版本。继续吗? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    # 步骤1: 删除标准文件
    if not delete_standard_files():
        print("❌ 删除标准文件失败")
        return
    
    # 步骤2: 上传FP16优化版本
    if not upload_to_huggingface_fp16_only():
        print("❌ 上传失败")
        return
    
    print("\n🎉 FP16优化完成!")
    print("模型现在只包含FP16版本，节省了50%的存储空间")

if __name__ == "__main__":
    main() 