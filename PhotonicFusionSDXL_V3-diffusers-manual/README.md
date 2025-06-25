---
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
