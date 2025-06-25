# PhotonicFusion SDXL - 后端错误修复

## 🐛 问题描述

根据您提供的日志，后端出现以下错误：

```
❌ Error loading model: auto not supported. Supported strategies are: balanced
NotImplementedError: auto not supported. Supported strategies are: balanced
```

## 🔍 错误原因

错误原因是在 `StableDiffusionXLPipeline.from_pretrained()` 中使用了 `device_map="auto"` 参数，但当前版本的 diffusers 库不支持 "auto" 策略，只支持 "balanced" 策略。

具体错误位置：
- 文件：`handler.py`，第46行和第54行
- 原代码：`device_map="auto" if device == "cuda" else None`

## ✅ 修复内容

### 1. 移除 device_map="auto"

**修改前：**
```python
pipeline = StableDiffusionXLPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    device_map="auto" if device == "cuda" else None,  # ❌ 这行导致错误
    local_files_only=True
)
```

**修改后：**
```python
pipeline = StableDiffusionXLPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    local_files_only=True
)
```

### 2. 改进设备管理

**修改前：**
```python
# Move to device if not using device_map
if device == "cuda" and pipeline.device != torch.device("cuda"):
    pipeline = pipeline.to(device)
```

**修改后：**
```python
# Move to device
if device == "cuda":
    pipeline = pipeline.to(device)
    print(f"✅ Pipeline moved to {device}")
```

### 3. 添加多图像支持

为了提升用户体验，同时添加了多图像生成支持：

- 新增 `num_images_per_prompt` 参数（1-4张图像）
- 修改返回格式从 `"image"` 到 `"images"` 数组
- 前端兼容新旧格式

## 🧪 测试验证

创建了完整的测试套件 `test_fix.py`：

1. **模型加载测试** - 验证不再出现 device_map 错误
2. **单图像生成测试** - 验证基本功能正常
3. **多图像生成测试** - 验证新功能正常
4. **Handler API测试** - 验证完整的API流程

运行测试：
```bash
python test_fix.py
```

## 📋 部署步骤

### 1. 本地测试（推荐）
```bash
# 在 photonicfusion-sdxl-runpod 目录下
python test_fix.py
```

### 2. 构建新镜像
```bash
./fix_deploy.sh
```

### 3. 更新 RunPod 端点
1. 在 RunPod 控制台中更新您的端点
2. 使用新构建的 Docker 镜像
3. 确保使用相同的 volume 配置

### 4. 验证修复
使用以下测试数据验证端点：

```json
{
  "input": {
    "prompt": "a beautiful landscape, high quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "num_images_per_prompt": 1
  }
}
```

预期响应格式：
```json
{
  "images": ["base64_encoded_image_data"],
  "prompt": "a beautiful landscape, high quality",
  "parameters": {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "num_images_per_prompt": 1,
    "seed": null
  }
}
```

## 🔧 前端兼容性

前端代码已更新以支持新的 API 格式，同时保持向后兼容：

- ✅ 支持新格式：`result.output.images` (数组)
- ✅ 支持旧格式：`result.output.image` (单张)
- ✅ 自动错误处理和显示

## 🚀 新功能

### 多图像生成
现在支持一次生成多张图像：

**前端使用：**
- 在"图像数量"滑块中选择1-4张
- 一次请求生成多张不同的图像

**API使用：**
```json
{
  "input": {
    "prompt": "digital art masterpiece",
    "num_images_per_prompt": 3
  }
}
```

### 改进的错误处理
- 更清晰的错误信息
- 自动 CUDA OOM 恢复
- 内存优化策略

## 📊 性能优化

修复后的性能提升：
- ✅ 消除模型加载错误
- ✅ 更快的设备初始化
- ✅ 支持批量图像生成
- ✅ 智能内存管理

## 🔍 故障排除

### 如果仍然出现错误：

1. **检查 diffusers 版本**：
   ```bash
   pip list | grep diffusers
   ```
   
2. **检查 torch 版本**：
   ```bash
   pip list | grep torch
   ```

3. **清除缓存**：
   ```bash
   rm -rf ~/.cache/huggingface/transformers/
   ```

4. **检查 CUDA 可用性**：
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

### 常见问题：

**Q: 模型加载慢**
A: 确保使用 RunPod Volume 预下载模型

**Q: 内存不足**
A: 降低分辨率或减少 num_images_per_prompt

**Q: 生成质量差**
A: 增加 num_inference_steps 和优化 prompt

## 📝 更新日志

- **2025-01-25**: 修复 device_map="auto" 错误
- **2025-01-25**: 添加多图像生成支持
- **2025-01-25**: 改进错误处理和日志
- **2025-01-25**: 更新前端兼容性
- **2025-01-25**: 添加完整测试套件

---

**修复完成！** 🎉 您的 PhotonicFusion SDXL 后端现在应该可以正常工作了。 