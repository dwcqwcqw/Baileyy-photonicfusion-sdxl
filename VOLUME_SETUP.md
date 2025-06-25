# RunPod Volume 设置指南

本指南将帮助您在 RunPod 上设置 Network Volume，以实现快速的模型加载。

## 🎯 为什么使用 Volume?

- **⚡ 极快启动**: 消除 3-7 秒的模型下载时间
- **💰 节省带宽**: 避免重复下载 6GB+ 模型文件
- **🔄 可靠性**: 不依赖外部网络连接
- **📈 扩展性**: 可以在多个端点间共享模型

## 📋 设置步骤

### 第一步：创建 Network Volume

1. **登录 RunPod Console**: https://www.runpod.io/console
2. **导航到 Storage**: 在左侧菜单选择 "Storage"
3. **创建新 Volume**:
   - 点击 "Create Network Volume"
   - 名称: `photonicfusion-models`
   - 大小: 至少 10GB (推荐 15GB)
   - 区域: 选择与您的端点相同的区域

### 第二步：上传模型文件

#### 方法 1: 使用 RunPod Pod

1. **创建临时 Pod**:
   ```bash
   # 选择任意 GPU 实例
   # 挂载刚创建的 volume 到 /workspace
   ```

2. **连接到 Pod 终端**:
   ```bash
   # 进入 Pod 的 JupyterLab 或终端
   cd /workspace
   ```

3. **克隆模型**:
   ```bash
   # 使用 git-lfs 克隆
   git lfs clone https://huggingface.co/Baileyy/photonicfusion-sdxl photonicfusion-sdxl
   
   # 或者使用 huggingface-hub
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id='Baileyy/photonicfusion-sdxl',
       local_dir='photonicfusion-sdxl',
       use_auth_token=False
   )
   "
   ```

4. **验证文件结构**:
   ```bash
   ls -la photonicfusion-sdxl/
   # 应该看到:
   # - model_index.json
   # - scheduler/
   # - text_encoder/
   # - text_encoder_2/
   # - unet/
   # - vae/
   # - README.md
   ```

#### 方法 2: 本地上传 (如果有快速网络)

如果您本地有模型文件，可以使用 `rsync` 或 `scp` 上传：

```bash
# 通过 SSH 上传 (需要 Pod 的 SSH 信息)
rsync -avz --progress /local/path/to/model/ user@pod-ip:/workspace/photonicfusion-sdxl/
```

### 第三步：验证文件结构

确保 volume 中的文件结构如下：

```
/workspace/photonicfusion-sdxl/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   └── model.safetensors
├── text_encoder_2/
│   └── model.safetensors
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── README.md
```

### 第四步：配置 Serverless 端点

1. **创建 Serverless 端点**时：
   - Docker 镜像: 您的 `photonicfusion-sdxl` 镜像
   - **Volume**: 挂载 `photonicfusion-models` 到 `/runpod-volume`

2. **环境变量**:
   ```json
   {
     "LOCAL_MODEL_PATH": "/runpod-volume/photonicfusion-sdxl",
     "MODEL_NAME": "Baileyy/photonicfusion-sdxl"
   }
   ```

## 🧪 测试 Volume 设置

创建测试请求验证 volume 加载：

```python
import requests

# 测试请求
payload = {
    "input": {
        "prompt": "test image to verify volume loading",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10
    }
}

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json=payload
)

# 检查日志中是否显示 "Loading model from local volume"
```

## 📊 性能比较

| 场景 | 冷启动时间 | 总响应时间 (512x512, 10步) |
|------|------------|----------------------------|
| 无 Volume (HF下载) | ~8-15秒 | ~15-20秒 |
| 有 Volume (本地) | ~1-3秒 | ~8-12秒 |

## 🔧 故障排除

### 常见问题

1. **Volume 路径不存在**:
   ```
   ❌ Local volume not found, loading from Hugging Face Hub
   ```
   - 检查 volume 挂载路径是否正确
   - 确认文件已正确上传到 volume

2. **文件权限问题**:
   ```bash
   # 在上传 Pod 中设置正确权限
   chmod -R 755 /workspace/photonicfusion-sdxl/
   ```

3. **模型文件损坏**:
   ```bash
   # 验证关键文件
   python -c "
   import safetensors
   safetensors.safe_open('unet/diffusion_pytorch_model.safetensors', framework='pt')
   print('UNet safetensors file is valid')
   "
   ```

### Volume 维护

- **更新模型**: 在临时 Pod 中重新下载并替换文件
- **备份**: 定期创建 volume 快照
- **监控**: 检查 volume 使用情况和性能

## 💡 最佳实践

1. **多区域部署**: 在每个区域创建专用 volume
2. **版本管理**: 使用不同文件夹管理模型版本
3. **监控**: 跟踪冷启动时间改善情况
4. **成本优化**: volume 按存储计费，考虑清理不用的模型

## 🔗 相关资源

- [RunPod Volume 文档](https://docs.runpod.io/storage/network-volumes)
- [Hugging Face Hub 文档](https://huggingface.co/docs/hub/index)
- [Git LFS 安装](https://git-lfs.github.io/)

---

**注意**: Volume 设置是一次性工作，设置完成后所有使用该 volume 的端点都会获得性能提升。 