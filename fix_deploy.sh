#!/bin/bash

# PhotonicFusion SDXL RunPod Serverless - 修复部署脚本
# 这个脚本部署修复后的版本到 RunPod

set -e

echo "🔧 PhotonicFusion SDXL - 修复部署脚本"
echo "====================================="

# 检查必需文件
echo "📋 检查必需文件..."
required_files=("handler.py" "requirements.txt" "Dockerfile" "runpod_config.json")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ 缺失文件: $file"
        exit 1
    fi
    echo "✅ $file"
done

# 验证 handler.py 中的修复
echo -e "\n🔍 验证 handler.py 修复..."
if grep -q "required_components" handler.py; then
    echo "✅ 模型结构验证已添加"
else
    echo "❌ 警告: handler.py 可能未包含最新修复"
fi

if grep -q "text_encoder.*model.safetensors" handler.py; then
    echo "✅ text_encoder 路径验证已添加"
else
    echo "❌ 警告: text_encoder 验证可能缺失"
fi

# 构建 Docker 镜像
echo -e "\n🐳 构建 Docker 镜像..."
IMAGE_NAME="photonicfusion-sdxl-fixed"
docker build -t $IMAGE_NAME . || {
    echo "❌ Docker 构建失败"
    exit 1
}
echo "✅ Docker 镜像构建成功: $IMAGE_NAME"

# 推送到容器注册表 (如果配置了)
if [[ ! -z "$RUNPOD_REGISTRY" ]]; then
    echo -e "\n📤 推送镜像到注册表..."
    docker tag $IMAGE_NAME $RUNPOD_REGISTRY/$IMAGE_NAME:latest
    docker push $RUNPOD_REGISTRY/$IMAGE_NAME:latest
    echo "✅ 镜像已推送到: $RUNPOD_REGISTRY/$IMAGE_NAME:latest"
else
    echo "ℹ️ 未配置 RUNPOD_REGISTRY，跳过推送"
fi

# 显示部署信息
echo -e "\n📋 部署信息"
echo "=========================="
echo "镜像名称: $IMAGE_NAME"
echo "修复版本: $(date '+%Y%m%d-%H%M%S')"
echo ""
echo "🔧 主要修复:"
echo "  ✅ 正确的 diffusers 模型结构验证"
echo "  ✅ text_encoder/model.safetensors 路径检查"
echo "  ✅ 智能 fallback 机制"
echo "  ✅ 改进的错误处理和日志"
echo "  ✅ 内存优化"
echo ""
echo "📊 预期性能:"
echo "  🚀 冷启动: 1-3秒 (使用 volume)"
echo "  💾 内存使用: ~8-12GB"
echo "  ⚡ 生成速度: ~2-4秒/张 (1024x1024)"

# 生成部署建议
echo -e "\n💡 部署建议"
echo "=========================="
echo "1. 确保 RunPod Volume 配置:"
echo "   - Volume 名称: photonicfusion-models"
echo "   - 挂载路径: /runpod-volume"
echo "   - 模型路径: /runpod-volume/photonicfusion-sdxl/"
echo ""
echo "2. 容器资源配置:"
echo "   - GPU: RTX 3090/4090 或更好"
echo "   - 内存: 24GB+"
echo "   - 磁盘: 20GB+"
echo ""
echo "3. 环境变量:"
echo "   export TORCH_CUDA_ARCH_LIST=\"7.0;7.5;8.0;8.6\""
echo "   export CUDA_VISIBLE_DEVICES=\"0\""
echo ""
echo "4. 测试端点:"
echo "   curl -X POST https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync \\"
echo "     -H \"Authorization: Bearer \$RUNPOD_API_KEY\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"input\": {\"prompt\": \"a beautiful landscape\"}}'"

# 创建测试请求文件
echo -e "\n📝 创建测试请求文件..."
cat > test_api_request.json << 'EOF'
{
  "input": {
    "prompt": "a beautiful photorealistic landscape with mountains and a serene lake, golden hour lighting, 8k resolution",
    "negative_prompt": "low quality, blurry, distorted, watermark",
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }
}
EOF
echo "✅ 测试请求已保存到 test_api_request.json"

echo -e "\n✨ 修复部署完成!"
echo "现在可以使用修复后的版本在 RunPod Serverless 上部署了。"
echo ""
echo "📚 更多信息请查看:"
echo "  - ERROR_FIX.md - 详细的错误分析和修复说明"
echo "  - VOLUME_SETUP.md - Volume 配置指南"
echo "  - DEPLOYMENT.md - 完整部署文档" 