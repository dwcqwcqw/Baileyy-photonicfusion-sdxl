# PhotonicFusion SDXL RunPod 部署指南

## 🎯 项目概述

基于我们手动转换的 diffusers 格式 PhotonicFusion SDXL V3 模型，创建了一个完整的 RunPod Serverless 解决方案。

## 📦 核心组件

- **handler.py**: RunPod Serverless 主处理器
- **requirements.txt**: Python 依赖包列表
- **Dockerfile**: Docker 容器配置
- **test_local.py**: 本地测试脚本
- **api_examples.py**: API 使用示例
- **deploy.sh**: 部署自动化脚本

## 🚀 快速部署

1. 克隆仓库: `git clone https://github.com/dwcqwcqw/Baileyy-photonicfusion-sdxl.git`
2. 构建镜像: `./deploy.sh build`
3. 推送到仓库: 更新 deploy.sh 中的 registry 地址
4. 在 RunPod 创建 Serverless Endpoint

## 📊 性能

- GPU: RTX A4000+
- 内存: 12GB+
- 生成时间: ~15秒 (1024x1024, 30步)

详细说明请查看 README.md
