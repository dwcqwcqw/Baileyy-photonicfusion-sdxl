#!/usr/bin/env python3
"""
验证 PhotonicFusion SDXL RunPod 部署
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from io import BytesIO
from PIL import Image

# 默认 RunPod 端点 ID
DEFAULT_ENDPOINT_ID = "9u6js61unnr7p1"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="验证 PhotonicFusion SDXL RunPod 部署")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT_ID,
                        help=f"RunPod 端点 ID (默认: {DEFAULT_ENDPOINT_ID})")
    parser.add_argument("--api-key", type=str, default=os.environ.get("RUNPOD_API_KEY"),
                        help="RunPod API 密钥 (默认: 从环境变量 RUNPOD_API_KEY 读取)")
    parser.add_argument("--prompt", type=str, 
                        default="a beautiful landscape with mountains, high quality, photorealistic",
                        help="测试提示词")
    parser.add_argument("--output", type=str, default="verification_output.png",
                        help="输出图像文件名 (默认: verification_output.png)")
    parser.add_argument("--request-file", type=str, default="test_api_request.json",
                        help="JSON 请求文件 (默认: test_api_request.json)")
    
    return parser.parse_args()

def load_request_data(file_path, prompt=None):
    """加载请求数据"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # 如果提供了提示词，则更新请求中的提示词
        if prompt:
            data["input"]["prompt"] = prompt
            
        return data
    except Exception as e:
        print(f"❌ 加载请求数据失败: {str(e)}")
        return None

def send_request(endpoint_id, api_key, data):
    """发送请求到 RunPod 端点"""
    if not api_key:
        print("❌ 未提供 API 密钥。请使用 --api-key 参数或设置 RUNPOD_API_KEY 环境变量。")
        return None
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        print(f"📤 发送请求到端点: {endpoint_id}")
        print(f"   提示词: {data['input']['prompt'][:50]}...")
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return None

def check_status(endpoint_id, api_key, task_id):
    """检查任务状态"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    max_attempts = 60  # 最多等待 5 分钟 (60 * 5 秒)
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            status = result.get("status")
            
            if status == "COMPLETED":
                return result
            elif status == "FAILED":
                print(f"❌ 任务失败: {result.get('error')}")
                return None
            
            # 继续等待
            attempt += 1
            wait_time = 5  # 每 5 秒检查一次
            print(f"⏳ 任务正在处理中... ({attempt}/{max_attempts})")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"❌ 检查状态失败: {str(e)}")
            return None
    
    print("❌ 等待超时")
    return None

def save_image(result, output_file):
    """保存结果图像"""
    try:
        # 检查是否有多张图像
        if "images" in result["output"]:
            images = result["output"]["images"]
            image_data = images[0] if isinstance(images, list) else images
        elif "image" in result["output"]:
            image_data = result["output"]["image"]
        else:
            print("❌ 响应中没有图像数据")
            return False
        
        # 解码 base64 图像
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # 保存图像
        image.save(output_file)
        print(f"✅ 图像已保存到: {output_file}")
        
        # 显示图像信息
        print(f"   尺寸: {image.size}")
        print(f"   格式: {image.format}")
        
        return True
    except Exception as e:
        print(f"❌ 保存图像失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔍 PhotonicFusion SDXL - 部署验证")
    print("================================")
    
    # 解析参数
    args = parse_args()
    
    # 检查 API 密钥
    if not args.api_key:
        print("❌ 未提供 RunPod API 密钥")
        print("请使用 --api-key 参数或设置 RUNPOD_API_KEY 环境变量")
        return False
    
    # 加载请求数据
    data = load_request_data(args.request_file, args.prompt)
    if not data:
        return False
    
    # 发送请求
    start_time = time.time()
    response = send_request(args.endpoint, args.api_key, data)
    if not response:
        return False
    
    # 获取任务 ID
    task_id = response.get("id")
    if not task_id:
        print("❌ 响应中没有任务 ID")
        return False
    
    print(f"✅ 请求已提交，任务 ID: {task_id}")
    
    # 检查任务状态
    result = check_status(args.endpoint, args.api_key, task_id)
    if not result:
        return False
    
    # 计算总时间
    total_time = time.time() - start_time
    print(f"✅ 任务完成，用时: {total_time:.2f} 秒")
    
    # 保存图像
    if not save_image(result, args.output):
        return False
    
    print("\n🎉 验证成功! 部署工作正常。")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 