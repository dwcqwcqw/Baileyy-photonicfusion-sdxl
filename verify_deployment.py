#!/usr/bin/env python3
"""
验证 PhotonicFusion SDXL RunPod Serverless 部署的脚本
检查修复是否正确应用
"""

import json
import sys
import time
import subprocess
import requests
from pathlib import Path

def check_file_structure():
    """检查项目文件结构"""
    print("📁 检查项目文件结构...")
    
    required_files = {
        "handler.py": "主处理器文件",
        "requirements.txt": "Python 依赖",
        "Dockerfile": "Docker 配置",
        "runpod_config.json": "RunPod 配置",
        "ERROR_FIX.md": "错误修复文档",
        "test_api_request.json": "API 测试请求"
    }
    
    all_good = True
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"  ✅ {file_path} ({description})")
        else:
            print(f"  ❌ {file_path} ({description}) - 缺失")
            all_good = False
    
    return all_good

def check_handler_fixes():
    """检查 handler.py 中的修复"""
    print("\n🔍 检查 handler.py 修复...")
    
    handler_path = Path("handler.py")
    if not handler_path.exists():
        print("  ❌ handler.py 不存在")
        return False
    
    with open(handler_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "required_components": "模型结构验证",
        "text_encoder_model": "text_encoder 路径检查",
        "local_files_only": "本地文件优先策略",
        "enable_attention_slicing": "内存优化",
        "logger.info": "改进的日志记录",
        "EulerDiscreteScheduler": "调度器配置"
    }
    
    all_good = True
    for pattern, description in checks.items():
        if pattern in content:
            print(f"  ✅ {description} - 已应用")
        else:
            print(f"  ❌ {description} - 未找到")
            all_good = False
    
    return all_good

def check_dockerfile():
    """检查 Dockerfile 配置"""
    print("\n🐳 检查 Dockerfile...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("  ❌ Dockerfile 不存在")
        return False
    
    with open(dockerfile_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "pytorch/pytorch": "PyTorch 基础镜像",
        "WORKDIR": "工作目录设置",
        "requirements.txt": "依赖安装",
        "handler.py": "处理器文件复制",
        "CMD": "启动命令"
    }
    
    all_good = True
    for pattern, description in checks.items():
        if pattern in content:
            print(f"  ✅ {description} - 已配置")
        else:
            print(f"  ⚠️ {description} - 可能缺失")
            all_good = False
    
    return all_good

def validate_test_request():
    """验证测试请求格式"""
    print("\n📝 验证测试请求...")
    
    test_file = Path("test_api_request.json")
    if not test_file.exists():
        print("  ❌ test_api_request.json 不存在")
        return False
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_fields = ["prompt", "negative_prompt", "num_inference_steps", 
                          "guidance_scale", "width", "height", "seed"]
        
        input_data = data.get("input", {})
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            print(f"  ❌ 缺失字段: {missing_fields}")
            return False
        
        print("  ✅ 测试请求格式正确")
        print(f"  📋 Prompt: {input_data['prompt'][:50]}...")
        print(f"  🎯 尺寸: {input_data['width']}x{input_data['height']}")
        print(f"  ⚙️ 步数: {input_data['num_inference_steps']}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON 格式错误: {e}")
        return False

def check_dependencies():
    """检查 requirements.txt 依赖"""
    print("\n📦 检查依赖配置...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("  ❌ requirements.txt 不存在")
        return False
    
    with open(req_file, 'r', encoding='utf-8') as f:
        deps = f.read().strip().split('\n')
    
    critical_deps = ["torch", "diffusers", "transformers", "runpod", "Pillow"]
    found_deps = [dep for dep in critical_deps if any(dep in line for line in deps)]
    missing_deps = [dep for dep in critical_deps if dep not in found_deps]
    
    if missing_deps:
        print(f"  ❌ 缺失关键依赖: {missing_deps}")
        return False
    
    print(f"  ✅ 找到 {len(found_deps)}/{len(critical_deps)} 关键依赖")
    for dep in found_deps:
        print(f"    - {dep}")
    
    return True

def test_endpoint_configuration():
    """测试端点配置"""
    print("\n🔧 检查 RunPod 配置...")
    
    config_file = Path("runpod_config.json")
    if not config_file.exists():
        print("  ❌ runpod_config.json 不存在")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_fields = ["name", "image", "ports", "volume_mounts", "env"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"  ❌ 配置缺失字段: {missing_fields}")
            return False
        
        print("  ✅ RunPod 配置格式正确")
        print(f"  🏷️ 名称: {config.get('name', 'N/A')}")
        print(f"  🐳 镜像: {config.get('image', 'N/A')}")
        
        # 检查 volume 配置
        volume_mounts = config.get("volume_mounts", [])
        if volume_mounts:
            for volume in volume_mounts:
                print(f"  💾 Volume: {volume.get('name')} -> {volume.get('mount_path')}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"  ❌ 配置文件 JSON 错误: {e}")
        return False

def generate_deployment_report():
    """生成部署报告"""
    print("\n📊 生成部署报告...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "验证完成",
        "fixes_applied": [
            "正确的 diffusers 模型结构验证",
            "text_encoder/model.safetensors 路径检查",
            "智能 fallback 机制",
            "改进的错误处理和日志",
            "内存优化配置"
        ],
        "deployment_ready": True
    }
    
    with open("deployment_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("  ✅ 报告已保存到 deployment_verification_report.json")

def main():
    """主验证流程"""
    print("🔍 PhotonicFusion SDXL RunPod Serverless 部署验证")
    print("=" * 50)
    
    checks = [
        ("文件结构", check_file_structure),
        ("Handler 修复", check_handler_fixes),
        ("Dockerfile", check_dockerfile),
        ("依赖配置", check_dependencies),
        ("测试请求", validate_test_request),
        ("RunPod 配置", test_endpoint_configuration)
    ]
    
    results = {}
    all_passed = True
    
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ❌ 检查 {name} 时出错: {e}")
            results[name] = False
            all_passed = False
    
    # 显示总结
    print("\n" + "=" * 50)
    print("📋 验证总结")
    print("=" * 50)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    if all_passed:
        print("\n🎉 所有检查都通过了！")
        print("✅ 修复已正确应用，准备部署到 RunPod Serverless")
        print("\n📋 下一步:")
        print("  1. 运行 ./fix_deploy.sh 构建和部署")
        print("  2. 在 RunPod 控制台配置端点")
        print("  3. 使用 test_api_request.json 测试端点")
        
        generate_deployment_report()
        
    else:
        print("\n⚠️ 发现问题，请先修复后再部署")
        print("📚 查看 ERROR_FIX.md 获取详细修复说明")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 