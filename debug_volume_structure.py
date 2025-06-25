#!/usr/bin/env python3
"""
调试脚本：检查RunPod Volume中的模型文件结构
找出导致 NoneType 错误的缺失文件
"""

import os
import json

def check_volume_structure():
    """检查Volume中的模型文件结构"""
    volume_path = "/runpod-volume/photonicfusion-sdxl"
    
    print(f"🔍 检查模型目录: {volume_path}")
    
    if not os.path.exists(volume_path):
        print("❌ Volume路径不存在!")
        return
    
    # 检查必需的SDXL组件
    required_components = [
        "model_index.json",
        "unet",
        "vae", 
        "text_encoder",
        "text_encoder_2",
        "scheduler",
        "tokenizer",
        "tokenizer_2"
    ]
    
    print("\n📋 检查必需组件:")
    missing_components = []
    
    for component in required_components:
        component_path = os.path.join(volume_path, component)
        if os.path.exists(component_path):
            if os.path.isdir(component_path):
                files = os.listdir(component_path)
                print(f"✅ {component}/ - {len(files)} 个文件")
                
                # 检查关键配置文件
                config_file = os.path.join(component_path, "config.json")
                if os.path.exists(config_file):
                    print(f"   └── config.json ✅")
                else:
                    print(f"   └── config.json ❌")
                    missing_components.append(f"{component}/config.json")
                
                # 检查模型文件
                model_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]
                if model_files:
                    print(f"   └── 模型文件: {model_files}")
                else:
                    print(f"   └── 模型文件: ❌ 未找到")
                    missing_components.append(f"{component}/model files")
            else:
                print(f"✅ {component} - 文件")
        else:
            print(f"❌ {component} - 缺失")
            missing_components.append(component)
    
    # 检查 model_index.json 内容
    model_index_path = os.path.join(volume_path, "model_index.json")
    if os.path.exists(model_index_path):
        print(f"\n📄 检查 model_index.json:")
        try:
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            print(f"   架构: {model_index.get('_class_name', 'Unknown')}")
            print(f"   组件映射:")
            
            for key, value in model_index.items():
                if not key.startswith('_'):
                    if isinstance(value, list) and len(value) >= 2:
                        component_type, component_name = value[0], value[1] if value[1] else "null"
                        status = "✅" if component_name != "null" else "❌"
                        print(f"     {key}: {component_type} -> {component_name} {status}")
                        
                        if component_name == "null":
                            missing_components.append(f"model_index.json:{key}")
                            
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            missing_components.append("model_index.json (corrupt)")
    
    # 查找FP16文件
    print(f"\n🔍 查找FP16文件:")
    fp16_files = []
    for root, dirs, files in os.walk(volume_path):
        for file in files:
            if file.endswith('.fp16.safetensors'):
                rel_path = os.path.relpath(os.path.join(root, file), volume_path)
                fp16_files.append(rel_path)
    
    if fp16_files:
        print(f"✅ 找到 {len(fp16_files)} 个FP16文件:")
        for fp16_file in fp16_files:
            print(f"   - {fp16_file}")
    else:
        print("❌ 未找到FP16文件")
    
    # 总结
    print(f"\n📊 检查总结:")
    if missing_components:
        print(f"❌ 发现 {len(missing_components)} 个问题:")
        for issue in missing_components:
            print(f"   - {issue}")
    else:
        print("✅ 所有必需组件都存在")
    
    return missing_components

def create_missing_configs():
    """创建缺失的配置文件"""
    volume_path = "/runpod-volume/photonicfusion-sdxl"
    
    # 标准SDXL配置模板
    configs = {
        "tokenizer/tokenizer_config.json": {
            "add_prefix_space": False,
            "bos_token": {"__type": "AddedToken", "content": "<|startoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
            "clean_up_tokenization_spaces": True,
            "do_lower_case": True,
            "eos_token": {"__type": "AddedToken", "content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
            "errors": "replace",
            "model_max_length": 77,
            "name_or_path": "openai/clip-vit-large-patch14",
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "CLIPTokenizer",
            "unk_token": {"__type": "AddedToken", "content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False}
        },
        
        "tokenizer_2/tokenizer_config.json": {
            "add_prefix_space": False,
            "bos_token": {"__type": "AddedToken", "content": "<|startoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
            "clean_up_tokenization_spaces": True,
            "do_lower_case": True,
            "eos_token": {"__type": "AddedToken", "content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
            "errors": "replace",
            "model_max_length": 77,
            "name_or_path": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "CLIPTokenizer",
            "unk_token": {"__type": "AddedToken", "content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False}
        }
    }
    
    print("🔧 创建缺失的配置文件:")
    
    for config_path, config_content in configs.items():
        full_path = os.path.join(volume_path, config_path)
        
        if not os.path.exists(full_path):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                json.dump(config_content, f, indent=2)
            
            print(f"✅ 创建: {config_path}")
        else:
            print(f"⏭️  已存在: {config_path}")

if __name__ == "__main__":
    print("🚀 开始调试Volume结构...")
    
    missing = check_volume_structure()
    
    if missing:
        print(f"\n🔧 尝试修复缺失的配置...")
        create_missing_configs()
        
        print(f"\n🔄 重新检查...")
        check_volume_structure()
    
    print(f"\n✅ 调试完成!") 