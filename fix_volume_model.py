#!/usr/bin/env python3
"""
RunPod Volume 模型修复脚本
专门解决 "expected str, bytes or os.PathLike object, not NoneType" 错误

使用方法:
python fix_volume_model.py
"""

import os
import json
import sys

def fix_model_index(volume_path):
    """修复 model_index.json 中的 None 值"""
    model_index_path = os.path.join(volume_path, "model_index.json")
    
    print(f"🔍 检查 {model_index_path}")
    
    if not os.path.exists(model_index_path):
        print(f"❌ model_index.json 不存在")
        return False
    
    try:
        # 读取现有文件
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        
        print(f"📋 当前组件映射:")
        
        # 检查并修复 None 值
        fixed = False
        for key, value in model_index.items():
            if not key.startswith('_') and isinstance(value, list) and len(value) >= 2:
                component_type, component_name = value[0], value[1]
                
                if component_name is None or component_name == "null":
                    print(f"⚠️  {key}: {component_type} -> None (需要修复)")
                    
                    # 自动修复策略
                    if key == "feature_extractor":
                        model_index[key] = ["transformers", "CLIPImageProcessor"]
                        fixed = True
                        print(f"🔧 修复: {key} -> CLIPImageProcessor")
                    elif key == "image_encoder": 
                        model_index[key] = ["transformers", "CLIPVisionModelWithProjection"]
                        fixed = True
                        print(f"🔧 修复: {key} -> CLIPVisionModelWithProjection")
                    elif key == "safety_checker":
                        model_index[key] = [None, None]
                        print(f"🔧 设置: {key} -> null (禁用安全检查)")
                    else:
                        print(f"❓ 未知组件: {key} (跳过)")
                else:
                    print(f"✅ {key}: {component_type} -> {component_name}")
        
        # 保存修复后的文件
        if fixed:
            # 备份原文件
            backup_path = model_index_path + ".backup"
            os.rename(model_index_path, backup_path)
            print(f"💾 备份: {backup_path}")
            
            # 保存修复后的文件
            with open(model_index_path, 'w') as f:
                json.dump(model_index, f, indent=2)
            print(f"✅ 保存修复后的 model_index.json")
            return True
        else:
            print(f"ℹ️  无需修复")
            return True
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

def create_missing_configs(volume_path):
    """创建缺失的配置文件"""
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
        },
        
        "scheduler/scheduler_config.json": {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.21.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear", 
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
            "use_karras_sigmas": False
        }
    }
    
    print(f"\n🔧 检查并创建缺失的配置文件:")
    created_count = 0
    
    for config_path, config_content in configs.items():
        full_path = os.path.join(volume_path, config_path)
        
        if not os.path.exists(full_path):
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    json.dump(config_content, f, indent=2)
                print(f"✅ 创建: {config_path}")
                created_count += 1
            except Exception as e:
                print(f"❌ 创建 {config_path} 失败: {e}")
        else:
            print(f"⏭️  已存在: {config_path}")
    
    return created_count

def main():
    """主函数"""
    volume_path = "/runpod-volume/photonicfusion-sdxl"
    
    print("🚀 RunPod Volume 模型修复工具")
    print("=" * 50)
    
    # 检查Volume路径
    if not os.path.exists(volume_path):
        print(f"❌ Volume路径不存在: {volume_path}")
        print("请确保模型已正确上传到Volume")
        sys.exit(1)
    
    print(f"📂 Volume路径: {volume_path}")
    
    # 检查必需组件
    required_components = ["model_index.json", "unet", "vae", "text_encoder", "text_encoder_2"]
    missing_components = []
    
    print(f"\n📋 检查必需组件:")
    for component in required_components:
        component_path = os.path.join(volume_path, component)
        if os.path.exists(component_path):
            if os.path.isdir(component_path):
                file_count = len(os.listdir(component_path))
                print(f"✅ {component}/ ({file_count} 文件)")
            else:
                print(f"✅ {component}")
        else:
            print(f"❌ {component} (缺失)")
            missing_components.append(component)
    
    if missing_components:
        print(f"\n❌ 关键组件缺失: {missing_components}")
        print("请重新上传完整的模型文件")
        sys.exit(1)
    
    # 修复 model_index.json
    print(f"\n🔧 修复 model_index.json:")
    if not fix_model_index(volume_path):
        print("❌ model_index.json 修复失败")
        sys.exit(1)
    
    # 创建缺失的配置文件
    created = create_missing_configs(volume_path)
    
    # 总结
    print(f"\n📊 修复总结:")
    print(f"✅ model_index.json: 已检查并修复")
    print(f"✅ 配置文件: 创建了 {created} 个")
    
    print(f"\n🎉 修复完成! 现在可以重启您的RunPod实例了")

if __name__ == "__main__":
    main() 