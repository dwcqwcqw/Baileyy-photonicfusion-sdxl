#!/usr/bin/env python3
"""
删除HuggingFace上的老文件(标准safetensors文件)
只保留FP16版本和必要的配置文件
"""

import os
from huggingface_hub import HfApi, delete_file

def delete_old_files_from_huggingface():
    """删除HuggingFace仓库中的老文件"""
    
    repo_id = "Baileyy/photonicfusion-sdxl"
    
    print("🗑️ HuggingFace 老文件清理器")
    print("=" * 40)
    print(f"🌐 仓库: {repo_id}")
    
    # 需要删除的老文件列表
    files_to_delete = [
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "unet/diffusion_pytorch_model.safetensors", 
        "vae/diffusion_pytorch_model.safetensors",
        "test_yaml_fix.py"  # 这个测试文件也不需要
    ]
    
    try:
        api = HfApi()
        print(f"✅ 已连接到HuggingFace Hub")
        
        # 获取当前文件列表
        print("\n🔍 检查当前文件...")
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        
        print("📁 当前仓库文件:")
        for file in sorted(repo_files):
            file_type = "fp16" if "fp16" in file else "standard" if "safetensors" in file else "config"
            print(f"   📄 {file} [{file_type}]")
        
        print(f"\n🗑️ 删除标准safetensors文件...")
        
        deleted_count = 0
        for file_path in files_to_delete:
            if file_path in repo_files:
                try:
                    delete_file(
                        path_in_repo=file_path,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"删除老文件: {file_path}"
                    )
                    print(f"   ✅ 已删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ❌ 删除失败 {file_path}: {e}")
            else:
                print(f"   ⚠️ 文件不存在: {file_path}")
        
        print(f"\n📊 删除统计: 已删除 {deleted_count} 个文件")
        
        # 验证剩余文件
        print(f"\n🔍 验证剩余文件...")
        updated_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        
        remaining_files = [
            "model_index.json",
            "text_encoder/model.fp16.safetensors", 
            "text_encoder_2/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json", 
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "scheduler/scheduler_config.json",
            "README.md",
            ".gitattributes"
        ]
        
        print("📁 期望的文件列表:")
        for file in remaining_files:
            if file in updated_files:
                file_type = "fp16" if "fp16" in file else "config"
                print(f"   ✅ {file} [{file_type}]")
            else:
                print(f"   ❌ 缺失: {file}")
        
        # 检查是否还有不应该存在的标准safetensors文件
        standard_files = [f for f in updated_files if "safetensors" in f and "fp16" not in f]
        if standard_files:
            print(f"\n⚠️ 仍存在标准文件: {standard_files}")
            return False
        else:
            print(f"\n✅ 清理完成！仓库现在只包含FP16版本")
            print(f"🌐 查看: https://huggingface.co/{repo_id}")
            return True
            
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 HuggingFace 老文件删除工具")
    print("此工具将删除HuggingFace上的标准safetensors文件")
    print("=" * 50)
    
    # 确认操作
    confirm = input("\n⚠️ 这将从HuggingFace永久删除标准safetensors文件。继续吗? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    if delete_old_files_from_huggingface():
        print("\n🎉 HuggingFace文件清理完成!")
        print("模型现在完全优化为FP16版本")
    else:
        print("\n❌ 清理过程中出现问题")

if __name__ == "__main__":
    main() 