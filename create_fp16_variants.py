#!/usr/bin/env python3
"""
为现有的 diffusers 模型创建 fp16 variant 文件
这是一个手动解决方案，避免完全重新转换模型
"""

import os
import shutil
import json

def create_fp16_variants():
    """为现有模型创建 fp16 variant 文件"""
    
    # 模型路径
    model_path = "../PhotonicFusionSDXL_V3-diffusers-manual"
    
    print("🔧 PhotonicFusion SDXL FP16 Variant 创建器")
    print("=" * 55)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    print(f"📁 模型路径: {model_path}")
    
    # 需要创建 fp16 variant 的文件
    files_to_copy = [
        ("text_encoder/model.safetensors", "text_encoder/model.fp16.safetensors"),
        ("text_encoder_2/model.safetensors", "text_encoder_2/model.fp16.safetensors"),
        ("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors"),
        ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors")
    ]
    
    print("\n🔄 创建 fp16 variant 文件...")
    
    success_count = 0
    total_size = 0
    
    for src_file, dst_file in files_to_copy:
        src_path = os.path.join(model_path, src_file)
        dst_path = os.path.join(model_path, dst_file)
        
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                try:
                    # 复制文件作为 fp16 variant
                    shutil.copy2(src_path, dst_path)
                    
                    # 获取文件大小
                    size = os.path.getsize(dst_path)
                    total_size += size
                    size_mb = size / (1024 * 1024)
                    
                    print(f"   ✅ {dst_file} ({size_mb:.1f} MB)")
                    success_count += 1
                except Exception as e:
                    print(f"   ❌ 复制失败 {dst_file}: {e}")
            else:
                size = os.path.getsize(dst_path)
                total_size += size
                size_mb = size / (1024 * 1024)
                print(f"   ✅ {dst_file} (已存在, {size_mb:.1f} MB)")
                success_count += 1
        else:
            print(f"   ❌ 源文件不存在: {src_file}")
    
    print(f"\n📊 创建结果:")
    print(f"   成功: {success_count}/{len(files_to_copy)}")
    print(f"   总大小: {total_size / (1024**3):.2f} GB")
    
    # 验证文件结构
    print("\n🔍 验证模型结构...")
    
    required_standard = [
        "model_index.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors", 
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors"
    ]
    
    required_fp16 = [
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors", 
        "vae/diffusion_pytorch_model.fp16.safetensors"
    ]
    
    print("\n📁 标准文件:")
    std_ok = True
    for file in required_standard:
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)
            print(f"   ✅ {file} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file}")
            std_ok = False
    
    print("\n📁 FP16 variant 文件:")
    fp16_ok = True
    for file in required_fp16:
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)
            print(f"   ✅ {file} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file}")
            fp16_ok = False
    
    if std_ok and fp16_ok:
        print("\n🎉 FP16 variant 创建成功!")
        print("\n📋 下一步:")
        print(f"   1. 将 {model_path} 上传到 RunPod Volume")
        print("   2. 使用更新后的 handler.py 部署")
        print("   3. 测试 fp16 variant 加载")
        return True
    else:
        print("\n⚠️ 有文件缺失，请检查上述错误")
        return False

def test_structure():
    """测试模型结构是否正确"""
    model_path = "../PhotonicFusionSDXL_V3-diffusers-manual"
    
    print("\n🧪 测试模型结构...")
    
    # 检查 model_index.json
    index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            print("✅ model_index.json 有效")
            print(f"   组件: {list(index_data.keys())}")
        except Exception as e:
            print(f"❌ model_index.json 解析失败: {e}")
    else:
        print("❌ model_index.json 不存在")
    
    # 检查每个组件目录
    components = ["text_encoder", "text_encoder_2", "unet", "vae", "scheduler"]
    
    for comp in components:
        comp_path = os.path.join(model_path, comp)
        if os.path.exists(comp_path):
            files = os.listdir(comp_path)
            print(f"✅ {comp}/: {files}")
        else:
            print(f"❌ {comp}/ 不存在")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_structure()
    else:
        success = create_fp16_variants()
        if success:
            test_structure()
        return success

if __name__ == "__main__":
    main() 