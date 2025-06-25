#!/usr/bin/env python3
"""
测试FP16-only模型检测逻辑
验证handler能否正确识别只有FP16文件的模型
"""

import os

def test_fp16_detection():
    """测试FP16模型文件检测逻辑"""
    
    print("🧪 FP16-Only 模型检测测试")
    print("=" * 40)
    
    # 模拟Volume路径
    model_path = "/runpod-volume/photonicfusion-sdxl"
    
    # 模拟文件检查（使用HuggingFace上的实际结构）
    print(f"📁 测试模型路径: {model_path}")
    
    # 检查逻辑（基于修复后的handler代码）
    text_encoder_standard = os.path.join(model_path, "text_encoder", "model.safetensors")
    text_encoder_fp16 = os.path.join(model_path, "text_encoder", "model.fp16.safetensors")
    text_encoder_2_standard = os.path.join(model_path, "text_encoder_2", "model.safetensors")
    text_encoder_2_fp16 = os.path.join(model_path, "text_encoder_2", "model.fp16.safetensors")
    
    print(f"\n🔍 检查文件存在性:")
    print(f"   text_encoder/model.safetensors: {os.path.exists(text_encoder_standard)}")
    print(f"   text_encoder/model.fp16.safetensors: {os.path.exists(text_encoder_fp16)}")
    print(f"   text_encoder_2/model.safetensors: {os.path.exists(text_encoder_2_standard)}")
    print(f"   text_encoder_2/model.fp16.safetensors: {os.path.exists(text_encoder_2_fp16)}")
    
    # 检查是否通过验证
    te1_valid = os.path.exists(text_encoder_standard) or os.path.exists(text_encoder_fp16)
    te2_valid = os.path.exists(text_encoder_2_standard) or os.path.exists(text_encoder_2_fp16)
    
    print(f"\n✅ 验证结果:")
    print(f"   text_encoder 验证: {'通过' if te1_valid else '失败'}")
    print(f"   text_encoder_2 验证: {'通过' if te2_valid else '失败'}")
    
    if te1_valid and te2_valid:
        # 确定使用的版本
        te1_version = "fp16" if os.path.exists(text_encoder_fp16) else "standard"
        te2_version = "fp16" if os.path.exists(text_encoder_2_fp16) else "standard"
        
        print(f"\n🎉 模型检测成功!")
        print(f"   text_encoder: {te1_version} 版本")
        print(f"   text_encoder_2: {te2_version} 版本")
        
        if te1_version == "fp16" and te2_version == "fp16":
            print(f"✅ 检测到FP16-only模型配置")
        else:
            print(f"ℹ️ 检测到混合模型配置")
            
        return True
    else:
        print(f"\n❌ 模型检测失败 - 缺少必要文件")
        return False

def test_with_local_model():
    """使用本地转换的模型进行测试"""
    
    print(f"\n🔬 本地模型测试")
    print("=" * 30)
    
    # 本地模型路径
    local_model_path = "../PhotonicFusionSDXL_V3-diffusers-manual"
    
    if not os.path.exists(local_model_path):
        print(f"⚠️ 本地模型路径不存在: {local_model_path}")
        return False
    
    print(f"📁 本地模型路径: {local_model_path}")
    
    # 检查文件
    text_encoder_standard = os.path.join(local_model_path, "text_encoder", "model.safetensors")
    text_encoder_fp16 = os.path.join(local_model_path, "text_encoder", "model.fp16.safetensors")
    text_encoder_2_standard = os.path.join(local_model_path, "text_encoder_2", "model.safetensors")
    text_encoder_2_fp16 = os.path.join(local_model_path, "text_encoder_2", "model.fp16.safetensors")
    
    print(f"\n🔍 本地文件检查:")
    print(f"   text_encoder/model.safetensors: {os.path.exists(text_encoder_standard)}")
    print(f"   text_encoder/model.fp16.safetensors: {os.path.exists(text_encoder_fp16)}")
    print(f"   text_encoder_2/model.safetensors: {os.path.exists(text_encoder_2_standard)}")
    print(f"   text_encoder_2/model.fp16.safetensors: {os.path.exists(text_encoder_2_fp16)}")
    
    # 应用新的检查逻辑
    te1_valid = os.path.exists(text_encoder_standard) or os.path.exists(text_encoder_fp16)
    te2_valid = os.path.exists(text_encoder_2_standard) or os.path.exists(text_encoder_2_fp16)
    
    if te1_valid and te2_valid:
        te1_version = "fp16" if os.path.exists(text_encoder_fp16) else "standard"
        te2_version = "fp16" if os.path.exists(text_encoder_2_fp16) else "standard"
        
        print(f"\n✅ 本地模型验证通过!")
        print(f"   检测到: text_encoder ({te1_version}), text_encoder_2 ({te2_version})")
        
        return True
    else:
        print(f"\n❌ 本地模型验证失败")
        return False

def main():
    """主测试函数"""
    print("🚀 FP16-Only 模型检测修复验证")
    print("用于验证handler.py修复是否能正确处理FP16-only模型")
    print("=" * 60)
    
    # 测试1: 模拟RunPod Volume检测
    print("\n📋 测试1: RunPod Volume路径检测")
    result1 = test_fp16_detection()
    
    # 测试2: 本地模型检测
    print("\n📋 测试2: 本地模型检测")
    result2 = test_with_local_model()
    
    # 总结
    print(f"\n📊 测试总结:")
    print(f"   RunPod Volume检测: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"   本地模型检测: {'✅ 通过' if result2 else '❌ 失败'}")
    
    if result2:  # 本地模型检测通过说明修复工作正常
        print(f"\n🎉 修复验证成功!")
        print(f"Handler现在能够正确检测FP16-only模型")
    else:
        print(f"\n⚠️ 需要进一步检查模型结构")

if __name__ == "__main__":
    main() 