#!/usr/bin/env python3
"""
测试 Meta Tensor 修复
"""

import torch
import os
import logging
from diffusers import StableDiffusionXLPipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_meta_tensors(model, model_name):
    """检查模型中的 meta tensors"""
    meta_tensors = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.is_meta:
            meta_tensors.append(name)
    
    logger.info(f"📊 {model_name}: {total_params} 参数, {len(meta_tensors)} meta tensors")
    
    if meta_tensors:
        logger.warning(f"⚠️ Meta tensors 在 {model_name}:")
        for name in meta_tensors[:5]:  # 只显示前5个
            logger.warning(f"   - {name}")
        if len(meta_tensors) > 5:
            logger.warning(f"   ... 还有 {len(meta_tensors) - 5} 个")
    
    return len(meta_tensors) == 0

def test_model_loading():
    """测试模型加载"""
    
    model_path = "/runpod-volume/photonicfusion-sdxl"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型路径不存在: {model_path}")
        return False
    
    logger.info(f"🔄 测试从 {model_path} 加载模型...")
    
    try:
        # 使用低内存模式加载
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
            device_map=None  # 先不移动到GPU
        )
        
        logger.info("✅ 模型加载成功!")
        
        # 检查各个组件的 meta tensors
        components_ok = True
        
        for component_name in ['vae', 'text_encoder', 'text_encoder_2', 'unet']:
            component = getattr(pipeline, component_name, None)
            if component is not None:
                is_ok = check_meta_tensors(component, component_name)
                components_ok = components_ok and is_ok
            else:
                logger.warning(f"⚠️ 组件不存在: {component_name}")
        
        if components_ok:
            logger.info("✅ 所有组件都没有 meta tensors")
            
            # 尝试移动到 GPU
            if torch.cuda.is_available():
                logger.info("🔄 尝试移动到 GPU...")
                try:
                    pipeline = pipeline.to("cuda")
                    logger.info("✅ 成功移动到 GPU")
                    
                    # 简单测试
                    logger.info("🧪 运行简单测试...")
                    result = pipeline(
                        prompt="a simple test",
                        num_inference_steps=1,
                        width=64,
                        height=64,
                        output_type="pil"
                    )
                    logger.info("✅ 测试成功!")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ GPU 移动失败: {e}")
                    return False
            else:
                logger.info("ℹ️ CUDA 不可用，跳过 GPU 测试")
                return True
        else:
            logger.error("❌ 发现 meta tensors，需要修复")
            return False
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return False

def main():
    """主函数"""
    
    logger.info("🚀 开始 Meta Tensor 检查...")
    logger.info("=" * 80)
    
    success = test_model_loading()
    
    logger.info("=" * 80)
    if success:
        logger.info("🎉 所有测试通过!")
    else:
        logger.error("💥 测试失败!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 