#!/usr/bin/env python3
"""
检查和修复配置文件中的 None 值
"""

import json
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_scheduler_config():
    """修复 scheduler 配置文件中的 None 值"""
    
    model_path = "/runpod-volume/photonicfusion-sdxl"
    scheduler_config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
    
    if not os.path.exists(scheduler_config_path):
        logger.error(f"❌ Scheduler 配置文件不存在: {scheduler_config_path}")
        return False
    
    try:
        # 读取当前配置
        with open(scheduler_config_path, 'r') as f:
            config = json.load(f)
        
        logger.info("🔍 检查 scheduler 配置...")
        
        # 检查并修复常见的 None 值问题
        fixes_applied = []
        
        # 检查关键参数
        critical_params = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
            "clip_sample": False,
            "set_alpha_to_one": False,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "skip_prk_steps": True,
            "use_karras_sigmas": False,
            "sample_max_value": 1.0
        }
        
        for key, default_value in critical_params.items():
            if key not in config or config[key] is None:
                logger.warning(f"⚠️ 修复 {key}: {config.get(key)} -> {default_value}")
                config[key] = default_value
                fixes_applied.append(key)
            else:
                logger.info(f"✅ {key}: {config[key]}")
        
        # 检查 trained_betas（这个可以是 None）
        if "trained_betas" not in config:
            config["trained_betas"] = None
            logger.info("ℹ️ 设置 trained_betas: null")
        
        # 如果有修复，保存文件
        if fixes_applied:
            # 备份原文件
            backup_path = scheduler_config_path + ".backup"
            os.rename(scheduler_config_path, backup_path)
            logger.info(f"💾 备份原文件: {backup_path}")
            
            # 保存修复后的配置
            with open(scheduler_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ 已修复 {len(fixes_applied)} 个参数: {fixes_applied}")
        else:
            logger.info("✅ Scheduler 配置无需修复")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 修复 scheduler 配置失败: {e}")
        return False

def fix_text_encoder_configs():
    """修复 text_encoder 配置中的 None 值"""
    
    model_path = "/runpod-volume/photonicfusion-sdxl"
    
    for encoder_name in ["text_encoder", "text_encoder_2"]:
        config_path = os.path.join(model_path, encoder_name, "config.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ {encoder_name} 配置不存在: {config_path}")
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"🔍 检查 {encoder_name} 配置...")
            
            fixes_applied = []
            
            # 检查关键参数
            critical_params = {
                "vocab_size": 49408 if encoder_name == "text_encoder" else 49408,
                "hidden_size": 768 if encoder_name == "text_encoder" else 1280,
                "intermediate_size": 3072 if encoder_name == "text_encoder" else 5120,
                "num_hidden_layers": 12 if encoder_name == "text_encoder" else 32,
                "num_attention_heads": 12 if encoder_name == "text_encoder" else 20,
                "max_position_embeddings": 77,
                "hidden_act": "quick_gelu",
                "layer_norm_eps": 1e-05,
                "attention_dropout": 0.0,
                "initializer_range": 0.02,
                "initializer_factor": 1.0,
                "pad_token_id": 1,
                "bos_token_id": 0,
                "eos_token_id": 2
            }
            
            for key, default_value in critical_params.items():
                if key in config and config[key] is None:
                    logger.warning(f"⚠️ 修复 {encoder_name}.{key}: None -> {default_value}")
                    config[key] = default_value
                    fixes_applied.append(key)
            
            # 如果有修复，保存文件
            if fixes_applied:
                backup_path = config_path + ".backup"
                os.rename(config_path, backup_path)
                logger.info(f"💾 备份 {encoder_name} 配置: {backup_path}")
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"✅ 已修复 {encoder_name} 的 {len(fixes_applied)} 个参数")
            else:
                logger.info(f"✅ {encoder_name} 配置无需修复")
                
        except Exception as e:
            logger.error(f"❌ 修复 {encoder_name} 配置失败: {e}")

def fix_unet_config():
    """修复 UNet 配置中的 None 值"""
    
    model_path = "/runpod-volume/photonicfusion-sdxl"
    config_path = os.path.join(model_path, "unet", "config.json")
    
    if not os.path.exists(config_path):
        logger.warning(f"⚠️ UNet 配置不存在: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info("🔍 检查 UNet 配置...")
        
        fixes_applied = []
        
        # 检查关键参数
        critical_params = {
            "sample_size": 128,
            "in_channels": 4,
            "out_channels": 4,
            "down_block_types": [
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D"
            ],
            "up_block_types": [
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D"
            ],
            "block_out_channels": [320, 640, 1280],
            "layers_per_block": 2,
            "attention_head_dim": [5, 10, 20],
            "num_attention_heads": None,  # 这个可以是 None
            "cross_attention_dim": [2048, 2048],
            "norm_num_groups": 32,
            "use_linear_projection": True,
            "class_embed_type": None,  # 这个可以是 None
            "num_class_embeds": None,  # 这个可以是 None
            "upcast_attention": None,  # 这个可以是 None
            "resnet_time_scale_shift": "default"
        }
        
        for key, default_value in critical_params.items():
            if key in config and config[key] is None and default_value is not None:
                logger.warning(f"⚠️ 修复 unet.{key}: None -> {default_value}")
                config[key] = default_value
                fixes_applied.append(key)
        
        # 如果有修复，保存文件
        if fixes_applied:
            backup_path = config_path + ".backup"
            os.rename(config_path, backup_path)
            logger.info(f"💾 备份 UNet 配置: {backup_path}")
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ 已修复 UNet 的 {len(fixes_applied)} 个参数")
        else:
            logger.info("✅ UNet 配置无需修复")
            
    except Exception as e:
        logger.error(f"❌ 修复 UNet 配置失败: {e}")

def main():
    """主函数"""
    
    logger.info("🚀 开始检查和修复配置文件...")
    logger.info("=" * 80)
    
    # 修复各个组件的配置
    success_count = 0
    
    if fix_scheduler_config():
        success_count += 1
    
    fix_text_encoder_configs()
    success_count += 1
    
    fix_unet_config()
    success_count += 1
    
    logger.info("=" * 80)
    logger.info(f"🎯 配置修复完成，成功处理 {success_count} 个组件")
    
    logger.info("\n💡 建议:")
    logger.info("1. 重新启动 RunPod 端点")
    logger.info("2. 检查日志确认 NoneType 错误是否消失")
    logger.info("3. 如果仍有问题，可能需要重新下载官方配置文件")

if __name__ == "__main__":
    main() 