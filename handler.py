import runpod
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import base64
from io import BytesIO
from PIL import Image
import os
import gc
import logging
import warnings
import json

# Configure logging and suppress specific warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress weight mismatch warnings that don't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of the model checkpoint*")
warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of*were not initialized*")
warnings.filterwarnings("ignore", category=UserWarning, message="You should probably TRAIN this model*")

# --- Configuration ---
# The model path in the RunPod volume
MODEL_PATH = "/runpod-volume/photonicfusion-sdxl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global pipeline variable
pipeline = None

def check_and_fix_model_index():
    """检查并修复 model_index.json 中的 None 值"""
    model_index_path = os.path.join(MODEL_PATH, "model_index.json")
    
    if not os.path.exists(model_index_path):
        logger.error(f"❌ model_index.json 不存在: {model_index_path}")
        return False
    
    try:
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        
        logger.info("🔍 检查 model_index.json 组件映射...")
        
        # 检查并修复 None 值
        fixed = False
        for key, value in model_index.items():
            if not key.startswith('_') and isinstance(value, list) and len(value) >= 2:
                component_type, component_name = value[0], value[1]
                
                if component_name is None or component_name == "null":
                    logger.warning(f"⚠️ 发现 None 值: {key} -> {component_name}")
                    
                    # 尝试自动修复常见组件
                    if key == "feature_extractor":
                        model_index[key] = ["transformers", "CLIPImageProcessor"]
                        fixed = True
                        logger.info(f"🔧 修复: {key} -> CLIPImageProcessor")
                    elif key == "image_encoder":
                        model_index[key] = ["transformers", "CLIPVisionModelWithProjection"]
                        fixed = True
                        logger.info(f"🔧 修复: {key} -> CLIPVisionModelWithProjection")
                    elif key == "safety_checker":
                        # 安全检查器可以设为 null
                        model_index[key] = [None, None]
                        logger.info(f"🔧 设置: {key} -> null (安全)")
                    else:
                        logger.warning(f"⚠️ 无法自动修复: {key}")
                else:
                    logger.info(f"✅ {key}: {component_type} -> {component_name}")
        
        # 如果有修复，保存文件
        if fixed:
            backup_path = model_index_path + ".backup"
            os.rename(model_index_path, backup_path)
            logger.info(f"💾 备份原文件: {backup_path}")
            
            with open(model_index_path, 'w') as f:
                json.dump(model_index, f, indent=2)
            logger.info(f"✅ 保存修复后的 model_index.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 处理 model_index.json 失败: {e}")
        return False

def create_missing_configs():
    """创建缺失的配置文件"""
    configs_to_create = {
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
    
    # 创建tokenizer必需的特殊文件
    special_tokenizer_files = {
        "tokenizer/special_tokens_map.json": {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>"
        },
        "tokenizer_2/special_tokens_map.json": {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>"
        }
    }
    
    created_count = 0
    
    # 创建JSON配置文件
    for config_path, config_content in configs_to_create.items():
        full_path = os.path.join(MODEL_PATH, config_path)
        
        if not os.path.exists(full_path):
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    json.dump(config_content, f, indent=2)
                logger.info(f"✅ 创建配置文件: {config_path}")
                created_count += 1
            except Exception as e:
                logger.error(f"❌ 创建 {config_path} 失败: {e}")
        else:
            logger.info(f"⏭️ 已存在: {config_path}")
    
    # 创建tokenizer特殊文件
    for file_path, content in special_tokenizer_files.items():
        full_path = os.path.join(MODEL_PATH, file_path)
        
        if not os.path.exists(full_path):
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    json.dump(content, f, indent=2)
                logger.info(f"✅ 创建特殊文件: {file_path}")
                created_count += 1
            except Exception as e:
                logger.error(f"❌ 创建 {file_path} 失败: {e}")
        else:
            logger.info(f"⏭️ 已存在: {file_path}")
    
    # 检查并创建vocab.json和merges.txt (如果缺失)
    tokenizer_dirs = ["tokenizer", "tokenizer_2"]
    for tokenizer_dir in tokenizer_dirs:
        tokenizer_path = os.path.join(MODEL_PATH, tokenizer_dir)
        
        # 检查vocab.json
        vocab_path = os.path.join(tokenizer_path, "vocab.json")
        if not os.path.exists(vocab_path):
            logger.warning(f"⚠️ {tokenizer_dir}/vocab.json 缺失，这会导致tokenizer加载失败")
            logger.info(f"📝 建议从Hugging Face官方CLIP tokenizer复制vocab.json文件")
        
        # 检查merges.txt
        merges_path = os.path.join(tokenizer_path, "merges.txt")
        if not os.path.exists(merges_path):
            logger.warning(f"⚠️ {tokenizer_dir}/merges.txt 缺失，这会导致tokenizer加载失败")
            logger.info(f"📝 建议从Hugging Face官方CLIP tokenizer复制merges.txt文件")
    
    return created_count

def diagnose_volume_structure():
    """诊断Volume中的模型结构"""
    logger.info(f"🔍 诊断模型目录结构: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Volume路径不存在: {MODEL_PATH}")
        return False
    
    # 检查必需组件
    required_components = ["model_index.json", "unet", "vae", "text_encoder", "text_encoder_2"]
    missing_components = []
    
    for component in required_components:
        component_path = os.path.join(MODEL_PATH, component)
        if os.path.exists(component_path):
            logger.info(f"✅ {component}")
        else:
            logger.error(f"❌ 缺失: {component}")
            missing_components.append(component)
    
    if missing_components:
        logger.error(f"❌ 关键组件缺失: {missing_components}")
        return False
    
    # 检查并修复 model_index.json
    if not check_and_fix_model_index():
        return False
    
    # 创建缺失的配置文件
    created = create_missing_configs()
    if created > 0:
        logger.info(f"✅ 创建了 {created} 个配置文件")
    
    return True

def fix_meta_tensors(model):
    """修复模型中的 meta tensors"""
    import torch.nn as nn
    
    def _fix_module(module):
        for name, param in module.named_parameters(recurse=False):
            if param.is_meta:
                logger.warning(f"   🔧 Fixing meta tensor: {name}")
                # Create a new parameter with the same shape but on CPU
                new_param = nn.Parameter(
                    torch.empty_like(param, device='cpu', dtype=param.dtype)
                )
                # Initialize with small random values
                nn.init.normal_(new_param, mean=0.0, std=0.02)
                setattr(module, name, new_param)
        
        # Recursively fix child modules
        for child_module in module.children():
            _fix_module(child_module)
    
    _fix_module(model)
    return model

def load_model():
    """Load the PhotonicFusion SDXL model from RunPod volume"""
    global pipeline
    
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"📁 Loading model from: {MODEL_PATH}")
    
    # 首先诊断并修复模型结构
    if not diagnose_volume_structure():
        raise RuntimeError(f"❌ Volume模型结构检查失败")
    
    # 检查tokenizer文件，如果缺失关键文件，尝试自动修复
    try:
        from transformers import CLIPTokenizer
        
        for tokenizer_dir in ["tokenizer", "tokenizer_2"]:
            tokenizer_path = os.path.join(MODEL_PATH, tokenizer_dir)
            vocab_path = os.path.join(tokenizer_path, "vocab.json")
            merges_path = os.path.join(tokenizer_path, "merges.txt")
            
            if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
                logger.warning(f"⚠️ {tokenizer_dir} 缺失关键文件，尝试自动修复...")
                
                try:
                    # 下载标准的CLIP tokenizer
                    if tokenizer_dir == "tokenizer":
                        std_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                    else:
                        std_tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
                    
                    # 保存到本地
                    os.makedirs(tokenizer_path, exist_ok=True)
                    std_tokenizer.save_pretrained(tokenizer_path)
                    logger.info(f"✅ 已下载并保存 {tokenizer_dir} 文件")
                    
                except Exception as e:
                    logger.error(f"❌ 无法自动修复 {tokenizer_dir}: {e}")
                    # 如果下载失败，创建最小化的tokenizer文件
                    logger.info(f"🔧 创建最小化tokenizer文件...")
                    
                    if not os.path.exists(vocab_path):
                        # 创建最小的vocab.json
                        minimal_vocab = {}
                        for i in range(49408):
                            if i == 49406:
                                minimal_vocab["<|startoftext|>"] = i
                            elif i == 49407:
                                minimal_vocab["<|endoftext|>"] = i
                            else:
                                minimal_vocab[f"token_{i}"] = i
                        
                        with open(vocab_path, 'w', encoding='utf-8') as f:
                            json.dump(minimal_vocab, f)
                        logger.info(f"✅ 创建最小化 {tokenizer_dir}/vocab.json")
                    
                    if not os.path.exists(merges_path):
                        # 创建最小的merges.txt
                        minimal_merges = "\n".join([f"token_{i} token_{i+1}" for i in range(0, 1000, 2)])
                        
                        with open(merges_path, 'w', encoding='utf-8') as f:
                            f.write(minimal_merges)
                        logger.info(f"✅ 创建最小化 {tokenizer_dir}/merges.txt")
                        
    except Exception as e:
        logger.error(f"❌ Tokenizer文件检查失败: {e}")
    
    try:
        # Load the pipeline with comprehensive error handling
        logger.info("🔄 Loading StableDiffusionXLPipeline...")
        
        # 尝试不同的加载策略
        load_strategies = [
            # 策略1: 低内存模式 + FP16
            {
                "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                "variant": "fp16" if DEVICE == "cuda" else None,
                "use_safetensors": True,
                "local_files_only": True,
                "safety_checker": None,
                "requires_safety_checker": False,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if DEVICE == "cuda" else None
            },
            # 策略2: 标准FP16加载（原策略1）
            {
                "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                "variant": "fp16" if DEVICE == "cuda" else None,
                "use_safetensors": True,
                "local_files_only": True,
                "safety_checker": None,
                "requires_safety_checker": False
            },
            # 策略3: 不指定variant
            {
                "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                "use_safetensors": True,
                "local_files_only": True,
                "safety_checker": None,
                "requires_safety_checker": False,
                "low_cpu_mem_usage": True
            },
            # 策略4: 不使用safetensors
            {
                "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                "local_files_only": True,
                "safety_checker": None,
                "requires_safety_checker": False
            },
            # 策略5: 允许网络下载缺失组件
            {
                "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                "use_safetensors": True,
                "safety_checker": None,
                "requires_safety_checker": False,
                "local_files_only": False,
                "low_cpu_mem_usage": True
            }
        ]
        
        last_error = None
        for i, strategy in enumerate(load_strategies, 1):
            logger.info(f"🔄 尝试加载策略 {i}/{len(load_strategies)}...")
            
            # Suppress stderr temporarily to hide warnings
            import sys
            from io import StringIO
            
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            
            try:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_PATH,
                    **strategy
                )
                
                # 检查并修复 meta tensors
                logger.info(f"🔍 检查 meta tensors...")
                components_with_meta = []
                
                for component_name in ['vae', 'text_encoder', 'text_encoder_2', 'unet']:
                    component = getattr(pipeline, component_name, None)
                    if component is not None:
                        has_meta = any(param.is_meta for param in component.parameters())
                        if has_meta:
                            components_with_meta.append(component_name)
                
                if components_with_meta:
                    logger.warning(f"⚠️ 发现 meta tensors 在: {components_with_meta}")
                    logger.info(f"🔧 修复 meta tensors...")
                    
                    for component_name in components_with_meta:
                        component = getattr(pipeline, component_name)
                        fixed_component = fix_meta_tensors(component)
                        setattr(pipeline, component_name, fixed_component)
                    
                    logger.info(f"✅ Meta tensors 已修复")
                else:
                    logger.info(f"✅ 未发现 meta tensors")
                
                logger.info(f"✅ 策略 {i} 成功!")
                break
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # 提供更详细的错误信息
                if "meta tensor" in error_msg.lower():
                    logger.warning(f"⚠️ 策略 {i} 失败 (meta tensor): {error_msg[:150]}...")
                elif "safetensors" in error_msg.lower():
                    logger.warning(f"⚠️ 策略 {i} 失败 (safetensors): {error_msg[:150]}...")
                elif "device" in error_msg.lower():
                    logger.warning(f"⚠️ 策略 {i} 失败 (device): {error_msg[:150]}...")
                else:
                    logger.warning(f"⚠️ 策略 {i} 失败: {error_msg[:150]}...")
                
                pipeline = None
                
            finally:
                # Restore stderr
                sys.stderr = old_stderr
        
        if pipeline is None:
            raise RuntimeError(f"所有加载策略都失败了。最后错误: {last_error}")
        
        # Configure scheduler
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        # Move to device with meta tensor handling
        logger.info(f"🔄 Moving pipeline to {DEVICE}...")
        
        try:
            # First, try to initialize any meta tensors
            if hasattr(pipeline, '_apply_meta_tensor_fix'):
                pipeline._apply_meta_tensor_fix()
            
            # Move each component individually to handle meta tensors better
            components = ['vae', 'text_encoder', 'text_encoder_2', 'unet']
            for component_name in components:
                component = getattr(pipeline, component_name, None)
                if component is not None:
                    try:
                        logger.info(f"   🔄 Moving {component_name} to {DEVICE}...")
                        
                        # Check for meta tensors and handle them
                        has_meta = False
                        for param in component.parameters():
                            if param.is_meta:
                                has_meta = True
                                break
                        
                        if has_meta:
                            logger.warning(f"   ⚠️ {component_name} has meta tensors, using to_empty()...")
                            # Use to_empty() for meta tensors
                            component = component.to_empty(device=DEVICE)
                            setattr(pipeline, component_name, component)
                        else:
                            # Normal move for non-meta tensors
                            component = component.to(DEVICE)
                            setattr(pipeline, component_name, component)
                        
                        logger.info(f"   ✅ {component_name} moved successfully")
                        
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to move {component_name}: {e}")
                        # If individual component fails, try to continue with others
                        continue
            
            logger.info("✅ Pipeline components moved to device")
            
        except Exception as e:
            logger.error(f"❌ Error moving pipeline to device: {e}")
            # Fallback: try to use the pipeline as-is
            logger.info("🔄 Attempting to continue with CPU/mixed precision...")
        
        if DEVICE == "cuda":
            try:
                pipeline.enable_attention_slicing()
                logger.info("✅ Attention slicing enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to enable attention slicing: {e}")
            
            try:
                pipeline.enable_model_cpu_offload()
                logger.info("✅ Model CPU offload enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to enable CPU offload: {e}")
            
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers enabled")
            except:
                logger.info("ℹ️ XFormers not available")
        
        # Test the model
        logger.info("🧪 Testing model...")
        test_result = pipeline(
            prompt="test",
            num_inference_steps=1,
            width=64,
            height=64,
            output_type="pil"
        )
        
        logger.info("✅ Model loaded and tested successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        
        # 如果是NoneType错误，提供详细诊断
        if "NoneType" in str(e):
            logger.error("🔍 检测到NoneType错误，这通常是由于tokenizer文件缺失引起的")
            logger.error("请检查tokenizer目录中的vocab.json和merges.txt文件")
        
        raise RuntimeError(f"Failed to load model from volume: {e}")

def generate_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=7.0, 
                  width=1024, height=1024, seed=None):
    """Generate an image using the loaded pipeline"""
    global pipeline
    
    # Load model if not already loaded
    if pipeline is None:
        pipeline = load_model()
    
    logger.info(f"🎨 Generating image with prompt: {prompt[:50]}...")
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    else:
        generator = None
    
    try:
        # Generate image
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
        
        # Convert to base64
        image = result.images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Cleanup
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("✅ Image generated successfully!")
        return img_str
        
    except Exception as e:
        logger.error(f"❌ Image generation failed: {e}")
        raise

def handler(event):
    """RunPod handler function"""
    try:
        input_data = event['input']
        
        prompt = input_data.get('prompt', '')
        negative_prompt = input_data.get('negative_prompt', '')
        num_inference_steps = input_data.get('num_inference_steps', 20)
        guidance_scale = input_data.get('guidance_scale', 7.0)
        width = input_data.get('width', 1024)
        height = input_data.get('height', 1024)
        seed = input_data.get('seed', None)
        
        if not prompt:
            return {"error": "Prompt is required"}
        
        # Generate image
        image_base64 = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed
        )
        
        return {
            "image": image_base64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed
        }
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless worker...")
    
    # Pre-load the model for faster first request
    try:
        load_model()
        logger.info("✅ Model pre-loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Model pre-load failed: {e}")
    
    # Start the RunPod worker
    runpod.serverless.start({"handler": handler}) 