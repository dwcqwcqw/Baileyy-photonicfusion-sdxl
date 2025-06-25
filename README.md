# PhotonicFusion SDXL RunPod Serverless

This repository contains a RunPod Serverless implementation for the PhotonicFusion SDXL model, converted to diffusers format for optimal performance and compatibility.

## 🎯 Features

- **High-Quality Image Generation**: Based on PhotonicFusion SDXL V3 model
- **Serverless Deployment**: Optimized for RunPod Serverless platform
- **Memory Efficient**: Includes GPU memory optimizations and fallback strategies
- **Flexible Parameters**: Support for various image sizes, steps, and guidance scales
- **Base64 Output**: Returns images as base64-encoded strings for easy API integration

## 🚀 Quick Start

### RunPod Deployment

1. **Build and Push Docker Image**:
   ```bash
   docker build -t your-registry/photonicfusion-sdxl:latest .
   docker push your-registry/photonicfusion-sdxl:latest
   ```

2. **Setup RunPod Volume** (for faster loading):
   - Create a RunPod Network Volume named `photonicfusion-models`
   - Upload the model files to `/photonicfusion-sdxl/` in the volume
   - This eliminates cold start time and reduces bandwidth usage

3. **Deploy to RunPod**:
   - Create a new serverless endpoint in RunPod console
   - Use the Docker image: `your-registry/photonicfusion-sdxl:latest`
   - Set GPU type: RTX A4000 or better
   - Mount the volume: `photonicfusion-models` → `/runpod-volume`
   - Configure environment variables as needed

3. **Test the Endpoint**:
   ```bash
   curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "input": {
         "prompt": "a beautiful sunset over mountains, photorealistic",
         "width": 1024,
         "height": 1024,
         "num_inference_steps": 30,
         "guidance_scale": 7.5
       }
     }'
   ```

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Local Tests**:
   ```bash
   python test_local.py
   ```

3. **Test Individual Components**:
   ```bash
   python -c "from handler import load_model; load_model()"
   ```

## 📋 API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the image to generate |
| `negative_prompt` | string | `"blurry, low quality, distorted, ugly"` | What to avoid in the image |
| `width` | integer | `1024` | Image width (512-1536) |
| `height` | integer | `1024` | Image height (512-1536) |
| `num_inference_steps` | integer | `30` | Number of denoising steps (10-100) |
| `guidance_scale` | float | `7.5` | How closely to follow the prompt (1.0-20.0) |
| `seed` | integer | `null` | Random seed for reproducible results |

### Output Format

```json
{
  "image": "base64_encoded_image_data",
  "prompt": "original_prompt",
  "parameters": {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

### Error Handling

```json
{
  "error": "Error description"
}
```

## 🔧 Configuration

### Environment Variables

- `MODEL_NAME`: Hugging Face model identifier (default: `Baileyy/photonicfusion-sdxl`)
- `TORCH_CUDA_ARCH_LIST`: CUDA architectures to support
- `PYTHONPATH`: Python path configuration

### Memory Optimization

The handler includes several memory optimization strategies:

1. **Attention Slicing**: Reduces memory usage during attention computation
2. **Model CPU Offloading**: Moves unused model components to CPU
3. **XFormers**: Uses memory-efficient attention when available
4. **CUDA OOM Recovery**: Automatically retries with smaller resolution on out-of-memory

## 📊 Performance

### Expected Performance

| GPU | Resolution | Steps | Time (Volume) | Time (Hub) | Memory |
|-----|------------|-------|---------------|------------|--------|
| RTX A4000 | 1024x1024 | 30 | ~8s | ~15s | ~12GB |
| RTX A5000 | 1024x1024 | 30 | ~6s | ~12s | ~14GB |
| RTX A6000 | 1024x1024 | 30 | ~5s | ~10s | ~16GB |

**Note**: Volume loading eliminates ~3-7s model loading time per cold start.

### Optimization Tips

1. **Use Volume**: Pre-upload model to RunPod volume for instant loading
2. **Reduce Steps**: Use 20-25 steps for faster generation
3. **Lower Resolution**: Use 768x768 for memory-constrained environments
4. **Batch Processing**: Process multiple requests with different seeds
5. **Persistent Workers**: Keep at least 1 worker to avoid cold starts

## 🧪 Testing

### Local Testing

```bash
# Run all tests
python test_local.py

# Test specific functionality
python -c "
from handler import handler
result = handler({
    'input': {
        'prompt': 'test image',
        'width': 512,
        'height': 512,
        'num_inference_steps': 10
    }
})
print('Success!' if 'image' in result else 'Failed!')
"
```

### API Testing

Update `api_examples.py` with your endpoint details and run:

```bash
python api_examples.py
```

## 📁 Project Structure

```
photonicfusion-sdxl-runpod/
├── handler.py              # Main RunPod handler
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── test_local.py          # Local testing script
├── api_examples.py        # API usage examples
├── runpod_config.json     # RunPod deployment configuration
└── README.md              # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `width` and `height` to 768x768 or lower
   - Decrease `num_inference_steps` to 20-25
   - The handler automatically attempts recovery

2. **Slow Cold Starts**:
   - Use RunPod Network Volume with pre-uploaded model
   - Enable "flashboot" feature in RunPod
   - Consider keeping min_workers > 0 for production

3. **Model Loading Errors**:
   - Ensure stable internet connection for Hugging Face downloads
   - Check if the model `Baileyy/photonicfusion-sdxl` is accessible
   - Verify authentication tokens if needed

### Debug Mode

Set environment variable for verbose logging:

```bash
export RUNPOD_DEBUG=1
python handler.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the same terms as the original PhotonicFusion SDXL model. Please refer to the model's license for usage restrictions.

## 🙏 Acknowledgments

- Original PhotonicFusion SDXL V3 model creators
- Hugging Face Diffusers team
- RunPod platform

---

**Note**: Make sure to update the Docker registry path and RunPod endpoint details before deployment. 