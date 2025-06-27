# ComfyUI MiniMax-Remover


A ComfyUI custom node for fast and effective video/image object removal using MiniMax optimization. Remove objects from videos and images with high quality results using only 6-12 inference steps.



## ✨ Features

- **🚀 Fast**: Only 6-12 inference steps required, no CFG needed
- **🎯 Effective**: Seamlessly removes objects from videos and images
- **💪 Robust**: Prevents regeneration of unwanted objects or artifacts
- **🔧 Easy to Use**: Simple ComfyUI nodes with intuitive workflows
- **📱 Flexible**: Supports both single images and video processing
- **⚡ Optimized**: Automatic model downloading and caching
- **💡 Smart Hints**: Automatic backend suggestions for optimal settings

![Image subject Remover](https://github.com/user-attachments/assets/55394115-d88a-4bca-b70d-29ce51e97ead)

## 🛠️ Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "MiniMax-Remover"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/1038lab/ComfyUI-MiniMax-Remover.git
```

3. Install dependencies:
```bash
cd ComfyUI-MiniMax-Remover
```

**For Portable ComfyUI (Windows):**
```bash
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

**For Standard Python Installation:**
```bash
pip install -r requirements.txt
```

4. Restart ComfyUI

## 📥 Model Download

Models will be automatically downloaded from HuggingFace when first used.

### Manual Model Download (Optional)

If automatic download fails or you prefer manual installation:

1. **Download using HuggingFace CLI:**
```bash
huggingface-cli download zibojia/minimax-remover --include vae transformer scheduler --local-dir ./models/MiniMax-Remover
```

2. **Or download manually from HuggingFace:**
   - Visit: https://huggingface.co/zibojia/minimax-remover
   - Download the `vae`, `transformer`, and `scheduler` folders
   - Place them in: `ComfyUI/models/MiniMax-Remover/`

3. **Verify installation:**
   - Check that `ComfyUI/models/MiniMax-Remover/` contains:
     - `vae/` folder
     - `transformer/` folder
     - `scheduler/` folder

## 📦 Available Nodes

| Node Name | Display Name | Purpose |
|-----------|--------------|---------|
| `MinimaxImageRemover` | MiniMax Image Object Remover | Remove objects from single images (requires external model loader) |
| `MinimaxVideoRemover` | MiniMax Video Object Remover | Remove objects from videos (requires external model loader) |
| `MinimaxModelLoader` | MiniMax Model Loader | Load VAE, Transformer, and Scheduler for separate nodes |
| `MinimaxVideoLoader` | MiniMax Video Loader | Load video files for processing |
| `ImageSizeAdjuster` | MiniMax Image Size Adjuster | Adjust image dimensions for compatibility |



## ⚙️ Recommended Settings

### MiniMax Model Loader (Advanced)
- **scheduler_type**: `FlowMatchEulerDiscreteScheduler` (recommended) or `UniPCMultistepScheduler`
- **torch_dtype**: `float16` (GPU recommended) or `float32` (CPU/compatibility)
- **device**: `auto` (automatic detection)

## 🎨 Mask Requirements

### Supported Formats
- ✅ **Grayscale images** (recommended)
- ✅ **RGB images** (automatically converted)
- ✅ **PNG, JPG, TIFF** and other common formats

### Mask Guidelines
- **Black background, white foreground** (standard mask format)
- **High contrast** for better edge detection
- **Complete coverage** of objects to remove
- **Clean edges** without excessive noise
- **Avoid overly complex details** in small areas

## 🚀 Performance Tips

### GPU Memory Optimization
1. Use `float16` precision (reduces memory by ~50%)
2. Keep image resolution reasonable (≤1024x1024 recommended)
3. Use fewer inference steps (6-12 usually sufficient)
4. Close other GPU applications to free VRAM

### Processing Speed
1. `FlowMatchEulerDiscreteScheduler` is typically faster than `UniPCMultistepScheduler`
2. `temporal_frames=3` provides best speed/quality balance
3. Use `ImageSizeAdjuster` node to optimize image dimensions for better performance

## 🔧 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ system RAM
- **VRAM**: 6GB+ GPU memory for optimal performance

### Dependencies
All dependencies are automatically installed via `requirements.txt`:

```
torch>=1.13.0
diffusers>=0.21.0
decord>=0.6.0
einops>=0.6.0
scipy>=1.9.0
opencv-python>=4.5.0
huggingface_hub>=0.16.0
accelerate>=0.20.0
```

## ❓ Troubleshooting

### Common Issues

**Model Download Fails**
- Try manual download from HuggingFace website
- Check internet connection and HuggingFace access
- Ensure sufficient disk space in `ComfyUI/models/MiniMax-Remover/`

**CUDA Out of Memory**
- Reduce image resolution
- Use `float16` instead of `float32`
- Reduce `num_inference_steps`
- Close other applications using GPU

**Tensor Dimension Errors**
- Ensure mask and image have compatible dimensions
- Enable `fix_dimensions` in the All-in-One node for automatic adjustment
- Use `ImageSizeAdjuster` node for manual dimension control
- Check that images are RGB (not RGBA)

**Dimension Warning Messages**
- If you see "Image dimensions not divisible by 16" warning, enable `fix_dimensions=True`
- The All-in-One node will automatically suggest this when needed

**Poor Removal Quality**
- Increase `num_inference_steps` (try 16-20)
- Improve mask quality (higher contrast, cleaner edges)
- Adjust `mask_dilation_iterations` based on mask precision
- Try different scheduler types

## 📚 Additional Resources

- [Original MiniMax-Remover Paper](https://arxiv.org/abs/2505.24873)
- [HuggingFace Model Repository](https://huggingface.co/zibojia/minimax-remover)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)

## 📄 License

![License](https://img.shields.io/badge/license-GPL--3.0-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green.svg)

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MiniMax-Remover Team** for the original research and implementation
- **ComfyUI Community** for the excellent framework and ecosystem
- **HuggingFace** for model hosting and distribution
- **Contributors** who helped improve this custom node

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## ⭐ Support

If you find this project helpful, please consider giving it a star on GitHub! It helps others discover the project and motivates continued development.

---

**Made with ❤️ for the ComfyUI community**
