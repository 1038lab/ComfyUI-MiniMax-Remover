import torch
import numpy as np
import os
import sys
from typing import Optional, Union

# Add the current directory to Python path to import the original modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from Minimax.pipeline_minimax_remover import Minimax_Remover_Pipeline
    from Minimax.transformer_minimax_remover import Transformer3DModel
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from .utils import safe_tensor_to_device
except ImportError as e:
    print(f"Warning: Failed to import MiniMax-Remover dependencies: {e}")
    print("Please ensure all required packages are installed.")

    # Create dummy utility function if import fails
    def safe_tensor_to_device(tensor, device): return tensor


class MinimaxVideoRemoverNode:
    
    CATEGORY = "🧪AILab/🧼MiniMax-Remover"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video_frames",)
    FUNCTION = "remove_objects"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {"tooltip": "Input video frames as image batch (F, H, W, C) in range [0, 1]"}),
                "masks": ("MASK", {"tooltip": "Object masks for removal (F, H, W) in range [0, 1]"}),
                "vae": ("VAE", {"tooltip": "MiniMax VAE model (AutoencoderKLWan)"}),
                "transformer": ("TRANSFORMER", {"tooltip": "MiniMax Transformer3D model"}),
                "scheduler": ("SCHEDULER", {"tooltip": "Diffusion scheduler"}),
                "num_inference_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Number of inference steps"}),
                "mask_dilation_iterations": ("INT", {"default": 6, "min": 0, "max": 20, "step": 1, "tooltip": "Mask dilation iterations"}),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, video_frames, masks, vae, transformer, scheduler, num_inference_steps, mask_dilation_iterations, **kwargs):
        """Validate input dimensions and formats"""
        return True
    
    def remove_objects(self, video_frames, masks, vae, transformer, scheduler,
                      num_inference_steps=12, mask_dilation_iterations=6, seed=42):
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get video dimensions
            num_frames, height, width = video_frames.shape[:3]

            # Fix channel mismatch between video_frames and masks
            if video_frames.shape[-1] == 4:  # RGBA video
                video_frames = video_frames[..., :3]  # Convert to RGB

            # Convert mask to proper format using ComfyUI standard approach
            # We want [F,H,W,C] with C = 1 for masks
            if len(masks.shape)==2: # we have [H,W], so insert F and C as dimension 1
                masks = masks[None,:,:,None]
            elif len(masks.shape)==3 and masks.shape[2]==1: # we have [H,W,C]
                masks = masks[None,:,:,:]
            elif len(masks.shape)==3: # we have [F,H,W]
                masks = masks[:,:,:,None]

            # Ensure video_frames and masks have exactly the same spatial dimensions
            if video_frames.shape[1:3] != masks.shape[1:3]:
                print(f"Warning: Video {video_frames.shape[1:3]} and mask {masks.shape[1:3]} dimension mismatch, resizing mask")
                import torch.nn.functional as F
                # Resize mask to match video dimensions
                mask_resized = F.interpolate(
                    masks.permute(0, 3, 1, 2),  # (F, H, W, C) -> (F, C, H, W)
                    size=(video_frames.shape[1], video_frames.shape[2]),
                    mode='bilinear',
                    align_corners=False
                )
                masks = mask_resized.permute(0, 2, 3, 1)  # (F, C, H, W) -> (F, H, W, C)

            # Check if mask has meaningful content
            if (masks > 0.1).sum().item() == 0:
                print("Warning: Mask appears to be empty")

            # Convert inputs to proper format for MiniMax pipeline
            images = video_frames * 2.0 - 1.0  # Convert [0,1] to [-1,1]
            masks = masks.clamp(0, 1)  # Keep as (F, H, W, 1)

            # Move tensors to device safely
            images = safe_tensor_to_device(images, device)
            masks = safe_tensor_to_device(masks, device)

            # Initialize the pipeline
            pipe = Minimax_Remover_Pipeline(
                vae=vae,
                transformer=transformer,
                scheduler=scheduler
            ).to(device)

            # Set up generator for reproducible results
            generator = torch.Generator(device=device).manual_seed(seed)

            # Run the removal process (using actual video dimensions)
            result = pipe(
                images=images,
                masks=masks,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                iterations=mask_dilation_iterations
            )

            # Handle different result formats (improved logic from image version)
            if hasattr(result, 'frames'):
                output_frames = result.frames
                if isinstance(output_frames, np.ndarray):
                    output_frames = torch.from_numpy(output_frames).float()
                elif isinstance(output_frames, (list, tuple)) and len(output_frames) > 0:
                    output_frames = output_frames[0]
                elif not isinstance(output_frames, torch.Tensor):
                    try:
                        output_frames = torch.tensor(output_frames, dtype=torch.float32)
                    except:
                        return (video_frames,)
            elif hasattr(result, 'images'):
                output_frames = result.images
                if isinstance(output_frames, np.ndarray):
                    output_frames = torch.from_numpy(output_frames).float()
            else:
                output_frames = result
                if isinstance(output_frames, np.ndarray):
                    output_frames = torch.from_numpy(output_frames).float()

            # Convert to tensor format expected by ComfyUI
            if isinstance(output_frames, list):
                import torchvision.transforms as transforms
                transform = transforms.Compose([transforms.ToTensor()])
                output_tensor = torch.stack([transform(frame) for frame in output_frames])
                output_tensor = output_tensor.permute(0, 2, 3, 1)  # (F, C, H, W) -> (F, H, W, C)
            elif isinstance(output_frames, torch.Tensor):
                output_tensor = output_frames
                # Handle different tensor formats - convert to (F, H, W, C)
                if output_tensor.dim() == 5:  # (B, C, F, H, W) or (B, F, H, W, C)
                    if output_tensor.shape[1] == 3:  # (B, C, F, H, W)
                        output_tensor = output_tensor.squeeze(0).permute(1, 2, 3, 0)
                    elif output_tensor.shape[-1] == 3:  # (B, F, H, W, C)
                        output_tensor = output_tensor.squeeze(0)
                    else:
                        B, dim1, dim2, H, W = output_tensor.shape
                        if dim1 == 3:  # Assume (B, C, F, H, W)
                            output_tensor = output_tensor.squeeze(0).permute(1, 2, 3, 0)
                        elif dim2 == 3:  # Assume (B, F, C, H, W)
                            output_tensor = output_tensor.squeeze(0).permute(0, 2, 3, 1)
                        else:
                            output_tensor = output_tensor.squeeze(0).permute(1, 2, 3, 0)
                elif output_tensor.dim() == 4:
                    if output_tensor.shape[1] == 3:  # (F, C, H, W)
                        output_tensor = output_tensor.permute(0, 2, 3, 1)  # -> (F, H, W, C)
                    elif output_tensor.shape[-1] != 3:
                        if output_tensor.shape[-1] == 1:  # (F, H, W, 1)
                            output_tensor = output_tensor.repeat(1, 1, 1, 3)  # Convert to RGB
                elif output_tensor.dim() == 3:  # Single frame (H, W, C)
                    output_tensor = output_tensor.unsqueeze(0)  # Add frame dimension -> (1, H, W, C)
                else:
                    # Try to reshape if possible
                    if output_tensor.numel() == num_frames * height * width * 3:
                        output_tensor = output_tensor.view(num_frames, height, width, 3)
                    else:
                        return (video_frames,)
            else:
                return (video_frames,)

            # Ensure values are in [0, 1] range
            if output_tensor.dtype == torch.float32 or output_tensor.dtype == torch.float16:
                # Convert from [-1, 1] to [0, 1] if needed
                if output_tensor.min() < -0.5:  # Likely in [-1, 1] range
                    output_tensor = (output_tensor + 1.0) / 2.0

                output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

            # Validate output frame count
            if output_tensor.shape[0] != num_frames:
                print(f"⚠️  WARNING: Output frames ({output_tensor.shape[0]}) != Input frames ({num_frames})")
                if output_tensor.shape[0] < num_frames:
                    print(f"❌ Frame loss detected! Expected {num_frames}, got {output_tensor.shape[0]}")
                    # Try to pad missing frames by repeating the last frame
                    missing_frames = num_frames - output_tensor.shape[0]
                    last_frame = output_tensor[-1:].repeat(missing_frames, 1, 1, 1)
                    output_tensor = torch.cat([output_tensor, last_frame], dim=0)
                    print(f"🔧 Padded {missing_frames} missing frames")
                elif output_tensor.shape[0] > num_frames:
                    print(f"✂️  Trimming extra frames: {output_tensor.shape[0]} -> {num_frames}")
                    output_tensor = output_tensor[:num_frames]

            # Final format validation for ComfyUI
            if output_tensor.dim() != 4 or output_tensor.shape[-1] != 3:
                # Try to fix common issues
                if output_tensor.dim() == 3:  # (H, W, C)
                    output_tensor = output_tensor.unsqueeze(0)  # -> (1, H, W, C)
                elif output_tensor.shape[-1] != 3:
                    if output_tensor.shape[-1] == 1:  # Grayscale
                        output_tensor = output_tensor.repeat(1, 1, 1, 3)  # -> RGB
                    else:
                        return (video_frames,)  # Return original as fallback

            print(f"✅ Video processing completed: {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            print(f"Error in MiniMax Video Remover: {str(e)}")
            return (video_frames,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when seed changes"""
        return kwargs.get("seed", 42)
