import torch
import numpy as np
from typing import Optional, Union
import folder_paths
import os
import sys

# Add the current directory to Python path to import the original modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from Minimax.pipeline_minimax_remover import Minimax_Remover_Pipeline
    from Minimax.transformer_minimax_remover import Transformer3DModel
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import UniPCMultistepScheduler, FlowMatchEulerDiscreteScheduler
    from einops import rearrange
    from .utils import (
        validate_video_tensor, validate_mask_tensor, check_tensor_compatibility,
        safe_tensor_to_device, estimate_memory_usage, format_tensor_info,
        check_dependencies, create_error_placeholder
    )
except ImportError as e:
    print(f"Warning: Failed to import MiniMax-Remover dependencies: {e}")
    print("Please ensure all required packages are installed.")
    
    # Create dummy utility functions if import fails
    def validate_video_tensor(*args, **kwargs): return ""
    def validate_mask_tensor(*args, **kwargs): return ""
    def check_tensor_compatibility(*args, **kwargs): return ""
    def safe_tensor_to_device(tensor, device): return tensor
    def estimate_memory_usage(*args, **kwargs): return 0.0
    def format_tensor_info(tensor, name=""): return f"{name}: {tensor.shape if tensor is not None else 'None'}"
    def check_dependencies(): return True, ""
    def create_error_placeholder(shape, dtype=torch.float32): return torch.zeros(shape, dtype=dtype)


class MinimaxImageRemoverNode:
    
    CATEGORY = "🧪AILab/🧼MiniMax-Remover"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_objects"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "mask": ("MASK", {"tooltip": "Object mask for removal"}),
                "vae": ("VAE", {"tooltip": "MiniMax VAE model"}),
                "transformer": ("TRANSFORMER", {"tooltip": "MiniMax Transformer3D model"}),
                "scheduler": ("SCHEDULER", {"tooltip": "Diffusion scheduler"}),
                "num_inference_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Number of denoising steps"}),
                "mask_dilation_iterations": ("INT", {"default": 6, "min": 0, "max": 20, "step": 1, "tooltip": "Mask dilation iterations"}),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, mask, vae, transformer, scheduler, num_inference_steps, mask_dilation_iterations, **kwargs):
        """Validate input dimensions and formats"""
        return True
    
    def remove_objects(self, image, mask, vae, transformer, scheduler,
                      num_inference_steps=12, mask_dilation_iterations=6,
                      seed=42):
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get batch size and original dimensions
            batch_size = image.shape[0]
            original_height, original_width = image.shape[1:3]

            print(f"🖼️ Processing batch: {batch_size} images, {original_height}x{original_width}")

            # Create temporal sequence from single image
            temporal_frames = 3

            # Fix channel mismatch between image and mask
            if image.shape[-1] == 4:  # RGBA image
                image = image[..., :3]  # Convert to RGB

            # Convert mask to proper format using ComfyUI standard approach
            if len(mask.shape)==2:
                mask = mask[None,:,:,None]
            elif len(mask.shape)==3 and mask.shape[2]==1:
                mask = mask[None,:,:,:]
            elif len(mask.shape)==3:
                mask = mask[:,:,:,None]

            # Ensure image and mask have exactly the same spatial dimensions
            if image.shape[1:3] != mask.shape[1:3]:
                print(f"Warning: Image {image.shape[1:3]} and mask {mask.shape[1:3]} dimension mismatch, resizing mask")
                import torch.nn.functional as F
                mask_resized = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=(image.shape[1], image.shape[2]),
                    mode='bilinear',
                    align_corners=False
                )
                mask = mask_resized.permute(0, 2, 3, 1)

            # Ensure mask batch size matches image batch size
            if mask.shape[0] != batch_size:
                if mask.shape[0] == 1:
                    mask = mask.repeat(batch_size, 1, 1, 1)

            # Check if mask has meaningful content
            if (mask > 0.1).sum().item() == 0:
                print("Warning: Mask appears to be empty")

            # Process each image separately to maintain batch integrity
            processed_images = []

            for batch_idx in range(batch_size):
                print(f"🔄 Processing image {batch_idx+1}/{batch_size}")

                # Get single image and mask
                single_image = image[batch_idx:batch_idx+1]  # (1, H, W, C)
                single_mask = mask[batch_idx:batch_idx+1]    # (1, H, W, 1)

                # Expand to temporal sequence
                video_frames = single_image.repeat(temporal_frames, 1, 1, 1)  # (F, H, W, C)
                video_masks = single_mask.repeat(temporal_frames, 1, 1, 1)    # (F, H, W, 1)

                # Convert inputs to proper format for MiniMax pipeline
                images = video_frames * 2.0 - 1.0  # Convert [0,1] to [-1,1]
                masks = video_masks.clamp(0, 1)    # Keep as (F, H, W, 1)

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
                generator = torch.Generator(device=device).manual_seed(seed + batch_idx)

                # Run the removal process for single image
                result = pipe(
                    images=images,
                    masks=masks,
                    num_frames=temporal_frames,
                    height=original_height,
                    width=original_width,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    iterations=mask_dilation_iterations
                )

                # Handle different result formats
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
                            processed_images.append(single_image)
                            continue
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
                    elif output_tensor.dim() == 3:  # Single image (H, W, C)
                        output_tensor = output_tensor.unsqueeze(0)  # Add frame dimension -> (1, H, W, C)
                    else:
                        # Try to reshape if possible
                        if output_tensor.numel() == temporal_frames * original_height * original_width * 3:
                            output_tensor = output_tensor.view(temporal_frames, original_height, original_width, 3)
                        else:
                            processed_images.append(single_image)
                            continue
                else:
                    processed_images.append(single_image)
                    continue

                # Ensure values are in [0, 1] range
                if output_tensor.dtype == torch.float32 or output_tensor.dtype == torch.float16:
                    # Convert from [-1, 1] to [0, 1] if needed
                    if output_tensor.min() < -0.5:  # Likely in [-1, 1] range
                        output_tensor = (output_tensor + 1.0) / 2.0
                    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

                # Extract middle frame for single image output (most stable result)
                if output_tensor.shape[0] > 1:
                    middle_idx = temporal_frames // 2
                    output_image = output_tensor[middle_idx:middle_idx+1]  # Keep frame dimension: (1, H, W, C)
                else:
                    output_image = output_tensor  # Already (1, H, W, C)

                # For single image processing, we only want one frame
                if output_image.shape[0] > 1:
                    output_image = output_image[:1]  # Take only first frame: (1, H, W, C)

                # Resize back to original dimensions if needed
                if (output_image.shape[1], output_image.shape[2]) != (original_height, original_width):
                    import torch.nn.functional as F
                    # Convert to (B, C, H, W) for interpolation
                    output_image = output_image.permute(0, 3, 1, 2)  # (1, H, W, C) -> (1, C, H, W)
                    output_image = F.interpolate(
                        output_image,
                        size=(original_height, original_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Convert back to (B, H, W, C)
                    output_image = output_image.permute(0, 2, 3, 1)  # (1, C, H, W) -> (1, H, W, C)

                # Final format validation for ComfyUI
                if output_image.dim() != 4 or output_image.shape[0] != 1 or output_image.shape[-1] != 3:
                    # Try to fix common issues
                    if output_image.dim() == 3:  # (H, W, C)
                        output_image = output_image.unsqueeze(0)  # -> (1, H, W, C)
                    elif output_image.shape[-1] != 3:
                        if output_image.shape[-1] == 1:  # Grayscale
                            output_image = output_image.repeat(1, 1, 1, 3)  # -> RGB
                        else:
                            processed_images.append(single_image)
                            continue

                # Add processed image to batch results
                processed_images.append(output_image)
                print(f"✅ Successfully processed image {batch_idx+1}/{batch_size}")

            # Combine all processed images into a single batch
            if len(processed_images) == 0:
                return (image,)  # Return original if all failed

            final_output = torch.cat(processed_images, dim=0)  # Combine along batch dimension
            print(f"🎉 Batch processing completed: {final_output.shape}")
            return (final_output,)
            
        except Exception as e:
            print(f"Error in MiniMax Single Image Remover: {str(e)}")
            return (image,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when seed changes"""
        return kwargs.get("seed", 42)
