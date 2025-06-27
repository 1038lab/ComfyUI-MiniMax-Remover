"""
Utility functions for MiniMax-Remover ComfyUI nodes
"""

import torch
import numpy as np
import os
from typing import Tuple, Optional, Union


def validate_video_tensor(tensor: torch.Tensor, name: str = "video") -> str:

    if tensor is None:
        return f"{name} tensor is None"
    
    if not isinstance(tensor, torch.Tensor):
        return f"{name} must be a torch.Tensor, got {type(tensor)}"
    
    if len(tensor.shape) != 4:
        return f"{name} must be 4D tensor (F, H, W, C), got shape {tensor.shape}"
    
    if tensor.shape[-1] not in [1, 3, 4]:
        return f"{name} must have 1, 3, or 4 channels, got {tensor.shape[-1]}"
    
    if tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        return f"{name} must be float tensor, got {tensor.dtype}"
    
    if torch.any(tensor < 0) or torch.any(tensor > 1):
        return f"{name} values must be in range [0, 1], got range [{tensor.min():.3f}, {tensor.max():.3f}]"
    
    return ""


def validate_mask_tensor(tensor: torch.Tensor, name: str = "mask") -> str:

    if tensor is None:
        return f"{name} tensor is None"
    
    if not isinstance(tensor, torch.Tensor):
        return f"{name} must be a torch.Tensor, got {type(tensor)}"
    
    if len(tensor.shape) != 3:
        return f"{name} must be 3D tensor (F, H, W), got shape {tensor.shape}"
    
    if tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        return f"{name} must be float tensor, got {tensor.dtype}"
    
    if torch.any(tensor < 0) or torch.any(tensor > 1):
        return f"{name} values must be in range [0, 1], got range [{tensor.min():.3f}, {tensor.max():.3f}]"
    
    return ""


def check_tensor_compatibility(video: torch.Tensor, mask: torch.Tensor) -> str:

    if video.shape[0] != mask.shape[0]:
        return f"Frame count mismatch: video has {video.shape[0]} frames, mask has {mask.shape[0]} frames"
    
    if video.shape[1:3] != mask.shape[1:3]:
        return f"Spatial dimension mismatch: video {video.shape[1:3]}, mask {mask.shape[1:3]}"
    
    return ""


def safe_tensor_to_device(tensor: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:

    try:
        return tensor.to(device)
    except Exception as e:
        print(f"Warning: Failed to move tensor to {device}: {e}")
        return tensor


def estimate_memory_usage(video_shape: Tuple[int, ...], dtype: torch.dtype = torch.float16) -> float:

    element_size = 2 if dtype == torch.float16 else 4  # bytes per element
    tensor_size = np.prod(video_shape) * element_size
    
    total_size = tensor_size * 3

    return total_size / (1024 ** 3)


def get_optimal_batch_size(video_shape: Tuple[int, ...], available_memory_gb: float = 8.0) -> int:

    frame_shape = video_shape[1:]  # (H, W, C)
    memory_per_frame = estimate_memory_usage((1,) + frame_shape)
    
    # Use 80% of available memory for safety
    safe_memory = available_memory_gb * 0.8
    
    batch_size = max(1, int(safe_memory / memory_per_frame))
    return min(batch_size, video_shape[0])  # Don't exceed total frames


def format_tensor_info(tensor: torch.Tensor, name: str = "tensor") -> str:

    if tensor is None:
        return f"{name}: None"
    
    return (f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
            f"device={tensor.device}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")


def check_dependencies() -> Tuple[bool, str]:

    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    try:
        import decord
    except ImportError:
        missing_deps.append("decord")
    
    try:
        import einops
    except ImportError:
        missing_deps.append("einops")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    if missing_deps:
        error_msg = f"Missing dependencies: {', '.join(missing_deps)}\n"
        error_msg += "Please install with: pip install " + " ".join(missing_deps)
        return False, error_msg
    
    return True, ""


def create_error_placeholder(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:

    return torch.zeros(shape, dtype=dtype)
