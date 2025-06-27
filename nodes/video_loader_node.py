import torch
import numpy as np
import os
import folder_paths
from typing import List, Tuple

try:
    from decord import VideoReader
    import cv2
except ImportError as e:
    print(f"Warning: Failed to import video processing dependencies: {e}")
    print("Please install decord and opencv-python")


class VideoLoaderNode:
    
    CATEGORY = "🧪AILab/🧼MiniMax-Remover"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("video_frames", "frame_count")
    FUNCTION = "load_video"
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        video_files = []
        
        if os.path.exists(input_dir):
            for file in os.listdir(input_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')):
                    video_files.append(file)
        
        return {
            "required": {
                "video_file": (sorted(video_files), {
                    "tooltip": "Select video file from ComfyUI input directory"
                }),
            },
            "optional": {
                "max_frames": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Maximum number of frames to load (0 = load all)"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Starting frame index"
                }),
                "frame_step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Frame sampling step (1 = every frame, 2 = every other frame)"
                }),
                "target_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Target width for resizing (0 = keep original)"
                }),
                "target_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Target height for resizing (0 = keep original)"
                }),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, video_file, **kwargs):
        if not video_file:
            return "No video file selected"

        input_dir = folder_paths.get_input_directory()
        video_path = os.path.join(input_dir, video_file)

        if not os.path.exists(video_path):
            return f"Video file not found: {video_file}"

        return True
    
    def load_video(self, video_file, max_frames=81, start_frame=0, frame_step=1,
                   target_width=0, target_height=0):

        try:
            input_dir = folder_paths.get_input_directory()
            video_path = os.path.join(input_dir, video_file)
            
            vr = VideoReader(video_path)
            total_frames = len(vr)
            
            end_frame = min(start_frame + max_frames * frame_step, total_frames) if max_frames > 0 else total_frames
            frame_indices = list(range(start_frame, end_frame, frame_step))
            
            if not frame_indices:
                raise ValueError("No frames to load with current settings")
            
            # Load frames
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Convert to torch tensor and normalize to [0, 1]
            frames = torch.from_numpy(frames).float() / 255.0
            
            if target_width > 0 and target_height > 0:
                frames = self._resize_frames(frames, target_height, target_width)
            
            frame_count = frames.shape[0]
            

            
            return (frames, frame_count)
            
        except Exception as e:
            print(f"Error loading video {video_file}: {str(e)}")
            
            dummy_frame = torch.zeros((1, 480, 640, 3), dtype=torch.float32)
            return (dummy_frame, 1)
    
    def _resize_frames(self, frames, target_height, target_width):
        import torch.nn.functional as F
        
        frames = frames.permute(0, 3, 1, 2)
        
        frames = F.interpolate(
            frames, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        frames = frames.permute(0, 2, 3, 1)
        
        return frames
    
    @classmethod
    def IS_CHANGED(cls, video_file, **kwargs):
        input_dir = folder_paths.get_input_directory()
        video_path = os.path.join(input_dir, video_file)
        
        if os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return video_file
