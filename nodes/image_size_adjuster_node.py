import torch
import torch.nn.functional as F

class ImageSizeAdjusterNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to adjust"}),
                "divisible_by": ([2, 4, 8, 16, 32, 64], {"default": 16, "tooltip": "Make dimensions divisible by this number"}),
                "adjustment_mode": (["crop", "pad", "resize"], {"default": "crop", "tooltip": "How to adjust size"}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for resize mode"}),
                "pad_color": (["black", "white", "edge"], {"default": "black", "tooltip": "Padding color"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to adjust"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("adjusted_image", "adjusted_mask")
    FUNCTION = "adjust_image_size"
    CATEGORY = "🧪AILab/🧼MiniMax-Remover"
    
    def adjust_image_size(self, image, divisible_by=16, adjustment_mode="crop",
                         scale_by=1.0, pad_color="black", mask=None):
        try:
            batch_size, orig_height, orig_width, channels = image.shape

            # Step 1: Apply scale_by first (actual scaling)
            if scale_by != 1.0:
                scaled_image = self._resize_image(image, int(orig_height * scale_by), int(orig_width * scale_by))
                scaled_mask = self._resize_mask(mask, int(orig_height * scale_by), int(orig_width * scale_by)) if mask is not None else None
                scaled_height = int(orig_height * scale_by)
                scaled_width = int(orig_width * scale_by)
            else:
                scaled_image = image
                scaled_mask = mask
                scaled_height = orig_height
                scaled_width = orig_width

            # Step 2: Calculate target size based on divisible_by
            if adjustment_mode == "crop":
                target_height = (scaled_height // divisible_by) * divisible_by
                target_width = (scaled_width // divisible_by) * divisible_by
            elif adjustment_mode == "pad":
                target_height = ((scaled_height + divisible_by - 1) // divisible_by) * divisible_by
                target_width = ((scaled_width + divisible_by - 1) // divisible_by) * divisible_by
            elif adjustment_mode == "resize":
                target_height = ((scaled_height + divisible_by - 1) // divisible_by) * divisible_by
                target_width = ((scaled_width + divisible_by - 1) // divisible_by) * divisible_by

            target_height = max(target_height, divisible_by)
            target_width = max(target_width, divisible_by)

            # Step 3: Apply adjustment mode to reach target size
            if (scaled_height, scaled_width) != (target_height, target_width):
                if adjustment_mode == "resize":
                    adjusted_image = self._resize_image(scaled_image, target_height, target_width)
                    adjusted_mask = self._resize_mask(scaled_mask, target_height, target_width) if scaled_mask is not None else None
                elif adjustment_mode == "crop":
                    adjusted_image = self._center_crop_image(scaled_image, target_height, target_width)
                    adjusted_mask = self._center_crop_mask(scaled_mask, target_height, target_width) if scaled_mask is not None else None
                elif adjustment_mode == "pad":
                    adjusted_image = self._center_pad_image(scaled_image, target_height, target_width, pad_color)
                    adjusted_mask = self._center_pad_mask(scaled_mask, target_height, target_width) if scaled_mask is not None else None
            else:
                adjusted_image = scaled_image
                adjusted_mask = scaled_mask

            # Print transformation info
            if (orig_height, orig_width) != (target_height, target_width):
                if scale_by != 1.0:
                    print(f"Image Size Adjuster: {orig_height}x{orig_width} → {scaled_height}x{scaled_width} (scale={scale_by:.1f}) → {target_height}x{target_width} ({adjustment_mode}, divisible_by={divisible_by})")
                else:
                    print(f"Image Size Adjuster: {orig_height}x{orig_width} → {target_height}x{target_width} ({adjustment_mode}, divisible_by={divisible_by})")

            if adjusted_mask is None:
                adjusted_mask = torch.zeros((batch_size, target_height, target_width), dtype=torch.float32)

            return (adjusted_image, adjusted_mask)

        except Exception as e:
            print(f"Error in Image Size Adjuster: {str(e)}")
            dummy_mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)
            return (image, mask if mask is not None else dummy_mask)
    
    def _resize_image(self, image, target_height, target_width):
        image_resized = image.permute(0, 3, 1, 2)
        image_resized = F.interpolate(image_resized, size=(target_height, target_width), mode='bilinear', align_corners=False)
        return image_resized.permute(0, 2, 3, 1)

    def _resize_mask(self, mask, target_height, target_width):
        if mask is None:
            return None
        mask_resized = mask.unsqueeze(1) if len(mask.shape) == 3 else mask
        mask_resized = F.interpolate(mask_resized, size=(target_height, target_width), mode='bilinear', align_corners=False)
        return mask_resized.squeeze(1)
    
    def _center_crop_image(self, image, target_height, target_width):
        """Center crop image to target size"""
        batch_size, orig_height, orig_width, channels = image.shape
        
        # Calculate crop coordinates
        start_y = max(0, (orig_height - target_height) // 2)
        start_x = max(0, (orig_width - target_width) // 2)
        end_y = min(orig_height, start_y + target_height)
        end_x = min(orig_width, start_x + target_width)
        
        # Crop the image
        cropped = image[:, start_y:end_y, start_x:end_x, :]
        
        # If cropped size is smaller than target, pad it
        if cropped.shape[1] < target_height or cropped.shape[2] < target_width:
            cropped = self._center_pad_image(cropped, target_height, target_width, "black")
        

        return cropped
    
    def _center_crop_mask(self, mask, target_height, target_width):
        """Center crop mask to target size"""
        if mask is None:
            return None
        
        batch_size, orig_height, orig_width = mask.shape
        
        # Calculate crop coordinates
        start_y = max(0, (orig_height - target_height) // 2)
        start_x = max(0, (orig_width - target_width) // 2)
        end_y = min(orig_height, start_y + target_height)
        end_x = min(orig_width, start_x + target_width)
        
        # Crop the mask
        cropped = mask[:, start_y:end_y, start_x:end_x]
        
        # If cropped size is smaller than target, pad it
        if cropped.shape[1] < target_height or cropped.shape[2] < target_width:
            cropped = self._center_pad_mask(cropped, target_height, target_width)
        
        return cropped
    
    def _center_pad_image(self, image, target_height, target_width, pad_color):
        """Center pad image to target size"""
        batch_size, orig_height, orig_width, channels = image.shape
        
        if orig_height >= target_height and orig_width >= target_width:
            # If image is larger, crop it first
            return self._center_crop_image(image, target_height, target_width)
        
        # Calculate padding
        pad_top = (target_height - orig_height) // 2
        pad_bottom = target_height - orig_height - pad_top
        pad_left = (target_width - orig_width) // 2
        pad_right = target_width - orig_width - pad_left
        
        # Determine padding value
        if pad_color == "black":
            pad_value = 0.0
        elif pad_color == "white":
            pad_value = 1.0
        elif pad_color == "edge":
            # Use edge padding mode
            padded = F.pad(
                image.permute(0, 3, 1, 2),  # (B, H, W, C) -> (B, C, H, W)
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
            return padded.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        else:
            pad_value = 0.0
        
        # Apply constant padding
        padded = F.pad(
            image.permute(0, 3, 1, 2),  # (B, H, W, C) -> (B, C, H, W)
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=pad_value
        )
        

        return padded.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    
    def _center_pad_mask(self, mask, target_height, target_width):
        """Center pad mask to target size"""
        if mask is None:
            return None
        
        batch_size, orig_height, orig_width = mask.shape
        
        if orig_height >= target_height and orig_width >= target_width:
            # If mask is larger, crop it first
            return self._center_crop_mask(mask, target_height, target_width)
        
        # Calculate padding
        pad_top = (target_height - orig_height) // 2
        pad_bottom = target_height - orig_height - pad_top
        pad_left = (target_width - orig_width) // 2
        pad_right = target_width - orig_width - pad_left
        
        # Apply constant padding with 0 (background)
        padded = F.pad(
            mask.unsqueeze(1),  # (B, H, W) -> (B, 1, H, W)
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0.0
        )
        
        return padded.squeeze(1)  # (B, 1, H, W) -> (B, H, W)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageSizeAdjuster": ImageSizeAdjusterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSizeAdjuster": "MiniMax Image Size Adjuster"
}
