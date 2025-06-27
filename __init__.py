from .nodes.minimax_video_remover_node import MinimaxVideoRemoverNode
from .nodes.minimax_image_remover_node import MinimaxImageRemoverNode
from .nodes.video_loader_node import VideoLoaderNode
from .nodes.model_loader_node import MinimaxModelLoaderNode
from .nodes.image_size_adjuster_node import ImageSizeAdjusterNode

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MinimaxVideoRemover": MinimaxVideoRemoverNode,
    "MinimaxImageRemover": MinimaxImageRemoverNode,
    "MinimaxVideoLoader": VideoLoaderNode,
    "MinimaxModelLoader": MinimaxModelLoaderNode,
    "ImageSizeAdjuster": ImageSizeAdjusterNode,
}

# Display name mappings for better UI readability
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxVideoRemover": "MiniMax Video Object Remover",
    "MinimaxImageRemover": "MiniMax Image Object Remover",
    "MinimaxVideoLoader": "MiniMax Video Loader",
    "MinimaxModelLoader": "MiniMax Model Loader",
    "ImageSizeAdjuster": "MiniMax Image Size Adjuster",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
