import torch
import os
import folder_paths
from typing import Tuple, Optional

try:
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
    import sys
    
    # Add current directory to path for importing transformer
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    from Minimax.transformer_minimax_remover import Transformer3DModel
    
except ImportError as e:
    print(f"Warning: Failed to import MiniMax model dependencies: {e}")
    print("Please ensure diffusers and other dependencies are installed")

class MinimaxModelLoaderNode:
  
    CATEGORY = "🧪AILab/🧼MiniMax-Remover"
    RETURN_TYPES = ("VAE", "TRANSFORMER", "SCHEDULER")
    RETURN_NAMES = ("vae", "transformer", "scheduler")
    FUNCTION = "load_models"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "scheduler": (["Auto", "UniPC", "FlowMatch"], {"default": "Auto", "tooltip": "Auto: try UniPC first (official), UniPC: official example, FlowMatch: alternative"}),
                "torch_dtype": (["float16", "float32", "bfloat16"], {"default": "float16", "tooltip": "Model precision: float16 (recommended, fast), float32 (high precision, may have issues), bfloat16 (experimental)"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "Device to load models on"}),
                "local_files_only": ("BOOLEAN", {"default": False, "tooltip": "Only use local files, don't download from HuggingFace"}),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs"""
        return True

    def load_models(self, scheduler="Auto", torch_dtype="float16", device="auto", local_files_only=False):
        try:
            # Use ComfyUI standard models directory
            models_dir = folder_paths.models_dir
            local_model_path = os.path.join(models_dir, "MiniMax-Remover")
            hf_model_path = "zibojia/minimax-remover"

            device_str = str(device)
            if device_str == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device_str

            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)

            # Determine which model path to use and set up download
            if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
                model_path = local_model_path
                load_kwargs = {
                    "torch_dtype": torch_dtype_obj,
                    "local_files_only": True,
                }
                print(f"📁 Using local MiniMax models from: {local_model_path}")
            else:
                if not local_files_only:
                    # Download models to ComfyUI models directory
                    print(f"🌐 Downloading MiniMax models to: {local_model_path}")
                    print("📥 This may take a while on first run...")

                    try:
                        from huggingface_hub import snapshot_download
                        # Create the directory if it doesn't exist
                        os.makedirs(local_model_path, exist_ok=True)
                        print(f"📁 Created directory: {local_model_path}")

                        # Download the entire model repository
                        downloaded_path = snapshot_download(
                            repo_id=hf_model_path,
                            local_dir=local_model_path,
                            local_dir_use_symlinks=False,  # Copy files instead of symlinks
                        )
                        print(f"✅ Models downloaded successfully to: {downloaded_path}")
                        print(f"📂 Contents: {os.listdir(local_model_path) if os.path.exists(local_model_path) else 'Directory not found'}")
                        model_path = local_model_path
                        load_kwargs = {
                            "torch_dtype": torch_dtype_obj,
                            "local_files_only": True,
                        }
                    except Exception as e:
                        print(f"⚠️  Download failed, falling back to HuggingFace cache: {e}")
                        print(f"🔍 Will try to load from: {hf_model_path}")
                        model_path = hf_model_path
                        load_kwargs = {
                            "torch_dtype": torch_dtype_obj,
                            "local_files_only": False,
                        }
                else:
                    model_path = hf_model_path
                    load_kwargs = {
                        "torch_dtype": torch_dtype_obj,
                        "local_files_only": True,
                    }
                    print(f"❌ Local models not found at {local_model_path} and local_files_only=True")

            # Load VAE
            if os.path.isdir(model_path):
                # Local directory
                vae_path = os.path.join(model_path, "vae")
                if os.path.exists(vae_path):
                    vae = AutoencoderKLWan.from_pretrained(vae_path, **load_kwargs)
                else:
                    raise FileNotFoundError(f"VAE not found at {vae_path}")
            else:
                # HuggingFace path
                vae = AutoencoderKLWan.from_pretrained(
                    model_path,
                    subfolder="vae",
                    **load_kwargs
                )
            
            # Load Transformer
            if os.path.isdir(model_path):
                # Local directory
                transformer_path = os.path.join(model_path, "transformer")
                if os.path.exists(transformer_path):
                    transformer = Transformer3DModel.from_pretrained(transformer_path, **load_kwargs)
                else:
                    raise FileNotFoundError(f"Transformer not found at {transformer_path}")
            else:
                # HuggingFace path
                transformer = Transformer3DModel.from_pretrained(
                    model_path,
                    subfolder="transformer",
                    **load_kwargs
                )

            # Load Scheduler based on user choice
            loaded_scheduler = None

            # Determine scheduler classes to try based on user selection
            if scheduler == "UniPC":
                scheduler_classes = [UniPCMultistepScheduler]
                mode_desc = "UniPC (official example)"
            elif scheduler == "FlowMatch":
                scheduler_classes = [FlowMatchEulerDiscreteScheduler]
                mode_desc = "FlowMatch (alternative)"
            else:  # Auto
                scheduler_classes = [UniPCMultistepScheduler, FlowMatchEulerDiscreteScheduler]
                mode_desc = "Auto mode"

            for i, scheduler_class in enumerate(scheduler_classes):
                try:
                    if os.path.isdir(model_path):
                        # Local directory
                        scheduler_path = os.path.join(model_path, "scheduler")
                        if os.path.exists(scheduler_path):
                            loaded_scheduler = scheduler_class.from_pretrained(scheduler_path)
                        else:
                            raise FileNotFoundError(f"Scheduler not found at {scheduler_path}")
                    else:
                        # HuggingFace path
                        loaded_scheduler = scheduler_class.from_pretrained(
                            model_path,
                            subfolder="scheduler"
                        )

                    # Provide clear feedback about what was loaded
                    if scheduler == "Auto":
                        if i == 0:
                            print(f"✅ {mode_desc}: Successfully loaded {scheduler_class.__name__} (official)")
                        else:
                            print(f"✅ {mode_desc}: Using {scheduler_class.__name__} (fallback)")
                    else:
                        print(f"✅ Loaded {scheduler_class.__name__} ({mode_desc})")
                    break

                except Exception as e:
                    if scheduler == "Auto":
                        if i == 0:
                            print(f"⚠️  {mode_desc}: {scheduler_class.__name__} failed, trying fallback...")
                        else:
                            print(f"❌ {mode_desc}: All schedulers failed")
                    else:
                        print(f"❌ Failed to load {scheduler_class.__name__}: {e}")
                        # If user specifically requested this scheduler, don't try others
                        raise e
                    continue

            if loaded_scheduler is None:
                raise RuntimeError("Failed to load any compatible scheduler")
            
            # Move models to device (use actual_device to ensure it's a string)
            print(f"🚀 Moving models to device: {actual_device}")
            vae = vae.to(actual_device)
            transformer = transformer.to(actual_device)

            return (vae, transformer, loaded_scheduler)
            
        except Exception as e:
            print(f"Error loading MiniMax models: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to provide helpful error messages
            if "not found" in str(e).lower():
                print("\nTroubleshooting:")
                print("1. Check your internet connection for downloading from HuggingFace")
                print("2. If using local_files_only=True, make sure model is downloaded:")
                print("   huggingface-cli download zibojia/minimax-remover --include vae transformer scheduler --local-dir ./models/MiniMax-Remover")
                print("3. Try setting local_files_only=False to download automatically")
                print("4. Check available disk space for model download")
            
            raise e
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Check if parameters have changed"""
        return kwargs.get("torch_dtype", "float16")
