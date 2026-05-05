import os
import sys
import uuid
import gc
import torch
import numpy as np
from PIL import Image
import shutil

# Add parent directory to path to allow importing anigen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anigen.pipelines import AnigenImageTo3DPipeline
from anigen.utils.random_utils import set_random_seed
from anigen.utils.ckpt_utils import ensure_ckpts

# Constants
SS_MODEL_CHOICES = ["ss_flow_duet", "ss_flow_solo", "ss_flow_epic"]
SLAT_MODEL_CHOICES = ["slat_flow_auto", "slat_flow_control"]
DEFAULT_SS_MODEL = "ss_flow_duet"
DEFAULT_SLAT_MODEL = "slat_flow_auto"

# Global pipeline instance
_pipeline = None
_current_ss_model = None
_current_slat_model = None


def _get_pipeline(ss_model_name, slat_model_name):
    global _pipeline, _current_ss_model, _current_slat_model
    
    ensure_ckpts()
    
    # Check if we need to create or update the pipeline
    if _pipeline is None:
        # Create new pipeline
        _pipeline = AnigenImageTo3DPipeline.from_pretrained(
            ss_flow_path=f'ckpts/anigen/{ss_model_name}',
            slat_flow_path=f'ckpts/anigen/{slat_model_name}',
            device='cuda',
            use_ema=False
        )
        _pipeline.cuda()
        _current_ss_model = ss_model_name
        _current_slat_model = slat_model_name
    else:
        # Check if models have changed
        if ss_model_name != _current_ss_model:
            _pipeline.load_ss_flow_model(f'ckpts/anigen/{ss_model_name}', device='cuda', use_ema=False)
            _current_ss_model = ss_model_name
        if slat_model_name != _current_slat_model:
            _pipeline.load_slat_flow_model(f'ckpts/anigen/{slat_model_name}', device='cuda', use_ema=False)
            _current_slat_model = slat_model_name
    
    return _pipeline


def _get_output_dir():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


class AniGenImageTo3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "ss_model": (SS_MODEL_CHOICES, {"default": DEFAULT_SS_MODEL}),
                "slat_model": (SLAT_MODEL_CHOICES, {"default": DEFAULT_SLAT_MODEL}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 15.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "joints_density": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
                "texture_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 256}),
                "simplify_ratio": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "smooth_skin_weights": ("BOOLEAN", {"default": True}),
                "filter_skin_weights": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mesh_path", "skeleton_path", "processed_image")
    FUNCTION = "generate_3d"
    CATEGORY = "AniGen"
    
    def generate_3d(
        self,
        image,
        seed,
        ss_model,
        slat_model,
        ss_guidance_strength,
        ss_sampling_steps,
        slat_guidance_strength,
        slat_sampling_steps,
        joints_density,
        texture_size,
        simplify_ratio,
        fill_holes,
        smooth_skin_weights,
        filter_skin_weights,
    ):
        # Convert ComfyUI image to PIL Image
        # ComfyUI image format: (batch, height, width, channels)
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[-1] == 4:
            # Convert RGBA to RGB if needed
            pil_image = Image.fromarray(image_np).convert('RGB')
        else:
            pil_image = Image.fromarray(image_np)
        
        # Get pipeline
        pipeline = _get_pipeline(ss_model, slat_model)
        
        # Create output directory
        run_id = uuid.uuid4().hex
        output_dir = os.path.join(_get_output_dir(), run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        output_glb_path = os.path.join(output_dir, 'mesh.glb')
        skeleton_glb_path = os.path.join(output_dir, 'skeleton.glb')
        
        # Run pipeline
        try:
            outputs = pipeline.run(
                pil_image,
                seed=seed,
                cfg_scale_ss=ss_guidance_strength,
                cfg_scale_slat=slat_guidance_strength,
                ss_steps=ss_sampling_steps,
                slat_steps=slat_sampling_steps,
                joints_density=joints_density,
                texture_size=texture_size,
                simplify_ratio=simplify_ratio,
                fill_holes=fill_holes,
                no_smooth_skin_weights=not smooth_skin_weights,
                no_filter_skin_weights=not filter_skin_weights,
                output_glb=output_glb_path,
            )
            
            # Process outputs
            processed_image = outputs['processed_image']
            
            # Convert PIL image to ComfyUI image format
            processed_image_np = np.array(processed_image).astype(np.float32) / 255.0
            if processed_image_np.ndim == 3:
                processed_image_tensor = torch.from_numpy(processed_image_np)[None, ...]
            else:
                processed_image_tensor = torch.from_numpy(processed_image_np)
            
            # Prepare outputs
            mesh_path = output_glb_path if os.path.exists(output_glb_path) else ""
            skeleton_path = skeleton_glb_path if os.path.exists(skeleton_glb_path) else ""
            
            return (mesh_path, skeleton_path, processed_image_tensor)
        finally:
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()


class AniGenCleanupTemp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "cleanup"
    CATEGORY = "AniGen"
    
    def cleanup(self, confirm):
        if not confirm:
            return ("Cleanup not confirmed",)
        
        temp_dir = _get_output_dir()
        try:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            return (f"Cleaned up {temp_dir}",)
        except Exception as e:
            return (f"Error cleaning up: {str(e)}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "AniGenImageTo3D": AniGenImageTo3D,
    "AniGenCleanupTemp": AniGenCleanupTemp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AniGenImageTo3D": "AniGen: Image to 3D",
    "AniGenCleanupTemp": "AniGen: Cleanup Temp Files",
}

