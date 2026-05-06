import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import gc
import folder_paths

from anigen.utils.ckpt_utils import ensure_ckpts
from anigen.utils.model_utils import load_model_from_path, load_decoder
from anigen.utils.image_utils import load_dsine, preprocess_image, encode_image
from anigen.pipelines import samplers
from anigen.modules import sparse as sp
from anigen.utils.general_utils import _keep_largest_connected_component_3d
from anigen.utils.skin_utils import repair_skeleton_parents, smooth_skin_weights_on_mesh, filter_skinning_weights
from anigen.utils.export_utils import convert_to_glb_from_data, _extract_vertex_rgb, visualize_skeleton_as_mesh
from anigen.utils.postprocessing_utils import (
    postprocess_mesh,
    barycentric_transfer_attributes,
    parametrize_mesh,
    bake_texture,
)


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


SS_MODEL_CHOICES = ["ss_flow_duet", "ss_flow_solo", "ss_flow_epic"]
SLAT_MODEL_CHOICES = ["slat_flow_auto", "slat_flow_control"]


class AniGenModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ss_model": (SS_MODEL_CHOICES, {"default": "ss_flow_duet"}),
                "slat_model": (SLAT_MODEL_CHOICES, {"default": "slat_flow_auto"}),
            },
        }

    RETURN_TYPES = ("ANIGEN_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_models"
    CATEGORY = "AniGen"

    def load_models(self, ss_model, slat_model):
        ensure_ckpts()

        dinov2_model = torch.hub.load('./ckpts/dinov2', 'dinov2_vitl14_reg', pretrained=True, source='local')
        dinov2_model.to('cuda').eval()

        dsine_model = load_dsine('cuda')

        ss_flow_model, ss_config = load_model_from_path(
            f'ckpts/anigen/{ss_model}', model_name_in_config='denoiser', device='cuda', use_ema=False
        )

        ss_dec_path = ss_config.dataset.args.get('ss_dec_path')
        ss_dec_ckpt = ss_config.dataset.args.get('ss_dec_ckpt')
        ss_decoder = load_decoder(ss_dec_path, ss_dec_ckpt, 'cuda')

        slat_flow_model, slat_config = load_model_from_path(
            f'ckpts/anigen/{slat_model}', model_name_in_config='denoiser', device='cuda', use_ema=False
        )

        slat_dec_path = slat_config.dataset.args.get('slat_dec_path')
        slat_dec_ckpt = slat_config.dataset.args.get('slat_dec_ckpt')
        slat_decoder = load_decoder(slat_dec_path, slat_dec_ckpt, 'cuda')

        models = {
            'dinov2': dinov2_model,
            'dsine': dsine_model,
            'ss_flow_model': ss_flow_model,
            'ss_decoder': ss_decoder,
            'slat_flow_model': slat_flow_model,
            'slat_decoder': slat_decoder,
            'ss_config': ss_config,
            'slat_config': slat_config,
        }
        return (models,)


class AniGenPreprocessImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ANIGEN_MODELS",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("processed_image", "normal_image")
    FUNCTION = "preprocess"
    CATEGORY = "AniGen"

    def preprocess(self, models, image):
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        processed, normal = preprocess_image(pil_image, models['dsine'], 'cuda')

        processed_np = np.array(processed).astype(np.float32) / 255.0
        normal_np = np.array(normal).astype(np.float32) / 255.0

        if processed_np.ndim == 2:
            processed_np = np.stack([processed_np] * 3, axis=-1)
        if normal_np.ndim == 2:
            normal_np = np.stack([normal_np] * 3, axis=-1)

        processed_tensor = torch.from_numpy(processed_np)[None, ...]
        normal_tensor = torch.from_numpy(normal_np)[None, ...]

        return (processed_tensor, normal_tensor)


class AniGenEncodeCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ANIGEN_MODELS",),
                "processed_image": ("IMAGE",),
                "normal_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("ANIGEN_COND_SS", "ANIGEN_COND_SLAT")
    RETURN_NAMES = ("cond_ss", "cond_slat")
    FUNCTION = "encode"
    CATEGORY = "AniGen"

    def encode(self, models, processed_image, normal_image):
        processed_pil = Image.fromarray((processed_image[0].cpu().numpy() * 255).astype(np.uint8))
        normal_pil = Image.fromarray((normal_image[0].cpu().numpy() * 255).astype(np.uint8))

        cond_rgb = encode_image(processed_pil, models['dinov2'], 'cuda')
        cond_normal = encode_image(normal_pil, models['dinov2'], 'cuda')

        normal_tensor = torch.from_numpy(np.array(normal_pil)).float() / 255.0
        normal_tensor = normal_tensor.permute(2, 0, 1).unsqueeze(0).to('cuda')

        rgb_tensor = torch.from_numpy(np.array(processed_pil.convert('RGB'))).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to('cuda')

        cond_ss = {
            'cond': cond_normal,
            'neg_cond': torch.zeros_like(cond_rgb),
            'normal': normal_tensor,
        }

        cond_slat = {
            'cond': cond_rgb,
            'neg_cond': torch.zeros_like(cond_rgb),
            'normal': rgb_tensor,
        }

        return (cond_ss, cond_slat)


class AniGenSampleSS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ANIGEN_MODELS",),
                "cond_ss": ("ANIGEN_COND_SS",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 15.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("ANIGEN_SS_RESULT",)
    RETURN_NAMES = ("ss_result",)
    FUNCTION = "sample"
    CATEGORY = "AniGen"

    def sample(self, models, cond_ss, seed, ss_guidance_strength, ss_sampling_steps):
        torch.manual_seed(seed)
        np.random.seed(seed)

        ss_model = models['ss_flow_model']
        ss_decoder = models['ss_decoder']

        ss_sampler = samplers.AniGenFlowEulerCfgSampler(sigma_min=1e-5)
        reso = ss_model.resolution

        noise = torch.randn(1, ss_model.in_channels, reso, reso, reso).to('cuda')
        if ss_model.z_is_global:
            noise = torch.randn(1, ss_model.global_token_num, ss_model.in_channels).to('cuda')

        noise_skl = torch.randn(1, ss_model.in_channels_skl, reso, reso, reso).to('cuda')
        if ss_model.z_skl_is_global:
            noise_skl = torch.randn(1, ss_model.global_token_num_skl, ss_model.in_channels_skl).to('cuda')

        z_s_out = ss_sampler.sample(
            ss_model,
            noise,
            noise_skl,
            **cond_ss,
            steps=ss_sampling_steps,
            cfg_strength=ss_guidance_strength,
            verbose=True,
        )

        z_s = z_s_out.samples
        z_s_skl = z_s_out.samples_skl

        decoded_ss, decoded_ss_skl = ss_decoder(z_s, z_s_skl)

        bsz, ch, d, h, w = decoded_ss_skl.shape
        for b in range(bsz):
            occ_3d = (decoded_ss_skl[b] > 0).any(dim=0).detach().cpu().numpy()
            if not np.any(occ_3d):
                continue
            mainland_3d = _keep_largest_connected_component_3d(occ_3d)
            mainland_t = torch.from_numpy(mainland_3d).to(device=decoded_ss_skl.device)
            mainland_cd = mainland_t.unsqueeze(0).expand(ch, -1, -1, -1)
            decoded_ss_skl[b] = torch.where(
                mainland_cd,
                decoded_ss_skl[b],
                torch.full_like(decoded_ss_skl[b], -1e9),
            )

        coords = torch.argwhere(decoded_ss > 0)[:, [0, 2, 3, 4]].int()
        coords_skl = torch.argwhere(decoded_ss_skl > 0)[:, [0, 2, 3, 4]].int()

        del z_s_out, z_s, z_s_skl, decoded_ss, decoded_ss_skl
        _cuda_cleanup()

        return ({"coords": coords, "coords_skl": coords_skl},)


class AniGenSampleSLat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ANIGEN_MODELS",),
                "cond_slat": ("ANIGEN_COND_SLAT",),
                "ss_result": ("ANIGEN_SS_RESULT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "joints_density": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
            },
        }

    RETURN_TYPES = ("ANIGEN_SLAT_RESULT",)
    RETURN_NAMES = ("slat_result",)
    FUNCTION = "sample"
    CATEGORY = "AniGen"

    def sample(self, models, cond_slat, ss_result, seed, slat_guidance_strength, slat_sampling_steps, joints_density):
        torch.manual_seed(seed)
        np.random.seed(seed)

        slat_model = models['slat_flow_model']
        slat_config = models['slat_config']
        coords = ss_result['coords']
        coords_skl = ss_result['coords_skl']

        gsn_enabled = False
        gsn_iters = 0
        gsn_alpha = 0.7
        if slat_config is not None:
            trainer_args = getattr(getattr(slat_config, 'trainer', None), 'args', None)
            if trainer_args is not None:
                gsn_enabled = bool(getattr(trainer_args, 'geodesic_smooth_noise', False))
                gsn_iters = int(getattr(trainer_args, 'geodesic_smooth_noise_iters', 0))
                gsn_alpha = float(getattr(trainer_args, 'geodesic_smooth_noise_alpha', 0.7))

        slat_sampler = samplers.AniGenFlowEulerCfgSampler(
            sigma_min=1e-5,
            geodesic_smooth_noise=gsn_enabled,
            geodesic_smooth_noise_iters=gsn_iters,
            geodesic_smooth_noise_alpha=gsn_alpha,
        )

        noise_slat = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], slat_model.in_channels + slat_model.in_channels_vert_skin).to('cuda'),
            coords=coords,
        )
        noise_skl = sp.SparseTensor(
            feats=torch.randn(coords_skl.shape[0], slat_model.in_channels_skl).to('cuda'),
            coords=coords_skl,
        )

        cond = cond_slat.copy()
        use_joint_num_cond = bool(getattr(slat_model, 'use_joint_num_cond', False))
        if use_joint_num_cond:
            joints_num = {0: 0, 1: 10, 2: 15, 3: 25, 4: 35}.get(joints_density, 10)
            cond['joints_num'] = joints_num
            cond['neg_joints_num'] = 0

        out = slat_sampler.sample(
            slat_model,
            noise_slat,
            noise_skl,
            **cond,
            steps=slat_sampling_steps,
            cfg_strength=slat_guidance_strength,
            verbose=True,
        )

        slat = out.samples
        slat_skl = out.samples_skl

        if 'dataset' in slat_config and 'args' in slat_config.dataset and 'normalization' in slat_config.dataset.args:
            norm_stats = slat_config.dataset.args.normalization

            def denormalize(tensor, mean, std):
                if tensor is None:
                    return None
                mean = torch.tensor(mean).to(tensor.device)
                std = torch.tensor(std).to(tensor.device)
                return tensor * std + mean

            if 'slat' in norm_stats:
                slat = slat.replace(feats=denormalize(slat.feats, norm_stats['slat']['mean'], norm_stats['slat']['std']))
            elif 'mean' in norm_stats and 'std' in norm_stats:
                slat = slat.replace(feats=denormalize(slat.feats, norm_stats['mean'], norm_stats['std']))

            if 'slat_skl' in norm_stats:
                slat_skl = slat_skl.replace(feats=denormalize(slat_skl.feats, norm_stats['slat_skl']['mean'], norm_stats['slat_skl']['std']))
            elif 'slat_skel' in norm_stats:
                slat_skl = slat_skl.replace(feats=denormalize(slat_skl.feats, norm_stats['slat_skel']['mean'], norm_stats['slat_skel']['std']))
            elif 'mean_skl' in norm_stats and 'std_skl' in norm_stats:
                slat_skl = slat_skl.replace(feats=denormalize(slat_skl.feats, norm_stats['mean_skl'], norm_stats['std_skl']))

        del out
        _cuda_cleanup()

        return ({"slat": slat, "slat_skl": slat_skl},)


class AniGenDecodeSLat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ANIGEN_MODELS",),
                "slat_result": ("ANIGEN_SLAT_RESULT",),
            },
        }

    RETURN_TYPES = ("ANIGEN_MESH_RESULT",)
    RETURN_NAMES = ("mesh_result",)
    FUNCTION = "decode"
    CATEGORY = "AniGen"

    def decode(self, models, slat_result):
        slat_decoder = models['slat_decoder']
        slat = slat_result['slat']
        slat_skl = slat_result['slat_skl']

        meshes, skeletons = slat_decoder(slat, slat_skl)
        mesh_result = meshes[0]
        skeleton_result = skeletons[0]

        del slat_result
        _cuda_cleanup()

        return ({"mesh": mesh_result, "skeleton": skeleton_result},)


class AniGenPostprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_result": ("ANIGEN_MESH_RESULT",),
                "simplify_ratio": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "smooth_skin_weights": ("BOOLEAN", {"default": True}),
                "filter_skin_weights": ("BOOLEAN", {"default": True}),
                "smooth_skin_weights_iters": ("INT", {"default": 100, "min": 0, "max": 500, "step": 1}),
                "smooth_skin_weights_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "texture_size": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 256}),
            },
        }

    RETURN_TYPES = ("ANIGEN_POST_RESULT",)
    RETURN_NAMES = ("post_result",)
    FUNCTION = "postprocess"
    CATEGORY = "AniGen"

    def postprocess(
        self,
        mesh_result,
        simplify_ratio,
        fill_holes,
        smooth_skin_weights,
        filter_skin_weights,
        smooth_skin_weights_iters,
        smooth_skin_weights_alpha,
        texture_size,
    ):
        import trimesh

        mesh_data = mesh_result['mesh']
        skeleton_data = mesh_result['skeleton']

        joints = skeleton_data.joints_grouped.cpu().numpy()
        parents = skeleton_data.parents_grouped.cpu().numpy().astype(np.int32)
        parents = repair_skeleton_parents(joints=joints, parents=parents, verbose=False).astype(np.int32)

        skin_weights = skeleton_data.skin_pred.cpu().numpy()
        vertex_colors = _extract_vertex_rgb(getattr(mesh_data, 'vertex_attrs', None))

        orig_vertices = mesh_data.vertices.cpu().numpy()
        orig_faces = mesh_data.faces.cpu().numpy()
        del skeleton_data
        _cuda_cleanup()

        new_vertices, new_faces = postprocess_mesh(
            orig_vertices, orig_faces,
            simplify=(simplify_ratio > 0),
            simplify_ratio=simplify_ratio,
            fill_holes=fill_holes,
            verbose=True,
        )

        if new_vertices.shape[0] != orig_vertices.shape[0]:
            orig_mesh = trimesh.Trimesh(vertices=orig_vertices, faces=orig_faces, process=False)
            skin_weights = barycentric_transfer_attributes(orig_mesh, skin_weights, new_vertices)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        if filter_skin_weights:
            skin_weights = filter_skinning_weights(mesh, skin_weights, joints, parents)

        if smooth_skin_weights:
            skin_weights = smooth_skin_weights_on_mesh(
                mesh, skin_weights,
                iterations=smooth_skin_weights_iters,
                alpha=smooth_skin_weights_alpha,
            )

        texture_image = None
        if texture_size > 0:
            uv_vertices, uv_faces, uvs, vmapping = parametrize_mesh(new_vertices, new_faces)
            skin_weights = skin_weights[vmapping]

            from anigen.utils.render_utils import render_multiview

            observations, extrinsics_mv, intrinsics_mv = render_multiview(
                mesh_data, resolution=1024, nviews=100,
            )
            masks = [np.any(obs > 0, axis=-1) for obs in observations]
            extrinsics_np = [e.cpu().numpy() for e in extrinsics_mv]
            intrinsics_np = [i.cpu().numpy() for i in intrinsics_mv]
            del extrinsics_mv, intrinsics_mv
            _cuda_cleanup()

            with torch.enable_grad():
                texture = bake_texture(
                    uv_vertices, uv_faces, uvs,
                    observations, masks, extrinsics_np, intrinsics_np,
                    texture_size=texture_size, mode='opt',
                    lambda_tv=0.01,
                    verbose=True,
                )
            texture_image = texture
            del observations, masks, extrinsics_np, intrinsics_np
            _cuda_cleanup()

            mesh = trimesh.Trimesh(
                vertices=uv_vertices,
                faces=uv_faces,
                visual=trimesh.visual.TextureVisuals(uv=uvs),
                process=False,
            )

        del mesh_data
        _cuda_cleanup()

        skeleton_mesh = visualize_skeleton_as_mesh(joints, parents)

        result = {
            'mesh': mesh,
            'skeleton_mesh': skeleton_mesh,
            'joints': joints,
            'parents': parents,
            'skin_weights': skin_weights,
            'vertex_colors': vertex_colors,
            'texture_image': texture_image,
        }

        return (result,)


class AniGenExportGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "post_result": ("ANIGEN_POST_RESULT",),
                "filename_prefix": ("STRING", {"default": "AniGen"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "skeleton_path")
    FUNCTION = "export"
    CATEGORY = "AniGen"
    OUTPUT_NODE = True

    def export(self, post_result, filename_prefix):
        output_dir = folder_paths.get_output_directory()

        mesh = post_result['mesh']
        joints = post_result['joints']
        parents = post_result['parents']
        skin_weights = post_result['skin_weights']
        vertex_colors = post_result['vertex_colors']
        texture_image = post_result['texture_image']
        skeleton_mesh = post_result['skeleton_mesh']

        mesh_path = os.path.join(output_dir, f"{filename_prefix}_mesh.glb")
        convert_to_glb_from_data(
            mesh, joints, parents, skin_weights, mesh_path,
            vertex_colors=vertex_colors,
            texture_image=texture_image,
        )

        skeleton_path = ""
        if skeleton_mesh is not None and len(skeleton_mesh.vertices) > 0:
            skeleton_path = os.path.join(output_dir, f"{filename_prefix}_skeleton.glb")
            skeleton_mesh.export(skeleton_path)

        _cuda_cleanup()

        return (mesh_path, skeleton_path)


NODE_CLASS_MAPPINGS = {
    "AniGenModelLoader": AniGenModelLoader,
    "AniGenPreprocessImage": AniGenPreprocessImage,
    "AniGenEncodeCondition": AniGenEncodeCondition,
    "AniGenSampleSS": AniGenSampleSS,
    "AniGenSampleSLat": AniGenSampleSLat,
    "AniGenDecodeSLat": AniGenDecodeSLat,
    "AniGenPostprocess": AniGenPostprocess,
    "AniGenExportGLB": AniGenExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AniGenModelLoader": "AniGen: Load Models",
    "AniGenPreprocessImage": "AniGen: Preprocess Image",
    "AniGenEncodeCondition": "AniGen: Encode Condition",
    "AniGenSampleSS": "AniGen: Sample Sparse Structure",
    "AniGenSampleSLat": "AniGen: Sample Structured Latent",
    "AniGenDecodeSLat": "AniGen: Decode SLat",
    "AniGenPostprocess": "AniGen: Postprocess Mesh",
    "AniGenExportGLB": "AniGen: Export GLB",
}
