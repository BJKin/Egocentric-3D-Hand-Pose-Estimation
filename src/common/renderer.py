"""
Differentiable MANO silhouette rasterizer used by the rendered segmentation loss.
"""
import math
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    BlendParams, MeshRasterizer, MeshRenderer, PerspectiveCameras,
    RasterizationSettings, SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from src.common.mano import build_mano_aa


class MANORenderer(nn.Module):
    def __init__(self, img_res: int):
        """
        Builds a silhouette rasterizer at the requested patch resolution.

        Arguments:
            img_res -- output silhouette mask resolution
        """
        super().__init__()
        blend_params = BlendParams(sigma=1e-5, gamma=1e-4)
        dist_eps = 1e-6
        raster_settings = RasterizationSettings(
            image_size=img_res,
            blur_radius=math.log(1.0 / dist_eps - 1.0) * blend_params.sigma,
            faces_per_pixel=10,
            perspective_correct=False,
        )
        rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.renderer = MeshRenderer(rasterizer=rasterizer, shader=SoftSilhouetteShader())

        faces_r = build_mano_aa(is_rhand=True).faces.astype("int64")
        faces_l = build_mano_aa(is_rhand=False).faces.astype("int64")
        self.register_buffer("faces_r", torch.from_numpy(faces_r))
        self.register_buffer("faces_l", torch.from_numpy(faces_l))

        intrx_to_ndc = torch.tensor([
            [2.0 / img_res, 0.0, -1.0],
            [0.0, 2.0 / img_res, -1.0],
            [0.0, 0.0, 1.0],
        ])
        self.register_buffer("intrx_to_ndc", intrx_to_ndc)
        self.img_res = img_res

    def forward(self, vertices: torch.Tensor, K: torch.Tensor, is_right: bool) -> torch.Tensor:
        """
        Rasterizes a batch of MANO meshes into soft silhouette masks.

        Arguments:
            vertices -- (B, 778, 3) MANO vertices in camera coordinates
            K -- (B, 3, 3) camera intrinsics for each sample in patch pixel space
            is_right -- True for right-hand faces, False for left-hand faces

        Returns:
            mask -- (B, 1, img_res, img_res) soft silhouette in [0, 1]
        """
        bz = vertices.shape[0]
        faces = (self.faces_r if is_right else self.faces_l).unsqueeze(0).expand(bz, -1, -1)

        K_ndc = torch.matmul(self.intrx_to_ndc.unsqueeze(0).expand(bz, -1, -1), K)
        focal = torch.diagonal(K_ndc, dim1=-1, dim2=-2)[:, :2]
        principal = K_ndc[:, :2, 2]
        cameras = PerspectiveCameras(focal, principal, device=vertices.device)

        meshes = Meshes(verts=vertices, faces=faces)
        meshes.textures = torch.zeros_like(vertices)

        image = self.renderer(meshes, cameras=cameras)
        image = torch.flip(image, dims=[1, 2]).transpose(-1, -2).transpose(-2, -3)
        _, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)
        return mask
