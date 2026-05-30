"""
Hand regression heads: HMRLayer iteratively refines a pose/shape/cam vector
from a flat feature; HandHMR wraps it for the MANO 16-joint hand parameterization.
"""
from typing import Dict
import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn


# ----------------------------- config -----------------------------
_HAND_SPECS = {"pose_6d": 6 * 16, "cam_t_wp": 3, "shape": 10}


class HMRLayer(nn.Module):
    def __init__(self, feat_dim: int, mid_dim: int, specs_dict: Dict[str, int]):
        """
        Iterative regressor that refines a concatenated vector dict in given number of rounds.

        Arguments:
            feat_dim -- input feature dimension
            mid_dim -- hidden dimension of the refine MLP
            specs_dict -- mapping from output name to its vector length
        """
        super().__init__()
        self.specs_dict = specs_dict
        vector_dim = sum(specs_dict.values())
        self.refine = nn.Sequential(
            nn.Linear(feat_dim + vector_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(),
        )
        decoders = {key: nn.Linear(mid_dim, vec_size) for key, vec_size in specs_dict.items()}
        self.decoders = nn.ModuleDict(decoders)
        for decoder in self.decoders.values():
            nn.init.xavier_uniform_(decoder.weight, gain=0.01)

    def forward(self, feat: torch.Tensor, init_vector_dict: Dict[str, torch.Tensor], n_iter: int) -> Dict[str, torch.Tensor]:
        """
        Runs residual refinement for given number of steps over the concatenated state vector.

        Arguments:
            feat -- (B, feat_dim) flat features
            init_vector_dict -- starting values for each key in self.specs_dict
            n_iter -- number of refinement iterations

        Returns:
            pred -- dict of refined vectors with the same keys as specs_dict
        """
        pred = dict(init_vector_dict)
        for _ in range(n_iter):
            xc = torch.cat([feat] + [pred[k] for k in self.specs_dict], dim=1)
            xc = self.refine(xc)
            for key, decoder in self.decoders.items():
                pred[key] = decoder(xc) + pred[key]
        return pred


class HandHMR(nn.Module):
    def __init__(self, feat_dim: int, is_rhand: bool, n_iter: int = 3):
        """
        Wraps HMRLayer for MANO hand regression with an MLP that bootstraps the
        initial weak perspective camera translation from features.

        Arguments:
            feat_dim -- input feature dimension
            is_rhand -- True for the right hand, False for the left hand
            n_iter -- number of HMRLayer refinement iterations
        """
        super().__init__()
        self.is_rhand = is_rhand
        self.n_iter = n_iter
        self.hmr_layer = HMRLayer(feat_dim, 1024, _HAND_SPECS)
        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def _init_vector_dict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Initializes pose (zero rotation in 6D), shape (zeros) and camera (MLP from features).

        Arguments:
            features -- (B, feat_dim) flat features

        Returns:
            init -- dict with keys pose_6d, shape, cam_t_wp
        """
        batch_size = features.shape[0]
        dev = features.device
        init_pose = rot_conv.matrix_to_rotation_6d(rot_conv.axis_angle_to_matrix(torch.zeros(16, 3, device=dev))).reshape(1, -1).repeat(batch_size, 1)
        init_shape = torch.zeros(batch_size, 10, device=dev)
        init_cam = self.cam_init(features)
        return {"pose_6d": init_pose, "shape": init_shape, "cam_t_wp": init_cam}

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs iterative MANO regression and converts the predicted 6D pose to rotation matrices.

        Arguments:
            features -- (B, feat_dim) flat features

        Returns:
            pred -- dict with keys pose (B, 16, 3, 3), shape (B, 10), cam_t.wp (B, 3),
                    cam_t.wp.init (B, 3 — pre-refinement initial camera)
        """
        batch_size = features.shape[0]
        init = self._init_vector_dict(features)
        init_cam = init["cam_t_wp"].clone()
        pred = self.hmr_layer(features, init, self.n_iter)
        pose_rotmat = rot_conv.rotation_6d_to_matrix(
            pred["pose_6d"].reshape(-1, 6)
        ).view(batch_size, 16, 3, 3)
        return {
            "pose": pose_rotmat,
            "shape": pred["shape"],
            "cam_t.wp": pred["cam_t_wp"],
            "cam_t.wp.init": init_cam,
        }
