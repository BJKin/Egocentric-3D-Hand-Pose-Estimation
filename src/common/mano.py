"""
MANO hand model factory and MANO regression head.
"""
import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn
from smplx import MANO
from pathlib import Path
from src.common import camera as cam_utils
from src.common import data_utils


# ----------------------------- config -----------------------------
MANO_DIR = Path("../mano")


def build_mano_aa(is_rhand: bool, create_transl: bool = False, flat_hand: bool = False) -> MANO:
    """
    Constructs a MANO layer that consumes axis angle pose parameters.

    Arguments:
        is_rhand -- True for the right hand, False for the left hand
        create_transl -- whether the MANO layer exposes a learnable translation
        flat_hand -- if True, uses a flat hand mean instead of the default MANO hand mean

    Returns:
        mano -- smplx.MANO instance configured for axis angle input
    """
    return MANO(
        str(MANO_DIR),
        create_transl=create_transl,
        use_pca=False,
        flat_hand_mean=flat_hand,
        is_rhand=is_rhand,
    )


class MANOHead(nn.Module):
    def __init__(self, is_rhand: bool, focal_length: float, img_res: int):
        """
        Builds a MANO regression head for one hand side.

        Arguments:
            is_rhand -- True for the right hand, False for the left hand
            focal_length -- fallback focal length in pixels
            img_res -- patch resolution used for weak-perspective scaling and 2D keypoint normalization
        """
        super().__init__()
        self.mano = build_mano_aa(is_rhand)
        self.is_rhand = is_rhand
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat: torch.Tensor, shape: torch.Tensor, cam: torch.Tensor, K: torch.Tensor) -> dict:
        """
        Runs MANO forward and projects the resulting joints into camera and image space.

        Arguments:
            rotmat -- (B, 16, 3, 3) rotation matrices for each joint
            shape -- (B, 10) MANO beta parameters
            cam -- (B, 3) predicted weak-perspective camera [s, tx, ty]
            K -- (B, 3, 3) camera intrinsics for each sample

        Returns:
            output -- dict of tensors with keys cam_t.wp, cam_t, joints3d, vertices,
                      j3d.cam, v3d.cam, j2d.norm, beta, pose, each postfixed with
                      '.r' or '.l' depending on which hand this head represents
        """
        rotmat_original = rotmat.clone()
        if rotmat.shape[-1] != 48:
            rotmat = rot_conv.matrix_to_axis_angle(rotmat.reshape(-1, 3, 3)).reshape(-1, 48)

        mano_output = self.mano(
            betas=shape,
            hand_pose=rotmat[:, 3:],
            global_orient=rotmat[:, :3],
        )

        avg_focal = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
        cam_t = cam_utils.weak_perspective_to_perspective_torch(
            cam, focal_length=avg_focal, img_res=self.img_res, min_s=0.1,
        )

        joints3d_cam = mano_output.joints + cam_t[:, None, :]
        v3d_cam = mano_output.vertices + cam_t[:, None, :]

        joints2d = cam_utils.project2d_batch(K, joints3d_cam)
        joints2d = data_utils.normalize_kp2d(joints2d, self.img_res)

        postfix = ".r" if self.is_rhand else ".l"
        return {
            f"cam_t.wp{postfix}": cam,
            f"cam_t{postfix}": cam_t,
            f"joints3d{postfix}": mano_output.joints,
            f"vertices{postfix}": mano_output.vertices,
            f"j3d.cam{postfix}": joints3d_cam,
            f"v3d.cam{postfix}": v3d_cam,
            f"j2d.norm{postfix}": joints2d,
            f"beta{postfix}": shape,
            f"pose{postfix}": rotmat_original,
        }
