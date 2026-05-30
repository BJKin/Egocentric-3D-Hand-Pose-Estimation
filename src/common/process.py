"""
Derives full MANO/camera GT targets
"""
import torch
from src.common import camera as cam_utils


@torch.no_grad()
def process_data(mano_r, mano_l, targets: dict, K: torch.Tensor) -> dict:
    """
    Runs MANO forward on the GT pose/beta and derives the camera vertex and
    weak perspective camera targets that the loss compares predictions against.

    Arguments:
        mano_r -- right-hand smplx.MANO layer
        mano_l -- left-hand smplx.MANO layer
        targets -- dict containing 'mano.pose.{r,l}' (B,48), 'mano.beta.{r,l}' (B,10),
                   and 'mano.j3d.full.{r,l}' (B,21,3)
        K -- (B, 3, 3) camera intrinsics for each sample

    Returns:
        targets -- same dict with additional keys mano.joints3d.{r,l},
                   mano.vertices.{r,l}, mano.v3d.cam.{r,l}, mano.j3d.cam.{r,l},
                   mano.cam_t.{r,l}, mano.cam_t.wp.{r,l}
    """
    img_res = 224

    for side, mano in (("r", mano_r), ("l", mano_l)):
        pose = targets[f"mano.pose.{side}"]
        beta = targets[f"mano.beta.{side}"]
        gt_out = mano(betas=beta, hand_pose=pose[:, 3:], global_orient=pose[:, :3], transl=None)
        joints_out = gt_out.joints
        verts_out = gt_out.vertices
        root_out = joints_out[:, 0]

        delta = (targets[f"mano.j3d.full.{side}"] - joints_out).mean(dim=1)
        verts_cam = verts_out + delta[:, None, :]

        root_cam = targets[f"mano.j3d.full.{side}"][:, 0]
        cam_t = root_cam - root_out

        targets[f"mano.joints3d.{side}"] = joints_out
        targets[f"mano.vertices.{side}"] = verts_out
        targets[f"mano.v3d.cam.{side}"] = verts_cam
        targets[f"mano.j3d.cam.{side}"] = targets[f"mano.j3d.full.{side}"]
        targets[f"mano.cam_t.{side}"] = cam_t

    avg_focal = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    targets["mano.cam_t.wp.r"] = cam_utils.perspective_to_weak_perspective_torch(
        targets["mano.cam_t.r"], avg_focal, img_res,
    )
    targets["mano.cam_t.wp.l"] = cam_utils.perspective_to_weak_perspective_torch(
        targets["mano.cam_t.l"], avg_focal, img_res,
    )
    return targets
