"""
Training losses.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
_mse = nn.MSELoss(reduction="none")


def _vector_loss(pred: torch.Tensor, gt: torch.Tensor, is_valid: torch.Tensor) -> torch.Tensor:
    """
    MSE between two tensors, computed for each sample and masked by an is_valid flag.

    Arguments:
        pred -- (B, ...) predicted tensor
        gt -- (B, ...) target tensor
        is_valid -- (B,) float mask (1 keeps the sample, 0 zeros it)

    Returns:
        loss -- (B, K) loss for each element of each sample
    """
    dist = _mse(pred, gt)
    bz = dist.shape[0]
    dist = dist.reshape(bz, -1)
    if is_valid.sum() == 0:
        return torch.zeros_like(dist)
    return dist * is_valid[..., None]


def _joints_loss(pred: torch.Tensor, gt: torch.Tensor, jts_valid: torch.Tensor) -> torch.Tensor:
    """
    MSE computed for each joint between two batched keypoint tensors, masked by the validity of each joint.

    Arguments:
        pred -- (B, N, D) predicted keypoints
        gt -- (B, N, D) target keypoints
        jts_valid -- (B, N) float mask

    Returns:
        loss -- (B, N, D) loss for each coordinate of each joint
    """
    return _mse(pred, gt) * jts_valid[:, :, None]


def _hand_kp3d_loss(pred_3d: torch.Tensor, gt_3d: torch.Tensor, jts_valid: torch.Tensor) -> torch.Tensor:
    """
    Root-relative 3D keypoint MSE (subtracts each tensor's joint 0 before comparing).

    Arguments:
        pred_3d -- (B, 21, 3) predicted 3D joints
        gt_3d -- (B, 21, 3) target 3D joints
        jts_valid -- (B, 21) float mask for each joint

    Returns:
        loss -- (B, 21, 3) loss for each coordinate of each joint
    """
    return _joints_loss(pred_3d - pred_3d[:, :1], gt_3d - gt_3d[:, :1], jts_valid)


def compute_loss(pred: dict, gt: dict, meta_info: dict) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Computes the WildHands training loss dict.

    Arguments:
        pred -- model output dict (keys: mano.{pose,beta,j3d.cam,j2d.norm,cam_t.wp,cam_t.wp.init}.{r,l},
                grasp.{r,l}, render.{r,l}, center.{r,l}, corner.{r,l})
        gt -- targets dict from the dataset, after the process_data step
        meta_info -- dataset/sample-level flags (is_*_loss) used to gate terms

    Returns:
        loss_dict -- mapping loss_name -> (scalar tensor, weight)
    """
    bz = meta_info["is_j2d_loss"].shape[0]
    right_valid = gt["right_valid"]
    left_valid = gt["left_valid"]
    jv_r = gt["joints_valid_r"]
    jv_l = gt["joints_valid_l"]

    gt_pose_r = axis_angle_to_matrix(gt["mano.pose.r"].reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt["mano.pose.l"].reshape(-1, 3)).reshape(-1, 16, 3, 3)

    loss_pose_r = _vector_loss(pred["mano.pose.r"], gt_pose_r, right_valid)
    loss_pose_l = _vector_loss(pred["mano.pose.l"], gt_pose_l, left_valid)
    loss_beta_r = _vector_loss(pred["mano.beta.r"], gt["mano.beta.r"], right_valid)
    loss_beta_l = _vector_loss(pred["mano.beta.l"], gt["mano.beta.l"], left_valid)

    loss_kp2d_r = _joints_loss(pred["mano.j2d.norm.r"], gt["mano.j2d.norm.r"], jv_r)
    loss_kp2d_l = _joints_loss(pred["mano.j2d.norm.l"], gt["mano.j2d.norm.l"], jv_l)
    loss_kp3d_r = _hand_kp3d_loss(pred["mano.j3d.cam.r"], gt["mano.j3d.cam.r"], jv_r)
    loss_kp3d_l = _hand_kp3d_loss(pred["mano.j3d.cam.l"], gt["mano.j3d.cam.l"], jv_l)

    loss_cam_r = _vector_loss(pred["mano.cam_t.wp.r"], gt["mano.cam_t.wp.r"], right_valid) \
               + _vector_loss(pred["mano.cam_t.wp.init.r"], gt["mano.cam_t.wp.r"], right_valid)
    loss_cam_l = _vector_loss(pred["mano.cam_t.wp.l"], gt["mano.cam_t.wp.l"], left_valid) \
               + _vector_loss(pred["mano.cam_t.wp.init.l"], gt["mano.cam_t.wp.l"], left_valid)

    loss_transl = _vector_loss(
        pred["mano.cam_t.wp.l"] - pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.l"] - gt["mano.cam_t.wp.r"],
        right_valid * left_valid,
    )

    j2d, j3d, beta, pose, cam = (meta_info[f"is_{n}_loss"][..., None] for n in ("j2d", "j3d", "beta", "pose", "cam"))
    loss_dict = {
        "loss/mano/pose/r":  ((loss_pose_r * pose).mean().view(-1), 10.0),
        "loss/mano/pose/l":  ((loss_pose_l * pose).mean().view(-1), 10.0),
        "loss/mano/beta/r":  ((loss_beta_r * beta).mean().view(-1), 0.001),
        "loss/mano/beta/l":  ((loss_beta_l * beta).mean().view(-1), 0.001),
        "loss/mano/kp2d/r":  ((loss_kp2d_r.reshape(bz, -1) * j2d).mean().view(-1), 5.0),
        "loss/mano/kp2d/l":  ((loss_kp2d_l.reshape(bz, -1) * j2d).mean().view(-1), 5.0),
        "loss/mano/kp3d/r":  ((loss_kp3d_r.reshape(bz, -1) * j3d).mean().view(-1), 5.0),
        "loss/mano/kp3d/l":  ((loss_kp3d_l.reshape(bz, -1) * j3d).mean().view(-1), 5.0),
        "loss/mano/cam_t/r": ((loss_cam_r * cam).mean().view(-1), 1.0),
        "loss/mano/cam_t/l": ((loss_cam_l * cam).mean().view(-1), 1.0),
        "loss/mano/transl/l": ((loss_transl * cam).mean().view(-1), 1.0),
    }

    is_grasp = meta_info["is_grasp_loss"][..., None]
    loss_grasp_r = F.cross_entropy(pred["grasp.r"], gt["grasp.r"], reduction="none") * gt["grasp_valid_r"]
    loss_grasp_l = F.cross_entropy(pred["grasp.l"], gt["grasp.l"], reduction="none") * gt["grasp_valid_l"]
    loss_dict["loss/grasp/r"] = ((loss_grasp_r.view(bz, -1) * is_grasp).mean().view(-1), 0.1)
    loss_dict["loss/grasp/l"] = ((loss_grasp_l.view(bz, -1) * is_grasp).mean().view(-1), 0.1)

    is_mask = meta_info["is_mask_loss"][..., None]
    loss_mask_r = F.l1_loss(pred["render.r"], gt["render.r"], reduction="none").view(bz, -1) * gt["render_valid_r"][..., None]
    loss_mask_l = F.l1_loss(pred["render.l"], gt["render.l"], reduction="none").view(bz, -1) * gt["render_valid_l"][..., None]
    loss_dict["loss/mask/r"] = ((loss_mask_r * is_mask).mean().view(-1), 10.0)
    loss_dict["loss/mask/l"] = ((loss_mask_l * is_mask).mean().view(-1), 10.0)

    return loss_dict
