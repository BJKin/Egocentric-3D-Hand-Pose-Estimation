"""
Camera intrinsics and projection helpers.
"""
import torch


def perspective_to_weak_perspective_torch(perspective_camera: torch.Tensor, focal_length: torch.Tensor, img_res: int) -> torch.Tensor:
    """
    Converts a perspective camera translation [tx, ty, tz] to
    [s, tx, ty] for the given focal length and patch resolution.

    Arguments:
        perspective_camera -- (B, 3) translation vectors
        focal_length -- (B,) focal length in pixels for each sample
        img_res -- patch resolution used to compute the scale

    Returns:
        weak -- (B, 3) weak perspective camera [s, tx, ty]
    """
    tx = perspective_camera[:, 0]
    ty = perspective_camera[:, 1]
    tz = perspective_camera[:, 2]
    return torch.stack([2 * focal_length / (img_res * tz + 1e-9), tx, ty], dim=-1)


def weak_perspective_to_perspective_torch(weak_perspective_camera: torch.Tensor, focal_length: torch.Tensor, img_res: int, min_s: float) -> torch.Tensor:
    """
    Converts a weak perspective camera [s, tx, ty] back to a 3D translation
    [tx, ty, tz] using the given focal length and patch resolution.

    Arguments:
        weak_perspective_camera -- (B, 3) weak perspective camera
        focal_length -- (B,) focal length in pixels for each sample
        img_res -- patch resolution
        min_s -- floor for the scale to avoid division by zero

    Returns:
        perspective -- (B, 3) 3D translation
    """
    s = torch.clamp(weak_perspective_camera[:, 0], min_s)
    tx = weak_perspective_camera[:, 1]
    ty = weak_perspective_camera[:, 2]
    return torch.stack([tx, ty, 2 * focal_length / (img_res * s + 1e-9)], dim=-1)


def project2d_batch(K: torch.Tensor, pts_cam: torch.Tensor) -> torch.Tensor:
    """
    Projects batched 3D camera coordinate points into pixel space using
    pinhole intrinsics.

    Arguments:
        K -- (B, 3, 3) intrinsics
        pts_cam -- (B, N, 3) points in camera coordinates

    Returns:
        pts2d -- (B, N, 2) projected pixel coordinates
    """
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[-1] == 3
    pts_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    return pts_homo[:, :, :2] / pts_homo[:, :, 2:3]
