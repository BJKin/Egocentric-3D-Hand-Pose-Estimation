"""
Eval metrics for each sample: root-aligned MPJPE, procrustes-aligned root-aligned
MPJPE, mean relative root position error (right vs left), and pixel error.
"""
import numpy as np
import torch


def _to_numpy(t):
    """
    Detach and convert a torch tensor to numpy on CPU.

    Arguments:
        t -- torch.Tensor or numpy.ndarray

    Returns:
        arr -- numpy.ndarray
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _per_sample_l2(gt: np.ndarray, pred: np.ndarray, per_joint_valid: np.ndarray) -> np.ndarray:
    """
    Mean joint L2 distance for each sample, averaged over the joints where
    per_joint_valid is non-zero. Samples with no valid joints return NaN.

    Arguments:
        gt -- (B, N, 3) ground-truth 3D joints
        pred -- (B, N, 3) predicted 3D joints
        per_joint_valid -- (B, N) validity for each joint in {0, 1}

    Returns:
        err -- (B,) mean joint error for each sample
    """
    dist = np.linalg.norm(gt - pred, axis=-1)  
    masked = dist * per_joint_valid
    counts = per_joint_valid.sum(axis=-1)
    out = np.where(counts > 0, masked.sum(axis=-1) / np.maximum(counts, 1e-9), np.nan)
    return out


def mpjpe_ra(pred_j3d: torch.Tensor, gt_j3d: torch.Tensor, per_joint_valid: torch.Tensor) -> np.ndarray:
    """
    Root-aligned MPJPE in millimeters. Both pred and GT are translated so the
    root joint (index 0) sits at the origin before the L2 distance is taken.

    Arguments:
        pred_j3d -- (B, 21, 3) predicted 3D joints in meters
        gt_j3d -- (B, 21, 3) GT 3D joints in meters
        per_joint_valid -- (B, 21) validity mask for each joint

    Returns:
        err -- (B,) MPJPE in mm for each sample (NaN where no joints are valid)
    """
    pred = _to_numpy(pred_j3d)
    gt = _to_numpy(gt_j3d)
    valid = _to_numpy(per_joint_valid)
    pred_ra = pred - pred[:, :1, :]
    gt_ra = gt - gt[:, :1, :]
    return _per_sample_l2(gt_ra, pred_ra, valid) * 1000.0


def _similarity_transform(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes: returns a sR + t alignment of S1 onto S2.

    Arguments:
        S1 -- (N, 3) source points
        S2 -- (N, 3) target points

    Returns:
        S1_hat -- (N, 3) aligned source (or all-NaN if SVD failed)
    """
    s1, s2 = S1.T, S2.T
    mu1 = s1.mean(axis=1, keepdims=True)
    mu2 = s2.mean(axis=1, keepdims=True)
    X1 = s1 - mu1
    X2 = s2 - mu2
    var1 = np.sum(X1 ** 2)
    K = X1 @ X2.T
    try:
        U, _, Vh = np.linalg.svd(K)
        V = Vh.T
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
        R = V @ (Z @ U.T)
        scale = np.trace(R @ K) / var1
        t = mu2 - scale * (R @ mu1)
        return (scale * R @ s1 + t).T
    except np.linalg.LinAlgError:
        return np.full_like(S1, np.nan)


def mpjpe_pa_ra(pred_j3d: torch.Tensor, gt_j3d: torch.Tensor, per_joint_valid: torch.Tensor) -> np.ndarray:
    """
    Procrustes-aligned (after root-alignment) MPJPE in millimeters. For each
    sample with at least one valid joint, fits an sR + t alignment of the
    root-aligned prediction onto the root-aligned GT (using only valid joints),
    then averages the distance of each valid joint.

    Arguments:
        pred_j3d -- (B, 21, 3) predicted 3D joints in meters
        gt_j3d -- (B, 21, 3) GT 3D joints in meters
        per_joint_valid -- (B, 21) validity mask for each joint

    Returns:
        err -- (B,) PA-MPJPE in mm for each sample (NaN if no joints valid or SVD failed)
    """
    pred = _to_numpy(pred_j3d)
    gt = _to_numpy(gt_j3d)
    valid = _to_numpy(per_joint_valid).astype(bool)
    pred_ra = pred - pred[:, :1, :]
    gt_ra = gt - gt[:, :1, :]
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = valid[i]
        if m.sum() == 0:
            continue
        aligned = _similarity_transform(pred_ra[i, m], gt_ra[i, m])
        out[i] = np.linalg.norm(gt_ra[i, m] - aligned, axis=-1).mean()
    return out * 1000.0


def mrrpe_rl(pred_root_r: torch.Tensor, pred_root_l: torch.Tensor, gt_root_r: torch.Tensor, gt_root_l: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    """
    Mean Relative Root Position Error between the two hand roots, in millimeters.

    Arguments:
        pred_root_r -- (B, 3) predicted right hand root in meters
        pred_root_l -- (B, 3) predicted left hand root in meters
        gt_root_r -- (B, 3) GT right hand root in meters
        gt_root_l -- (B, 3) GT left hand root in meters
        valid -- (B,) sample level validity

    Returns:
        err -- (B,) displacement vector error in mm for each sample (NaN where invalid)
    """
    pr = _to_numpy(pred_root_r) - _to_numpy(pred_root_l)
    gr = _to_numpy(gt_root_r) - _to_numpy(gt_root_l)
    v = _to_numpy(valid).astype(bool)
    dist = np.linalg.norm(pr - gr, axis=-1)
    return np.where(v, dist, np.nan) * 1000.0


def pix_err(pred_j2d_pix: torch.Tensor, gt_j2d_pix: torch.Tensor, per_joint_valid: torch.Tensor) -> np.ndarray:
    """
    Mean 2D pixel error, averaged over the valid joints.

    Arguments:
        pred_j2d_pix -- (B, 21, 2) predicted 2D joints in patch pixels
        gt_j2d_pix -- (B, 21, 2) GT 2D joints in patch pixels
        per_joint_valid -- (B, 21) validity mask for each joint

    Returns:
        err -- (B,) mean pixel error for each sample (NaN if no joints valid)
    """
    pred = _to_numpy(pred_j2d_pix)
    gt = _to_numpy(gt_j2d_pix)
    valid = _to_numpy(per_joint_valid)
    dist = np.linalg.norm(pred - gt, axis=-1) 
    masked = dist * valid
    counts = valid.sum(axis=-1)
    return np.where(counts > 0, masked.sum(axis=-1) / np.maximum(counts, 1e-9), np.nan)
