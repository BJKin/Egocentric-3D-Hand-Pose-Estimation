"""
Shared constants and small helpers used by every training dataset.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
from src.common import data_utils


# ----------------------------- config -----------------------------
EGO4D_FRAMES_DIR = Path("../data_reduced/ego4d/ego4d_frames")
VISOR_FRAMES_DIR = Path("../data_reduced/visor/rgb_frames")

IMG_RES = 224
FOCAL_LENGTH = 1000.0
ROT_FACTOR = 30.0
NOISE_FACTOR = 0.4
SCALE_FACTOR = 0.25
FLIP_PROB = 0.0

# Mean MANO betas (right, left) computed from the WildHands val set.
MEAN_BETA_R = [0.82747316, 0.13775729, -0.39435294, 0.17889787, -0.73901576, 0.7788163, -0.5702684, 0.4947751, -0.24890041, 1.5943261]
MEAN_BETA_L = [-0.19330633, -0.08867972, -2.5790455, -0.10344583, -0.71684015, -0.28285977, 0.55171007, -0.8403888, -0.8490544, -1.3397144]

def pos_enc_angles(bbox: np.ndarray, intrx) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the (center, corner) angular positional encodings.

    Arguments:
        bbox -- (4,) [x0, y0, x1, y1] hand bbox in patch pixels
        intrx -- (3, 3) tensor or array of camera intrinsics

    Returns:
        center_angle -- (2,) float32 [yaw, pitch] of the bbox center
        corner_angle -- (8,) float32 [yaw, pitch] for each of the four bbox corners
    """
    K = np.asarray(intrx) if not isinstance(intrx, np.ndarray) else intrx
    if hasattr(intrx, "numpy"):
        K = intrx.numpy()
    cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]

    center = (bbox[:2] + bbox[2:]) / 2.0
    center_angle = np.array([
        np.arctan2(center[0] - cx, fx),
        np.arctan2(center[1] - cy, fy),
    ], dtype=np.float32)

    corners = np.array([
        [bbox[0], bbox[1]], [bbox[0], bbox[3]],
        [bbox[2], bbox[1]], [bbox[2], bbox[3]],
    ])
    corners = np.stack([corners[:, 0] - cx, corners[:, 1] - cy], axis=-1)
    corner_angle = np.arctan2(corners, np.array([[fx, fy]])).flatten().astype(np.float32)
    return center_angle, corner_angle


def visor_path(pickle_key: str):
    """
    Remaps a VISOR pickle key to the local file under data/visor/rgb_frames.

    Arguments:
        pickle_key -- the absolute path stored as the pickle key

    Returns:
        path -- local pathlib.Path to the JPG
    """
    parts = pickle_key.replace("\\", "/").split("/")
    split, participant, _seq, filename = parts[-4:]
    return VISOR_FRAMES_DIR / split / participant / filename


def ego_path(pickle_key: str):
    """
    Remaps an Ego4D pickle key to the local file under data/ego4d/ego4d_frames.

    Arguments:
        pickle_key -- the absolute path stored as the pickle key

    Returns:
        path -- local pathlib.Path to the PNG
    """
    return EGO4D_FRAMES_DIR / Path(pickle_key).name


def pad_jts2d(jts: np.ndarray) -> np.ndarray:
    """
    Appends a constant '1' confidence column to a 2D keypoint array.

    Arguments:
        jts -- (N, 2) array of pixel-space 2D keypoints

    Returns:
        jts_pad -- (N, 3) array with the third column set to 1
    """
    out = np.ones((jts.shape[0], 3))
    out[:, :2] = jts
    return out


def transform_bbox_to_patch(bbox_xyxy, center, scale, augm, img_res):
    """
    Maps a [x0, y0, x1, y1] bbox from source pixels to a [x0, y0, w, h] bbox in
    the augmented img_res patch by passing the corners through j2d_processing.

    Arguments:
        bbox_xyxy -- (4,) [x0, y0, x1, y1] in source pixels
        center -- [cx, cy] bbox center for j2d_processing
        scale -- bbox dim in scale units for j2d_processing
        augm -- augmentation dict (with keys sc, rot)
        img_res -- output patch resolution

    Returns:
        bbox -- [x0, y0, w, h] in patch pixels (float)
    """
    end_pts = np.array([[bbox_xyxy[0], bbox_xyxy[1]], [bbox_xyxy[2], bbox_xyxy[3]]])
    end_pts = data_utils.j2d_processing(pad_jts2d(end_pts), center, scale, augm, img_res)
    end_pts = ((end_pts[..., :2] + 1) / 2) * img_res
    end_pts = end_pts.flatten()
    return [end_pts[0], end_pts[1], end_pts[2] - end_pts[0], end_pts[3] - end_pts[1]]


def dummy_egocam_intrx(width: int, height: int) -> np.ndarray:
    """
    Builds the dummy intrinsics used by every fixed-cam egocentric
    dataset (VISOR, Ego4D). No real K is available so we use a fixed focal
    length scaled to the IMG_RES patch by the image-resize ratio.

    Arguments:
        width -- source image width in pixels
        height -- source image height in pixels

    Returns:
        intrx -- (3, 3) float32 intrinsics
    """
    fixed_focal = FOCAL_LENGTH * (IMG_RES / max(width, height))
    scale = max(width, height) / 200.0
    return data_utils.get_aug_intrix(
        np.zeros((3, 3), dtype=np.float32),
        fixed_focal,
        IMG_RES,
        use_gt_k=False,
        bbox_cx=width / 2.0, bbox_cy=height / 2.0,
        scale=scale,
    ).numpy().astype(np.float32)

