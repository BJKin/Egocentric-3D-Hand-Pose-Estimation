"""
H2O zero-shot eval dataset (subject3 egocentric).
"""
from typing import Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from src.common import data_utils
from src.common.camera import project2d_batch
from src.common.data_utils import IMG_NORM_MEAN, IMG_NORM_STD
from src.datasets._common import (
    FLIP_PROB, FOCAL_LENGTH, IMG_RES, NOISE_FACTOR, ROT_FACTOR, SCALE_FACTOR,
    pad_jts2d, pos_enc_angles,
)
from pathlib import Path


# ----------------------------- config -----------------------------
H2O_DIR = Path("../data_reduced/h2o")
BBOX_SCALE = 1.5

# H2O native joint order -> MANO 21-joint order
H2O_TO_MANO = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20])


def _read_intrinsics(seq_dir) -> np.ndarray:
    """
    Reads a 'fx fy cx cy w h' row from cam_intrinsics.txt and returns a 3x3 K matrix.

    Arguments:
        seq_dir -- Path to a sequence directory containing cam_intrinsics.txt

    Returns:
        K -- (3, 3) float32 camera intrinsics
    """
    vals = np.loadtxt(seq_dir / "cam_intrinsics.txt")
    fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


class H2ODataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the H2O subject3 egocentric val/test split list.

        Arguments:
            split -- 'val' or 'test'
        """
        assert split in ("val", "test"), f"H2O split must be 'val' or 'test', got {split!r}"
        list_p = H2O_DIR / "label_split" / f"pose_{split}.txt"
        with open(list_p) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.samples = []
        for ln in lines:
            parts = ln.split("/")
            seq_rel = "/".join(["subject3_ego"] + parts[1:-2])  # subject3_ego/k2/0/cam4
            frame_idx = parts[-1].split(".")[0]
            self.samples.append((seq_rel, frame_idx))

        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.aug_data = False
        logger.info(f"# samples in H2O {split}: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        seq_rel, frame_idx = self.samples[idx]
        seq_dir = H2O_DIR / seq_rel
        img_path = str(seq_dir / "rgb" / f"{frame_idx}.png")
        cv_img, _ = data_utils.read_img(img_path, (720, 1280, 3))

        hand_info = np.loadtxt(seq_dir / "hand_pose" / f"{frame_idx}.txt")
        left_valid = float(hand_info[0])
        joints3d_l = hand_info[1:64].reshape(21, 3)[H2O_TO_MANO].astype(np.float32)
        right_valid = float(hand_info[64])
        joints3d_r = hand_info[65:128].reshape(21, 3)[H2O_TO_MANO].astype(np.float32)

        mano_info = np.loadtxt(seq_dir / "hand_pose_mano" / f"{frame_idx}.txt")
        left_pose = mano_info[4:52].astype(np.float32)
        left_beta = mano_info[52:62].astype(np.float32)
        right_pose = mano_info[66:114].astype(np.float32)
        right_beta = mano_info[114:124].astype(np.float32)

        intrx = _read_intrinsics(seq_dir)

        K_b = torch.from_numpy(intrx).unsqueeze(0)
        joints2d_r = project2d_batch(K_b, torch.from_numpy(joints3d_r).unsqueeze(0)).squeeze(0).numpy()
        joints2d_l = project2d_batch(K_b, torch.from_numpy(joints3d_l).unsqueeze(0)).squeeze(0).numpy()
        joints2d_r = pad_jts2d(joints2d_r)
        joints2d_l = pad_jts2d(joints2d_l)

        img_h, img_w = cv_img.shape[:2]
        bbox = [img_w / 2, img_h / 2, max(img_w, img_h) / 200]
        center, scale = [bbox[0], bbox[1]], bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0
        joints2d_r = data_utils.j2d_processing(joints2d_r, center, scale, augm, IMG_RES)
        joints2d_l = data_utils.j2d_processing(joints2d_l, center, scale, augm, IMG_RES)
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)

        right_bbox, _ = self._joint_bbox(joints2d_r)
        left_bbox, _ = self._joint_bbox(joints2d_l)

        r_img, r_bbox = data_utils.crop_and_pad(img, right_bbox, IMG_RES, IMG_RES, scale=BBOX_SCALE)
        l_img, l_bbox = data_utils.crop_and_pad(img, left_bbox, IMG_RES, IMG_RES, scale=BBOX_SCALE)
        norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
        norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

        img_ds = data_utils.generate_patch_image_clean(
            img.transpose(1, 2, 0),
            [IMG_RES / 2, IMG_RES / 2, IMG_RES, IMG_RES],
            1.0, 0.0, [IMG_RES, IMG_RES], cv2.INTER_CUBIC,
        )[0].transpose(2, 0, 1)
        norm_img = self.normalize_img(torch.from_numpy(np.clip(img_ds, 0, 1)).float())

        intrx_aug = data_utils.get_aug_intrix(
            intrx.copy(), FOCAL_LENGTH, IMG_RES, use_gt_k=True,
            bbox_cx=img_w / 2.0, bbox_cy=img_h / 2.0,
            scale=augm["sc"] * max(img_w, img_h) / 200.0,
        )
        intrx_aug_np = intrx_aug.numpy() if isinstance(intrx_aug, torch.Tensor) else np.asarray(intrx_aug)

        r_center_angle, r_corner_angle = pos_enc_angles(r_bbox, intrx_aug_np)
        l_center_angle, l_corner_angle = pos_enc_angles(l_bbox, intrx_aug_np)

        inputs = {
            "img": norm_img, "r_img": norm_r_img, "l_img": norm_l_img,
            "r_center_angle": r_center_angle, "l_center_angle": l_center_angle,
            "r_corner_angle": r_corner_angle, "l_corner_angle": l_corner_angle,
        }
        targets = {
            "mano.pose.r": torch.from_numpy(right_pose).float(),
            "mano.pose.l": torch.from_numpy(left_pose).float(),
            "mano.beta.r": torch.from_numpy(right_beta).float(),
            "mano.beta.l": torch.from_numpy(left_beta).float(),
            "mano.j2d.norm.r": torch.from_numpy(joints2d_r[:, :2]).float(),
            "mano.j2d.norm.l": torch.from_numpy(joints2d_l[:, :2]).float(),
            "mano.j3d.full.r": torch.from_numpy(joints3d_r).float(),
            "mano.j3d.full.l": torch.from_numpy(joints3d_l).float(),
            "grasp.r": 8, "grasp.l": 8,
            "grasp_valid_r": 0, "grasp_valid_l": 0,
            "render.r": torch.zeros((1, IMG_RES, IMG_RES)),
            "render.l": torch.zeros((1, IMG_RES, IMG_RES)),
            "render_valid_r": 0, "render_valid_l": 0,
            "is_valid": 1.0,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "joints_valid_r": np.ones(21, dtype=np.float32) * right_valid,
            "joints_valid_l": np.ones(21, dtype=np.float32) * left_valid,
        }
        meta_info = {
            "imgname": img_path,
            "intrinsics": torch.from_numpy(intrx_aug_np).float(),
            "dataset": "h2o",
            "is_j2d_loss": 1, "is_j3d_loss": 1,
            "is_beta_loss": 0, "is_pose_loss": 0, "is_cam_loss": 0,
            "is_grasp_loss": 0, "is_mask_loss": 0,
        }
        return inputs, targets, meta_info

    @staticmethod
    def _joint_bbox(joints2d_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derives a [x0, y0, w, h] bbox from normalized 2D joints inside the IMG_RES patch.

        Arguments:
            joints2d_norm -- (N, 3) joints with xy in [-1, 1]

        Returns:
            bbox -- [x0, y0, w, h] int16
            bbox_og -- [x0, y0, x1, y1]
        """
        pix = ((joints2d_norm[..., :2] + 1) / 2) * (IMG_RES - 1)
        x0, y0 = pix[:, 0].min(), pix[:, 1].min()
        x1, y1 = pix[:, 0].max(), pix[:, 1].max()
        x0, y0, x1, y1 = (np.clip([x0, y0, x1, y1], 0, IMG_RES - 1)).astype(np.int16)
        w, h = x1 - x0, y1 - y0
        if w == 0 or h == 0:
            return None, np.array([0, 0, IMG_RES - 1, IMG_RES - 1], dtype=np.int16)
        bb = np.array([x0, y0, w, h], dtype=np.int16)
        return bb, bb.copy()
