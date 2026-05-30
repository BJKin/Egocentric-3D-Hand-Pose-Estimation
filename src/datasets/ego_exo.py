"""
Ego-Exo4D zero-shot eval dataset (right hand only).
"""
from typing import Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import pickle
from src.common import data_utils
from src.common.data_utils import IMG_NORM_MEAN, IMG_NORM_STD
from src.datasets._common import (
    FLIP_PROB, IMG_RES, MEAN_BETA_L, MEAN_BETA_R, NOISE_FACTOR, ROT_FACTOR,
    SCALE_FACTOR, pad_jts2d, pos_enc_angles,
)
from pathlib import Path


# ----------------------------- config -----------------------------
EGO4D_HANDS_DIR = Path("../data_reduced/ego4d_hands")
BBOX_SCALE = 1.5

# Ego-Exo4D's joint ordering per hand-> MANO 21-joint index.
EGOEXO_JOINT_NAMES = [
    "wrist",
    "index_1", "index_2", "index_3",
    "middle_1", "middle_2", "middle_3",
    "pinky_1", "pinky_2", "pinky_3",
    "ring_1", "ring_2", "ring_3",
    "thumb_1", "thumb_2", "thumb_3", "thumb_4",
    "index_4", "middle_4", "ring_4", "pinky_4",
]


class EgoExo4DDataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the Ego-Exo4D val annotations.

        Arguments:
            split -- 'val'
        """
        assert split == "val", f"Ego-Exo4D split must be 'val', got {split!r}"
        with open(EGO4D_HANDS_DIR / "joint_annotations_egoexo_val.pkl", "rb") as f:
            self.data = pickle.load(f)
        self.imgnames = list(self.data.keys())
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.aug_data = False
        logger.info(f"# samples in Ego-Exo4D {split}: {len(self.imgnames)}")

    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        key = self.imgnames[idx]
        data = self.data[key]
        cv_img = data["img"].astype(np.float32)  
        img_h, img_w = data["crop_size"]

        j2d_r, j2d_l, jv2d_r, jv2d_l = self._collect_joints(data["j2d"], dim=2)
        j3d_r, j3d_l, jv3d_r, jv3d_l = self._collect_joints(data["j3d"], dim=3)

        bbox = [img_w / 2, img_h / 2, max(img_w, img_h) / 200]
        center, scale = [bbox[0], bbox[1]], bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0

        joints2d_r = data_utils.j2d_processing(pad_jts2d(j2d_r), center, scale, augm, IMG_RES)
        joints2d_l = data_utils.j2d_processing(pad_jts2d(j2d_l), center, scale, augm, IMG_RES)
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)

        right_bbox, _ = self._joint_bbox(joints2d_r, jv2d_r)
        left_bbox, _ = self._joint_bbox(joints2d_l, jv2d_l)

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

        intrx = data["crop_intrx"].copy().astype(np.float32)
        ratio = IMG_RES / max(img_w, img_h)
        intrx[0, 0] *= ratio
        intrx[1, 1] *= ratio
        intrx[0, 2] *= ratio
        intrx[1, 2] *= ratio

        r_center_angle, r_corner_angle = pos_enc_angles(r_bbox, intrx)
        l_center_angle, l_corner_angle = pos_enc_angles(l_bbox, intrx)

        right_valid = float(jv2d_r.sum() > 3)
        left_valid = float(jv2d_l.sum() > 3) 

        inputs = {
            "img": norm_img, "r_img": norm_r_img, "l_img": norm_l_img,
            "r_center_angle": r_center_angle, "l_center_angle": l_center_angle,
            "r_corner_angle": r_corner_angle, "l_corner_angle": l_corner_angle,
        }
        targets = {
            "mano.pose.r": torch.zeros(48), "mano.pose.l": torch.zeros(48),
            "mano.beta.r": torch.tensor(MEAN_BETA_R), "mano.beta.l": torch.tensor(MEAN_BETA_L),
            "mano.j2d.norm.r": torch.from_numpy(joints2d_r[:, :2]).float(),
            "mano.j2d.norm.l": torch.from_numpy(joints2d_l[:, :2]).float(),
            "mano.j3d.full.r": torch.from_numpy(j3d_r).float(),
            "mano.j3d.full.l": torch.from_numpy(j3d_l).float(),
            "grasp.r": 8, "grasp.l": 8,
            "grasp_valid_r": 0, "grasp_valid_l": 0,
            "render.r": torch.zeros((1, IMG_RES, IMG_RES)),
            "render.l": torch.zeros((1, IMG_RES, IMG_RES)),
            "render_valid_r": 0, "render_valid_l": 0,
            "is_valid": 1.0,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "joints_valid_r": jv2d_r * right_valid,
            "joints_valid_l": jv2d_l * left_valid,
            "joints3d_valid_r": jv3d_r * right_valid,
            "joints3d_valid_l": jv3d_l * left_valid,
        }
        meta_info = {
            "imgname": key,
            "intrinsics": torch.from_numpy(intrx).float(),
            "dataset": "egoexo",
            "is_j2d_loss": 1, "is_j3d_loss": 1,
            "is_beta_loss": 0, "is_pose_loss": 0, "is_cam_loss": 0,
            "is_grasp_loss": 0, "is_mask_loss": 0,
        }
        return inputs, targets, meta_info

    @staticmethod
    def _collect_joints(joint_dict: dict, dim: int):
        """
        Extracts joint arrays for each hand from Ego-Exo4D's name-keyed dicts.

        Arguments:
            joint_dict -- mapping from 'right_<name>' / 'left_<name>' to {'x','y'[,'z']}
            dim -- 2 for 2D joints, 3 for 3D

        Returns:
            j_r -- (21, dim) right-hand joints in MANO order (zeros where missing)
            j_l -- (21, dim) left-hand joints
            v_r -- (21,) right-hand validity (1 if present, 0 if missing)
            v_l -- (21,) left-hand validity
        """
        j_r = np.zeros((21, dim), dtype=np.float32)
        j_l = np.zeros((21, dim), dtype=np.float32)
        v_r = np.zeros(21, dtype=np.float32)
        v_l = np.zeros(21, dtype=np.float32)
        for i, name in enumerate(EGOEXO_JOINT_NAMES):
            for side, j, v in (("right", j_r, v_r), ("left", j_l, v_l)):
                key = f"{side}_{name}"
                if key in joint_dict:
                    pt = joint_dict[key]
                    j[i] = [pt["x"], pt["y"]] if dim == 2 else [pt["x"], pt["y"], pt["z"]]
                    v[i] = 1
        return j_r, j_l, v_r, v_l

    @staticmethod
    def _joint_bbox(joints2d_norm: np.ndarray, joints_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derives a [x0, y0, w, h] bbox from valid 2D joints inside the IMG_RES patch.

        Arguments:
            joints2d_norm -- (N, 3) joints with xy in [-1, 1]
            joints_valid -- (N,) validity mask for each joint

        Returns:
            bbox -- [x0, y0, w, h] int16
            bbox_og -- [x0, y0, x1, y1]
        """
        pix = ((joints2d_norm[..., :2] + 1) / 2) * (IMG_RES - 1)
        pix = pix[joints_valid > 0]
        if pix.shape[0] == 0:
            return None, np.array([0, 0, IMG_RES - 1, IMG_RES - 1], dtype=np.int16)
        x0, y0 = pix[:, 0].min(), pix[:, 1].min()
        x1, y1 = pix[:, 0].max(), pix[:, 1].max()
        x0, y0, x1, y1 = (np.clip([x0, y0, x1, y1], 0, IMG_RES - 1)).astype(np.int16)
        w, h = x1 - x0, y1 - y0
        if w == 0 or h == 0:
            return None, np.array([0, 0, IMG_RES - 1, IMG_RES - 1], dtype=np.int16)
        bb = np.array([x0, y0, w, h], dtype=np.int16)
        return bb, bb.copy()
