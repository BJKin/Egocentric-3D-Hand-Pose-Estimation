"""
EPIC-HandKps zero-shot eval dataset (2D-only).
"""
import pickle
from typing import Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from src.common import data_utils
from src.common.data_utils import IMG_NORM_MEAN, IMG_NORM_STD
from src.datasets._common import (
    FLIP_PROB, IMG_RES, MEAN_BETA_L, MEAN_BETA_R, NOISE_FACTOR, ROT_FACTOR,
    SCALE_FACTOR, dummy_egocam_intrx, pad_jts2d, pos_enc_angles, visor_path,
)
from src.datasets.assembly import ASSEMBLY_TO_MANO_R 
from pathlib import Path


# ----------------------------- config -----------------------------
EPIC_HANDS_DIR = Path("../data_reduced/epic_hands")
BBOX_SCALE = 1.5
VISOR_W, VISOR_H = 1920, 1080
DUMMY_IMG_SHAPE = (VISOR_H, VISOR_W, 3)


def _dummy_side() -> dict:
    """
    Returns the placeholder entry for one side, used when a sample annotates only
    one of the two hands.

    Returns:
        side -- dict with bbox=None, joints zeros (21,2), joints_valid zeros (21,)
    """
    return {"bbox": None, "joints": np.zeros((21, 2)), "joints_valid": [0] * 21}


class EPICHandKpsDataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the EPIC-HandKps annotations.

        Arguments:
            split -- 'test' (the 5k eval set in hands_5000.pkl) or 'val' (250 sample preview)
        """
        assert split in ("test", "val"), f"EPIC-HandKps split must be 'test' or 'val', got {split!r}"
        filename = "hands_5000.pkl" if split == "test" else "hands_250.pkl"
        with open(EPIC_HANDS_DIR / filename, "rb") as f:
            self.data = pickle.load(f)
        self.imgnames = list(self.data.keys())
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.aug_data = False
        self.intrx = dummy_egocam_intrx(VISOR_W, VISOR_H)
        logger.info(f"# samples in EPIC-HandKps {split}: {len(self.imgnames)}")

    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        pickle_key = self.imgnames[idx]
        data = self.data[pickle_key]
        data_r = data.get("right", _dummy_side())
        data_l = data.get("left", _dummy_side())

        joints2d_r = pad_jts2d(np.asarray(data_r["joints"])[ASSEMBLY_TO_MANO_R])
        joints2d_l = pad_jts2d(np.asarray(data_l["joints"])[ASSEMBLY_TO_MANO_R])
        joints_valid_r = np.asarray(data_r["joints_valid"])[ASSEMBLY_TO_MANO_R].astype(np.float32)
        joints_valid_l = np.asarray(data_l["joints_valid"])[ASSEMBLY_TO_MANO_R].astype(np.float32)

        imgname = str(visor_path(pickle_key))
        cv_img, _ = data_utils.read_img(imgname, DUMMY_IMG_SHAPE)

        bbox = [VISOR_W / 2, VISOR_H / 2, max(VISOR_W, VISOR_H) / 200]
        center, scale = [bbox[0], bbox[1]], bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0

        joints2d_r = data_utils.j2d_processing(joints2d_r, center, scale, augm, IMG_RES)
        joints2d_l = data_utils.j2d_processing(joints2d_l, center, scale, augm, IMG_RES)
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)

        right_bbox, _ = self._joint_bbox(joints2d_r, joints_valid_r)
        left_bbox, _ = self._joint_bbox(joints2d_l, joints_valid_l)

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

        r_center_angle, r_corner_angle = pos_enc_angles(r_bbox, self.intrx)
        l_center_angle, l_corner_angle = pos_enc_angles(l_bbox, self.intrx)

        right_valid = float(joints_valid_r.sum() > 3)
        left_valid = float(joints_valid_l.sum() > 3)

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
            "mano.j3d.full.r": torch.zeros(21, 3),
            "mano.j3d.full.l": torch.zeros(21, 3),
            "grasp.r": 8, "grasp.l": 8,
            "grasp_valid_r": 0, "grasp_valid_l": 0,
            "render.r": torch.zeros((1, IMG_RES, IMG_RES)),
            "render.l": torch.zeros((1, IMG_RES, IMG_RES)),
            "render_valid_r": 0, "render_valid_l": 0,
            "is_valid": 1.0,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "joints_valid_r": joints_valid_r * right_valid,
            "joints_valid_l": joints_valid_l * left_valid,
        }
        meta_info = {
            "imgname": imgname,
            "intrinsics": torch.from_numpy(self.intrx).float(),
            "dataset": "epic_handkps",
            "is_j2d_loss": 1, "is_j3d_loss": 0,
            "is_beta_loss": 0, "is_pose_loss": 0, "is_cam_loss": 0,
            "is_grasp_loss": 0, "is_mask_loss": 0,
        }
        return inputs, targets, meta_info

    @staticmethod
    def _joint_bbox(joints2d_norm: np.ndarray, joints_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derives a [x0, y0, w, h] bbox from the visible 2D joints inside the
        IMG_RES patch. Uses only joints with validity > 0.

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
