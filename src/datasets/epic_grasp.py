"""
EPIC-Kitchens VISOR grasp dataset.
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
    SCALE_FACTOR, dummy_egocam_intrx, pos_enc_angles, transform_bbox_to_patch,
    visor_path,
)
from pathlib import Path


# ----------------------------- config -----------------------------
EPIC_HANDS_DIR = Path("../data_reduced/epic_hands")
BBOX_SCALE = 1.5
VISOR_W, VISOR_H = 1920, 1080
DUMMY_IMG_SHAPE = (VISOR_H, VISOR_W, 3)

_GRASP_LABEL = {
    "NP-Palm": 0, "NP-Fin": 1, "Pow-Pris": 2, "Pre-Pris": 3,
    "Pow-Circ": 4, "Pre-Circ": 5, "Later": 6, "Other": 7,
}
NO_GRASP = 8


class EPICGraspDataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the VISOR train grasp annotations.

        Arguments:
            split -- 'train'
        """
        assert split == "train", f"EPICGraspDataset only supports the 'train' split, got {split!r}"
        with open(EPIC_HANDS_DIR / "grasp_visor_train.pkl", "rb") as f:
            self.data = pickle.load(f)
        self.imgnames = list(self.data.keys())
        self.aug_data = True
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.intrx = dummy_egocam_intrx(VISOR_W, VISOR_H)
        logger.info(f"# samples in EPIC grasp train: {len(self.imgnames)}")

    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        pickle_key = self.imgnames[idx]
        data = self.data[pickle_key]
        imgname = str(visor_path(pickle_key))
        cv_img, _ = data_utils.read_img(imgname, DUMMY_IMG_SHAPE)

        bbox = [VISOR_W / 2, VISOR_H / 2, max(VISOR_W, VISOR_H) / 200]
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0  
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)

        right_bbox = np.asarray(data["right_bbox"], dtype=np.float32) if data["right_bbox"] is not None else None
        left_bbox = np.asarray(data["left_bbox"], dtype=np.float32) if data["left_bbox"] is not None else None

        r_xywh = transform_bbox_to_patch(right_bbox, center, scale, augm, IMG_RES) if right_bbox is not None else None
        l_xywh = transform_bbox_to_patch(left_bbox, center, scale, augm, IMG_RES) if left_bbox is not None else None

        r_img, r_bbox = data_utils.crop_and_pad(img, r_xywh, IMG_RES, IMG_RES, scale=BBOX_SCALE)
        l_img, l_bbox = data_utils.crop_and_pad(img, l_xywh, IMG_RES, IMG_RES, scale=BBOX_SCALE)
        norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
        norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

        img_ds = data_utils.generate_patch_image_clean(
            img.transpose(1, 2, 0),
            [IMG_RES / 2, IMG_RES / 2, IMG_RES, IMG_RES],
            1.0, 0.0, [IMG_RES, IMG_RES], cv2.INTER_CUBIC,
        )[0].transpose(2, 0, 1)
        img_ds = np.clip(img_ds, 0, 1)
        norm_img = self.normalize_img(torch.from_numpy(img_ds).float())

        r_center_angle, r_corner_angle = pos_enc_angles(r_bbox, self.intrx)
        l_center_angle, l_corner_angle = pos_enc_angles(l_bbox, self.intrx)

        right_valid = float(right_bbox is not None)
        left_valid = float(left_bbox is not None)

        inputs = {
            "img": norm_img, "r_img": norm_r_img, "l_img": norm_l_img,
            "r_center_angle": r_center_angle, "l_center_angle": l_center_angle,
            "r_corner_angle": r_corner_angle, "l_corner_angle": l_corner_angle,
        }
        targets = {
            "mano.pose.r": torch.zeros(48), "mano.pose.l": torch.zeros(48),
            "mano.beta.r": torch.tensor(MEAN_BETA_R), "mano.beta.l": torch.tensor(MEAN_BETA_L),
            "mano.j2d.norm.r": torch.zeros(21, 2), "mano.j2d.norm.l": torch.zeros(21, 2),
            "mano.j3d.full.r": torch.zeros(21, 3), "mano.j3d.full.l": torch.zeros(21, 3),
            "grasp.r": _GRASP_LABEL[data["right_grasp"]] if data["right_grasp"] is not None else NO_GRASP,
            "grasp.l": _GRASP_LABEL[data["left_grasp"]] if data["left_grasp"] is not None else NO_GRASP,
            "grasp_valid_r": right_valid,
            "grasp_valid_l": left_valid,
            "render.r": torch.zeros((1, IMG_RES, IMG_RES)),
            "render.l": torch.zeros((1, IMG_RES, IMG_RES)),
            "render_valid_r": 0,
            "render_valid_l": 0,
            "is_valid": 1.0,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "joints_valid_r": np.zeros(21, dtype=np.float32),
            "joints_valid_l": np.zeros(21, dtype=np.float32),
        }
        meta_info = {
            "imgname": imgname,
            "intrinsics": torch.from_numpy(self.intrx).float(),
            "dataset": "epic_grasp",
            "is_j2d_loss": 0,
            "is_j3d_loss": 0,
            "is_beta_loss": 0,
            "is_pose_loss": 0,
            "is_cam_loss": 0,
            "is_grasp_loss": 1,
            "is_mask_loss": 0,
        }
        return inputs, targets, meta_info
