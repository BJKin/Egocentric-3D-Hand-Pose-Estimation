"""
Ego4D segmentation dataset.
"""
import pickle
import random
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
    SCALE_FACTOR, dummy_egocam_intrx, ego_path, pos_enc_angles,
    transform_bbox_to_patch,
)
from pathlib import Path


# ----------------------------- config -----------------------------
EGO4D_HANDS_DIR = Path("../data_reduced/ego4d_hands")
BBOX_SCALE = 1.5
EGO_W, EGO_H = 1440, 1080
DUMMY_IMG_SHAPE = (EGO_H, EGO_W, 3)
RIGHT_MASK_VAL = 255
LEFT_MASK_VAL = 127


class Ego4DSegDataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the Ego4D grasp bboxes and pred hand masks.

        Arguments:
            split -- 'train'
        """
        assert split == "train", f"Ego4DSegDataset only supports the 'train' split, got {split!r}"
        with open(EGO4D_HANDS_DIR / "grasp_ego.pkl", "rb") as f:
            self.bbox = pickle.load(f)

        self.masks_path = EGO4D_HANDS_DIR / "ego_blur_pred_masks.npz"
        with np.load(self.masks_path, allow_pickle=True) as npz:
            mask_keys = set(npz.files)
        self.masks_npz = None
        self.imgnames = sorted(set(self.bbox) & mask_keys)

        self.aug_data = True
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.intrx = dummy_egocam_intrx(EGO_W, EGO_H)
        logger.info(f"# samples in Ego4D seg train: {len(self.imgnames)}")

    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        if self.masks_npz is None:
            self.masks_npz = np.load(self.masks_path, allow_pickle=True)
        while True:
            pickle_key = self.imgnames[idx]
            try:
                mask_npz = self.masks_npz[pickle_key][..., 0]
                break
            except Exception:
                idx = random.randrange(len(self.imgnames))

        bbox_data = self.bbox[pickle_key]
        imgname = str(ego_path(pickle_key))
        cv_img, _ = data_utils.read_img(imgname, DUMMY_IMG_SHAPE)

        bbox = [EGO_W / 2, EGO_H / 2, max(EGO_W, EGO_H) / 200]
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)

        right_bbox = np.asarray(bbox_data["right_bbox"], dtype=np.float32) if bbox_data["right_bbox"] is not None else None
        left_bbox = np.asarray(bbox_data["left_bbox"], dtype=np.float32) if bbox_data["left_bbox"] is not None else None
        right_bbox_og = right_bbox.astype(np.int16) if right_bbox is not None else np.array([0, 0, IMG_RES - 1, IMG_RES - 1])
        left_bbox_og = left_bbox.astype(np.int16) if left_bbox is not None else np.array([0, 0, IMG_RES - 1, IMG_RES - 1])

        mask_r = self._bbox_clipped_mask(mask_npz, RIGHT_MASK_VAL, right_bbox_og, right_bbox is not None)
        mask_l = self._bbox_clipped_mask(mask_npz, LEFT_MASK_VAL, left_bbox_og, left_bbox is not None)
        mask_augm = {**augm, "pn": np.ones_like(augm["pn"])}
        mask_img_r = data_utils.mask_processing(mask_r, center, scale, mask_augm, IMG_RES)
        mask_img_l = data_utils.mask_processing(mask_l, center, scale, mask_augm, IMG_RES)

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
            "grasp.r": 8, "grasp.l": 8,
            "grasp_valid_r": 0, "grasp_valid_l": 0,
            "render.r": torch.from_numpy(mask_img_r[0:1]).float(),
            "render.l": torch.from_numpy(mask_img_l[0:1]).float(),
            "render_valid_r": right_valid,
            "render_valid_l": left_valid,
            "is_valid": 1.0,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "joints_valid_r": np.zeros(21, dtype=np.float32),
            "joints_valid_l": np.zeros(21, dtype=np.float32),
        }
        meta_info = {
            "imgname": imgname,
            "intrinsics": torch.from_numpy(self.intrx).float(),
            "dataset": "ego_seg",
            "is_j2d_loss": 0,
            "is_j3d_loss": 0,
            "is_beta_loss": 0,
            "is_pose_loss": 0,
            "is_cam_loss": 0,
            "is_grasp_loss": 0,
            "is_mask_loss": 1,
        }
        return inputs, targets, meta_info

    @staticmethod
    def _bbox_clipped_mask(mask_npz: np.ndarray, pixel_val: int, bbox_og: np.ndarray, bbox_present: bool) -> np.ndarray:
        """
        Builds the GT segmentation mask for one hand side.

        Arguments:
            mask_npz -- HxW source mask
            pixel_val -- 255 for right hand, 127 for left hand
            bbox_og -- (4,) [x0, y0, x1, y1] bbox in source pixels, or
                       [0, 0, IMG_RES-1, IMG_RES-1] when the hand is missing
            bbox_present -- if False, returns an all zero mask

        Returns:
            mask -- HxWx3 float mask in {0, 255}
        """
        if not bbox_present:
            return np.zeros((mask_npz.shape[0], mask_npz.shape[1], 3), dtype=np.float32)
        hand = (mask_npz == pixel_val)
        clip = np.zeros_like(hand)
        clip[bbox_og[1]:bbox_og[3], bbox_og[0]:bbox_og[2]] = True
        m = (hand & clip).astype(np.float32) * 255.0
        return np.stack([m, m, m], axis=-1)
