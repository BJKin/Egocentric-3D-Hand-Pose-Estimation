"""
ARCTIC egocentric dataset.
"""
import json
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
    FLIP_PROB, FOCAL_LENGTH, IMG_RES, NOISE_FACTOR, ROT_FACTOR, SCALE_FACTOR,
    pos_enc_angles,
)
from pathlib import Path


# ----------------------------- config -----------------------------
ARCTIC_IMAGES_DIR = Path("../data_reduced/arctic/data/cropped_images")
ARCTIC_META_DIR = Path("../data_reduced/arctic/data/meta")
ARCTIC_SPLITS_DIR = Path("../data_reduced/arctic/data/splits")

EGO_IMAGE_SCALE = 0.3 
BBOX_SCALE = 1.5        
DUMMY_IMG_SHAPE = (600, 840, 3)


def _load_misc() -> dict:
    """
    Loads the ARCTIC metadata for each subject (image sizes, allocentric intrinsics, frame offsets).

    Returns:
        misc -- dict keyed by subject id with keys 'image_size', 'intris_mat', 'ioi_offset'
    """
    with open(ARCTIC_META_DIR / "misc.json") as f:
        return json.load(f)


def _pad_jts2d(jts: np.ndarray) -> np.ndarray:
    """
    Appends a constant '1' confidence column to a 2D keypoint array.

    Arguments:
        jts -- (N, 2) array of 2D keypoints

    Returns:
        jts_pad -- (N, 3) array with the third column set to 1
    """
    out = np.ones((jts.shape[0], 3))
    out[:, :2] = jts
    return out


class ArcticDataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the ARCTIC egocentric split and the metadata for each subject.

        Arguments:
            split -- one of 'train', 'val', 'test'; selects the p2_{split}.npy file
        """
        assert split in ("train", "val", "test"), f"unknown ARCTIC split {split!r}"
        self.split = split
        self.aug_data = split == "train"
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

        misc = _load_misc()
        self.intris_mat = {sid: misc[sid]["intris_mat"] for sid in misc}
        self.image_sizes = {sid: misc[sid]["image_size"] for sid in misc}
        self.ioi_offset = {sid: misc[sid]["ioi_offset"] for sid in misc}

        data_p = ARCTIC_SPLITS_DIR / f"p2_{split}.npy"
        logger.info(f"Loading {data_p}")
        d = np.load(data_p, allow_pickle=True).item()
        self.data = d["data_dict"]
        self.imgnames = d["imgnames"]
        self.egocam_k = None 
        logger.info(f"# samples in ARCTIC {split}: {len(self.imgnames)}")

    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        return self._getitem(self.imgnames[idx])

    def _getitem(self, imgname_rel: str) -> Tuple[dict, dict, dict]:
        """
        Loads one egocentric ARCTIC sample and assembles the (inputs, targets, meta_info)
        tuple that WildHands expects.

        Arguments:
            imgname_rel -- relative imgname stored in the split file

        Returns:
            inputs -- dict with img, r_img, l_img, r_bbox, l_bbox, r_bbox_og, l_bbox_og,
                      r_center_angle, l_center_angle, r_corner_angle, l_corner_angle
            targets -- dict of MANO/grasp/render/center/corner GT plus validity flags for each sample
            meta_info -- dict with imgname, intrinsics, center, rot_angle, dataset name,
                         is_flipped, and boolean masks for each loss term
        """
        sid, seq_name, view_idx_str, image_idx = imgname_rel.split("/")[-4:]
        view_idx = int(view_idx_str)
        assert view_idx == 0, f"ARCTIC egocentric split should only contain view 0, got {view_idx}"
        seq_data = self.data[f"{sid}/{seq_name}"]
        data_cam, data_2d = seq_data["cam_coord"], seq_data["2d"]
        data_bbox, data_params = seq_data["bbox"], seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        is_valid = data_cam["is_valid"][vidx, view_idx]
        right_valid = data_cam["right_valid"][vidx, view_idx]
        left_valid = data_cam["left_valid"][vidx, view_idx]

        intrx = data_params["K_ego"][vidx].copy()
        joints2d_r = _pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()
        joints2d_l = _pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        pose_l = data_params["pose_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()
        rot_r = data_cam["rot_r_cam"][vidx, view_idx]
        rot_l = data_cam["rot_l_cam"][vidx, view_idx]

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}
        bbox = data_bbox[vidx, view_idx]

        joints2d_r[:, :2] *= EGO_IMAGE_SCALE
        joints2d_l[:, :2] *= EGO_IMAGE_SCALE
        bbox = np.array(bbox) * EGO_IMAGE_SCALE

        imgname = str(ARCTIC_IMAGES_DIR / sid / seq_name / view_idx_str / image_idx)
        cv_img, _ = data_utils.read_img(imgname, DUMMY_IMG_SHAPE)

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0 

        joints2d_r = data_utils.j2d_processing(joints2d_r, center, scale, augm, IMG_RES)
        joints2d_l = data_utils.j2d_processing(joints2d_l, center, scale, augm, IMG_RES)
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)


        right_bbox, _ = self._joint_bbox(joints2d_r)
        left_bbox, _ = self._joint_bbox(joints2d_l)

        if self.aug_data:
            right_bbox = data_utils.jitter_bbox(right_bbox)
            left_bbox = data_utils.jitter_bbox(left_bbox)
            right_bbox = self._clip_bbox(right_bbox)
            left_bbox = self._clip_bbox(left_bbox)

        r_img, r_bbox = data_utils.crop_and_pad(img, right_bbox, IMG_RES, IMG_RES, scale=BBOX_SCALE)
        l_img, l_bbox = data_utils.crop_and_pad(img, left_bbox, IMG_RES, IMG_RES, scale=BBOX_SCALE)
        norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
        norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

        img_ds = data_utils.generate_patch_image_clean(
            img.transpose(1, 2, 0),
            [IMG_RES / 2, IMG_RES / 2, IMG_RES, IMG_RES],
            1.0, 0.0, [IMG_RES, IMG_RES], cv2.INTER_CUBIC,
        )[0].transpose(2, 0, 1)
        img_ds = np.clip(img_ds, 0, 1)
        norm_img = self.normalize_img(torch.from_numpy(img_ds).float())

        intrx = data_utils.get_aug_intrix(
            intrx,
            FOCAL_LENGTH,
            IMG_RES,
            use_gt_k=True,
            bbox_cx=image_size["width"] / 2.0,
            bbox_cy=image_size["height"] / 2.0,
            scale=augm["sc"] * max(image_size["width"], image_size["height"]) / 200.0,
        )
        if self.egocam_k is None:
            self.egocam_k = intrx
        else:
            intrx = self.egocam_k

        r_center_angle, r_corner_angle = pos_enc_angles(r_bbox, intrx)
        l_center_angle, l_corner_angle = pos_enc_angles(l_bbox, intrx)

        pose_r = np.concatenate([rot_r, pose_r], axis=0)
        pose_l = np.concatenate([rot_l, pose_l], axis=0)

        inputs = {
            "img": norm_img,
            "r_img": norm_r_img,
            "l_img": norm_l_img,
            "r_center_angle": r_center_angle,
            "l_center_angle": l_center_angle,
            "r_corner_angle": r_corner_angle,
            "l_corner_angle": l_corner_angle,
        }
        targets = {
            "mano.pose.r": torch.from_numpy(data_utils.pose_processing(pose_r, augm)).float(),
            "mano.pose.l": torch.from_numpy(data_utils.pose_processing(pose_l, augm)).float(),
            "mano.beta.r": torch.from_numpy(betas_r).float(),
            "mano.beta.l": torch.from_numpy(betas_l).float(),
            "mano.j2d.norm.r": torch.from_numpy(joints2d_r[:, :2]).float(),
            "mano.j2d.norm.l": torch.from_numpy(joints2d_l[:, :2]).float(),
            "mano.j3d.full.r": torch.from_numpy(joints3d_r[:, :3]).float(),
            "mano.j3d.full.l": torch.from_numpy(joints3d_l[:, :3]).float(),
            "grasp.r": 8,
            "grasp.l": 8,
            "grasp_valid_r": 0,
            "grasp_valid_l": 0,
            "render.r": torch.zeros((1, IMG_RES, IMG_RES)),
            "render.l": torch.zeros((1, IMG_RES, IMG_RES)),
            "render_valid_r": 0,
            "render_valid_l": 0,
            "is_valid": float(is_valid),
            "left_valid": float(left_valid) * float(is_valid),
            "right_valid": float(right_valid) * float(is_valid),
            "joints_valid_r": np.ones(21, dtype=np.float32) * float(right_valid) * float(is_valid),
            "joints_valid_l": np.ones(21, dtype=np.float32) * float(left_valid) * float(is_valid),
        }
        meta_info = {
            "imgname": imgname,
            "intrinsics": torch.from_numpy(np.asarray(intrx)).float(),
            "dataset": "arctic",
            "is_j2d_loss": 1,
            "is_j3d_loss": 1,
            "is_beta_loss": 1,
            "is_pose_loss": 1,
            "is_cam_loss": 1,
            "is_grasp_loss": 0,
            "is_mask_loss": 0,
        }
        return inputs, targets, meta_info

    @staticmethod
    def _joint_bbox(joints2d_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derives an [x0, y0, w, h] bbox from normalized 2D joints inside the IMG_RES patch.

        Arguments:
            joints2d_norm -- (N, 3) joints with xy in [-1, 1] coordinates

        Returns:
            bbox -- [x0, y0, w, h] int16 array
            bbox_og -- [x0, y0, w, h] int16 array
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

    @staticmethod
    def _clip_bbox(bbox):
        """
        Clips a jittered bbox to the IMG_RES patch and returns None if it collapsed.

        Arguments:
            bbox -- [x0, y0, w, h] or None

        Returns:
            bbox -- clipped [x0, y0, w, h] (np.int16) or None if non positive after clipping
        """
        if bbox is None:
            return None
        x0, y0, w, h = bbox
        x1, y1 = np.clip([x0 + w, y0 + h], 0, IMG_RES - 1).astype(np.int16)
        x0, y0 = np.clip([x0, y0], 0, IMG_RES - 1).astype(np.int16)
        if x1 - x0 == 0 or y1 - y0 == 0:
            return None
        return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.int16)
