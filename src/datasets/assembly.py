"""
AssemblyHands egocentric dataset.
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
    FLIP_PROB, FOCAL_LENGTH, IMG_RES, MEAN_BETA_L, MEAN_BETA_R, NOISE_FACTOR,
    ROT_FACTOR, SCALE_FACTOR, pad_jts2d, pos_enc_angles, transform_bbox_to_patch,
)
from pathlib import Path


# ----------------------------- config -----------------------------
ASSEMBLY_DIR = Path("../data_reduced/assembly")
ASSEMBLY_ANNOT_DIR = ASSEMBLY_DIR / "annotations"
BBOX_SCALE = 1.75
ANNOT_VERSION = "v1-1"

# Maps AssemblyHands' 42-joint index space to the 21-joint MANO ordering per hand.
ASSEMBLY_TO_MANO_R = np.array([20, 7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3,  2, 1, 0, 4,  8, 12, 16])
ASSEMBLY_TO_MANO_L = np.array([41, 28, 27, 26, 32, 31, 30, 40, 39, 38, 36, 35, 34, 24, 23, 22, 21, 25, 29, 33, 37])


class AssemblyDataset(Dataset):
    def __init__(self, split: str):
        """
        Loads the AssemblyHands ego data/calib/joint-3d JSONs for the requested split.

        Arguments:
            split -- 'train' or 'val'
        """
        assert split in ("train", "val"), f"AssemblyHands split must be 'train' or 'val', got {split!r}"
        self.split = split
        self.aug_data = split == "train"
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

        data_p = ASSEMBLY_ANNOT_DIR / split / f"assemblyhands_{split}_ego_data_{ANNOT_VERSION}.json"
        calib_p = ASSEMBLY_ANNOT_DIR / split / f"assemblyhands_{split}_ego_calib_{ANNOT_VERSION}.json"
        joint_p = ASSEMBLY_ANNOT_DIR / split / f"assemblyhands_{split}_joint_3d_{ANNOT_VERSION}.json"
        logger.info(f"Loading AssemblyHands {split} annotations (3 JSONs, ~1.6 GB for train)…")
        with open(data_p) as f:
            d = json.load(f)
        with open(calib_p) as f:
            self.cameras = json.load(f)["calibration"]
        with open(joint_p) as f:
            self.joints_3d = json.load(f)["annotations"]

        self.images_by_id = {im["id"]: im for im in d["images"]}
        self.anns = d["annotations"]
        logger.info(f"# samples in AssemblyHands {split}: {len(self.anns)}")

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        ann = self.anns[idx]
        img_info = self.images_by_id[ann["image_id"]]
        seq, cam, frame_idx = img_info["seq_name"], img_info["camera"], img_info["frame_idx"]
        cam_key = f"{cam}_mono10bit"
        frame_key = f"{frame_idx:06d}"

        K = np.asarray(self.cameras[seq]["intrinsics"][cam_key], dtype=np.float32)
        Rt = np.asarray(self.cameras[seq]["extrinsics"][frame_key][cam_key], dtype=np.float32)
        world = np.asarray(self.joints_3d[seq][frame_key]["world_coord"], dtype=np.float32)
        R, t = Rt[:, :3], Rt[:, 3]
        joints_cam = world @ R.T + t
        proj = joints_cam @ K.T
        joints_img = proj[:, :2] / proj[:, 2:3]

        joints2d_r = pad_jts2d(joints_img[ASSEMBLY_TO_MANO_R])
        joints2d_l = pad_jts2d(joints_img[ASSEMBLY_TO_MANO_L])
        joints3d_r = joints_cam[ASSEMBLY_TO_MANO_R] / 1000.0 
        joints3d_l = joints_cam[ASSEMBLY_TO_MANO_L] / 1000.0
        joint_valid = np.asarray(ann["joint_valid"], dtype=np.float32)
        joints_valid_r = joint_valid[ASSEMBLY_TO_MANO_R]
        joints_valid_l = joint_valid[ASSEMBLY_TO_MANO_L]

        img_w, img_h = img_info["width"], img_info["height"]
        img_path = str(ASSEMBLY_DIR / img_info["file_name"])
        cv_img, _ = data_utils.read_img(img_path, (img_h, img_w, 3))

        bbox = [img_w / 2, img_h / 2, max(img_w, img_h) / 200.0]
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        augm = data_utils.augm_params(self.aug_data, flip_prob=FLIP_PROB, noise_factor=NOISE_FACTOR, rot_factor=ROT_FACTOR, scale_factor=SCALE_FACTOR)
        augm["sc"] = 1.0 

        joints2d_r = data_utils.j2d_processing(joints2d_r, center, scale, augm, IMG_RES)
        joints2d_l = data_utils.j2d_processing(joints2d_l, center, scale, augm, IMG_RES)
        img = data_utils.rgb_processing(self.aug_data, cv_img, center, scale, augm, IMG_RES)

        right_xyxy = ann["bbox"]["right"]
        left_xyxy = ann["bbox"]["left"]
        r_xywh = transform_bbox_to_patch(np.asarray(right_xyxy, dtype=np.float32), center, scale, augm, IMG_RES) if right_xyxy is not None else None
        l_xywh = transform_bbox_to_patch(np.asarray(left_xyxy, dtype=np.float32), center, scale, augm, IMG_RES) if left_xyxy is not None else None

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

        intrx = data_utils.get_aug_intrix(
            K.copy(), FOCAL_LENGTH, IMG_RES, use_gt_k=True,
            bbox_cx=img_w / 2.0, bbox_cy=img_h / 2.0,
            scale=augm["sc"] * max(img_w, img_h) / 200.0,
        )
        intrx_np = intrx.numpy() if isinstance(intrx, torch.Tensor) else np.asarray(intrx)

        r_center_angle, r_corner_angle = pos_enc_angles(r_bbox, intrx_np)
        l_center_angle, l_corner_angle = pos_enc_angles(l_bbox, intrx_np)

        right_valid = float(right_xyxy is not None)
        left_valid = float(left_xyxy is not None)

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
            "joints_valid_r": joints_valid_r,
            "joints_valid_l": joints_valid_l,
        }
        meta_info = {
            "imgname": img_path,
            "intrinsics": torch.from_numpy(intrx_np).float(),
            "dataset": "assembly",
            "is_j2d_loss": 1,
            "is_j3d_loss": 1,
            "is_beta_loss": 0,
            "is_pose_loss": 0,
            "is_cam_loss": 0,
            "is_grasp_loss": 0,
            "is_mask_loss": 0,
        }
        return inputs, targets, meta_info
