"""
Image, keypoint, and pose augmentation helpers used by the dataset pipelines.
"""
import cv2
import numpy as np
import torch
from loguru import logger


# ----------------------------- config -----------------------------
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


def read_img(img_fn: str, dummy_shape: tuple) -> tuple[np.ndarray, bool]:
    """
    Reads an image as a float32 RGB array, returning a zero array on failure.

    Arguments:
        img_fn -- path to the image file
        dummy_shape -- (H, W, 3) shape used to fabricate a zero image if loading fails

    Returns:
        cv_img -- HxWx3 float32 image in RGB order
        ok -- True if the file loaded, False otherwise
    """
    try:
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
        return img.astype(np.float32), True
    except Exception:
        logger.warning(f"Unable to load {img_fn}")
        return np.zeros(dummy_shape, dtype=np.float32), False


def augm_params(is_train: bool, flip_prob: float, noise_factor: float, rot_factor: float, scale_factor: float) -> dict:
    """
    Samples augmentation parameters (flip flag, pixel noise for each channel, rotation, scale).

    Arguments:
        is_train -- if False, no augmentation is applied
        flip_prob -- probability of horizontal flip
        noise_factor -- max multiplicative pixel noise magnitude for each channel
        rot_factor -- degrees standard deviation for rotation
        scale_factor -- relative scale standard deviation around 1.0

    Returns:
        augm_dict -- dict with keys flip, pn, rot, sc
    """
    flip = 0
    pn = np.ones(3)
    rot = 0
    sc = 1
    if is_train:
        if np.random.uniform() <= flip_prob:
            flip = 1
        pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)
        rot = min(2 * rot_factor, max(-2 * rot_factor, np.random.randn() * rot_factor))
        sc = min(1 + scale_factor, max(1 - scale_factor, np.random.randn() * scale_factor + 1))
        if np.random.uniform() <= 0.6:
            rot = 0
    return {"flip": flip, "pn": pn, "rot": rot, "sc": sc}


def rotate_2d(pt_2d: np.ndarray, rot_rad: float) -> np.ndarray:
    """
    Rotates a 2D point around the origin by rot_rad radians.

    Arguments:
        pt_2d -- (2,) point
        rot_rad -- rotation angle in radians

    Returns:
        rotated -- (2,) rotated point
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([pt_2d[0] * cs - pt_2d[1] * sn,
                     pt_2d[0] * sn + pt_2d[1] * cs], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float, src_width: float, src_height: float, dst_width: float, dst_height: float, scale: float, rot: float, inv: bool = False) -> np.ndarray:
    """
    Builds an OpenCV style 2x3 affine matrix that maps a source bbox patch into the
    destination patch with scaling and rotation.

    Arguments:
        c_x, c_y -- source patch center in source pixel coordinates
        src_width, src_height -- source patch size in pixels
        dst_width, dst_height -- destination patch size in pixels
        scale -- multiplicative scale applied to the source patch
        rot -- rotation in degrees
        inv -- if True, returns the inverse mapping (dst -> src)

    Returns:
        trans -- (2, 3) float32 affine matrix
    """
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.stack([src_center, src_center + src_downdir, src_center + src_rightdir])
    dst = np.stack([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans.astype(np.float32)


def generate_patch_image(cvimg: np.ndarray, bbox: list, scale: float, rot: float, out_shape: list, interpl_strategy: int, gauss_kernel: int = 5, gauss_sigma: float = 8.0) -> tuple:
    """
    Crops a bbox patch from an image with optional rotation/scaling and
    Gaussian anti-aliasing prior to warping.

    Arguments:
        cvimg -- HxWx3 source image
        bbox -- [cx, cy, w, h] bbox in source pixel space
        scale -- multiplicative scale on the bbox
        rot -- rotation in degrees
        out_shape -- [out_h, out_w] destination patch size
        interpl_strategy -- OpenCV interpolation flag
        gauss_kernel -- Gaussian kernel size for anti-aliasing
        gauss_sigma -- Gaussian sigma for anti-aliasing

    Returns:
        img_patch -- warped patch as float32 array
        trans -- (2, 3) forward affine matrix
        inv_trans -- (2, 3) inverse affine matrix
    """
    bb_c_x, bb_c_y, bb_w, bb_h = (float(x) for x in bbox)
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_w, bb_h, out_shape[1], out_shape[0], scale, rot)
    blur = cv2.GaussianBlur(cvimg, (gauss_kernel, gauss_kernel), gauss_sigma)
    img_patch = cv2.warpAffine(blur, trans, (int(out_shape[1]), int(out_shape[0])),  flags=interpl_strategy).astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_w, bb_h, out_shape[1], out_shape[0], scale, rot, inv=True)
    return img_patch, trans, inv_trans


def generate_patch_image_clean(cvimg: np.ndarray, bbox: list, scale: float, rot: float, out_shape: list, interpl_strategy: int) -> tuple:
    """
    Same as generate_patch_image but without the Gaussian anti-aliasing blur step.

    Arguments:
        cvimg -- HxWx3 source image
        bbox -- [cx, cy, w, h] bbox in source pixel space
        scale -- multiplicative scale on the bbox
        rot -- rotation in degrees
        out_shape -- [out_h, out_w] destination patch size
        interpl_strategy -- OpenCV interpolation flag

    Returns:
        img_patch -- warped patch as float32 array
        trans -- (2, 3) forward affine matrix
        inv_trans -- (2, 3) inverse affine matrix
    """
    bb_c_x, bb_c_y, bb_w, bb_h = (float(x) for x in bbox)
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_w, bb_h, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(cvimg, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy).astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_w, bb_h, out_shape[1], out_shape[0], scale, rot, inv=True)
    return img_patch, trans, inv_trans


def rgb_processing(is_train: bool, rgb_img: np.ndarray, center: list, bbox_dim: float, augm_dict: dict, img_res: int) -> np.ndarray:
    """
    Crops and augments an RGB image patch around a bbox center, returning a
    CHW float image normalized to [0, 1].

    Arguments:
        is_train -- training mode flag
        rgb_img -- HxWx3 source image
        center -- [cx, cy] bbox center in source pixel space
        bbox_dim -- bbox dimension in scale units
        augm_dict -- dict from augm_params with keys rot, sc, pn
        img_res -- output patch resolution

    Returns:
        img -- 3xRxR float32 image in [0, 1]
    """
    rot = augm_dict["rot"]
    scale = augm_dict["sc"] * bbox_dim
    pn = augm_dict["pn"]
    crop_dim = int(scale * 200)
    rgb_img = generate_patch_image(rgb_img, [center[0], center[1], crop_dim, crop_dim],
                                   1.0, rot, [img_res, img_res], cv2.INTER_CUBIC)[0]
    rgb_img[:, :, 0] = np.clip(rgb_img[:, :, 0] * pn[0], 0.0, 255.0)
    rgb_img[:, :, 1] = np.clip(rgb_img[:, :, 1] * pn[1], 0.0, 255.0)
    rgb_img[:, :, 2] = np.clip(rgb_img[:, :, 2] * pn[2], 0.0, 255.0)
    return np.transpose(rgb_img, (2, 0, 1)).astype(np.float32) / 255.0


def mask_processing(mask_img: np.ndarray, center: list, bbox_dim: float, augm_dict: dict, img_res: int) -> np.ndarray:
    """
    Same crop/rotate transform as rgb_processing but with nearest neighbor
    interpolation and no color noise (used for segmentation masks).

    Arguments:
        mask_img -- HxWx3 source mask
        center -- [cx, cy] bbox center in source pixel space
        bbox_dim -- bbox dimension
        augm_dict -- dict from augm_params with keys rot, sc
        img_res -- output patch resolution

    Returns:
        mask -- 3xRxR float32 mask in [0, 1]
    """
    scale = augm_dict["sc"] * bbox_dim
    crop_dim = int(scale * 200)
    warped = generate_patch_image_clean(
        mask_img, [center[0], center[1], crop_dim, crop_dim],
        1.0, augm_dict["rot"], [img_res, img_res], cv2.INTER_NEAREST,
    )[0]
    return np.transpose(warped.astype("float32"), (2, 0, 1)) / 255.0


def get_transform(center: list, scale: float, res: list, rot: float = 0) -> np.ndarray:
    """
    Builds the 3x3 affine that maps a (center, scale) bbox into a res patch
    with optional in-plane rotation around the patch center.

    Arguments:
        center -- [cx, cy] bbox center in source pixel space
        scale -- bbox dimension
        res -- [out_h, out_w] destination resolution
        rot -- rotation in degrees

    Returns:
        t -- (3, 3) affine matrix
    """
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if rot != 0:
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = t_inv @ rot_mat @ t_mat @ t
    return t


def transform(pt: np.ndarray, center: list, scale: float, res: list, invert: int = 0, rot: float = 0) -> np.ndarray:
    """
    Applies the (center, scale, rot) crop transform to a single 2D pixel point.

    Arguments:
        pt -- (2,) point in source pixel space
        center -- [cx, cy] bbox center
        scale -- bbox dimension in scale units
        res -- [out_h, out_w] destination resolution
        invert -- if non-zero, applies the inverse transform
        rot -- rotation in degrees

    Returns:
        new_pt -- (2,) integer point in the destination patch
    """
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def normalize_kp2d_np(kp2d: np.ndarray, img_res: int) -> np.ndarray:
    """
    Maps 2D keypoints into [-1, 1].

    Arguments:
        kp2d -- (N, 3) keypoints
        img_res -- patch resolution used for normalization

    Returns:
        kp2d_norm -- (N, 3) keypoints with xy in [-1, 1]
    """
    assert kp2d.shape[1] == 3
    out = kp2d.copy()
    out[:, :2] = 2.0 * kp2d[:, :2] / img_res - 1.0
    return out


def j2d_processing(kp: np.ndarray, center: list, bbox_dim: float, augm_dict: dict, img_res: int) -> np.ndarray:
    """
    Crops and normalizes 2D keypoints into the [-1, 1] patch coordinate space.

    Arguments:
        kp -- (N, 3) keypoints in source pixel space
        center -- [cx, cy] bbox center
        bbox_dim -- bbox dimension in scale units
        augm_dict -- dict from augm_params with keys sc, rot
        img_res -- output patch resolution

    Returns:
        kp -- (N, 3) float32 keypoints with xy in [-1, 1]
    """
    scale = augm_dict["sc"] * bbox_dim
    rot = augm_dict["rot"]
    for i in range(kp.shape[0]):
        kp[i, :2] = transform(kp[i, :2] + 1, center, scale, [img_res, img_res], rot=rot)
    return normalize_kp2d_np(kp, img_res).astype("float32")


def rot_aa(aa: np.ndarray, rot: float) -> np.ndarray:
    """
    Applies a rotation to an axis angle global orientation.

    Arguments:
        aa -- (3,) axis angle global orientation
        rot -- rotation in degrees to compose on the left

    Returns:
        aa -- (3,) rotated axis angle vector
    """
    R = np.array([
        [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
        [np.sin(np.deg2rad(-rot)),  np.cos(np.deg2rad(-rot)), 0],
        [0, 0, 1],
    ])
    per_rdg, _ = cv2.Rodrigues(aa)
    resrot, _ = cv2.Rodrigues(R @ per_rdg)
    return resrot.T[0]


def pose_processing(pose: np.ndarray, augm_dict: dict) -> np.ndarray:
    """
    Applies the augmentation rotation to the global orientation of a MANO pose vector.

    Arguments:
        pose -- (48,) MANO axis-angle pose
        augm_dict -- dict from augm_params with key rot

    Returns:
        pose -- (48,) float32 rotated MANO pose
    """
    pose[:3] = rot_aa(pose[:3], augm_dict["rot"])
    return pose.astype("float32")


def get_wp_intrix(fixed_focal: float, img_res: int) -> torch.Tensor:
    """
    Builds a 3x3 weak-perspective intrinsics matrix centered on the patch.

    Arguments:
        fixed_focal -- focal length in pixels
        img_res -- patch resolution

    Returns:
        intrx -- (3, 3) intrinsics tensor
    """
    intrx = torch.zeros([3, 3])
    intrx[0, 0] = fixed_focal
    intrx[1, 1] = fixed_focal
    intrx[2, 2] = 1.0
    intrx[0, -1] = img_res // 2
    intrx[1, -1] = img_res // 2
    return intrx


def get_aug_intrix(intrx: np.ndarray, fixed_focal: float, img_res: int, use_gt_k: bool, bbox_cx: float, bbox_cy: float, scale: float):
    """
    Returns camera intrinsics consistent with the augmented patch.

    Arguments:
        intrx -- (3, 3) ground truth intrinsics in full image space
        fixed_focal -- fallback focal length in pixels when not using GT
        img_res -- patch resolution
        use_gt_k -- whether to scale the GT intrinsics or use the fallback
        bbox_cx, bbox_cy -- bbox center in full image pixels
        scale -- bbox dimension

    Returns:
        intrx -- (3, 3) intrinsics
    """
    if not use_gt_k:
        return get_wp_intrix(fixed_focal, img_res)
    dim = scale * 200.0
    k_scale = float(img_res) / dim
    intrx[0, 0] *= k_scale
    intrx[1, 1] *= k_scale
    intrx[0, 2] -= bbox_cx - dim / 2.0
    intrx[1, 2] -= bbox_cy - dim / 2.0
    intrx[0, 2] *= k_scale
    intrx[1, 2] *= k_scale
    return intrx


def jitter_bbox(bbox, t_stdev: float = 0.2):
    """
    Randomly translates a [x0, y0, w, h] bbox by a Gaussian jitter scaled to bbox size.
    Returns None unchanged if the input is None.

    Arguments:
        bbox -- [x0, y0, w, h] or None
        t_stdev -- translation jitter standard deviation as a fraction of bbox size

    Returns:
        new_bbox -- float32 [x0, y0, w, h] or None
    """
    if bbox is None:
        return bbox
    x0, y0, w, h = bbox
    center = np.array([x0 + w / 2, y0 + h / 2])
    size = np.array([w, h])
    jitter = (np.random.rand(2) * 2 - 1) * t_stdev * size
    new_center = center + jitter
    return np.array([new_center[0] - size[0] / 2,
                     new_center[1] - size[1] / 2,
                     size[0], size[1]]).astype(np.float32)


def crop_and_pad(img: np.ndarray, bbox, img_res: int, img_res_ds: int, scale: float = 1.0):
    """
    Crops a square patch around a [x0, y0, w, h] bbox and resizes to img_res_ds.
    Falls back to a full-frame downscale if bbox is None.

    Arguments:
        img -- 3xHxW input image
        bbox -- [x0, y0, w, h] in pixel space, or None
        img_res -- full image resolution (used for the fallback bbox)
        img_res_ds -- output patch resolution
        scale -- multiplicative scale on the cropped square

    Returns:
        img_crop -- 3xRxR float32 patch in [0, 1]
        new_bbox -- (4,) int16 [x0, y0, x1, y1] in original coordinates
    """
    if bbox is None:
        img_crop = generate_patch_image_clean(
            img.transpose(1, 2, 0),
            [img_res / 2, img_res / 2, img_res, img_res],
            1.0, 0.0, [img_res_ds, img_res_ds], cv2.INTER_CUBIC,
        )[0].transpose(2, 0, 1)
        img_crop = np.clip(img_crop, 0, 1)
        return img_crop, np.array([0, 0, img_res - 1, img_res - 1])

    x0, y0 = int(bbox[0]), int(bbox[1])
    x1, y1 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    x_mid, y_mid = (x0 + x1) // 2, (y0 + y1) // 2
    size = max(x1 - x0, y1 - y0)
    img_crop = generate_patch_image_clean(
        img.transpose(1, 2, 0),
        [x_mid, y_mid, size * scale, size * scale],
        1.0, 0.0, [img_res_ds, img_res_ds], cv2.INTER_CUBIC,
    )[0]
    img_crop = np.clip(img_crop, 0, 1).transpose(2, 0, 1)
    new_bbox = np.array([
        x_mid - (size * scale) // 2, y_mid - (size * scale) // 2,
        x_mid + (size * scale) // 2, y_mid + (size * scale) // 2,
    ]).clip(0, img_res - 1).astype(np.int16)
    return img_crop, new_bbox


def normalize_kp2d(kp2d: torch.Tensor, img_res: int) -> torch.Tensor:
    """
    Normalizes a batched 2D keypoint tensor from pixel space to [-1, 1].

    Arguments:
        kp2d -- (B, N, 2 or 3) keypoint tensor
        img_res -- patch resolution used for normalization

    Returns:
        kp2d_norm -- same shape as input with xy in [-1, 1]
    """
    assert kp2d.dim() == 3
    out = kp2d.clone()
    out[:, :, :2] = 2.0 * kp2d[:, :, :2] / img_res - 1.0
    return out


def unormalize_kp2d(kp2d_normalized: torch.Tensor, img_res: int) -> torch.Tensor:
    """
    Inverts normalize_kp2d, mapping (B, N, 2) keypoints from [-1, 1] back to pixels.

    Arguments:
        kp2d_normalized -- (B, N, 2) keypoint tensor in [-1, 1]
        img_res -- patch resolution

    Returns:
        kp2d -- (B, N, 2) keypoints in pixel space
    """
    assert kp2d_normalized.dim() == 3 and kp2d_normalized.shape[2] == 2
    return 0.5 * img_res * (kp2d_normalized + 1)
