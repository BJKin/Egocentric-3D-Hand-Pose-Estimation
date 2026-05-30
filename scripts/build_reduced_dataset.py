"""
This script READS the full ./data tree and WRITES the subset to ./data_reduced.
"""
import json
import pickle
import random
import shutil
from pathlib import Path
import numpy as np
from loguru import logger


# ----------------------------- config -----------------------------
ARCTIC_SPLITS_DIR = Path("data/arctic/data/splits")
ARCTIC_META_DIR = Path("data/arctic/data/meta")
ARCTIC_IMAGES_DIR = Path("data/arctic/data/cropped_images")
ARCTIC_CKPT = Path("data/arctic/arctic_sf_allocentric/last.ckpt")
ASSEMBLY_DIR = Path("data/assembly")
ASSEMBLY_ANNOT_DIR = ASSEMBLY_DIR / "annotations"
EGO4D_FRAMES_DIR = Path("data/ego4d/ego4d_frames")
EGO4D_HANDS_DIR = Path("data/ego4d_hands")
EPIC_HANDS_DIR = Path("data/epic_hands")
H2O_DIR = Path("data/h2o")
VISOR_FRAMES_DIR = Path("data/visor/rgb_frames")

OUT_DIR = "data_reduced"
SEED = 42

# sample counts per dataset
ARCTIC_TRAIN = 2000
ARCTIC_VAL = 1000
ARCTIC_TRAIN_SEQS = 20        # upper bound on ARCTIC sequences sampled for train
ARCTIC_VAL_SEQS = 20          # upper bound on ARCTIC sequences sampled for val
ASSEMBLY_TRAIN = 2000
ASSEMBLY_VAL = 1000
EPIC_GRASP = 1000
EPIC_SEG = 1000
EGO_GRASP = 1000
EGO_SEG = 1000
EPIC_HANDKPS = 1000
EGOEXO_VAL = 1000
H2O_VAL = 1000


def _ensure_dir(p: Path) -> None:
    """
    Creates a directory and any missing parent directories.

    Arguments:
        p -- directory path to create
    """
    p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> bool:
    """
    Copies one file if it exists, creating the destination's parent directories.

    Arguments:
        src -- source file path
        dst -- destination file path

    Returns:
        copied -- True if the file was copied, False if the source was missing
    """
    if not src.exists():
        return False
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _total_size_mb(root: Path) -> float:
    """
    Sums the size of every file under a directory tree.

    Arguments:
        root -- directory to measure recursively

    Returns:
        size_mb -- total size of all files in megabytes
    """
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6


def _read_valid_npz_entries(npz, keys) -> dict:
    """
    Reads each given key, dropping any entry whose decompression raises,
    and casting to uint8.

    Arguments:
        npz -- np.load(...) NpzFile object
        keys -- iterable of keys to attempt to read

    Returns:
        valid -- {key: uint8 array} for every key that loaded cleanly
    """
    valid = {}
    for k in keys:
        try:
            valid[k] = npz[k].astype(np.uint8)
        except Exception:
            logger.warning(f"    skipping corrupt npz entry: {Path(k).name}")
    return valid


def reduce_arctic(out_root: Path, n_train: int, n_val: int, n_train_seqs: int, n_val_seqs: int) -> None:
    """
    Subsets ARCTIC's egocentric split files by first sampling N whole
    sequences and then sampling up to (n_split / n_seqs) frames from each.

    Arguments:
        out_root -- root directory of the reduced dataset to write into
        n_train -- target number of train frames to keep
        n_val -- target number of val frames to keep
        n_train_seqs -- upper bound on distinct sequences sampled for train
        n_val_seqs -- upper bound on distinct sequences sampled for val
    """
    rng = random.Random(SEED)
    for split, n, n_seqs in (("train", n_train, n_train_seqs), ("val", n_val, n_val_seqs)):
        src_split = ARCTIC_SPLITS_DIR / f"p2_{split}.npy"
        if not src_split.exists():
            logger.warning(f"  ARCTIC {split}: split file missing ({src_split})")
            continue
        d = np.load(src_split, allow_pickle=True).item()

        by_seq: dict = {}
        for img in d["imgnames"]:
            sid_seq = f"{img.split('/')[-4]}/{img.split('/')[-3]}"
            by_seq.setdefault(sid_seq, []).append(img)
        keep_seqs = rng.sample(list(by_seq), min(n_seqs, len(by_seq)))
        per_seq = max(1, n // max(1, len(keep_seqs)))
        keep: list = []
        for s in keep_seqs:
            keep.extend(rng.sample(by_seq[s], min(per_seq, len(by_seq[s]))))

        data_dict_sub = {s: d["data_dict"][s] for s in keep_seqs if s in d["data_dict"]}
        out_split = out_root / "arctic/data/splits" / f"p2_{split}.npy"
        _ensure_dir(out_split.parent)
        np.save(out_split, {"data_dict": data_dict_sub, "imgnames": keep})

        n_copied = 0
        for img_rel in keep:
            sid, seq, view, img = img_rel.split("/")[-4:]
            src = ARCTIC_IMAGES_DIR / sid / seq / view / img
            dst = out_root / "arctic/data/cropped_images" / sid / seq / view / img
            if _copy_file(src, dst):
                n_copied += 1
        logger.info(f"  ARCTIC {split}: kept {len(keep)} frames from {len(keep_seqs)} seqs, "
                    f"copied {n_copied} images")

    misc_src = ARCTIC_META_DIR / "misc.json"
    if misc_src.exists():
        _copy_file(misc_src, out_root / "arctic/data/meta/misc.json")
        logger.info(f"  ARCTIC meta/misc.json: copied")
    if ARCTIC_CKPT.exists():
        out_ckpt = out_root / "arctic/arctic_sf_allocentric/last.ckpt"
        _ensure_dir(out_ckpt.parent)
        shutil.copy2(ARCTIC_CKPT, out_ckpt)
        logger.info(f"  ARCTIC init ckpt (resnet50-arctic): copied ({ARCTIC_CKPT.stat().st_size/1e6:.0f} MB)")


def reduce_assembly(out_root: Path, n_train: int, n_val: int) -> None:
    """
    Subsets AssemblyHands three JSONs per split (ego_data, ego_calib, joint_3d)
    to only contain entries for the sampled images, and copies those images.

    Arguments:
        out_root -- root directory of the reduced dataset to write into
        n_train -- target number of train annotations to keep
        n_val -- target number of val annotations to keep
    """
    rng = random.Random(SEED)
    for split, n in (("train", n_train), ("val", n_val)):
        data_p = ASSEMBLY_ANNOT_DIR / split / f"assemblyhands_{split}_ego_data_v1-1.json"
        calib_p = ASSEMBLY_ANNOT_DIR / split / f"assemblyhands_{split}_ego_calib_v1-1.json"
        joint_p = ASSEMBLY_ANNOT_DIR / split / f"assemblyhands_{split}_joint_3d_v1-1.json"
        if not data_p.exists():
            logger.warning(f"  Assembly {split}: data JSON missing ({data_p})")
            continue

        with open(data_p) as f:
            d = json.load(f)
        n_keep = min(n, len(d["annotations"]))
        keep_anns = rng.sample(d["annotations"], n_keep)
        keep_image_ids = {a["image_id"] for a in keep_anns}
        keep_images = [im for im in d["images"] if im["id"] in keep_image_ids]
        keep_seq_frames = {(im["seq_name"], f"{im['frame_idx']:06d}") for im in keep_images}
        keep_seqs = {sq for sq, _ in keep_seq_frames}

        out_dir = out_root / "assembly/annotations" / split
        _ensure_dir(out_dir)
        with open(out_dir / data_p.name, "w") as f:
            json.dump({**d, "images": keep_images, "annotations": keep_anns}, f)

        if calib_p.exists():
            with open(calib_p) as f:
                calib = json.load(f)
            cal_out = {}
            for seq in keep_seqs:
                if seq not in calib["calibration"]:
                    continue
                entry = calib["calibration"][seq]
                ext_sub = {fk: v for fk, v in entry["extrinsics"].items()
                           if (seq, fk) in keep_seq_frames}
                cal_out[seq] = {"intrinsics": entry["intrinsics"], "extrinsics": ext_sub}
            with open(out_dir / calib_p.name, "w") as f:
                json.dump({**calib, "calibration": cal_out}, f)

        if joint_p.exists():
            with open(joint_p) as f:
                joints = json.load(f)
            j_out = {}
            for seq in keep_seqs:
                if seq not in joints["annotations"]:
                    continue
                j_out[seq] = {fk: v for fk, v in joints["annotations"][seq].items()
                              if (seq, fk) in keep_seq_frames}
            with open(out_dir / joint_p.name, "w") as f:
                json.dump({**joints, "annotations": j_out}, f)

        for fname in ("skeleton.txt", "README.txt"):
            _copy_file(ASSEMBLY_ANNOT_DIR / fname, out_root / "assembly/annotations" / fname)

        n_copied = 0
        for im in keep_images:
            src = ASSEMBLY_DIR / im["file_name"]
            dst = out_root / "assembly" / im["file_name"]
            if _copy_file(src, dst):
                n_copied += 1
        logger.info(f"  Assembly {split}: kept {n_keep} anns / {len(keep_images)} images, "f"copied {n_copied}")


def _visor_local_path(pickle_key: str) -> Path:
    """
    Mirrors src.datasets._common.visor_path so we know which local frame to copy.

    Arguments:
        pickle_key -- the absolute Linux path stored as the VISOR pickle key

    Returns:
        path -- local path to the corresponding VISOR frame under data/visor/rgb_frames
    """
    parts = pickle_key.replace("\\", "/").split("/")
    split, participant, _, filename = parts[-4:]
    return VISOR_FRAMES_DIR / split / participant / filename


def reduce_epic(out_root: Path, n_grasp: int, n_seg: int, n_handkps: int) -> None:
    """
    Subsets the EPIC grasp pickle, the EPIC seg trio (modal annot + bbox pkl + mask
    npz), and the EPIC-HandKps test pickle. Copies the union of referenced VISOR
    frames into data_reduced/visor/.

    Arguments:
        out_root -- root directory of the reduced dataset to write into
        n_grasp -- target number of EPIC grasp samples to keep
        n_seg -- target number of EPIC segmentation samples to keep
        n_handkps -- target number of EPIC-HandKps samples to keep
    """
    rng = random.Random(SEED)
    all_keys: set = set()

    grasp_src = EPIC_HANDS_DIR / "grasp_visor_train.pkl"
    if grasp_src.exists():
        with open(grasp_src, "rb") as f:
            grasp = pickle.load(f)
        keep = rng.sample(list(grasp.keys()), min(n_grasp, len(grasp)))
        out_pkl = out_root / "epic_hands/grasp_visor_train.pkl"
        _ensure_dir(out_pkl.parent)
        with open(out_pkl, "wb") as f:
            pickle.dump({k: grasp[k] for k in keep}, f)
        all_keys.update(keep)
        logger.info(f"  EPIC grasp: kept {len(keep)} / {len(grasp)} samples")

    modal_src = EPIC_HANDS_DIR / "modal_amodal_annot.pkl"
    masks_src = EPIC_HANDS_DIR / "visor_pred_masks_train.npz"
    if modal_src.exists() and masks_src.exists() and grasp_src.exists():
        with open(modal_src, "rb") as f:
            modal = pickle.load(f)

        masks = np.load(masks_src, allow_pickle=True)
        seg_pool = sorted(set(grasp) & set(modal) & set(masks.files))
        keep_seg = rng.sample(seg_pool, min(n_seg, len(seg_pool)))
        with open(out_root / "epic_hands/modal_amodal_annot.pkl", "wb") as f:
            pickle.dump({k: modal[k] for k in keep_seg}, f)

        out_grasp = out_root / "epic_hands/grasp_visor_train.pkl"
        with open(out_grasp, "rb") as f:
            cur_grasp = pickle.load(f)
        for k in keep_seg:
            cur_grasp.setdefault(k, grasp[k])
        with open(out_grasp, "wb") as f:
            pickle.dump(cur_grasp, f)
        valid_masks = _read_valid_npz_entries(masks, keep_seg)
        np.savez_compressed(out_root / "epic_hands/visor_pred_masks_train.npz", **valid_masks)
        all_keys.update(valid_masks.keys())
        logger.info(f"  EPIC seg: kept {len(valid_masks)} / {len(seg_pool)} samples")

    handkps_src = EPIC_HANDS_DIR / "hands_5000.pkl"
    if handkps_src.exists():
        with open(handkps_src, "rb") as f:
            handkps = pickle.load(f)
        keep = rng.sample(list(handkps.keys()), min(n_handkps, len(handkps)))
        with open(out_root / "epic_hands/hands_5000.pkl", "wb") as f:
            pickle.dump({k: handkps[k] for k in keep}, f)
        all_keys.update(keep)
        logger.info(f"  EPIC-HandKps: kept {len(keep)} / {len(handkps)} samples")

    n_copied = 0
    for key in all_keys:
        src = _visor_local_path(key)
        rel = src.relative_to(VISOR_FRAMES_DIR)
        if _copy_file(src, out_root / "visor/rgb_frames" / rel):
            n_copied += 1
    logger.info(f"  VISOR: copied {n_copied} / {len(all_keys)} unique frames")


def reduce_ego4d(out_root: Path, n_grasp: int, n_seg: int) -> None:
    """
    Subsets the Ego4D grasp pickle and its mask npz, then copies the referenced
    flat PNG frames.

    Arguments:
        out_root -- root directory of the reduced dataset to write into
        n_grasp -- target number of Ego4D grasp samples to keep
        n_seg -- target number of Ego4D segmentation samples to keep
    """
    rng = random.Random(SEED)
    all_keys: set = set()

    grasp_src = EGO4D_HANDS_DIR / "grasp_ego.pkl"
    masks_src = EGO4D_HANDS_DIR / "ego_blur_pred_masks.npz"
    if grasp_src.exists():
        with open(grasp_src, "rb") as f:
            grasp = pickle.load(f)
        keep_grasp = rng.sample(list(grasp.keys()), min(n_grasp, len(grasp)))
        if masks_src.exists():
            masks = np.load(masks_src, allow_pickle=True)
            seg_pool = sorted(set(grasp) & set(masks.files))
            keep_seg = rng.sample(seg_pool, min(n_seg, len(seg_pool)))
        else:
            masks = None
            keep_seg = []
        union = sorted(set(keep_grasp) | set(keep_seg))
        out_pkl = out_root / "ego4d_hands/grasp_ego.pkl"
        _ensure_dir(out_pkl.parent)
        with open(out_pkl, "wb") as f:
            pickle.dump({k: grasp[k] for k in union}, f)
        if masks is not None and keep_seg:
            valid_masks = _read_valid_npz_entries(masks, keep_seg)
            np.savez_compressed(out_root / "ego4d_hands/ego_blur_pred_masks.npz", **valid_masks)
            n_seg_kept = len(valid_masks)
        else:
            n_seg_kept = 0
        all_keys.update(union)
        logger.info(f"  Ego4D: kept {len(union)} samples ({len(keep_grasp)} grasp, "
                    f"{n_seg_kept} seg)")

    n_copied = 0
    for key in all_keys:
        src = EGO4D_FRAMES_DIR / Path(key).name
        if _copy_file(src, out_root / "ego4d/ego4d_frames" / Path(key).name):
            n_copied += 1
    logger.info(f"  Ego4D: copied {n_copied} / {len(all_keys)} frames")


def reduce_egoexo(out_root: Path, n: int) -> None:
    """
    Truncate Ego-Exo4D's pickle as it embeds the 256x256 image crops directly.

    Arguments:
        out_root -- root directory of the reduced dataset to write into
        n -- target number of Ego-Exo4D val samples to keep
    """
    src = EGO4D_HANDS_DIR / "joint_annotations_egoexo_val.pkl"
    if not src.exists():
        logger.warning(f"  Ego-Exo4D: pickle missing ({src})")
        return
    rng = random.Random(SEED)
    with open(src, "rb") as f:
        d = pickle.load(f)
    keep = rng.sample(list(d.keys()), min(n, len(d)))
    out_pkl = out_root / "ego4d_hands/joint_annotations_egoexo_val.pkl"
    _ensure_dir(out_pkl.parent)
    with open(out_pkl, "wb") as f:
        pickle.dump({k: d[k] for k in keep}, f)
    logger.info(f"  Ego-Exo4D: kept {len(keep)} / {len(d)} samples (images embedded)")


def reduce_h2o(out_root: Path, n: int) -> None:
    """
    Subsets H2O's pose_val.txt and copies the per frame rgb + hand_pose +
    hand_pose_mano txt files. cam_intrinsics.txt is copied once per camera dir.

    Arguments:
        out_root -- root directory of the reduced dataset to write into
        n -- target number of H2O val samples to keep
    """
    list_p = H2O_DIR / "label_split" / "pose_val.txt"
    if not list_p.exists():
        logger.warning(f"  H2O: pose_val.txt missing")
        return
    rng = random.Random(SEED)
    with open(list_p) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    keep = sorted(rng.sample(lines, min(n, len(lines))))
    out_list = out_root / "h2o/label_split/pose_val.txt"
    _ensure_dir(out_list.parent)
    with open(out_list, "w") as f:
        f.write("\n".join(keep) + "\n")

    n_copied = 0
    seen_cam_dirs: set = set()
    for line in keep:
        parts = line.split("/")
        seq_path = Path("subject3_ego", *parts[1:-2])
        frame = parts[-1].rsplit(".", 1)[0]
        for sub, ext in (("rgb", ".png"), ("hand_pose", ".txt"), ("hand_pose_mano", ".txt")):
            src = H2O_DIR / seq_path / sub / f"{frame}{ext}"
            dst = out_root / "h2o" / seq_path / sub / f"{frame}{ext}"
            if _copy_file(src, dst) and sub == "rgb":
                n_copied += 1
        if seq_path not in seen_cam_dirs:
            _copy_file(H2O_DIR / seq_path / "cam_intrinsics.txt",
                       out_root / "h2o" / seq_path / "cam_intrinsics.txt")
            seen_cam_dirs.add(seq_path)
    logger.info(f"  H2O: kept {len(keep)} samples, copied {n_copied} rgb frames "
                f"across {len(seen_cam_dirs)} camera dirs")


def main() -> None:
    out_root = Path(OUT_DIR).resolve()
    _ensure_dir(out_root)
    logger.info(f"Writing reduced dataset to {out_root}")

    logger.info("=== ARCTIC ===")
    reduce_arctic(out_root, ARCTIC_TRAIN, ARCTIC_VAL, ARCTIC_TRAIN_SEQS, ARCTIC_VAL_SEQS)
    logger.info("=== AssemblyHands ===")
    reduce_assembly(out_root, ASSEMBLY_TRAIN, ASSEMBLY_VAL)
    logger.info("=== EPIC + VISOR ===")
    reduce_epic(out_root, EPIC_GRASP, EPIC_SEG, EPIC_HANDKPS)
    logger.info("=== Ego4D ===")
    reduce_ego4d(out_root, EGO_GRASP, EGO_SEG)
    logger.info("=== Ego-Exo4D ===")
    reduce_egoexo(out_root, EGOEXO_VAL)
    logger.info("=== H2O ===")
    reduce_h2o(out_root, H2O_VAL)

    size_mb = _total_size_mb(out_root)
    logger.info("")
    logger.info(f"Reduced dataset built: {out_root}")
    logger.info(f"Total size: {size_mb:.0f} MB ({size_mb/1024:.1f} GB)")


if __name__ == "__main__":
    main()
