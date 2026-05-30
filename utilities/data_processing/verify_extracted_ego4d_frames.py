"""
Verify extracted frames against pkl bounding boxes.
"""
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
import cv2


# ----------------------------- config -----------------------------
PKL_PATH = Path("../../data/ego4d_hands/grasp_ego.pkl")
FRAMES_DIR = Path("../../data/ego4d/ego4d_frames")
OUT_DIR = Path("../data_verification/ego4d_verify_extracted")
N_SAMPLES = 10
SEED = 42


def load_pkl_by_uid(pkl_path):
    """Group pkl entries by UID. Returns {uid: {pkl_idx: meta}}."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    by_uid = defaultdict(dict)
    for key, meta in data.items():
        fn = os.path.basename(key)
        stem = fn[:-4]
        uid, _, idx_str = stem.rpartition("_")
        if len(uid) != 36 or not idx_str.isdigit():
            continue
        by_uid[uid][int(idx_str)] = meta
    return by_uid


def parse_bbox(bbox):
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def annotate_frame(frame, meta, uid, pkl_idx, w, h):
    """Draw bboxes and labels onto a frame copy. Returns annotated frame."""
    annotated = frame.copy()

    for side, color in [("right_bbox", (0, 255, 0)),
                        ("left_bbox", (0, 0, 255))]:
        bb = parse_bbox(meta.get(side))
        if bb is None:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bb]

        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w - 1, x1), min(h - 1, y1)
        cv2.rectangle(annotated, (x0c, y0c), (x1c, y1c), color, 3)

        grasp_key = side.replace("_bbox", "_grasp")
        label = f"{side.split('_')[0]}: {meta.get(grasp_key, '?')}"
        label_y = max(y0c - 8, 20)
        cv2.putText(annotated, label, (x0c, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            cv2.putText(annotated, "BBOX OUT OF FRAME!", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    header = f"{uid[:8]} pkl_idx={pkl_idx} frame={w}x{h}"
    cv2.putText(annotated, header, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, header, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    return annotated


def verify_uid(uid, idx_to_meta, frames_dir, out_dir, n_samples, rng):
    """Sample frames for one UID and write annotated versions."""
    available = []
    for pkl_idx in idx_to_meta:
        png_path = frames_dir / f"{uid}_{pkl_idx}.png"
        if png_path.exists():
            available.append((pkl_idx, png_path))

    if not available:
        print(f"  {uid}: NO extracted PNGs found, skipping")
        return 0, 0

    sample_n = min(n_samples, len(available))
    samples = rng.sample(available, sample_n)

    written = 0
    for pkl_idx, png_path in samples:
        frame = cv2.imread(str(png_path))
        if frame is None:
            print(f"  failed to read {png_path}")
            continue
        h, w = frame.shape[:2]
        meta = idx_to_meta[pkl_idx]
        annotated = annotate_frame(frame, meta, uid, pkl_idx, w, h)
        out_path = out_dir / f"verify_{uid}_{pkl_idx:05d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        written += 1

    print(f"  {uid}: extracted={len(available)}, sampled={written}")
    return len(available), written


def main():
    if not PKL_PATH.exists():
        sys.exit(f"pkl not found: {PKL_PATH}")
    if not FRAMES_DIR.exists():
        sys.exit(f"frames-dir not found: {FRAMES_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"Loading {PKL_PATH}...")
    by_uid = load_pkl_by_uid(PKL_PATH)
    print(f"  {len(by_uid)} UIDs in pkl")

    uids = sorted(by_uid.keys())

    print(f"\nWriting annotated samples to {OUT_DIR}\n")
    total_avail = total_written = 0
    for i, uid in enumerate(uids, 1):
        print(f"[{i}/{len(uids)}]")
        a, w = verify_uid(uid, by_uid[uid], FRAMES_DIR, OUT_DIR,
                          N_SAMPLES, rng)
        total_avail += a
        total_written += w

    print(f"\nDone. Verified {total_written} frames from {total_avail} extracted "
          f"across {len(uids)} videos.")
    print(f"Open files in {OUT_DIR} and visually confirm bboxes land on "
          f"hand-object regions.")


if __name__ == "__main__":
    main()