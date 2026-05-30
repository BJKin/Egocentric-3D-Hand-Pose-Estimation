"""
Extract specific frames from Ego4D videos to match the WildHands grasp_ego.pkl keys.
"""

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import cv2


# ----------------------------- config -----------------------------
STRIDE = 15
PKL_PATH = Path("../../data/ego4d_hands/grasp_ego.pkl")
VIDEO_DIR = Path("../../data/ego4d/v1/full_scale")
OUT_DIR = Path("../../data/ego4d/ego4d_frames")


def load_needed_frames(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    needed = defaultdict(dict)
    for key, val in data.items():
        fn = os.path.basename(key)
        stem = fn[:-4]
        uid, _, idx_str = stem.rpartition("_")
        if len(uid) != 36 or not idx_str.isdigit():
            print(f"  skipping malformed key: {key}")
            continue
        needed[uid][int(idx_str)] = val
    return needed


def pkl_idx_to_video_frame(pkl_idx):
    """Convert 1-indexed pkl sample number to 0-indexed original video frame."""
    return (pkl_idx - 1) * STRIDE


def extract_video(video_path, idx_to_meta, out_dir, uid):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  FAILED to open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pkl_idxs = set(idx_to_meta.keys())
    print(f"  video: {w}x{h}, {total_frames} frames | "
          f"{len(pkl_idxs)} samples, stride={STRIDE}")

    target_frames = {pkl_idx_to_video_frame(p): p for p in pkl_idxs}

    out_of_range = [tf for tf in target_frames if tf >= total_frames]
    if out_of_range:
        print(f"  WARNING: {len(out_of_range)} target frames exceed video length; will skip")
        for tf in out_of_range:
            del target_frames[tf]

    written = 0
    cv2_frame_n = 0
    remaining = dict(target_frames)
    while remaining:
        ret, frame = cap.read()
        if not ret:
            break
        if cv2_frame_n in remaining:
            pkl_idx = remaining.pop(cv2_frame_n)
            out_path = out_dir / f"{uid}_{pkl_idx}.png"
            cv2.imwrite(str(out_path), frame)
            written += 1
        cv2_frame_n += 1

    cap.release()
    if remaining:
        print(f"  missed {len(remaining)} frames (video ended early?)")
    print(f"  wrote {written}")
    return written


def main():
    if not PKL_PATH.exists():
        sys.exit(f"pkl not found: {PKL_PATH}")
    if not VIDEO_DIR.exists():
        sys.exit(f"video-dir not found: {VIDEO_DIR}")

    print(f"Loading {PKL_PATH}...")
    needed = load_needed_frames(PKL_PATH)
    total = sum(len(v) for v in needed.values())
    print(f"  {len(needed)} videos, {total} total samples\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {OUT_DIR}\n")

    uids = sorted(needed.keys())

    total_written = 0
    for i, uid in enumerate(uids, 1):
        video_path = VIDEO_DIR / f"{uid}.mp4"
        print(f"[{i}/{len(uids)}] {uid}")
        if not video_path.exists():
            print(f"  MISSING video: {video_path}")
            continue
        written = extract_video(video_path, needed[uid], OUT_DIR, uid)
        total_written += written

    print(f"\nDone. Wrote {total_written} new frames to {OUT_DIR}")


if __name__ == "__main__":
    main()