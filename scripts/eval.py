"""
Zero-shot evaluation.

Loads a WildHands checkpoint and runs the model over four eval datasets:
  assembly       - AssemblyHands ego val (3D + 2D + both hands)
  h2o            - H2O subject3 ego val (full MANO + 3D + 2D + both hands)
  epic_handkps   - EPIC-HandKps test (2D only)
  egoexo         - Ego-Exo4D val (3D + 2D, right hand only)

Reports per dataset metrics where applicable:
  mpjpe          - root-aligned mean per joint position error (MPJPE), mm
  mrrpe          - mean relative root position error between hands, mm
  pix_err        - pixel error for each joint in the IMG_RES patch, px

Metrics are averaged with np.nanmean across samples and then across hands.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from src import metrics as M
from src.backbones import build_backbone
from src.common.data_utils import unormalize_kp2d
from src.datasets.arctic import ArcticDataset
from src.datasets.assembly import AssemblyDataset
from src.datasets.ego_exo import EgoExo4DDataset
from src.datasets.epic_handkps import EPICHandKpsDataset
from src.datasets.h2o import H2ODataset
from src.model import WildHands
import pandas as pd


# ----------------------------- config -----------------------------
NUM_WORKERS = 4
IMG_RES = 224
PIX_ERR_CAP = 316.0   # maximum pixel error allowed (the 224px patch diagonal)
MRRPE_CAP = 600.0     # maximum MRRPE (mm) (physically plausible distance between left and right hands ~2 ft in frame)
OUT_CSV = Path("../results/ablation.csv")

EVAL_DATASETS = {
    "assembly":     (lambda: AssemblyDataset(split="val"),     ("mpjpe", "mrrpe", "pix_err")),
    "h2o":          (lambda: H2ODataset(split="val"),          ("mpjpe", "mrrpe", "pix_err")),
    "epic_handkps": (lambda: EPICHandKpsDataset(split="test"), ("pix_err",)),
    "egoexo":       (lambda: EgoExo4DDataset(split="val"),     ("mpjpe", "pix_err")),
    "arctic":       (lambda: ArcticDataset(split="val"),       ("mpjpe", "mrrpe", "pix_err")),
}

MODELS = {
    "resnet50":         Path("../logs/resnet50/0529-1138_bs32_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "resnet50-arctic":  Path("../logs/resnet50-arctic/0528-2256_bs32_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "resnet101":        Path("../logs/resnet101/0529-1544_bs32_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "mobilenet_v3_l":   Path("../logs/mobilenet_v3_l/0529-0143_bs32_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "convnext_l":       Path("../logs/convnext_l/0529-2101_bs8_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "mobilevit_s":      Path("../logs/mobilevit_s/0530-2320_bs8_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "swinv2_tiny":      Path("../logs/swinv2_tiny/0601-0917_bs8_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "swin_tiny":        Path("../logs/swin_tiny/0531-2240_bs16_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
    "wave_vit_s":       Path("../logs/wave_vit_s/0531-0325_bs16_lr1e-05_ep100_seed1/checkpoints/best.ckpt"),
}

BATCH_SIZE = 32


def load_checkpoint(model: WildHands, ckpt_path: Path) -> None:
    """
    Loads a trained WildHands checkpoint into the model.

    Arguments:
        model -- the WildHands model to load weights into
        ckpt_path -- path to the .ckpt file written by scripts/train.py
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")


def evaluate(model: WildHands, dataset, metric_names, device: torch.device, batch_size: int, num_workers: int) -> dict:
    """
    Runs the model over the dataset and aggregates the requested metrics.

    Arguments:
        model -- WildHands instance in eval mode
        dataset -- torch Dataset
        metric_names -- tuple of metric strings
        device -- target device
        batch_size -- DataLoader batch size
        num_workers -- DataLoader workers

    Returns:
        results -- dict mapping metric_name to float
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    per_sample = {m: {"r": [], "l": []} for m in metric_names if m != "mrrpe"}
    if "mrrpe" in metric_names:
        per_sample["mrrpe"] = []


    for inputs, targets, meta_info in loader:
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
        meta_info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in meta_info.items()}
        with torch.no_grad():
            pred = model(inputs, meta_info)

        for side in ("r", "l"):
            gt_j3d = targets[f"mano.j3d.full.{side}"]
            pred_j3d = pred[f"mano.j3d.cam.{side}"]
            jv2d = targets[f"joints_valid_{side}"]
            jv3d = targets.get(f"joints3d_valid_{side}", jv2d)

            if "mpjpe" in metric_names:
                per_sample["mpjpe"][side].extend(M.mpjpe(pred_j3d, gt_j3d, jv3d).tolist())
            if "pix_err" in metric_names:
                pred_pix = unormalize_kp2d(pred[f"mano.j2d.norm.{side}"], IMG_RES)
                gt_pix = unormalize_kp2d(targets[f"mano.j2d.norm.{side}"], IMG_RES)
                per_sample["pix_err"][side].extend(np.minimum(M.pix_err(pred_pix, gt_pix, jv2d), PIX_ERR_CAP).tolist())

        if "mrrpe" in metric_names:
            valid = targets["right_valid"] * targets["left_valid"]
            per_sample["mrrpe"].extend(np.minimum(M.mrrpe(
                pred["mano.j3d.cam.r"][:, 0], pred["mano.j3d.cam.l"][:, 0],
                targets["mano.j3d.full.r"][:, 0],
                targets["mano.j3d.full.l"][:, 0],
                valid,
            ), MRRPE_CAP).tolist())

    results = {}
    for m in metric_names:
        if m == "mrrpe":
            results[m] = float(np.nanmean(per_sample[m])) if per_sample[m] else float("nan")
        else:
            r_mean = float(np.nanmean(per_sample[m]["r"])) if per_sample[m]["r"] else float("nan")
            l_mean = float(np.nanmean(per_sample[m]["l"])) if per_sample[m]["l"] else float("nan")
            hand_means = [x for x in (r_mean, l_mean) if not np.isnan(x)]
            results[m] = float(np.mean(hand_means)) if hand_means else float("nan")

    return results


def _save_results_csv(path: Path, backbone: str, all_results: dict) -> None:
    """
    Appends this run's metrics to a tidy CSV. Existing rows for the same 
    backbone are dropped first, so re-running a backbone overwrites its rows.

    Arguments:
        path -- destination CSV path
        backbone -- backbone name tag stored on every row of this run
        all_results -- dict mapping dataset name
    """
    rows = [
        {"backbone": backbone, "dataset": ds, "metric": m, "value": v}
        for ds, metrics in all_results.items()
        for m, v in metrics.items()
    ]
    new = pd.DataFrame(rows, columns=["backbone", "dataset", "metric", "value"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["backbone"] != backbone], new], ignore_index=True)
    new.to_csv(path, index=False)
    logger.info(f"wrote {len(rows)} rows for backbone {backbone!r} to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for backbone, ckpt in MODELS.items():
        model = WildHands(build_backbone(backbone, pretrained=False), build_backbone(backbone, pretrained=False)).to(device).eval()
        load_checkpoint(model, ckpt)

        all_results = {}
        for name in EVAL_DATASETS:
            builder, metric_names = EVAL_DATASETS[name]
            logger.info(f"=== eval: {name} ===")
            try:
                ds = builder()
            except FileNotFoundError as e:
                logger.warning(f"skipping {name}: missing data file {getattr(e, 'filename', None) or e}")
                continue
            all_results[name] = evaluate(model, ds, metric_names, device, BATCH_SIZE, NUM_WORKERS)
            for m, v in all_results[name].items():
                logger.info(f"  {m:14s} = {v:8.3f}")

        if OUT_CSV:
            _save_results_csv(OUT_CSV, backbone, all_results)

        metric_set = sorted({m for d in all_results.values() for m in d})
        header = f"{'dataset':14s}  " + "  ".join(f"{m:>12s}" for m in metric_set)
        print()
        print(header)
        print("-" * len(header))
        for name in EVAL_DATASETS:
            if name not in all_results:
                continue
            row = f"{name:14s}  " + "  ".join(f"{all_results[name].get(m, float('nan')):>12.3f}" for m in metric_set)
            print(row)


if __name__ == "__main__":
    main()
