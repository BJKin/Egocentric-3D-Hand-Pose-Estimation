"""
Training loop.
Runs a full epoch loop with TensorBoard logging and last+best checkpointing.
Every backbone is initialized from its timm/Hugging Face ImageNet weights.
"""
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backbones import build_backbone
from src.common.mano import build_mano_aa
from src.datasets.arctic import ArcticDataset
from src.datasets.assembly import AssemblyDataset
from src.datasets.ego_grasp import Ego4DGraspDataset
from src.datasets.ego_seg import Ego4DSegDataset
from src.datasets.epic_grasp import EPICGraspDataset
from src.datasets.epic_seg import EPICSegDataset
from src.losses import compute_loss
from src.model import WildHands
from src.common.process import process_data


# ----------------------------- config -----------------------------
NUM_WORKERS = 4
LOG_EVERY = 50
SEED = 1

# possible backbones: resnet50, resnet50-arctic, resnet101, mobilenet_v3_l, convnext_l, mobilevit_s, swinv2_tiny, swin_tiny
BACKBONE = "swinv2_b"
# possible training sets: ArcticDataset, AssemblyDataset, EPICGraspDataset, EPICSegDataset, Ego4DGraspDataset, Ego4DSegDataset
TRAIN_DATASETS = [ArcticDataset, AssemblyDataset, EPICGraspDataset, EPICSegDataset, Ego4DGraspDataset, Ego4DSegDataset]

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-5


def forward_step(model: WildHands, mano_r, mano_l, batch: tuple, device: torch.device):
    """
    Runs one forward pass: moves batch to device, derive full MANO/camera GT targets, 
    runs the model, computes the loss dict.

    Arguments:
        model -- WildHands instance
        mano_r -- right-hand MANO layer
        mano_l -- left-hand MANO layer
        batch -- (inputs, targets, meta_info) from the dataloader
        device -- target device

    Returns:
        loss_dict -- mapping loss_name -> (scalar tensor, weight)
        total -- weighted scalar sum of all loss terms
        batch_size -- number of samples in the batch
    """
    inputs, targets, meta_info = batch
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
    meta_info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in meta_info.items()}
    targets = process_data(mano_r, mano_l, targets, meta_info["intrinsics"])
    pred = model(inputs, meta_info)
    loss_dict = compute_loss(pred, targets, meta_info)
    total = sum(loss * weight for loss, weight in loss_dict.values()).squeeze()
    return loss_dict, total, inputs["img"].shape[0]


def validate(model: WildHands, mano_r, mano_l, loader: DataLoader, device: torch.device) -> tuple:
    """
    Runs the model over the validation set and returns the average weighted loss
    plus the average of each loss term.

    Arguments:
        model -- WildHands instance
        mano_r, mano_l -- MANO layers
        loader -- validation DataLoader
        device -- target device

    Returns:
        avg_total -- mean weighted loss across processed samples
        avg_terms -- dict mapping loss_name -> mean loss for each term across batches
    """
    was_training = model.training
    model.eval()
    total_sum, total_n = 0.0, 0
    term_sums = {}
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            loss_dict, total, B = forward_step(model, mano_r, mano_l, batch, device)
            total_sum += total.item() * B
            total_n += B
            for k, (loss, _) in loss_dict.items():
                term_sums[k] = term_sums.get(k, 0.0) + loss.item()
            n_batches += 1
    if was_training:
        model.train()
    avg_terms = {k: v / max(n_batches, 1) for k, v in term_sums.items()}
    return total_sum / max(total_n, 1), avg_terms


def _save_checkpoint(path: Path, model: WildHands, optimizer, epoch: int, global_step: int, best_val_loss: float) -> None:
    """
    Writes a checkpoint to given path, replacing any existing file there.

    Arguments:
        path -- destination .ckpt path
        model -- WildHands instance whose state_dict is saved
        optimizer -- optimizer whose state_dict is saved
        epoch -- current epoch index
        global_step -- current global step
        best_val_loss -- best validation loss seen so far
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, tmp)
    tmp.replace(path)


def train() -> None:
    logger.info(f"=== WildHands training: backbone={BACKBONE} bs={BATCH_SIZE} lr={LR:g} epochs={EPOCHS} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info(f"device: {device}, seed: {SEED}")

    run_name = f"{datetime.now():%m%d-%H%M}_bs{BATCH_SIZE}_lr{LR:g}_ep{EPOCHS}_seed{SEED}"
    log_dir = Path("../logs") / BACKBONE / run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("building datasets...")
    train_ds = ConcatDataset([make("train") for make in TRAIN_DATASETS])
    val_ds = ArcticDataset(split="val")

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    if NUM_WORKERS > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    logger.info(f"building model ({BACKBONE})...")
    model = WildHands(build_backbone(BACKBONE, pretrained=True), build_backbone(BACKBONE, pretrained=True)).to(device)

    mano_r = build_mano_aa(is_rhand=True).to(device)
    mano_l = build_mano_aa(is_rhand=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter(log_dir=str(log_dir))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"model: WildHands({BACKBONE}) params={n_params:.2f}M  "
                f"train={len(train_ds)} val={len(val_ds)}")
    logger.info(f"train datasets: {[d.__name__ for d in TRAIN_DATASETS]}")
    logger.info(f"logs+checkpoints -> {log_dir}")

    logger.info(f"starting training loop: {EPOCHS} epochs, {len(train_loader)} steps/epoch")
    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS - 1}", leave=True)
        for batch in pbar:
            loss_dict, total, _ = forward_step(model, mano_r, mano_l, batch, device)
            optimizer.zero_grad()
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 150.0)
            optimizer.step()

            running_loss += total.item()
            n_seen += 1
            pbar.set_postfix(loss=f"{total.item():.4f}", avg=f"{running_loss / n_seen:.4f}")

            if global_step % LOG_EVERY == 0:
                writer.add_scalar("train/loss", total.item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                for k, (loss, _) in loss_dict.items():
                    writer.add_scalar(f"train/{k}", loss.item(), global_step)

            global_step += 1
        pbar.close()
        train_loss = running_loss / max(n_seen, 1)

        val_loss, val_terms = validate(model, mano_r, mano_l, val_loader, device)
        writer.add_scalar("val/loss", val_loss, global_step)
        for k, v in val_terms.items():
            writer.add_scalar(f"val/{k}", v, global_step)
        logger.info(f"epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        _save_checkpoint(ckpt_dir / "last.ckpt", model, optimizer, epoch, global_step, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(ckpt_dir / "best.ckpt", model, optimizer, epoch, global_step, best_val_loss)
            logger.info(f"new best val_loss={best_val_loss:.4f} (epoch {epoch})")

    writer.close()
    logger.info("training done")


if __name__ == "__main__":
    train()
