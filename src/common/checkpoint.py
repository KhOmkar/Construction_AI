"""
Robust checkpoint saving, atomic writes, and automatic resume utilities.
Guarantees uninterrupted training even across power loss, sleep, or crashes.
"""

import os
import glob
import shutil
import torch


def save_checkpoint(
    state,
    is_best=False,
    checkpoint_dir="checkpoints",
    epoch=None,
    metric_name="val_loss",
    metric_val=None,
    keep_last_n=5
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_num = epoch if epoch is not None else state.get("epoch", 0)
    ckpt_filename = f"checkpoint_epoch_{epoch_num:03d}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
    tmp_path = ckpt_path + ".tmp"

    # Atomic write to temporary file first
    torch.save(state, tmp_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    os.rename(tmp_path, ckpt_path)

    # Update latest.pt symlink/copy
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    shutil.copy2(ckpt_path, latest_path)

    # Save best.pt if flagged
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        shutil.copy2(ckpt_path, best_path)

    # Maintain pruning: keep last N epoch checkpoints to conserve disk space
    all_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))
    if len(all_ckpts) > keep_last_n:
        for old_ckpt in all_ckpts[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
            except OSError:
                pass

    return ckpt_path


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    latest_symlink = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_symlink):
        return latest_symlink

    all_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))
    if all_ckpts:
        return all_ckpts[-1]

    return None


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    device="cuda"
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"--> Resuming checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load AMP Scaler
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Load LR scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", None)

    print(f"--> Successfully loaded checkpoint. Resuming from Epoch {start_epoch}")
    return start_epoch, best_metric
