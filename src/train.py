"""
Unified Training Runner for Construction Site AI.
Supports:
  1. Autoencoder (Anomaly Detection on Compliant Crops)
  2. Conditional VAE (Smooth Latent Space & Interpolation)
  3. Conditional GAN (DiffAugment + Spectral Norm Violation Synthesis)
  4. Class-Conditional Diffusion (DDPM + Classifier-Free Guidance)
  5. VQ-VAE Tokenizer & ImageGPT Transformer

Features:
  - Mixed precision FP16 with torch.amp.autocast and GradScaler
  - Logging contract: config.yaml, env.json, train_log.csv, epoch_log.csv, eval_metrics.json, timing.json
  - Visual grid output in runs/{run_dir}/samples/
  - Atomic checkpoint save every epoch & automatic resume
  - Smoke test mode (--smoke-test) for 2-epoch validation and runtime projection
"""

import os
import sys
import time
import math
import copy
import json
import shutil
import torchvision.utils as vutils
import argparse
import platform
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from src.common.dataset import get_crop_dataloader
from src.common.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.common.visualization import save_image_grid
from src.common.metrics import compute_reconstruction_metrics
from src.common.diffaugment import diff_augment
from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim_torchmetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Construction Site AI Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch dummy smoke test")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def log_environment(run_dir, device_str):
    os.makedirs(run_dir, exist_ok=True)
    is_cuda = "cuda" in device_str and torch.cuda.is_available()
    env_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": is_cuda,
        "cuda_version": torch.version.cuda if is_cuda else "N/A",
        "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if is_cuda else 0,
    }
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    return env_info


def save_config_copy(cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ==========================================
# 1. AUTOENCODER TRAINER
# ==========================================
def train_autoencoder(cfg, run_dir, device, smoke_test=False, resume=True):
    from src.models.ae import AnomalyAutoencoder

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    epochs = 2 if smoke_test else train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 128)
    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    ssim_weight = float(model_cfg.get("ssim_weight", 0.20))
    filter_class = train_cfg.get("filter_class", 0)
    norm_mode = data_cfg.get("normalize_mode", "sigmoid")

    train_loader = get_crop_dataloader(data_cfg["train_npz"], batch_size=batch_size, shuffle=True, filter_class=filter_class, normalize_mode=norm_mode)
    val_loader = get_crop_dataloader(data_cfg["val_npz"], batch_size=batch_size, shuffle=False, filter_class=filter_class, normalize_mode=norm_mode)

    model = AnomalyAutoencoder(
        in_channels=model_cfg.get("in_channels", 3),
        bottleneck_dim=model_cfg.get("bottleneck_dim", 256),
        channels=tuple(model_cfg.get("channels", [64, 128, 256, 512]))
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(device)

    start_epoch = 1
    best_val_loss = float("inf")
    epoch_logs = []
    train_logs = []

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Resume check
    if resume:
        latest = find_latest_checkpoint(run_dir)
        if latest:
            start_epoch, best_metric = load_checkpoint(latest, model, optimizer=optimizer, scaler=scaler, scheduler=scheduler, device=device)
            if best_metric is not None:
                best_val_loss = best_metric
            print(f"Resumed AE from epoch {start_epoch-1} (best val loss: {best_val_loss:.6f})")

    global_step = 0
    t_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t_ep_start = time.time()
        running_loss = 0.0
        n_samples = 0

        for b_idx, batch in enumerate(train_loader):
            if smoke_test and b_idx >= 5:
                break
            images = batch["image"].to(device)
            optimizer.zero_grad()

            with autocast(device):
                recons, _ = model(images)
                loss_mse = F.mse_loss(recons, images)
                loss_l1 = F.l1_loss(recons, images)
                loss = loss_mse + 0.1 * loss_l1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(images)
            n_samples += len(images)
            global_step += 1

            train_logs.append({
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "loss_mse": loss_mse.item(),
                "loss_l1": loss_l1.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })

        scheduler.step()
        ep_duration = time.time() - t_ep_start
        epoch_times.append(ep_duration)
        train_loss = running_loss / max(1, n_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_mse = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if smoke_test and b_idx >= 5:
                    break
                images = batch["image"].to(device)
                with autocast(device):
                    recons, _ = model(images)
                    l_mse = F.mse_loss(recons, images)
                    l_l1 = F.l1_loss(recons, images)
                    l_tot = l_mse + 0.1 * l_l1

                ssim_val = compute_ssim_torchmetrics(recons.float(), images.float(), data_range=1.0).item()
                val_loss += l_tot.item() * len(images)
                val_mse += l_mse.item() * len(images)
                val_ssim += ssim_val * len(images)
                val_samples += len(images)

                if b_idx == 0:
                    # Save reconstruction comparison grid (real vs recon)
                    paired = torch.cat([images[:8], recons[:8]], dim=0)
                    save_image_grid(paired, os.path.join(samples_dir, f"recon_epoch_{epoch:03d}.png"), nrow=8, normalize=False)

        val_loss /= max(1, val_samples)
        val_mse /= max(1, val_samples)
        val_ssim /= max(1, val_samples)
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

        print(f"AE Epoch {epoch:03d}/{epochs:03d} [{ep_duration:.2f}s] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} (MSE: {val_mse:.5f}, SSIM: {val_ssim:.4f}) | VRAM: {vram_mb:.1f} MB")

        is_best = (not math.isnan(val_loss)) and (val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "val_mse": val_mse,
                "val_ssim": val_ssim,
                "best_metric": best_val_loss
            },
            is_best=is_best,
            checkpoint_dir=run_dir,
            epoch=epoch,
            keep_last_n=5
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_ssim": val_ssim,
            "epoch_sec": ep_duration,
            "vram_mb": vram_mb
        })

    # Save CSV logs
    pd.DataFrame(epoch_logs).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
    pd.DataFrame(train_logs).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)


    avg_ep_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
    timing_info = {
        "seconds_per_epoch": avg_ep_sec,
        "projected_100_epochs_sec": avg_ep_sec * 100,
        "projected_100_epochs_hours": round((avg_ep_sec * 100) / 3600, 2),
        "total_elapsed_sec": time.time() - t_start,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    }
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing_info, f, indent=2)

    eval_summary = {
        "model": "autoencoder",
        "best_val_loss": float(best_val_loss),
        "epochs_completed": epochs,
        "smoke_test": smoke_test
    }

    # If full training (not smoke test) or if best checkpoint exists, run full anomaly detection evaluation
    if not smoke_test:
        best_ckpt_path = os.path.join(run_dir, "best.pt")
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
        anomaly_metrics = evaluate_autoencoder_anomaly(model, data_cfg, run_dir, device)
        eval_summary.update(anomaly_metrics)

    with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    return timing_info


def evaluate_autoencoder_anomaly(model, data_cfg, run_dir, device):
    from src.common.visualization import plot_reconstruction_trio
    from src.common.metrics import compute_anomaly_metrics

    print("\n--- Running Autoencoder Anomaly Detection Evaluation ---")
    val_loader = get_crop_dataloader(data_cfg["val_npz"], batch_size=128, shuffle=False, filter_class=None, normalize_mode="sigmoid")
    test_loader = get_crop_dataloader(data_cfg["test_npz"], batch_size=128, shuffle=False, filter_class=None, normalize_mode="sigmoid")

    model.eval()

    # 1. Validation: tune optimal threshold tau*
    val_scores = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].cpu().numpy()
            recons, _ = model(imgs)
            diff = (imgs - recons) ** 2
            mse_per_crop = diff.view(imgs.size(0), -1).mean(dim=1).cpu().numpy()
            val_scores.extend(mse_per_crop)
            val_labels.extend((lbls > 0).astype(int))

    val_metrics = compute_anomaly_metrics(val_scores, val_labels)
    opt_threshold = val_metrics["optimal_threshold"]
    print(f"Optimal Anomaly Threshold tau* tuned on Val: {opt_threshold:.6f} (Val AUROC: {val_metrics['auroc']:.4f}, F1: {val_metrics['best_f1']:.4f})")

    # 2. Test: evaluate on test split with frozen tau*
    test_scores = []
    test_labels = []
    sample_crops_by_class = {}

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].cpu().numpy()
            recons, _ = model(imgs)
            diff = (imgs - recons) ** 2
            mse_per_crop = diff.view(imgs.size(0), -1).mean(dim=1).cpu().numpy()
            test_scores.extend(mse_per_crop)
            test_labels.extend((lbls > 0).astype(int))

            for i in range(len(lbls)):
                cid = int(lbls[i])
                if cid not in sample_crops_by_class:
                    sample_crops_by_class[cid] = (imgs[i].cpu(), recons[i].cpu(), float(mse_per_crop[i]))

    test_metrics = compute_anomaly_metrics(test_scores, test_labels)
    test_preds = (np.array(test_scores) > opt_threshold).astype(int)
    test_targets = np.array(test_labels)
    acc = float(np.mean(test_preds == test_targets))

    # Save trios for all 4 classes
    class_names = {0: "compliant", 1: "no_helmet", 2: "no_vest", 3: "both_missing"}
    trios_dir = os.path.join(run_dir, "anomaly_trios")
    os.makedirs(trios_dir, exist_ok=True)
    for cid, (orig, recon, err) in sample_crops_by_class.items():
        cname = class_names.get(cid, f"class_{cid}")
        trio_path = os.path.join(trios_dir, f"trio_{cname}.png")
        plot_reconstruction_trio(orig, recon, trio_path, threshold=opt_threshold, error_val=err)

    eval_results = {
        "optimal_threshold": float(opt_threshold),
        "val_auroc": float(val_metrics["auroc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "val_best_f1": float(val_metrics["best_f1"]),
        "test_auroc": float(test_metrics["auroc"]),
        "test_auprc": float(test_metrics["auprc"]),
        "test_accuracy": acc,
        "test_fpr_at_95_tpr": float(test_metrics["fpr_at_95_tpr"]),
        "test_f1_at_val_threshold": float(test_metrics["best_f1"]),
        "sample_counts": {
            "test_total": len(test_labels),
            "test_compliant": int(np.sum(test_targets == 0)),
            "test_violations": int(np.sum(test_targets == 1))
        }
    }

    print(f"Test AUROC: {eval_results['test_auroc']:.4f} | Test AUPRC: {eval_results['test_auprc']:.4f} | Test Acc: {eval_results['test_accuracy']*100:.2f}% | FPR@95%TPR: {eval_results['test_fpr_at_95_tpr']:.4f}")
    with open(os.path.join(run_dir, "anomaly_eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    return eval_results


# ==========================================
# 2. CONDITIONAL VAE TRAINER
# ==========================================
def train_cvae(cfg, run_dir, device, smoke_test=False, resume=True):
    from src.models.vae import ConditionalVAE

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    epochs = 2 if smoke_test else train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 128)
    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    target_beta = float(model_cfg.get("beta", 1.0))
    kl_warmup_epochs = train_cfg.get("kl_warmup_epochs", 10)
    norm_mode = data_cfg.get("normalize_mode", "sigmoid")

    train_loader = get_crop_dataloader(data_cfg["train_npz"], batch_size=batch_size, shuffle=True, normalize_mode=norm_mode)
    val_loader = get_crop_dataloader(data_cfg["val_npz"], batch_size=batch_size, shuffle=False, normalize_mode=norm_mode)

    model = ConditionalVAE(
        in_channels=model_cfg.get("in_channels", 3),
        latent_dim=model_cfg.get("latent_dim", 128),
        num_classes=model_cfg.get("num_classes", 4),
        channels=tuple(model_cfg.get("channels", [64, 128, 256, 512]))
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(device)

    start_epoch = 1
    best_val_loss = float("inf")
    epoch_logs = []
    train_logs = []

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    if resume:
        latest = find_latest_checkpoint(run_dir)
        if latest:
            start_epoch, best_metric = load_checkpoint(latest, model, optimizer=optimizer, scaler=scaler, scheduler=scheduler, device=device)
            if best_metric is not None:
                best_val_loss = best_metric
            print(f"Resumed C-VAE from epoch {start_epoch-1}")

    global_step = 0
    t_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t_ep_start = time.time()
        # Linear KL warmup schedule
        beta = target_beta * min(1.0, epoch / max(1, kl_warmup_epochs))
        running_loss = 0.0
        n_samples = 0

        for b_idx, batch in enumerate(train_loader):
            if smoke_test and b_idx >= 5:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            with autocast(device):
                recons, mu, logvar, _ = model(images, labels)
                recon_loss = F.mse_loss(recons, images, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(images)
            n_samples += len(images)
            global_step += 1

            train_logs.append({
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "beta": beta
            })

        scheduler.step()
        ep_duration = time.time() - t_ep_start
        epoch_times.append(ep_duration)
        train_loss = running_loss / max(1, n_samples)

        # Validation & sampling
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if smoke_test and b_idx >= 5:
                    break
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                with autocast(device):
                    recons, mu, logvar, _ = model(images, labels)
                    r_loss = F.mse_loss(recons, images, reduction="mean")
                    k_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    l_tot = r_loss + beta * k_loss
                val_loss += l_tot.item() * len(images)
                val_samples += len(images)

            # Generate conditional sample grid: 4 crops per class (16 total)
            sample_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=device)
            z_sample = torch.randn(len(sample_labels), model.latent_dim, device=device)
            gen_samples = model.decode(z_sample, sample_labels)
            save_image_grid(gen_samples, os.path.join(samples_dir, f"sample_epoch_{epoch:03d}.png"), nrow=4, normalize=False)

        val_loss /= max(1, val_samples)
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        print(f"C-VAE Epoch {epoch:03d}/{epochs:03d} [{ep_duration:.2f}s] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} (beta: {beta:.2f}) | VRAM: {vram_mb:.1f} MB")

        is_best = (not math.isnan(val_loss)) and (val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "best_metric": best_val_loss
            },
            is_best=is_best,
            checkpoint_dir=run_dir,
            epoch=epoch,
            keep_last_n=5
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "beta": beta,
            "epoch_sec": ep_duration,
            "vram_mb": vram_mb
        })

    pd.DataFrame(epoch_logs).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
    pd.DataFrame(train_logs).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)


    avg_ep_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
    timing_info = {
        "seconds_per_epoch": avg_ep_sec,
        "projected_100_epochs_sec": avg_ep_sec * 100,
        "projected_100_epochs_hours": round((avg_ep_sec * 100) / 3600, 2),
        "total_elapsed_sec": time.time() - t_start,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    }
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing_info, f, indent=2)

    return timing_info


def export_synthetic_crops(model_name, generate_fn, num_per_class=250, out_base="outputs/synthetic"):
    """
    Exports synthetic crops for all safety classes into organized folders:
    outputs/synthetic/{model_name}/{class_name}/{class_name}_{i:04d}.png
    along with a manifest.csv
    """
    class_map = {0: "compliant", 1: "no_helmet", 2: "no_vest", 3: "both_missing"}
    out_dir = os.path.join(out_base, model_name)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    print(f"\n--- Exporting Synthetic Crops for {model_name.upper()} ({num_per_class} per class) ---")
    for cid, cname in class_map.items():
        cdir = os.path.join(out_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        imgs = generate_fn(cid, num_per_class)
        for i in range(len(imgs)):
            fname = f"{cname}_{i:04d}.png"
            fpath = os.path.join(cdir, fname)
            vutils.save_image(imgs[i], fpath, normalize=False)
            manifest.append({
                "filename": fname,
                "relative_path": os.path.join(cname, fname).replace("\\", "/"),
                "class_id": cid,
                "class_name": cname,
                "generator": model_name
            })
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"Exported {len(manifest)} synthetic images to {out_dir} with manifest.csv\n")


# ==========================================
# 3. CONDITIONAL GAN TRAINER
# ==========================================
def train_cgan(cfg, run_dir, device, smoke_test=False, resume=True):
    from src.models.gan import ConditionalGenerator, ConditionalDiscriminator

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    epochs = 2 if smoke_test else train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 64)
    lr_g = float(train_cfg.get("lr_g", 0.0002))
    lr_d = float(train_cfg.get("lr_d", 0.0002))
    betas = tuple(train_cfg.get("betas", [0.5, 0.999]))
    latent_dim = model_cfg.get("latent_dim", 128)
    num_classes = model_cfg.get("num_classes", 4)
    policy = model_cfg.get("diffaugment", "color,translation,cutout")
    norm_mode = data_cfg.get("normalize_mode", "tanh")

    train_loader = get_crop_dataloader(data_cfg["train_npz"], batch_size=batch_size, shuffle=True, normalize_mode=norm_mode)

    netG = ConditionalGenerator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    netD = ConditionalDiscriminator(num_classes=num_classes).to(device)

    optG = Adam(netG.parameters(), lr=lr_g, betas=betas)
    optD = Adam(netD.parameters(), lr=lr_d, betas=betas)
    scalerG = GradScaler(device)
    scalerD = GradScaler(device)

    start_epoch = 1
    epoch_logs = []
    train_logs = []

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Fixed noise & labels for tracking progress across epochs
    fixed_noise = torch.randn(16, latent_dim, device=device)
    fixed_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=device)

    if resume:
        latest = find_latest_checkpoint(run_dir)
        if latest:
            ckpt = torch.load(latest, map_location=device)
            netG.load_state_dict(ckpt["netG_state_dict"])
            netD.load_state_dict(ckpt["netD_state_dict"])
            optG.load_state_dict(ckpt["optG_state_dict"])
            optD.load_state_dict(ckpt["optD_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed CGAN from epoch {ckpt['epoch']}")

    global_step = 0
    t_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, epochs + 1):
        t_ep_start = time.time()
        running_d_loss = 0.0
        running_g_loss = 0.0
        n_batches = 0

        netG.train()
        netD.train()

        for b_idx, batch in enumerate(train_loader):
            if smoke_test and b_idx >= 5:
                break
            real_imgs = batch["image"].to(device)
            real_labels = batch["label"].to(device)
            b_size = real_imgs.size(0)

            # ---------------------
            # Train Discriminator (Hinge Loss)
            # ---------------------
            optD.zero_grad()
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)

            with autocast(device):
                with torch.no_grad():
                    fake_imgs = netG(noise, fake_labels)

                real_pred = netD(real_imgs, real_labels, apply_diffaugment=True, policy=policy)
                fake_pred = netD(fake_imgs.detach(), fake_labels, apply_diffaugment=True, policy=policy)

                # Hinge loss for D
                d_loss = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()

            scalerD.scale(d_loss).backward()
            scalerD.unscale_(optD)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            scalerD.step(optD)
            scalerD.update()

            # ---------------------
            # Train Generator
            # ---------------------
            optG.zero_grad()
            with autocast(device):
                gen_imgs = netG(noise, fake_labels)
                gen_pred = netD(gen_imgs, fake_labels, apply_diffaugment=True, policy=policy)
                # Hinge loss for G
                g_loss = -gen_pred.mean()

            scalerG.scale(g_loss).backward()
            scalerG.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            scalerG.step(optG)
            scalerG.update()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            n_batches += 1
            global_step += 1

            train_logs.append({
                "step": global_step,
                "epoch": epoch,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item()
            })

        ep_duration = time.time() - t_ep_start
        epoch_times.append(ep_duration)
        ep_d_loss = running_d_loss / max(1, n_batches)
        ep_g_loss = running_g_loss / max(1, n_batches)

        # Generate sample grid
        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise, fixed_labels)
            save_image_grid(samples, os.path.join(samples_dir, f"sample_epoch_{epoch:03d}.png"), nrow=4, normalize=True)

        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        print(f"CGAN Epoch {epoch:03d}/{epochs:03d} [{ep_duration:.2f}s] - D Loss: {ep_d_loss:.4f} | G Loss: {ep_g_loss:.4f} | VRAM: {vram_mb:.1f} MB")

        # Save checkpoint
        save_checkpoint(
            state={
                "epoch": epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optG_state_dict": optG.state_dict(),
                "optD_state_dict": optD.state_dict(),
                "d_loss": ep_d_loss,
                "g_loss": ep_g_loss
            },
            is_best=False,
            checkpoint_dir=run_dir,
            epoch=epoch,
            keep_last_n=5
        )

        epoch_logs.append({
            "epoch": epoch,
            "d_loss": ep_d_loss,
            "g_loss": ep_g_loss,
            "epoch_sec": ep_duration,
            "vram_mb": vram_mb
        })

    pd.DataFrame(epoch_logs).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
    pd.DataFrame(train_logs).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    # Save best.pt
    latest_ckpt_path = os.path.join(run_dir, "latest.pt")
    best_ckpt_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(latest_ckpt_path):
        shutil.copy2(latest_ckpt_path, best_ckpt_path)

    if not smoke_test:
        netG.eval()
        def cgan_gen_batch(class_id, count):
            batches = []
            cur = 0
            while cur < count:
                b = min(64, count - cur)
                z = torch.randn(b, latent_dim, device=device)
                y = torch.full((b,), class_id, dtype=torch.long, device=device)
                with torch.no_grad():
                    imgs = netG(z, y)
                    imgs = ((imgs + 1.0) * 0.5).clamp(0.0, 1.0)
                batches.append(imgs.cpu())
                cur += b
            return torch.cat(batches, dim=0)

        export_synthetic_crops("cgan", cgan_gen_batch, num_per_class=250)

    avg_ep_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
    timing_info = {
        "seconds_per_epoch": avg_ep_sec,
        "projected_100_epochs_sec": avg_ep_sec * 100,
        "projected_100_epochs_hours": round((avg_ep_sec * 100) / 3600, 2),
        "total_elapsed_sec": time.time() - t_start,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    }
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing_info, f, indent=2)

    return timing_info


# ==========================================
# 4. CONDITIONAL DIFFUSION TRAINER
# ==========================================
def train_diffusion(cfg, run_dir, device, smoke_test=False, resume=True):
    from src.models.diffusion import get_diffusion_unet, GaussianDiffusion

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    epochs = 2 if smoke_test else train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 32)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    lr = float(train_cfg.get("lr", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    norm_mode = data_cfg.get("normalize_mode", "tanh")

    train_loader = get_crop_dataloader(data_cfg["train_npz"], batch_size=batch_size, shuffle=True, normalize_mode=norm_mode)

    unet = get_diffusion_unet(
        input_resolution=model_cfg.get("input_resolution", 64),
        base_channels=model_cfg.get("base_channels", 64),
        channel_mult=tuple(model_cfg.get("channel_mult", [1, 2, 2, 4])),
        num_classes=model_cfg.get("num_classes", 4)
    ).to(device)

    diffusion = GaussianDiffusion(
        unet=unet,
        timesteps=model_cfg.get("timesteps", 1000),
        beta_schedule=model_cfg.get("beta_schedule", "cosine"),
        num_classes=model_cfg.get("num_classes", 4),
        cfg_dropout_prob=model_cfg.get("cfg_dropout_prob", 0.10)
    ).to(device)

    # EMA model copy
    ema_unet = copy.deepcopy(unet).to(device)
    for p in ema_unet.parameters():
        p.requires_grad = False

    optimizer = AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(device)

    start_epoch = 1
    epoch_logs = []
    train_logs = []

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    if resume:
        latest = find_latest_checkpoint(run_dir)
        if latest:
            ckpt = torch.load(latest, map_location=device)
            unet.load_state_dict(ckpt["unet_state_dict"])
            ema_unet.load_state_dict(ckpt["ema_unet_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed Diffusion from epoch {ckpt['epoch']}")

    global_step = 0
    t_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, epochs + 1):
        t_ep_start = time.time()
        unet.train()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for b_idx, batch in enumerate(train_loader):
            if smoke_test and b_idx >= 5:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(device):
                loss, _ = diffusion(images, labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update EMA
                with torch.no_grad():
                    for p_ema, p in zip(ema_unet.parameters(), unet.parameters()):
                        p_ema.data.mul_(0.9999).add_(p.data, alpha=1.0 - 0.9999)

            running_loss += loss.item() * grad_accum
            n_batches += 1
            global_step += 1

            train_logs.append({
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item() * grad_accum
            })

        ep_duration = time.time() - t_ep_start
        epoch_times.append(ep_duration)
        ep_loss = running_loss / max(1, n_batches)

        # Sampling with DDIM (fast mode in smoke test: 10 steps; normal: 50 steps)
        steps = 10 if smoke_test else 15
        with torch.no_grad():
            samples_list = []
            for c in range(4):
                samples_c, _ = diffusion.sample_ddim(
                    num_samples=4,
                    class_id=c,
                    steps=steps,
                    guidance_scale=3.0,
                    device=device
                )
                samples_list.append(samples_c)
            samples = torch.cat(samples_list, dim=0)
            save_image_grid(samples, os.path.join(samples_dir, f"sample_epoch_{epoch:03d}.png"), nrow=4, normalize=True)

        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        print(f"Diffusion Epoch {epoch:03d}/{epochs:03d} [{ep_duration:.2f}s] - Loss: {ep_loss:.5f} | VRAM: {vram_mb:.1f} MB")

        save_checkpoint(
            state={
                "epoch": epoch,
                "unet_state_dict": unet.state_dict(),
                "ema_unet_state_dict": ema_unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": ep_loss
            },
            is_best=False,
            checkpoint_dir=run_dir,
            epoch=epoch,
            keep_last_n=5
        )

        epoch_logs.append({
            "epoch": epoch,
            "loss": ep_loss,
            "epoch_sec": ep_duration,
            "vram_mb": vram_mb
        })

    pd.DataFrame(epoch_logs).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
    pd.DataFrame(train_logs).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    # Save best.pt as final converged generator
    latest_ckpt_path = os.path.join(run_dir, "latest.pt")
    best_ckpt_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(latest_ckpt_path):
        shutil.copy2(latest_ckpt_path, best_ckpt_path)

    # Export synthetic violation crops & guidance sweep if full training
    if not smoke_test:
        print("\n--- Generating Classifier-Free Guidance Scale Sweep ---")
        sweep_scales = [1.0, 2.0, 3.0, 5.0, 7.5]
        sweep_imgs = []
        with torch.no_grad():
            for s in sweep_scales:
                for c in range(4):
                    samp, _ = diffusion.sample_ddim(num_samples=1, class_id=c, steps=50, guidance_scale=s, device=device)
                    sweep_imgs.append(((samp + 1.0) * 0.5).clamp(0.0, 1.0))
            sweep_grid = torch.cat(sweep_imgs, dim=0)
            save_image_grid(sweep_grid, os.path.join(samples_dir, "guidance_scale_sweep.png"), nrow=4, normalize=False)

        def diffusion_gen_batch(class_id, count):
            batches = []
            cur = 0
            while cur < count:
                b = min(32, count - cur)
                samp, _ = diffusion.sample_ddim(num_samples=b, class_id=class_id, steps=50, guidance_scale=3.0, device=device)
                imgs = ((samp + 1.0) * 0.5).clamp(0.0, 1.0)
                batches.append(imgs.cpu())
                cur += b
            return torch.cat(batches, dim=0)

        export_synthetic_crops("diffusion", diffusion_gen_batch, num_per_class=250)

    avg_ep_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
    timing_info = {
        "seconds_per_epoch": avg_ep_sec,
        "projected_100_epochs_sec": avg_ep_sec * 100,
        "projected_100_epochs_hours": round((avg_ep_sec * 100) / 3600, 2),
        "total_elapsed_sec": time.time() - t_start,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    }
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing_info, f, indent=2)

    return timing_info


# ==========================================
# 5. VQ-VAE & TRANSFORMER TRAINER
# ==========================================
def train_vqvae_and_transformer(cfg, run_dir, device, smoke_test=False, resume=True):
    from src.models.vqvae import VQVAE
    from src.models.transformer import ImageTokenTransformer
    from torch.utils.data import TensorDataset, DataLoader

    v_cfg = cfg["stage1_vqvae"]
    t_cfg = cfg["stage2_transformer"]
    data_cfg = cfg["data"]

    epochs = 2 if smoke_test else v_cfg.get("epochs", 100)
    batch_size = v_cfg.get("batch_size", 128)
    lr = float(v_cfg.get("lr", 0.0005))
    norm_mode = data_cfg.get("normalize_mode", "sigmoid")

    # Stage 1: VQ-VAE
    vqvae = VQVAE(
        in_channels=v_cfg.get("in_channels", 3),
        codebook_size=v_cfg.get("codebook_size", 512),
        embedding_dim=v_cfg.get("embedding_dim", 64),
        commitment_beta=v_cfg.get("commitment_beta", 0.25)
    ).to(device)

    opt_vq = AdamW(vqvae.parameters(), lr=lr)
    scaler_vq = GradScaler(device)

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    t_start = time.time()
    epoch_times = []

    # Check for pre-trained VQ-VAE Stage 1 checkpoint
    stage1_ckpt_path = os.path.join(run_dir, "vqvae_stage1_best.pt")
    if not os.path.exists(stage1_ckpt_path):
        stage1_ckpt_path = os.path.join(run_dir, "latest.pt")

    stage1_done = False
    if resume and os.path.exists(stage1_ckpt_path):
        ckpt_s1 = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
        if "vqvae_state_dict" in ckpt_s1:
            vqvae.load_state_dict(ckpt_s1["vqvae_state_dict"])
            print(f"\n--- Resumed pre-trained Stage 1 VQ-VAE from {stage1_ckpt_path} (Skipping Stage 1 training) ---", flush=True)
            stage1_done = True

    if not stage1_done:
        train_loader = get_crop_dataloader(data_cfg["train_npz"], batch_size=batch_size, shuffle=True, normalize_mode=norm_mode)
        print("\n--- Training Stage 1: VQ-VAE Tokenizer ---", flush=True)
        for epoch in range(1, epochs + 1):
            t_ep_start = time.time()
            vqvae.train()
            running_loss = 0.0
            n_samples = 0

            for b_idx, batch in enumerate(train_loader):
                if smoke_test and b_idx >= 5:
                    break
                images = batch["image"].to(device)
                opt_vq.zero_grad()

                with autocast(device):
                    x_recon, loss, _ = vqvae(images)

                scaler_vq.scale(loss).backward()
                scaler_vq.unscale_(opt_vq)
                torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
                scaler_vq.step(opt_vq)
                scaler_vq.update()

                running_loss += loss.item() * len(images)
                n_samples += len(images)

            ep_duration = time.time() - t_ep_start
            epoch_times.append(ep_duration)
            ep_loss = running_loss / max(1, n_samples)

            # Save sample reconstruction
            vqvae.eval()
            with torch.no_grad():
                paired = torch.cat([images[:8], x_recon[:8]], dim=0)
                save_image_grid(paired, os.path.join(samples_dir, f"vqvae_recon_epoch_{epoch:03d}.png"), nrow=8, normalize=False)

            vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"VQ-VAE Epoch {epoch:03d}/{epochs:03d} [{ep_duration:.2f}s] - Loss: {ep_loss:.5f} | VRAM: {vram_mb:.1f} MB", flush=True)

    # Fast GPU-accelerated pre-encoding of training set into discrete codebook tokens
    token_cache_path = "processed/crops/train_tokens_vqvae.pt"
    if os.path.exists(token_cache_path):
        print(f"\n--- Loading pre-cached tokens from {token_cache_path} ---", flush=True)
        cached = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        all_tokens = cached["tokens"]
        all_labels = cached["labels"]
    else:
        print("\n--- Pre-encoding training images into discrete codebook tokens (GPU Batching) ---", flush=True)
        vqvae.eval()
        cached_tokens = []
        cached_labels = []
        encode_loader = get_crop_dataloader(data_cfg["train_npz"], batch_size=256, shuffle=False, normalize_mode=norm_mode)

        with torch.no_grad():
            for b_idx, batch in enumerate(encode_loader):
                if smoke_test and b_idx >= 5:
                    break
                imgs = batch["image"].to(device)
                lbls = batch["label"]
                toks = vqvae.get_codebook_indices(imgs).cpu() # (B, 256)
                cached_tokens.append(toks)
                cached_labels.append(lbls)

        all_tokens = torch.cat(cached_tokens, dim=0) # (N, 256)
        all_labels = torch.cat(cached_labels, dim=0) # (N,)
        os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)
        torch.save({"tokens": all_tokens, "labels": all_labels}, token_cache_path)
        print(f"Pre-cached and saved {len(all_tokens)} sequences to {token_cache_path}", flush=True)

    # Prepend class token: visual tokens are [0, 511], class tokens are [512, 515]
    class_tokens = (512 + all_labels).unsqueeze(1) # (N, 1)
    all_seqs = torch.cat([class_tokens, all_tokens], dim=1) # (N, 257)
    token_ram_mb = (all_seqs.element_size() * all_seqs.nelement()) / (1024**2)
    print(f"Full training sequence dataset ready: {len(all_seqs)} sequences ({token_ram_mb:.2f} MB in RAM).", flush=True)

    # Offload VQ-VAE to CPU during Transformer training to preserve 6GB VRAM budget
    vqvae.to("cpu")
    torch.cuda.empty_cache()

    # Fast in-memory DataLoader for Transformer training
    transformer_dataset = TensorDataset(all_seqs)
    transformer_loader = DataLoader(
        transformer_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Stage 2: Transformer
    print("\n--- Training Stage 2: Autoregressive ImageGPT (FlashAttention Native) ---", flush=True)
    gpt = ImageTokenTransformer(
        vocab_size=t_cfg.get("vocab_size", 516),
        seq_len=t_cfg.get("seq_len", 257),
        n_layer=t_cfg.get("n_layer", 6),
        n_head=t_cfg.get("n_head", 8),
        n_embd=t_cfg.get("n_embd", 384)
    ).to(device)

    opt_gpt = AdamW(gpt.parameters(), lr=float(t_cfg.get("lr", 0.0005)))
    scaler_gpt = GradScaler(device)
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        gpt.train()
        running_loss = 0.0
        n_batches = 0

        for b_idx, (b_seq,) in enumerate(transformer_loader):
            b_seq = b_seq.to(device)
            inputs = b_seq[:, :-1]
            targets = b_seq[:, 1:]

            opt_gpt.zero_grad()
            with autocast(device):
                logits, loss = gpt(inputs, targets=targets)

            scaler_gpt.scale(loss).backward()
            scaler_gpt.unscale_(opt_gpt)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)
            scaler_gpt.step(opt_gpt)
            scaler_gpt.update()

            running_loss += loss.item()
            n_batches += 1

        ep_duration = time.time() - t_ep_start
        epoch_times.append(ep_duration)
        ep_loss = running_loss / max(1, n_batches)

        # Generate sample grid via GPT autoregressive sampling and VQ-VAE decoder every 5 epochs and at epoch 1 & final
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            gpt.eval()
            vqvae.to(device)
            with torch.no_grad():
                sample_list = []
                for c in range(4):
                    sample_tokens = gpt.generate(class_id=c, num_samples=4, device=device) # (4, 256)
                    crops_c = vqvae.decode_indices(sample_tokens) # (4, 3, 64, 64)
                    sample_list.append(crops_c)
                samples = torch.cat(sample_list, dim=0)
                save_image_grid(samples, os.path.join(samples_dir, f"gpt_sample_epoch_{epoch:03d}.png"), nrow=4, normalize=False)
            vqvae.to("cpu")
            torch.cuda.empty_cache()

        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        print(f"GPT Epoch {epoch:03d}/{epochs:03d} [{ep_duration:.2f}s] - Cross-Entropy Loss: {ep_loss:.4f} | VRAM: {vram_mb:.1f} MB", flush=True)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": ep_loss,
            "epoch_sec": ep_duration,
            "vram_mb": vram_mb
        })
        pd.DataFrame(epoch_logs).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)

        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "vqvae_state_dict": vqvae.state_dict(),
                    "gpt_state_dict": gpt.state_dict(),
                    "opt_vq_state_dict": opt_vq.state_dict(),
                    "opt_gpt_state_dict": opt_gpt.state_dict(),
                },
                is_best=False,
                checkpoint_dir=run_dir,
                epoch=epoch,
                keep_last_n=5
            )

    avg_ep_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
    timing_info = {
        "seconds_per_epoch_combined": avg_ep_sec,
        "projected_100_epochs_sec": avg_ep_sec * 100,
        "projected_100_epochs_hours": round((avg_ep_sec * 100) / 3600, 2),
        "total_elapsed_sec": time.time() - t_start,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    }
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing_info, f, indent=2)

    latest_ckpt_path = os.path.join(run_dir, "latest.pt")
    best_ckpt_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(latest_ckpt_path):
        shutil.copy2(latest_ckpt_path, best_ckpt_path)

    if not smoke_test:
        gpt.eval()
        vqvae.to(device)
        def transformer_gen_batch(class_id, count):
            batches = []
            cur = 0
            while cur < count:
                b = min(32, count - cur)
                sample_tokens = gpt.generate(class_id=class_id, num_samples=b, device=device)
                crops = vqvae.decode_indices(sample_tokens).clamp(0.0, 1.0)
                batches.append(crops.cpu())
                cur += b
            return torch.cat(batches, dim=0)

        export_synthetic_crops("transformer", transformer_gen_batch, num_per_class=250)
        vqvae.to("cpu")
        torch.cuda.empty_cache()

    return timing_info


# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_type = cfg.get("model", {}).get("type", None)
    if not model_type and "stage1_vqvae" in cfg:
        model_type = "vqvae_transformer"

    run_dir = cfg.get("logging", {}).get("run_dir", f"runs/{model_type}_run")
    if args.smoke_test:
        run_dir = f"runs/smoke_test_{model_type}"

    os.makedirs(run_dir, exist_ok=True)
    log_environment(run_dir, args.device)
    save_config_copy(cfg, run_dir)

    print(f"\n=======================================================")
    print(f"Model: {model_type.upper()}")
    print(f"Run Directory: {run_dir}")
    print(f"Device: {args.device}")
    print(f"Smoke Test Mode: {args.smoke_test}")
    print(f"=======================================================\n")

    if model_type == "autoencoder":
        train_autoencoder(cfg, run_dir, args.device, smoke_test=args.smoke_test, resume=args.resume)
    elif model_type in ["cvae", "vae"]:
        train_cvae(cfg, run_dir, args.device, smoke_test=args.smoke_test, resume=args.resume)
    elif model_type in ["cgan", "gan"]:
        train_cgan(cfg, run_dir, args.device, smoke_test=args.smoke_test, resume=args.resume)
    elif model_type == "diffusion":
        train_diffusion(cfg, run_dir, args.device, smoke_test=args.smoke_test, resume=args.resume)
    elif model_type == "vqvae_transformer":
        train_vqvae_and_transformer(cfg, run_dir, args.device, smoke_test=args.smoke_test, resume=args.resume)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
