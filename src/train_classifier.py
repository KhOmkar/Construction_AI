"""
Training script for the reference worker crop classifier.
Trained on crops_train_64.npz, validated on crops_val_64.npz, tested on crops_test_64.npz.
Provides the frozen classifier oracle used for Class-Conditional Accuracy (CCA).
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np

from src.models.classifier import ReferenceClassifier
from src.common.dataset import get_crop_dataloader


def train_reference_classifier(epochs=12, batch_size=128, lr=1e-3, device="cuda"):
    os.makedirs("checkpoints/classifier", exist_ok=True)
    os.makedirs("outputs/classifier", exist_ok=True)

    print(f"--- Training Reference Crop Classifier on {device} ---")
    train_loader = get_crop_dataloader("processed/crops/crops_train_64.npz", batch_size=batch_size, shuffle=True, normalize_mode="tanh")
    val_loader = get_crop_dataloader("processed/crops/crops_val_64.npz", batch_size=batch_size, shuffle=False, normalize_mode="tanh")
    test_loader = get_crop_dataloader("processed/crops/crops_test_64.npz", batch_size=batch_size, shuffle=False, normalize_mode="tanh")

    # Calculate class weights for balanced cross entropy
    class_counts = torch.tensor([13297.0, 1326.0, 3735.0, 1410.0])
    total = class_counts.sum()
    class_weights = total / (4.0 * class_counts)
    class_weights = class_weights.to(device)
    print(f"Class Weights: {class_weights.tolist()}")

    model = ReferenceClassifier(num_classes=4, in_channels=3, base_channels=32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    best_val_acc = 0.0
    best_path = "checkpoints/classifier/best_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += len(labels)

        scheduler.step()
        train_loss = total_loss / total_samples
        train_acc = correct / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                with autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += len(labels)

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        epoch_sec = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_sec:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss
            }, best_path)

    print(f"\nTraining Complete. Best Val Accuracy: {best_val_acc*100:.2f}% saved to {best_path}")

    # Evaluate on Test Split
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    per_class_correct = [0] * 4
    per_class_total = [0] * 4

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += len(labels)

            for c in range(4):
                mask = (labels == c)
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += ((preds == c) & mask).sum().item()

    overall_test_acc = test_correct / test_total
    per_class_acc = [per_class_correct[c] / max(1, per_class_total[c]) for c in range(4)]

    class_names = ["Compliant (0)", "No Helmet (1)", "No Vest (2)", "Both Missing (3)"]
    print("\n--- Test Set Evaluation ---")
    print(f"Overall Test Accuracy: {overall_test_acc*100:.2f}%")
    for c in range(4):
        print(f"  {class_names[c]}: {per_class_acc[c]*100:.2f}% ({per_class_correct[c]}/{per_class_total[c]})")

    results = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(overall_test_acc),
        "per_class_test_acc": {class_names[c]: float(per_class_acc[c]) for c in range(4)}
    }
    with open("outputs/classifier/test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_reference_classifier(epochs=12, batch_size=128, lr=1e-3, device=device)
