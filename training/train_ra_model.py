# training/train_ra_model.py
# MobileNetV2 RA Regression Model — PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from PIL import Image
import os
import time


# ── Dataset ────────────────────────────────────────────────────────────────
class RADataset(Dataset):
    """
    CSV format: img_path, ra_value, asset_type
    Images: road sign/marking crops, 224×224
    """
    AUGMENT = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    EVAL = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    def __init__(self, csv_path: str, img_dir: str, augment: bool = True):
        self.data    = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tf      = self.AUGMENT if augment else self.EVAL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row  = self.data.iloc[idx]
        path = os.path.join(self.img_dir, row["img_path"])
        img  = Image.open(path).convert("RGB")
        ra   = float(row["ra_value"])
        return self.tf(img), torch.tensor(ra, dtype=torch.float32)


# ── Model ──────────────────────────────────────────────────────────────────
class MobileNetV2_RA(nn.Module):
    """
    MobileNetV2 backbone (ImageNet pretrained) + custom regression head.

    Architecture:
        MobileNetV2 features → GlobalAvgPool → 1280-dim feature vector
        → FC(1280→512) → BN → ReLU → Dropout(0.3)
        → FC(512→128)  → BN → ReLU → Dropout(0.2)
        → FC(128→64)   → BN → ReLU
        → RA head:   FC(64→1) → ReLU  [output: RA ≥ 0]
        → Conf head: FC(64→1) → Sigmoid [output: confidence 0-1]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Freeze first 10 layers of backbone for fine-tuning
        for i, layer in enumerate(base.features[:10]):
            for param in layer.parameters():
                param.requires_grad = False

        self.backbone = base.features   # outputs (B, 1280, 7, 7)

        self.regressor = nn.Sequential(
            nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,  128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,   64), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.ra_head   = nn.Sequential(nn.Linear(64, 1), nn.ReLU())
        self.conf_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)            # (B, 1280, 7, 7)
        feat = feat.mean(dim=[2, 3])       # GlobalAvgPool → (B, 1280)
        feat = self.regressor(feat)        # (B, 64)
        ra   = self.ra_head(feat).squeeze(1)    # (B,)
        conf = self.conf_head(feat).squeeze(1)  # (B,)
        return ra, conf


# ── Loss Functions ─────────────────────────────────────────────────────────
def huber_loss(pred: torch.Tensor, target: torch.Tensor,
               delta: float = 25.0) -> torch.Tensor:
    """
    Huber loss — robust to outlier RA measurements.
    L = 0.5·(pred-target)²       if |pred-target| < δ
      = δ·(|pred-target| - 0.5δ) otherwise
    More robust than MSE for RA measurements which can have large outliers.
    """
    diff = torch.abs(pred - target)
    return torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    ).mean()


def confidence_loss(pred: torch.Tensor, target: torch.Tensor,
                    conf: torch.Tensor, sigma: float = 25.0) -> torch.Tensor:
    """
    Confidence calibration loss.
    c_target = exp(-|RA_pred - RA_true| / σ)
    L_conf = BCE(conf_pred, c_target)

    Forces model to output high confidence when prediction error is small,
    low confidence when prediction error is large.
    """
    c_target = torch.exp(-torch.abs(pred.detach() - target) / sigma)
    return nn.functional.binary_cross_entropy(conf, c_target)


def total_loss(ra_pred, ra_true, conf_pred,
               lambda1: float = 0.3, lambda2: float = 0.1) -> torch.Tensor:
    """
    L_total = L_ra + λ₁·L_conf + λ₂·L_consistency
    (L_consistency = variance of ra_pred to encourage smooth predictions)
    """
    l_ra   = huber_loss(ra_pred, ra_true)
    l_conf = confidence_loss(ra_pred, ra_true, conf_pred)
    l_cons = ra_pred.var()   # penalize high variance predictions
    return l_ra + lambda1 * l_conf + lambda2 * l_cons


# ── Training Loop ─────────────────────────────────────────────────────────
def train(csv_path: str, img_dir: str, epochs: int = 50,
          batch_size: int = 32, lr: float = 1e-4,
          save_dir: str = "checkpoints", patience: int = 10,
          warmup_epochs: int = 5):
    """
    Complete training pipeline with:
    - Training/validation split (80/20)
    - Learning rate warmup for first `warmup_epochs` epochs
    - Cosine annealing LR schedule after warmup
    - Early stopping with patience
    - Model checkpoint saving (best validation loss)
    - RMSE and MAE metrics per epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # ── Dataset preparation ───────────────────────────────────────────────
    dataset = RADataset(csv_path, img_dir, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Override augmentation for validation
    val_ds.dataset = RADataset(csv_path, img_dir, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print(f"[TRAIN] Train: {train_size} | Val: {val_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = MobileNetV2_RA(pretrained=True).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # ── LR Scheduler: Warmup + Cosine Annealing ──────────────────────────
    def warmup_cosine_schedule(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: scale from 0.1 to 1.0
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    # ── Training state ────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_rmse": [], "val_rmse": [],
               "train_mae": [], "val_mae": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | "
          f"{'Val RMSE':>9} | {'Val MAE':>8} | {'LR':>10}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        train_preds, train_targets = [], []

        for images, targets in train_dl:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            ra_pred, conf_pred = model(images)
            loss = total_loss(ra_pred, targets, conf_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(ra_pred.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, targets in val_dl:
                images, targets = images.to(device), targets.to(device)
                ra_pred, conf_pred = model(images)
                loss = total_loss(ra_pred, targets, conf_pred)

                val_losses.append(loss.item())
                val_preds.extend(ra_pred.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # ── Metrics ───────────────────────────────────────────────────────
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        train_preds_arr = np.array(train_preds)
        train_targets_arr = np.array(train_targets)
        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)

        train_rmse = np.sqrt(np.mean((train_preds_arr - train_targets_arr) ** 2))
        val_rmse = np.sqrt(np.mean((val_preds_arr - val_targets_arr) ** 2))
        train_mae = np.mean(np.abs(train_preds_arr - train_targets_arr))
        val_mae = np.mean(np.abs(val_preds_arr - val_targets_arr))

        current_lr = optimizer.param_groups[0]['lr']

        # Log history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(f"{epoch+1:>6} | {train_loss:>11.4f} | {val_loss:>11.4f} | "
              f"{val_rmse:>9.2f} | {val_mae:>8.2f} | {current_lr:>10.6f}  ({elapsed:.1f}s)")

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            }
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"       ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # ── Early stopping ────────────────────────────────────────────────
        if patience_counter >= patience:
            print(f"\n[TRAIN] Early stopping at epoch {epoch+1}. "
                  f"Best: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
            break

        # Step scheduler
        scheduler.step()

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, final_path)

    # ── Save training history ─────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"[TRAIN] Complete! Best epoch: {best_epoch}")
    print(f"[TRAIN] Best val_loss: {best_val_loss:.4f}")
    print(f"[TRAIN] Model saved to: {checkpoint_path}")
    print(f"[TRAIN] History saved to: {os.path.join(save_dir, 'training_history.csv')}")

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MobileNetV2 RA Regression Model")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with img_path, ra_value")
    parser.add_argument("--img-dir", type=str, required=True, help="Root directory for images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args.csv, args.img_dir, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr,
          save_dir=args.save_dir, patience=args.patience)
