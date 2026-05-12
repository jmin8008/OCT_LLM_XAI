#!/usr/bin/env python3
"""
APTOS 2021 AutoResearch Exp #13: 5-Seed Ensemble + Warmup=2
Exp #12(0.9491)의 3-seed 앙상블을 5-seed로 확장.
각 seed가 epoch 2-3에서 peak → warmup 5→2로 단축하여 더 효과적 학습.
5개 seed로 variance 추가 감소 목표.

Runs 5 seeds sequentially within TIME_BUDGET:
- Each seed gets TIME_BUDGET/5 ≈ 180s (~2-3 epochs)
- warmup=2 (was 5): lr reaches full value by epoch 3
- Predictions averaged across all 5 seeds

Uses prepare.py harness for data loading, splitting, bag building, and AUC.
"""

import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Import from prepare.py (read-only harness)
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

from prepare import (
    load_pic_csv,
    load_case_csv,
    get_patient_split,
    build_mil_bags,
    compute_auc,
    TIME_BUDGET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIOMARKER_COLS = ["IRF", "SRF", "PED", "HRF"]
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 8
MAX_BAG_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
POS_WEIGHT_CAP = 10.0
EARLY_STOPPING_PATIENCE = 15
CUDA_DEVICE = 2

# Multi-seed ensemble
ENSEMBLE_SEEDS = [42, 123, 2024, 7, 999]
SEED_TIME_BUDGET = TIME_BUDGET // len(ENSEMBLE_SEEDS)  # 180s per seed

# MixUp parameters (same as Exp #5 best)
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5

# OCT cropping
OCT_X_START = 632
OCT_X_END = 1264
OCT_Y_START = 0
OCT_Y_END = 596


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Model: ConvNeXt-Tiny + MIL (Max Pooling)
# ===========================================================================

class MILClassifier(nn.Module):
    """ConvNeXt-Tiny backbone + MIL head with max pooling aggregation."""

    def __init__(self, backbone_name="convnext_tiny",
                 num_classes=NUM_CLASSES, pretrained=True,
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x, mask=None):
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x_flat = x.view(B * N, C, H, W)
            features = self.extract_features(x_flat)
            features = features.view(B, N, -1)
            if mask is not None:
                features = features.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
            pooled = features.max(dim=1)[0]
            logits = self.classifier(pooled)
        else:
            features = self.extract_features(x)
            logits = self.classifier(features)
        return logits


# ===========================================================================
# MixUp for MIL bags
# ===========================================================================

def mixup_data(images, labels, masks, alpha=MIXUP_ALPHA, prob=MIXUP_PROB):
    if np.random.random() > prob:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, preds, labels_a, labels_b, lam):
    return lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)


# ===========================================================================
# Dataset
# ===========================================================================

def _extract_macular_oct(img):
    return img.crop((OCT_X_START, OCT_Y_START, OCT_X_END, OCT_Y_END))


def _load_image(path):
    try:
        img = Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError):
        w = OCT_X_END - OCT_X_START
        h = OCT_Y_END - OCT_Y_START
        img = Image.new("RGB", (w, h), (0, 0, 0))
    return img


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class MILBagDataset(Dataset):
    def __init__(self, bags, transform=None, max_bag_size=MAX_BAG_SIZE):
        self.bags = bags
        self.transform = transform
        self.max_bag_size = max_bag_size

    def __len__(self):
        return len(self.bags)

    def _load_instance(self, image_path):
        img = _load_image(image_path)
        img = _extract_macular_oct(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        bag = self.bags[idx]
        image_paths = bag["image_paths"]
        images = []
        for p in image_paths[:self.max_bag_size]:
            images.append(self._load_instance(p))
        n_valid = len(images)
        while len(images) < self.max_bag_size:
            images.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        images_tensor = torch.stack(images)
        labels = torch.tensor(bag["labels"], dtype=torch.float32)
        mask = torch.zeros(self.max_bag_size, dtype=torch.float32)
        mask[:n_valid] = 1.0
        return images_tensor, labels, mask


# ===========================================================================
# Class weights
# ===========================================================================

def compute_pos_weights(bags):
    all_labels = np.stack([bag["labels"] for bag in bags], axis=0)
    weights = []
    for i in range(NUM_CLASSES):
        pos = all_labels[:, i].sum()
        neg = len(all_labels) - pos
        w = neg / max(pos, 1)
        w = max(w, 1.0)
        w = min(w, POS_WEIGHT_CAP)
        weights.append(w)
    return weights


# ===========================================================================
# Training & Validation
# ===========================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    for images, labels, mask in loader:
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mask)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(mixed_images, mask)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return avg_loss, all_preds, all_labels


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    for images, labels, mask in loader:
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        with autocast(device_type="cuda"):
            logits = model(images, mask)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return avg_loss, all_preds, all_labels


class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, mode="max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ===========================================================================
# Train one seed
# ===========================================================================

def train_one_seed(seed, train_bags, val_bags, device, seed_time_budget):
    """Train model with a specific seed, return best val predictions."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed} — budget: {seed_time_budget}s")
    print(f"{'='*60}")

    set_seed(seed)

    train_dataset = MILBagDataset(train_bags, transform=get_train_transforms())
    val_dataset = MILBagDataset(val_bags, transform=get_val_transforms())

    # Use seed-specific generator for DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=g, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model (re-initialized with this seed)
    set_seed(seed)
    model = MILClassifier(dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ConvNeXt-Tiny + MIL Max Pool + MixUp (dropout=0.3)")
    print(f"Total params: {total_params:,}")

    # Class weights
    weights = compute_pos_weights(train_bags)
    pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"pos_weight: {dict(zip(BIOMARKER_COLS, [f'{w:.2f}' for w in weights]))}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler("cuda")
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")

    best_auc = 0.0
    best_epoch = 0
    best_preds = None
    best_labels = None
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > seed_time_budget:
            print(f"  Seed {seed}: TIME_BUDGET ({seed_time_budget}s) exceeded at epoch {epoch+1}")
            break

        if epoch < WARMUP_EPOCHS:
            lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_loss, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_auc_dict, val_mean_auc = compute_auc(val_labels, val_preds)

        irf_auc = val_auc_dict.get("IRF", 0.0)
        srf_auc = val_auc_dict.get("SRF", 0.0)
        ped_auc = val_auc_dict.get("PED", 0.0)
        hrf_auc = val_auc_dict.get("HRF", 0.0)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        print(f"  seed={seed}, epoch: {epoch+1}, train_loss: {train_loss:.4f}, "
              f"val_mean_auc: {val_mean_auc:.4f}, irf: {irf_auc:.4f}, srf: {srf_auc:.4f}, "
              f"ped: {ped_auc:.4f}, hrf: {hrf_auc:.4f}")

        if val_mean_auc > best_auc:
            best_auc = val_mean_auc
            best_epoch = epoch + 1
            best_preds = val_preds.copy()
            best_labels = val_labels.copy()

        if early_stopper.step(val_mean_auc):
            print(f"  Seed {seed}: Early stopping at epoch {epoch+1}")
            break

    seed_time = time.time() - start_time
    print(f"  Seed {seed}: Best val_mean_auc: {best_auc:.4f} at epoch {best_epoch} ({seed_time:.1f}s)")

    # Free memory
    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()

    return best_auc, best_epoch, best_preds, best_labels, seed_time


# ===========================================================================
# Main
# ===========================================================================

def main():
    overall_start = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nMulti-seed Ensemble: {ENSEMBLE_SEEDS}")
    print(f"Per-seed budget: {SEED_TIME_BUDGET}s, Total budget: {TIME_BUDGET}s")

    # Load data
    print("\nLoading data via prepare.py...")
    pic_df = load_pic_csv()
    case_df = load_case_csv()

    # Use fixed seed for data split (same split for all models)
    train_patients, val_patients = get_patient_split(pic_df, val_ratio=0.15, seed=42)
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

    train_bags = build_mil_bags(pic_df, train_patients)
    val_bags = build_mil_bags(pic_df, val_patients)
    print(f"Train bags: {len(train_bags)}, Val bags: {len(val_bags)}")

    # Train each seed
    all_best_aucs = []
    all_best_preds = []
    all_best_labels = []
    all_best_epochs = []
    all_seed_times = []

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        remaining = TIME_BUDGET - (time.time() - overall_start)
        if remaining < 120:  # Need at least 2 min for a seed
            print(f"\nNot enough time for seed {seed} ({remaining:.0f}s remaining)")
            break

        # Give remaining seeds equal time
        remaining_seeds = len(ENSEMBLE_SEEDS) - i
        seed_budget = int(remaining / remaining_seeds)

        best_auc, best_epoch, best_preds, best_labels, seed_time = train_one_seed(
            seed, train_bags, val_bags, device, seed_budget
        )

        all_best_aucs.append(best_auc)
        all_best_preds.append(best_preds)
        all_best_labels.append(best_labels)
        all_best_epochs.append(best_epoch)
        all_seed_times.append(seed_time)

    # =========================================================================
    # Ensemble: average predictions across seeds
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE RESULTS")
    print(f"{'='*60}")

    for i, seed in enumerate(ENSEMBLE_SEEDS[:len(all_best_aucs)]):
        print(f"  Seed {seed}: AUC={all_best_aucs[i]:.4f} (epoch {all_best_epochs[i]}, {all_seed_times[i]:.0f}s)")

    # Ensemble: average predictions
    ensemble_preds = np.mean(all_best_preds, axis=0)
    # Labels should be the same across seeds (same val set)
    ensemble_labels = all_best_labels[0]

    # Compute ensemble AUC
    ensemble_auc_dict, ensemble_mean_auc = compute_auc(ensemble_labels, ensemble_preds)
    print(f"\n  Ensemble Mean AUC: {ensemble_mean_auc:.4f}")
    for k, v in ensemble_auc_dict.items():
        print(f"    {k}: {v:.4f}")

    # Compare vs best single seed
    best_single_idx = np.argmax(all_best_aucs)
    best_single = all_best_aucs[best_single_idx]
    print(f"\n  Best single seed AUC: {best_single:.4f} (seed={ENSEMBLE_SEEDS[best_single_idx]})")
    print(f"  Ensemble improvement: {ensemble_mean_auc - best_single:+.4f}")

    print(f"\n  peak_vram_mb: {torch.cuda.max_memory_allocated()/1e6:.0f}")

    total_time = time.time() - overall_start
    print(f"\nTotal time: {total_time:.1f}s / {TIME_BUDGET}s budget")


if __name__ == "__main__":
    main()
