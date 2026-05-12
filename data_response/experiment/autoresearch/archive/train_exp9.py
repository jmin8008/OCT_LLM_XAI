#!/usr/bin/env python3
"""
APTOS 2021 AutoResearch Exp #9: ConvNeXt-Tiny + MIL Max Pool + MixUp + Label Smoothing + TTA
Exp #5(0.9475) 기반: Label Smoothing 0.05 추가 (gentlest regularization — overconfidence 방지).
TTA(Test-Time Augmentation) 추가: validation 시 3 views (original + hflip + vflip) 평균으로 variance 감소.
SWA 제거 (Exp #8에서 hurt convergence), dropout 0.3 유지 (0.5는 underfitting).

Uses prepare.py harness for data loading, splitting, bag building, and AUC.
The agent is free to modify this file entirely.

Key architecture:
- Backbone: timm convnext_tiny (ImageNet pretrained, 28M params)
- Feature dim: 768
- Head: LayerNorm -> Linear(768,512) -> GELU -> Dropout(0.3) -> Linear(512,4)
- MIL aggregation: Max Pool across bag
- Loss: LabelSmoothingBCEWithLogitsLoss (smoothing=0.05) with pos_weight
- MixUp: alpha=0.2, probability=0.5
- TTA: 3 views (original + hflip + vflip) at validation
- Exp #9: ConvNeXt-Tiny + MixUp + Label Smoothing + TTA, batch=8, lr=1e-4, dropout=0.3, warmup=5, wd 1e-4
"""

import os
import sys
import time

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
WARMUP_EPOCHS = 5
POS_WEIGHT_CAP = 10.0
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED = 42
CUDA_DEVICE = 2

# MixUp parameters
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5

# OCT cropping: raw 1264x596 -> right half (OCT region)
OCT_X_START = 632
OCT_X_END = 1264
OCT_Y_START = 0
OCT_Y_END = 596


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
        self.num_features = self.backbone.num_features  # 768 for convnext_tiny

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
        """
        Args:
            x: (B, N, 3, H, W) MIL bag or (B, 3, H, W) single image
            mask: (B, N) valid-instance mask (1=valid, 0=padding)
        Returns:
            logits: (B, num_classes)
        """
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x_flat = x.view(B * N, C, H, W)

            features = self.extract_features(x_flat)  # (B*N, feat_dim)
            features = features.view(B, N, -1)         # (B, N, feat_dim)

            # Max Pool: mask out padding before pooling
            if mask is not None:
                features = features.masked_fill(mask.unsqueeze(-1) == 0, -1e4)

            pooled = features.max(dim=1)[0]  # (B, feat_dim)

            logits = self.classifier(pooled)
        else:
            features = self.extract_features(x)
            logits = self.classifier(features)

        return logits


# ===========================================================================
# MixUp for MIL bags
# ===========================================================================

def mixup_data(images, labels, masks, alpha=MIXUP_ALPHA, prob=MIXUP_PROB):
    """Apply MixUp to MIL bag data.

    Args:
        images: (B, N, 3, H, W)
        labels: (B, 4)
        masks: (B, N)
        alpha: Beta distribution parameter
        prob: probability of applying MixUp

    Returns:
        mixed_images, labels_a, labels_b, lam
    """
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
    """Compute MixUp loss."""
    return lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)


# ===========================================================================
# Label Smoothing BCEWithLogitsLoss
# ===========================================================================

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


# ===========================================================================
# Dataset: MILBagDataset (uses prepare.py build_mil_bags output)
# ===========================================================================

def _extract_macular_oct(img):
    """Extract macular OCT region from raw image (1264x596 -> right half)."""
    return img.crop((OCT_X_START, OCT_Y_START, OCT_X_END, OCT_Y_END))


def _load_image(path):
    """Load image, extract OCT region, return PIL Image."""
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class MILBagDataset(Dataset):
    """MIL Bag Dataset using build_mil_bags() results from prepare.py."""

    def __init__(self, bags, transform=None, max_bag_size=MAX_BAG_SIZE):
        self.bags = bags
        self.transform = transform
        self.max_bag_size = max_bag_size

    def __len__(self):
        return len(self.bags)

    def _load_instance(self, image_path):
        """Load and transform a single image instance by absolute path."""
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
# Class weights computation
# ===========================================================================

def compute_pos_weights(bags):
    """Compute pos_weight for BCEWithLogitsLoss from bag-level labels."""
    all_labels = np.stack([bag["labels"] for bag in bags], axis=0)
    weights = []
    for i in range(NUM_CLASSES):
        pos = all_labels[:, i].sum()
        neg = len(all_labels) - pos
        w = neg / max(pos, 1)
        w = max(w, 1.0)
        w = min(w, POS_WEIGHT_CAP)
        weights.append(w)
    # HRF (index 3) is majority class — do NOT cap below 1.0 (Exp #6 showed it causes AUC collapse)
    return weights


# ===========================================================================
# Training loop
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

        # MixUp
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

        # Use original labels (not mixed) for AUC tracking
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

        preds = torch.sigmoid(logits)

        # TTA: horizontal flip
        if images.dim() == 5:
            images_hf = images.flip(-1)  # flip last dim (W)
        else:
            images_hf = images.flip(-1)
        with autocast(device_type="cuda"):
            logits_hf = model(images_hf, mask)
        preds_hf = torch.sigmoid(logits_hf)

        # TTA: vertical flip
        if images.dim() == 5:
            images_vf = images.flip(-2)  # flip H dim
        else:
            images_vf = images.flip(-2)
        with autocast(device_type="cuda"):
            logits_vf = model(images_vf, mask)
        preds_vf = torch.sigmoid(logits_vf)

        # Average TTA predictions
        avg_preds = (preds + preds_hf + preds_vf) / 3.0
        all_preds.append(avg_preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return avg_loss, all_preds, all_labels


class EarlyStopping:
    """Early stopping on val_mean_auc (mode=max)."""

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
# Main
# ===========================================================================

def main():
    start_time = time.time()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data via prepare.py
    print("\nLoading data via prepare.py...")
    pic_df = load_pic_csv()
    case_df = load_case_csv()

    train_patients, val_patients = get_patient_split(pic_df, val_ratio=0.15, seed=RANDOM_SEED)
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

    train_bags = build_mil_bags(pic_df, train_patients)
    val_bags = build_mil_bags(pic_df, val_patients)
    print(f"Train bags: {len(train_bags)}, Val bags: {len(val_bags)}")

    train_dataset = MILBagDataset(train_bags, transform=get_train_transforms())
    val_dataset = MILBagDataset(val_bags, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = MILClassifier(dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ConvNeXt-Tiny + MIL Max Pool + MixUp + Label Smoothing + TTA (dropout=0.3)")
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    print(f"MixUp: alpha={MIXUP_ALPHA}, prob={MIXUP_PROB}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print(f"TTA: 3 views (original + hflip + vflip)")

    # Class weights
    weights = compute_pos_weights(train_bags)
    pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"pos_weight: {dict(zip(BIOMARKER_COLS, [f'{w:.2f}' for w in weights]))}")

    criterion = LabelSmoothingBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=LABEL_SMOOTHING)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )

    # AMP
    scaler = GradScaler("cuda")

    # Early stopping
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")

    # Training loop
    best_auc = 0.0
    best_epoch = 0
    best_auc_dict = {}

    print(f"\nStarting training for {NUM_EPOCHS} epochs (TIME_BUDGET={TIME_BUDGET}s)...")
    print("-" * 90)

    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET:
            print(f"\nTIME_BUDGET ({TIME_BUDGET}s) exceeded at epoch {epoch+1}. Stopping.")
            break

        if epoch < WARMUP_EPOCHS:
            lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        val_auc_dict, val_mean_auc = compute_auc(val_labels, val_preds)

        irf_auc = val_auc_dict.get("IRF", 0.0)
        srf_auc = val_auc_dict.get("SRF", 0.0)
        ped_auc = val_auc_dict.get("PED", 0.0)
        hrf_auc = val_auc_dict.get("HRF", 0.0)

        # Step scheduler after warmup
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        print(f"epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_mean_auc: {val_mean_auc:.4f}, "
              f"irf_auc: {irf_auc:.4f}, srf_auc: {srf_auc:.4f}, ped_auc: {ped_auc:.4f}, hrf_auc: {hrf_auc:.4f}")

        if val_mean_auc > best_auc:
            best_auc = val_mean_auc
            best_epoch = epoch + 1
            best_auc_dict = val_auc_dict.copy()

        if early_stopper.step(val_mean_auc):
            print(f"\nEarly stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
            break

    print(f"val_mean_auc: {best_auc:.4f}")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated()/1e6:.0f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete. Best val_mean_auc: {best_auc:.4f} at epoch {best_epoch}")
    print(f"Total time: {total_time:.1f}s / {TIME_BUDGET}s budget")


if __name__ == "__main__":
    main()
