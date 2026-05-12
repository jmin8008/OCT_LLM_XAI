#!/usr/bin/env python3
"""
APTOS 2021 AutoResearch Exp #33: 3 Seeds with New Seed 7 (replacing weak seed 42)
Hypothesis: Seed 42 is the consistent weak link (HRF=0.88-0.90 regardless of LR).
Replacing it with seed 7 at lr=5e-5 might give 3 uniformly strong seeds.
Seeds 123 and 2024 at lr=5e-5 both achieve HRF=0.92+. If seed 7 can match,
the 3-seed ensemble should beat the 2-seed (0.9636) by adding genuine diversity.

Key changes from Exp #32 (0.9636):
- 3 seeds: [7, 123, 2024] (seed 7 replaces seed 42)
- All seeds lr=5e-5
- 290s per seed budget
"""

import os
import sys
import time
import json
import copy

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
    CLASS_NAMES,
)

# ---------------------------------------------------------------------------
BIOMARKER_COLS = ["IRF", "SRF", "PED", "HRF"]
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 8
MAX_BAG_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
POS_WEIGHT_CAP = 10.0
EARLY_STOPPING_PATIENCE = 15
CUDA_DEVICE = 2

BACKBONE_NAME = "convnext_tiny"

EMA_DECAY = 0.999

# Replace seed 42 with seed 7
ENSEMBLE_SEEDS = [7, 123, 2024]

MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5

OCT_X_START = 632
OCT_X_END = 1264
OCT_Y_START = 0
OCT_Y_END = 596


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMAModel:
    """Exponential Moving Average of model parameters with warmup."""
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for p_shadow, p_model in zip(self.shadow.parameters(), model.parameters()):
                p_shadow.data.mul_(decay).add_(p_model.data, alpha=1 - decay)


class MILClassifier(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME,
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
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
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


def train_one_epoch(model, ema_model, loader, criterion, optimizer, scaler, device):
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

        ema_model.update(model)

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
def validate_ema(ema_model, loader, device):
    ema_model.shadow.eval()
    all_labels = []
    all_preds = []

    for images, labels, mask in loader:
        images = images.to(device)
        mask = mask.to(device)

        with autocast(device_type="cuda"):
            logits = ema_model.shadow(images, mask)

        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds, all_labels


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


def train_one_seed(seed, train_bags, val_bags, device, seed_time_budget):
    print(f"\n{'='*60}")
    print(f"  SEED {seed} (lr={LEARNING_RATE}) — budget: {seed_time_budget}s")
    print(f"{'='*60}")

    set_seed(seed)

    train_dataset = MILBagDataset(train_bags, transform=get_train_transforms())
    val_dataset = MILBagDataset(val_bags, transform=get_val_transforms())

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

    set_seed(seed)
    model = MILClassifier(dropout=0.3).to(device)
    ema_model = EMAModel(model, decay=EMA_DECAY)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {BACKBONE_NAME} + MIL Max Pool + MixUp + EMA({EMA_DECAY})")
    print(f"Total params: {total_params:,}")

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
    best_mean_preds = None
    best_labels = None

    per_class_best_auc = [0.0] * NUM_CLASSES
    per_class_best_epoch = [0] * NUM_CLASSES
    per_class_best_preds = [None] * NUM_CLASSES

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

        train_loss, _, _ = train_one_epoch(model, ema_model, train_loader, criterion, optimizer, scaler, device)

        ema_preds, val_labels = validate_ema(ema_model, val_loader, device)
        ema_auc_dict, ema_mean_auc = compute_auc(val_labels, ema_preds)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        irf_auc = ema_auc_dict.get("IRF", 0.0)
        srf_auc = ema_auc_dict.get("SRF", 0.0)
        ped_auc = ema_auc_dict.get("PED", 0.0)
        hrf_auc = ema_auc_dict.get("HRF", 0.0)

        print(f"  seed={seed}, epoch: {epoch+1}, train_loss: {train_loss:.4f}, "
              f"ema_mean_auc: {ema_mean_auc:.4f}, irf: {irf_auc:.4f}, srf: {srf_auc:.4f}, "
              f"ped: {ped_auc:.4f}, hrf: {hrf_auc:.4f}")

        class_aucs = [irf_auc, srf_auc, ped_auc, hrf_auc]
        for c in range(NUM_CLASSES):
            if class_aucs[c] > per_class_best_auc[c]:
                per_class_best_auc[c] = class_aucs[c]
                per_class_best_epoch[c] = epoch + 1
                per_class_best_preds[c] = ema_preds[:, c].copy()

        if ema_mean_auc > best_auc:
            best_auc = ema_mean_auc
            best_epoch = epoch + 1
            best_mean_preds = ema_preds.copy()
            best_labels = val_labels.copy()

        if early_stopper.step(ema_mean_auc):
            print(f"  Seed {seed}: Early stopping at epoch {epoch+1}")
            break

    seed_time = time.time() - start_time
    print(f"  Seed {seed}: Best mean AUC: {best_auc:.4f} at epoch {best_epoch} ({seed_time:.1f}s)")
    for c in range(NUM_CLASSES):
        print(f"    {BIOMARKER_COLS[c]}: best AUC={per_class_best_auc[c]:.4f} at epoch {per_class_best_epoch[c]}")

    del model, ema_model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()

    return {
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "best_mean_preds": best_mean_preds,
        "best_labels": best_labels,
        "seed_time": seed_time,
        "per_class_best_auc": per_class_best_auc,
        "per_class_best_epoch": per_class_best_epoch,
        "per_class_best_preds": per_class_best_preds,
    }


def main():
    overall_start = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n3-Seed (7+123+2024) lr={LEARNING_RATE} EMA({EMA_DECAY}) + Per-Class Best")
    print(f"Seeds: {ENSEMBLE_SEEDS} (seed 7 replaces weak seed 42)")
    print(f"Total budget: {TIME_BUDGET}s")

    print("\nLoading data via prepare.py...")
    pic_df = load_pic_csv()
    case_df = load_case_csv()

    train_patients, val_patients = get_patient_split(pic_df, val_ratio=0.15, seed=42)
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

    train_bags = build_mil_bags(pic_df, train_patients)
    val_bags = build_mil_bags(pic_df, val_patients)
    print(f"Train bags: {len(train_bags)}, Val bags: {len(val_bags)}")

    all_results = []

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        remaining = TIME_BUDGET - (time.time() - overall_start)
        if remaining < 60:
            print(f"\nNot enough time for seed {seed} ({remaining:.0f}s remaining)")
            break

        remaining_seeds = len(ENSEMBLE_SEEDS) - i
        seed_budget = int(remaining / remaining_seeds)

        result = train_one_seed(seed, train_bags, val_bags, device, seed_budget)
        all_results.append(result)

    # =========================================================================
    # Ensemble Results
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE RESULTS")
    print(f"{'='*60}")

    for i, seed in enumerate(ENSEMBLE_SEEDS[:len(all_results)]):
        r = all_results[i]
        print(f"  Seed {seed}: Best mean AUC={r['best_auc']:.4f} (epoch {r['best_epoch']}, {r['seed_time']:.0f}s)")
        print(f"    Per-class best: IRF={r['per_class_best_auc'][0]:.4f}(ep{r['per_class_best_epoch'][0]}), "
              f"SRF={r['per_class_best_auc'][1]:.4f}(ep{r['per_class_best_epoch'][1]}), "
              f"PED={r['per_class_best_auc'][2]:.4f}(ep{r['per_class_best_epoch'][2]}), "
              f"HRF={r['per_class_best_auc'][3]:.4f}(ep{r['per_class_best_epoch'][3]})")

    labels = all_results[0]["best_labels"]
    n_seeds = len(all_results)

    # --- Method 1: Simple mean ---
    simple_preds = np.mean([r["best_mean_preds"] for r in all_results], axis=0)
    simple_auc_dict, simple_mean_auc = compute_auc(labels, simple_preds)
    print(f"\n  [1] Simple Mean (best mean-AUC epoch): {simple_mean_auc:.4f}")
    for k, v in simple_auc_dict.items():
        print(f"      {k}: {v:.4f}")

    # --- Method 2: Per-class best + simple mean ---
    pc_simple_preds = np.zeros_like(simple_preds)
    for c in range(NUM_CLASSES):
        class_preds_list = [r["per_class_best_preds"][c] for r in all_results]
        pc_simple_preds[:, c] = np.mean(class_preds_list, axis=0)

    pc_auc_dict, pc_mean_auc = compute_auc(labels, pc_simple_preds)
    print(f"\n  [2] Per-Class Best + Simple Mean: {pc_mean_auc:.4f}")
    for k, v in pc_auc_dict.items():
        print(f"      {k}: {v:.4f}")

    # --- Method 3: Per-class best + squared-AUC weighted ---
    pc_sq_preds = np.zeros_like(simple_preds)
    for c in range(NUM_CLASSES):
        class_aucs = np.array([r["per_class_best_auc"][c] for r in all_results])
        sq_weights = (class_aucs ** 2) / (class_aucs ** 2).sum()
        class_preds_list = [r["per_class_best_preds"][c] for r in all_results]
        for w, p in zip(sq_weights, class_preds_list):
            pc_sq_preds[:, c] += w * p

    pcsq_auc_dict, pcsq_mean_auc = compute_auc(labels, pc_sq_preds)
    print(f"\n  [3] Per-Class Best + Squared-AUC Weighted: {pcsq_mean_auc:.4f}")
    for k, v in pcsq_auc_dict.items():
        print(f"      {k}: {v:.4f}")

    # --- Method 4: Per-class best + cubed-AUC weighted ---
    pc_cb_preds = np.zeros_like(simple_preds)
    for c in range(NUM_CLASSES):
        class_aucs = np.array([r["per_class_best_auc"][c] for r in all_results])
        cb_weights = (class_aucs ** 3) / (class_aucs ** 3).sum()
        class_preds_list = [r["per_class_best_preds"][c] for r in all_results]
        for w, p in zip(cb_weights, class_preds_list):
            pc_cb_preds[:, c] += w * p

    pcb_auc_dict, pcb_mean_auc = compute_auc(labels, pc_cb_preds)
    print(f"\n  [4] Per-Class Best + Cubed-AUC Weighted: {pcb_mean_auc:.4f}")
    for k, v in pcb_auc_dict.items():
        print(f"      {k}: {v:.4f}")

    # Summary
    methods = [
        ("Simple Mean (baseline)", simple_mean_auc),
        ("Per-Class Best + Simple Mean", pc_mean_auc),
        ("Per-Class Best + Squared-AUC Weighted", pcsq_mean_auc),
        ("Per-Class Best + Cubed-AUC Weighted", pcb_mean_auc),
    ]
    best_method = max(methods, key=lambda x: x[1])
    print(f"\n  Best method: {best_method[0]} = {best_method[1]:.4f}")
    for name, auc in methods:
        delta = auc - simple_mean_auc
        print(f"    {name}: {auc:.4f} ({delta:+.4f})")

    print(f"\n  peak_vram_mb: {torch.cuda.max_memory_allocated()/1e6:.0f}")

    total_time = time.time() - overall_start
    print(f"\nTotal time: {total_time:.1f}s / {TIME_BUDGET}s budget")


if __name__ == "__main__":
    main()
