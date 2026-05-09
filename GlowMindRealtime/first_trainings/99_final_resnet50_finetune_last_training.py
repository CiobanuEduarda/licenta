"""Extracted from last_training.ipynb — notebook cell index 3."""

import os
import cv2
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

# --- REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cudnn.benchmark = True

# --- CONFIG ---
BASE_PATH         = "/content/AffectNet/AffectNet"
OUTPUT_MODEL_PATH = "/content/drive/MyDrive/resnet50_va_finetune.pth"
CHECKPOINT_DIR    = "/content/drive/MyDrive/checkpoints_finetune"
LOG_PATH          = "/content/drive/MyDrive/training_log_finetune.txt"
RESUME_PATH       = "/content/drive/MyDrive/checkpoints/epoch_22_cccV0.562_cccA0.463.pth"

BATCH_SIZE  = 64
EPOCHS      = 20
LR_HEAD     = 3e-4
LR_FULL     = 5e-6
PATIENCE    = 6
EMA_DECAY   = 0.999
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA    = DEVICE == "cuda"

TEST_SPLIT_PATH = "/content/drive/MyDrive/test_split_finetune.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- LOG ---
def log(msg):
    print(msg)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')

log(f"Using device: {DEVICE}")

# --- METRICS ---
def concordance_cc(true, pred):
    true, pred           = true.flatten(), pred.flatten()
    mean_true, mean_pred = true.mean(), pred.mean()
    var_true, var_pred   = np.var(true), np.var(pred)
    cov = np.mean((true - mean_true) * (pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

# --- LOSS ---
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.9):  # stronger CCC focus
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        pred_v, pred_a = pred[:, 0], pred[:, 1]
        true_v, true_a = target[:, 0], target[:, 1]

        def ccc(p, t):
            pm, tm = p.mean(), t.mean()
            cov = ((p - pm) * (t - tm)).mean()
            return (2 * cov) / (p.var() + t.var() + (pm - tm) ** 2 + 1e-8)

        ccc_loss = 1 - (ccc(pred_v, true_v) + ccc(pred_a, true_a)) / 2
        mse_loss = self.mse(pred, target)
        return self.alpha * ccc_loss + (1 - self.alpha) * mse_loss

criterion = CombinedLoss(alpha=0.9)

# --- DATASET ---
class AffectNetVA(Dataset):
    def __init__(self, df, root, transform=None):
        self.df        = df.reset_index(drop=True)
        self.root      = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for attempt in range(5):
            row      = self.df.iloc[idx if attempt == 0 else np.random.randint(len(self.df))]
            img_path = os.path.join(self.root, row["subDirectory_filePath"])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                x, y, w, h = map(int, [row.face_x, row.face_y, row.face_width, row.face_height])
                buf   = 0.1
                x_min = max(0, int(x - w * buf))
                y_min = max(0, int(y - h * buf))
                x_max = min(img.shape[1], int(x + w * (1 + buf)))
                y_max = min(img.shape[0], int(y + h * (1 + buf)))

                face = img[y_min:y_max, x_min:x_max]
                if face.size == 0:
                    continue

                face = Image.fromarray(face)
                if self.transform:
                    face = self.transform(face)

                va = torch.tensor(
                    [float(row.valence), float(row.arousal)],
                    dtype=torch.float32
                )
                return face, va
            except Exception:
                continue
        raise ValueError(f"Failed to load valid image after 5 attempts (idx={idx}).")

# --- TRANSFORMS ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),  # noise after normalize
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- LOAD DATA ---
df = pd.read_csv("/content/drive/MyDrive/balanced_sample.csv")
df['valence'] = df['valence'].clip(-1.0, 1.0)
df['arousal'] = df['arousal'].clip(-1.0, 1.0)
df = df.dropna(subset=['valence', 'arousal', 'subDirectory_filePath'])

log(f"Total samples: {len(df)}")

train_df  = df.sample(frac=0.75, random_state=42)
remaining = df.drop(train_df.index)
val_df    = remaining.sample(frac=0.5, random_state=42)
test_df   = remaining.drop(val_df.index)

log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

test_df.to_csv(TEST_SPLIT_PATH, index=False)
log(f"✅ Test split saved to {TEST_SPLIT_PATH}")

train_loader = DataLoader(
    AffectNetVA(train_df, BASE_PATH, train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=USE_CUDA,
    persistent_workers=True, prefetch_factor=2,
)
val_loader = DataLoader(
    AffectNetVA(val_df, BASE_PATH, val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=USE_CUDA,
    persistent_workers=True, prefetch_factor=2,
)
test_loader = DataLoader(
    AffectNetVA(test_df, BASE_PATH, val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=USE_CUDA,
    persistent_workers=True, prefetch_factor=2,
)

# --- MODEL ---
class VAResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features   = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out = self.backbone(x)
        if not self.training:
            out = torch.clamp(out, -1, 1)
        return out

model     = VAResNet().to(DEVICE)
ema_model = VAResNet().to(DEVICE)

# --- LOAD CHECKPOINT (handle _orig_mod. prefix from torch.compile) ---
if os.path.exists(RESUME_PATH):
    ckpt       = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict']
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    bad = model.load_state_dict(state_dict, strict=False)
    ema_model.load_state_dict(state_dict, strict=False)
    if bad.missing_keys:
        log(f"   ⚠️ missing_keys ({len(bad.missing_keys)}): {bad.missing_keys[:8]}{'...' if len(bad.missing_keys) > 8 else ''}")
    if bad.unexpected_keys:
        log(f"   ⚠️ unexpected_keys ({len(bad.unexpected_keys)}): {bad.unexpected_keys[:8]}{'...' if len(bad.unexpected_keys) > 8 else ''}")
    log(f"✅ Loaded checkpoint from {RESUME_PATH}")
    log(f"   Starting from CCC_V={ckpt['ccc_v']:.3f}  CCC_A={ckpt['ccc_a']:.3f}")
else:
    raise RuntimeError(f"❌ Checkpoint not found: {RESUME_PATH}")

if DEVICE == 'cuda':
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {total_mem:.1f} GB")

# --- OPTIMIZER + SCHEDULER ---
optimizer = optim.Adam([
    {'params': model.backbone.fc.parameters(),
     'lr': LR_HEAD},
    {'params': [p for name, p in model.named_parameters()
                if 'backbone.fc' not in name],
     'lr': LR_FULL},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.3,
    patience=3,   # was 1, too aggressive
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

# --- EVALUATE (uses EMA model) ---
def evaluate(loader):
    ema_model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=USE_CUDA)
            with torch.amp.autocast("cuda", enabled=USE_CUDA):
                out = ema_model(imgs)
            preds.append(out.cpu().numpy())
            trues.append(labels.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    rmse  = np.sqrt(((preds - trues) ** 2).mean())
    ccc_v = concordance_cc(trues[:, 0], preds[:, 0])
    ccc_a = concordance_cc(trues[:, 1], preds[:, 1])
    return rmse, ccc_v, ccc_a

# --- TRAINING LOOP ---
best_ccc   = -1.0
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs   = imgs.to(DEVICE, non_blocking=USE_CUDA)
        labels = labels.to(DEVICE, non_blocking=USE_CUDA)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_CUDA):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # EMA update every batch (params decay-averaged; BN buffers copied — running stats are buffers, not params)
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)
            for ema_b, b in zip(ema_model.buffers(), model.buffers()):
                ema_b.copy_(b)

    rmse, ccc_v, ccc_a = evaluate(val_loader)
    avg_ccc = (ccc_v + ccc_a) / 2

    scheduler.step(avg_ccc)

    log(f"Epoch {epoch+1:02d} | Loss: {running_loss/len(train_loader):.4f} "
        f"| RMSE: {rmse:.4f} | CCC_V: {ccc_v:.3f} | CCC_A: {ccc_a:.3f} "
        f"| Avg: {avg_ccc:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if avg_ccc > best_ccc:
        best_ccc   = avg_ccc
        no_improve = 0
        torch.save(ema_model.state_dict(), OUTPUT_MODEL_PATH)
        torch.save({
            'epoch':            epoch,
            'model_state_dict': ema_model.state_dict(),
            'ccc_v':            ccc_v,
            'ccc_a':            ccc_a,
        }, os.path.join(CHECKPOINT_DIR, f"best_epoch_{epoch+1:02d}_cccV{ccc_v:.3f}_cccA{ccc_a:.3f}.pth"))
        log(f"  ⭐ New best EMA model saved (avg CCC={avg_ccc:.3f})")
    else:
        no_improve += 1
        log(f"  No improvement for {no_improve}/{PATIENCE} epochs")
        if no_improve >= PATIENCE:
            log("⏹ Early stopping")
            break

log(f"\n=== FINE-TUNING COMPLETE ===")
log(f"Best avg CCC: {best_ccc:.3f}")

# --- FINAL TEST ---
log("\n=== FINAL TEST RESULTS ===")
ema_model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=DEVICE, weights_only=False))
rmse, ccc_v, ccc_a = evaluate(test_loader)
log(f"Test RMSE : {rmse:.4f}")
log(f"CCC V     : {ccc_v:.3f}")
log(f"CCC A     : {ccc_a:.3f}")
log(f"Avg CCC   : {(ccc_v + ccc_a) / 2:.3f}")