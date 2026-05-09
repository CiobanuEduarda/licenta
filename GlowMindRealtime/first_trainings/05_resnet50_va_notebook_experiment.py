"""Extracted from GlowMind_Emotion_Model.ipynb — cell 42. Tag: varesnet_resnet50."""

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
import matplotlib.pyplot as plt

# --- REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION ---
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
BASE_PATH         = "/content/AffectNet/AffectNet"
OUTPUT_MODEL_PATH = "/content/drive/MyDrive/resnet50_va_v4.pth"
CHECKPOINT_DIR    = "/content/drive/MyDrive/checkpoints_v4"
LOG_PATH          = "/content/drive/MyDrive/training_log_v4.txt"
BATCH_SIZE        = 64
NUM_WORKERS       = 2
EPOCHS            = 35
LR_HEAD           = 1e-3
LR_FULL           = 3e-5
UNFREEZE_EPOCH    = 2
PATIENCE          = 8
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- LOGGING ---
def log(msg):
    print(msg)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')

log(f"Using device: {DEVICE}")

# --- DATA SETUP ---
if not os.path.exists(BASE_PATH):
    raise RuntimeError("❌ AffectNet not found. Run the extraction script first.")
else:
    log("✅ Data found locally.")

# --- METRICS ---
def concordance_cc(true, pred):
    true, pred           = true.flatten(), pred.flatten()
    mean_true, mean_pred = true.mean(), pred.mean()
    var_true, var_pred   = np.var(true), np.var(pred)
    cov = np.mean((true - mean_true) * (pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

# --- LOSS ---
class WeightedCombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, w_v=1.0, w_a=1.5):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.w_v   = w_v
        self.w_a   = w_a

    def forward(self, pred, target):
        pv, pa = pred[:, 0],   pred[:, 1]
        tv, ta = target[:, 0], target[:, 1]

        def ccc(p, t):
            pm, tm = p.mean(), t.mean()
            cov = ((p - pm) * (t - tm)).mean()
            return (2 * cov) / (p.var() + t.var() + (pm - tm) ** 2 + 1e-8)

        ccc_v    = ccc(pv, tv)
        ccc_a    = ccc(pa, ta)
        ccc_loss = 1 - (self.w_v * ccc_v + self.w_a * ccc_a) / (self.w_v + self.w_a)
        mse_loss = self.mse(pred, target)
        return self.alpha * ccc_loss + (1 - self.alpha) * mse_loss

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
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- PREPARE DATA ---
df = pd.read_csv("/content/drive/MyDrive/balanced_sample.csv")
df['valence'] = df['valence'].clip(-1.0, 1.0)
df['arousal'] = df['arousal'].clip(-1.0, 1.0)
df = df.dropna(subset=['valence', 'arousal', 'subDirectory_filePath'])

log(f"Total samples : {len(df)}")
log(f"Valence — mean: {df['valence'].mean():.3f}  std: {df['valence'].std():.3f}")
log(f"Arousal — mean: {df['arousal'].mean():.3f}  std: {df['arousal'].std():.3f}")

# --- VALIDATE PATHS ---
log("🔍 Validating dataset paths...")
missing = sum(
    1 for path in df['subDirectory_filePath']
    if not os.path.exists(os.path.join(BASE_PATH, path))
)
log(f"Missing files: {missing} / {len(df)}")
if missing > len(df) * 0.95:
    raise RuntimeError(f"Too many missing files ({missing}). Re-run extraction.")

# --- SPLIT ---
train_df  = df.sample(frac=0.75, random_state=42)
remaining = df.drop(train_df.index)
val_df    = remaining.sample(frac=0.5, random_state=42)
test_df   = remaining.drop(val_df.index)

log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
test_df.to_csv("/content/drive/MyDrive/test_split.csv", index=False)
log("✅ Test split saved.")

# --- DATALOADERS ---
train_loader = DataLoader(
    AffectNetVA(train_df, BASE_PATH, train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2,
)
val_loader = DataLoader(
    AffectNetVA(val_df, BASE_PATH, val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2,
)
test_loader = DataLoader(
    AffectNetVA(test_df, BASE_PATH, val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2,
)

# --- MODEL ---
class VAResNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Use resnet50")

        in_features = self.backbone.fc.in_features
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

model = VAResNet(backbone="resnet50").to(DEVICE)
model = model.to(memory_format=torch.channels_last)

if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
        log("✅ Model compiled with torch.compile")
    except Exception as e:
        log(f"⚠️ torch.compile failed: {e}")

if DEVICE == 'cuda':
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {total_mem:.1f} GB")

# --- FREEZE / UNFREEZE ---
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'backbone.fc' not in name:
            param.requires_grad = False
    log("🔒 Backbone frozen — training head only")

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    log(f"🔓 Backbone unfrozen — LR_FULL: {LR_FULL}")

freeze_backbone(model)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD
)
# Start with no scheduler during frozen phase
scheduler      = None
backbone_unfrozen = False
scaler         = torch.amp.GradScaler('cuda')
criterion      = WeightedCombinedLoss(alpha=0.7, w_v=1.0, w_a=1.5)

# --- SAVE CONFIG ---
config = {
    'backbone':       'resnet50',
    'batch_size':     BATCH_SIZE,
    'epochs':         EPOCHS,
    'lr_head':        LR_HEAD,
    'lr_full':        LR_FULL,
    'unfreeze_epoch': UNFREEZE_EPOCH,
    'patience':       PATIENCE,
    'loss':           'WeightedCombinedLoss(alpha=0.7, w_v=1.0, w_a=1.5)',
    'train_samples':  len(train_df),
    'val_samples':    len(val_df),
    'test_samples':   len(test_df),
}
with open("/content/drive/MyDrive/training_config_v4.json", 'w') as f:
    json.dump(config, f, indent=2)
log("✅ Config saved.")

# --- RESUME ---
RESUME_PATH = None  # set to checkpoint path to resume e.g. ".../epoch_10_....pth"
start_epoch = 0

if RESUME_PATH and os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    start_epoch = ckpt['epoch'] + 1
    log(f"✅ Resumed from epoch {ckpt['epoch']+1} "
        f"(CCC_V={ckpt['ccc_v']:.3f}  CCC_A={ckpt['ccc_a']:.3f})")

# --- EVALUATION ---
def evaluate(loader, model, criterion):
    model.eval()
    preds, trues = [], []
    total_loss   = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out  = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item()
            preds.append(out.cpu().numpy())
            trues.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    rmse  = np.sqrt(((preds - trues) ** 2).mean())
    ccc_v = concordance_cc(trues[:, 0], preds[:, 0])
    ccc_a = concordance_cc(trues[:, 1], preds[:, 1])
    return total_loss / len(loader), rmse, ccc_v, ccc_a

# --- TRAINING LOOP ---
history    = {'train_loss': [], 'val_loss': [], 'ccc_v': [], 'ccc_a': []}
best_ccc   = -1.0
best_ckpt  = None
no_improve = 0

for e in range(start_epoch, EPOCHS):

    # Unfreeze backbone and start ReduceLROnPlateau
    if e == UNFREEZE_EPOCH and not backbone_unfrozen:
        unfreeze_backbone(model)
        backbone_unfrozen = True
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
            factor=0.5,
            patience=4,
        )
        log(f"🔓 Optimizer rebuilt + ReduceLROnPlateau started at epoch {e+1}")

    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS}"):
        imgs   = imgs.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    v_loss, v_rmse, v_ccc_v, v_ccc_a = evaluate(val_loader, model, criterion)
    avg_ccc = (v_ccc_v + v_ccc_a) / 2

    # Step scheduler after evaluation (ReduceLROnPlateau needs the metric)
    if scheduler is not None:
        scheduler.step(avg_ccc)

    history['train_loss'].append(running_loss / len(train_loader))
    history['val_loss'].append(v_loss)
    history['ccc_v'].append(v_ccc_v)
    history['ccc_a'].append(v_ccc_a)

    log(f"Epoch {e+1:02d} | Loss: {v_loss:.4f} | RMSE: {v_rmse:.4f} "
        f"| CCC_V: {v_ccc_v:.3f} | CCC_A: {v_ccc_a:.3f} "
        f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        f"epoch_{e+1:02d}_cccV{v_ccc_v:.3f}_cccA{v_ccc_a:.3f}.pth"
    )
    torch.save({
        'epoch':            e,
        'model_state_dict': model.state_dict(),
        'ccc_v':            v_ccc_v,
        'ccc_a':            v_ccc_a,
    }, ckpt_path)

    if avg_ccc > best_ccc:
        best_ccc   = avg_ccc
        best_ckpt  = ckpt_path
        no_improve = 0
        torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        log(f"  ⭐ New best model saved (avg CCC={avg_ccc:.3f})")
    else:
        no_improve += 1
        log(f"  No improvement for {no_improve}/{PATIENCE} epochs")
        if no_improve >= PATIENCE:
            log("⏹ Early stopping triggered")
            break

log(f"\n=== TRAINING COMPLETE ===")
log(f"Best avg CCC: {best_ccc:.3f}  →  {best_ckpt}")

# --- PLOT ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'],   label='Val Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['ccc_v'], label='CCC Valence')
plt.plot(history['ccc_a'], label='CCC Arousal')
plt.title('CCC History')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/training_history_v4.png", dpi=150)
plt.show()
log("✅ Plot saved.")

# --- FINAL TEST ---
log("\n=== FINAL TEST RESULTS ===")
best_model = VAResNet(backbone="resnet50").to(DEVICE)
state_dict = torch.load(OUTPUT_MODEL_PATH, map_location=DEVICE, weights_only=False)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
best_model.load_state_dict(state_dict)
test_loss, test_rmse, test_ccc_v, test_ccc_a = evaluate(test_loader, best_model, criterion)
log(f"Test Loss : {test_loss:.4f}")
log(f"Test RMSE : {test_rmse:.4f}")
log(f"CCC V     : {test_ccc_v:.3f}")
log(f"CCC A     : {test_ccc_a:.3f}")
log(f"Avg CCC   : {(test_ccc_v + test_ccc_a) / 2:.3f}")