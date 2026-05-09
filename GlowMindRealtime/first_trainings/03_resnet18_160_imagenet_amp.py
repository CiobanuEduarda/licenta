"""Extracted from GlowMind_Emotion_Model.ipynb — cell 7. Tag: resnet18_linear_head."""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from torch.cuda.amp import autocast, GradScaler # For Mixed Precision (Optimization)

# --- CONFIGURATION ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BASE_PATH = "/content/GlowMind_subset"
OUTPUT_MODEL_PATH = "/content/drive/MyDrive/resnet18_va_model.pth"
BATCH_SIZE = 64 # Increased batch size due to Mixed Precision (Test 32 if 64 runs out of memory)
NUM_WORKERS = 2 # Increased workers for faster data loading
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def concordance_cc(true, pred):
    """Calculates the Concordance Correlation Coefficient (CCC)."""
    true = true.flatten()
    pred = pred.flatten()
    mean_true, mean_pred = true.mean(), pred.mean()
    var_true, var_pred = np.var(true), np.var(pred)
    cov = np.mean((true - mean_true) * (pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

# --- DATASET CLASS ---
class AffectNetVA(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # --- OPTIMIZED AFFECTNETVA __getitem__ METHOD ---

    def __getitem__(self, idx):
        # We start with the requested index
        row = self.df.iloc[idx]

        # --- Bounded Retry Loop ---
        for attempt in range(5):
            row = self.df.iloc[np.random.randint(len(self.df))] if attempt > 0 else self.df.iloc[idx]
            img_path = os.path.join(self.root, row["subDirectory_filePath"])

            try:
                # 1. Image Check
                img = cv2.imread(img_path)
                if img is None:
                    continue # Try next random sample
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 2. Face Crop Check
                x, y, w, h = map(int, [row.face_x, row.face_y, row.face_width, row.face_height])
                buffer = 0.1
                x_min = max(0, int(x - w * buffer))
                y_min = max(0, int(y - h * buffer))
                x_max = min(img.shape[1], int(x + w * (1 + buffer)))
                y_max = min(img.shape[0], int(y + h * (1 + buffer)))

                face = img[y_min:y_max, x_min:x_max]

                if face.size == 0:
                    continue # Try next random sample

                # If we reached here, the sample is good!
                face = Image.fromarray(face)

                if self.transform:
                    face = self.transform(face)

                va = torch.tensor([float(row.valence), float(row.arousal)], dtype=torch.float32)
                return face, va

            except Exception as e:
                # Catch any other reading/processing errors
                print(f"Error processing {img_path}: {e}")
                continue

        # If the loop completes without a return (i.e., 5 failed attempts)
        raise ValueError("Failed to find a valid image after 5 attempts. Check dataset integrity.")

# --- TRANSFORMS (OPTIMIZED NORMALIZATION) ---
train_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) # ImageNet Normalization
])

val_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) # ImageNet Normalization
])

# --- DATA LOADING ---
df = pd.read_csv(BASE_PATH + "/training.csv")

# Split 85% train / 15% val
train_df = df.sample(frac=0.85, random_state=42)
val_df = df.drop(train_df.index)

train_set = AffectNetVA(train_df, BASE_PATH, train_tf)
val_set = AffectNetVA(val_df, BASE_PATH, val_tf)

# DataLoader with Pinned Memory (Optimization)
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# --- MODEL SETUP (Transfer Learning) ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace last layer for 2 outputs (Valence and Arousal)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min')

# Initialize GradScaler for Mixed Precision (Optimization)
scaler = GradScaler()

# --- EVALUATION FUNCTION ---
def evaluate(loader, model):
    model.eval()
    preds, trues = [], []
    val_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with autocast(): # Use autocast even in eval for speed/consistency
                out = model(imgs)
                loss = criterion(out, labels)

            val_loss += loss.item()

            preds.append(out.cpu().numpy())
            trues.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    rmse = np.sqrt(((preds - trues) ** 2).mean())
    corr_val = pearsonr(trues[:,0], preds[:,0])[0]
    corr_aro = pearsonr(trues[:,1], preds[:,1])[0]

    ccc_val = concordance_cc(trues[:,0], preds[:,0])
    ccc_aro = concordance_cc(trues[:,1], preds[:,1])

    return val_loss / len(loader), rmse, corr_val, corr_aro, ccc_val, ccc_aro

# --- TRAINING LOOP (Optimized with Mixed Precision) ---
print(f"Starting training on device: {DEVICE} with batch size {BATCH_SIZE}")

for e in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # 🌟 Mixed Precision Context Manager
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)

        # 🌟 Scaler for backward pass and step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    val_loss, rmse, corr_v, corr_a, ccc_v, ccc_a = evaluate(val_loader, model)
    scheduler.step(val_loss)

    print(f"\nEpoch {e+1}/{EPOCHS}")
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"Corr(V,A):  {corr_v:.3f}, {corr_a:.3f}")
    print(f"CCC (V,A):  {ccc_v:.3f}, {ccc_a:.3f}")

# --- SAVE MODEL ---
torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
print(f"\nModel saved to: {OUTPUT_MODEL_PATH}")