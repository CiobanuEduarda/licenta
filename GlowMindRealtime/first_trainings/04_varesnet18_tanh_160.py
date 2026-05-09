"""Extracted from GlowMind_Emotion_Model.ipynb — cell 9. Tag: varesnet_resnet18."""

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
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BASE_PATH = "/content/GlowMind_subset"
OUTPUT_MODEL_PATH = "/content/drive/MyDrive/resnet18_va_model.pth"
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(BASE_PATH):
    print("🚀 Local data not found. Copying and unzipping...")
    !cp /content/drive/MyDrive/GlowMind_subset.zip /content/GlowMind_subset.zip
    !unzip -q /content/GlowMind_subset.zip -d /content/
    !rm /content/GlowMind_subset.zip
    print("✅ Done!")
else:
    print("        already exists locally. Skipping copy.")


# --- METRICS ---
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        for attempt in range(5):
            # If fail, try a random index
            current_row = self.df.iloc[np.random.randint(len(self.df))] if attempt > 0 else row
            img_path = os.path.join(self.root, current_row["subDirectory_filePath"])

            try:
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                x, y, w, h = map(int, [current_row.face_x, current_row.face_y, current_row.face_width, current_row.face_height])
                buffer = 0.1
                x_min, y_min = max(0, int(x - w * buffer)), max(0, int(y - h * buffer))
                x_max, y_max = min(img.shape[1], int(x + w * (1 + buffer))), min(img.shape[0], int(y + h * (1 + buffer)))

                face = img[y_min:y_max, x_min:x_max]
                if face.size == 0: continue

                face = Image.fromarray(face)
                if self.transform: face = self.transform(face)

                va = torch.tensor([float(current_row.valence), float(current_row.arousal)], dtype=torch.float32)
                return face, va
            except Exception:
                continue
        raise ValueError("Failed to find valid image after 5 attempts.")

# --- MODEL ARCHITECTURE ---
class VAResNet(nn.Module):
    def __init__(self):
        super(VAResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.activation = nn.Tanh() # Keeps outputs in [-1, 1] range

    def forward(self, x):
        return self.activation(self.backbone(x))

# --- TRANSFORMS ---
train_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# --- PREPARE DATA ---
df = pd.read_csv(os.path.join(BASE_PATH, "training.csv"))
train_df = df.sample(frac=0.85, random_state=42)
val_df = df.drop(train_df.index)

train_loader = DataLoader(AffectNetVA(train_df, BASE_PATH, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(AffectNetVA(val_df, BASE_PATH, val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- INITIALIZE ---
model = VAResNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min')
scaler = torch.amp.GradScaler('cuda')

# --- EVALUATION FUNCTION ---
def evaluate(loader, model, criterion):
    model.eval()
    preds, trues = [], []
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            val_loss += loss.item()
            preds.append(out.cpu().numpy())
            trues.append(labels.cpu().numpy())

    preds, trues = np.vstack(preds), np.vstack(trues)
    rmse = np.sqrt(((preds - trues) ** 2).mean())
    ccc_v = concordance_cc(trues[:,0], preds[:,0])
    ccc_a = concordance_cc(trues[:,1], preds[:,1])
    return val_loss / len(loader), rmse, ccc_v, ccc_a

# --- TRAINING LOOP ---
history = {'train_loss': [], 'val_loss': [], 'ccc_v': [], 'ccc_a': []}

for e in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    v_loss, v_rmse, v_ccc_v, v_ccc_a = evaluate(val_loader, model, criterion)
    scheduler.step(v_loss)

    history['train_loss'].append(running_loss/len(train_loader))
    history['val_loss'].append(v_loss)
    history['ccc_v'].append(v_ccc_v)
    history['ccc_a'].append(v_ccc_a)

    print(f"Loss: {v_loss:.4f} | RMSE: {v_rmse:.4f} | CCC_V: {v_ccc_v:.3f} | CCC_A: {v_ccc_a:.3f}")

# --- SAVE & PLOT ---
torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['ccc_v'], label='CCC Valence'); plt.plot(history['ccc_a'], label='CCC Arousal')
plt.title('CCC History'); plt.legend()
plt.show()