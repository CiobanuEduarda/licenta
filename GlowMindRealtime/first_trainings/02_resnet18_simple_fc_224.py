"""Extracted from GlowMind_Emotion_Model.ipynb — cell 4. Tag: resnet18_linear_head."""

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

def concordance_cc(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    mean_true, mean_pred = true.mean(), pred.mean()
    var_true, var_pred = np.var(true), np.var(pred)
    cov = np.mean((true - mean_true) * (pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

class AffectNetVA(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["subDirectory_filePath"])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Missing image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # crop
        x, y, w, h = map(int, [row.face_x, row.face_y, row.face_width, row.face_height])
        face = img[y:y+h, x:x+w]

        face = Image.fromarray(face)

        if self.transform:
            face = self.transform(face)

        va = torch.tensor([float(row.valence), float(row.arousal)], dtype=torch.float32)
        return face, va

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

base = "/content/drive/MyDrive/GlowMind_subset"
df = pd.read_csv(base + "/training.csv")

# split 85% train / 15% val
train_df = df.sample(frac=0.85, random_state=42)
val_df = df.drop(train_df.index)

train_set = AffectNetVA(train_df, base, train_tf)
val_set = AffectNetVA(val_df, base, val_tf)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# replace last layer for 2 outputs
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

def evaluate(loader, model):
    model.eval()
    preds, trues = [], []
    val_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
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


epochs = 20

for e in range(epochs):
    model.train()
    running = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running += loss.item()

    val_loss, rmse, corr_v, corr_a, ccc_v, ccc_a = evaluate(val_loader, model)
    scheduler.step(val_loss)

    print(f"\nEpoch {e+1}/{epochs}")
    print(f"Train Loss: {running/len(train_loader):.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"Corr(V,A):  {corr_v:.3f}, {corr_a:.3f}")
    print(f"CCC (V,A):  {ccc_v:.3f}, {ccc_a:.3f}")


torch.save(model.state_dict(), "/content/drive/MyDrive/resnet18_va_model.pth")
