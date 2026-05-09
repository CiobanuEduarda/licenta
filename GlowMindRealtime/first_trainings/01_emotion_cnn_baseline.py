"""Extracted from GlowMind_Emotion_Model.ipynb — cell 3. Tag: 01_cnn_baseline."""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
device

class AffectNetSubset(Dataset):
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

        x, y, w, h = int(row.face_x), int(row.face_y), int(row.face_width), int(row.face_height)
        h_img, w_img, _ = img.shape

        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        face = img[y:y+h, x:x+w]

        face = Image.fromarray(face)

        if self.transform:
            face = self.transform(face)

        valence = torch.tensor(float(row.valence), dtype=torch.float32)
        arousal = torch.tensor(float(row.arousal), dtype=torch.float32)

        return face, torch.stack([valence, arousal])


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

base = "/content/drive/MyDrive/GlowMind_subset"
df = pd.read_csv(base + "/training.csv")

dataset = AffectNetSubset(df, base, transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)


class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, 2)    # output: valence, arousal
        )

    def forward(self, x):
        return self.net(x)


model = EmotionCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")


torch.save(model.state_dict(), "/content/drive/MyDrive/glowmind_model.pth")


