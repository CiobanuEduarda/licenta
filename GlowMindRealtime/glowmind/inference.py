"""ResNet VA model, face cascade, and image preprocessing."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50


def build_va_resnet(device: str) -> torch.nn.Module:
    model = resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(128, 2),
    )
    model.to(device)
    model.eval()
    return model


def _strip_checkpoint_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    fixed: dict[str, Any] = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("."):
            k = k[1:]
        if k.startswith("backbone."):
            k = k[len("backbone.") :]
        fixed[k] = value
    return fixed


def load_model_weights(model: torch.nn.Module, weights_path: str, device: str) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: '{weights_path}'.\n"
            "- If the file is in this folder, set MODEL_WEIGHTS to its filename.\n"
            "- Or set env var MODEL_WEIGHTS to an absolute path, e.g.:\n"
            "  MODEL_WEIGHTS=/path/to/your_model.pth python main.py"
        )
    try:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_checkpoint_prefixes(state)
    model.load_state_dict(state, strict=True)


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets", "haarcascade_frontalface_default.xml"
    )
    if not os.path.exists(cascade_path):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        except AttributeError:
            cascade_path = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(
            "Could not load Haar cascade for face detection. "
            "Install OpenCV data files or provide a valid cascade path."
        )
    return cascade


def select_primary_face(
    faces: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Choose one face when Haar returns multiple detections.

    Prefer the largest bounding box (typically the nearer / main subject); break
    ties by proximity to the frame center to reduce flicker when two faces are
    similar in size.
    """
    if faces.ndim == 1:
        faces = faces.reshape(1, -1)
    fw = float(frame_width)
    fh = float(frame_height)
    cx0 = fw / 2.0
    cy0 = fh / 2.0

    def sort_key(row: np.ndarray) -> tuple[float, float]:
        x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        area = w * h
        fcx = x + w / 2.0
        fcy = y + h / 2.0
        dist_sq = (fcx - cx0) ** 2 + (fcy - cy0) ** 2
        return (area, -dist_sq)

    best = max(faces, key=sort_key)
    return int(best[0]), int(best[1]), int(best[2]), int(best[3])


def expand_face_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
    buffer: float = 0.1,
) -> tuple[int, int, int, int]:
    """Pad the detection box like AffectNet training (face box + margin)."""
    x_min = max(0, int(x - w * buffer))
    y_min = max(0, int(y - h * buffer))
    x_max = min(frame_width, int(x + w * (1.0 + buffer)))
    y_max = min(frame_height, int(y + h * (1.0 + buffer)))
    out_w = max(1, x_max - x_min)
    out_h = max(1, y_max - y_min)
    return x_min, y_min, out_w, out_h


def forward_va(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Run the VA head and clamp to [-1, 1] (matches fine-tune eval)."""
    out = model(batch)
    return torch.clamp(out, -1.0, 1.0)


def face_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
