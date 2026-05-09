"""
Grad-CAM analysis for GlowMind VA regression.

What it does:
- Samples N images from a dataset CSV
- Reproduces the same face crop + preprocessing used at runtime
- Computes Grad-CAM heatmaps for valence and arousal outputs
- Quantifies "energy" in coarse face regions (upper/middle/lower bands)
- Exports overlays + a per-image CSV + an aggregate bar chart

Notes:
- The region split is a lightweight proxy (upper ~ eyes/eyebrows, lower ~ mouth).
- The heatmaps explain the model's sensitivity, not ground-truth attention.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

# Allow running as a script: `python tools/gradcam_va_analysis.py ...`
# (When executed as a file, Python sets sys.path[0] to `tools/`, so we add repo root.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glowmind.inference import build_va_resnet, expand_face_bbox, face_transform, forward_va, load_model_weights

try:
    from pytorch_grad_cam import GradCAM
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency for Grad-CAM. Install with:\n"
        "  pip install -e '.[analysis]'\n"
        "or:\n"
        "  pip install grad-cam\n"
    ) from e

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency for plotting. Install with:\n"
        "  pip install -e '.[analysis]'\n"
        "or:\n"
        "  pip install matplotlib\n"
    ) from e


@dataclass(frozen=True)
class CsvSpec:
    path_col: str
    valence_col: str
    arousal_col: str
    face_x_col: str
    face_y_col: str
    face_w_col: str
    face_h_col: str


class VAIndexTarget:
    """Grad-CAM target for regression: maximize one scalar output index."""

    def __init__(self, idx: int) -> None:
        self.idx = int(idx)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # pytorch-grad-cam passes either:
        # - a single output tensor shaped (2,) for one image, or
        # - a batched tensor shaped (B, 2)
        if model_output.ndim == 1:
            return model_output[self.idx]
        return model_output[:, self.idx].sum()


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _safe_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _safe_int(row: dict[str, str], key: str) -> int | None:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _load_face_from_row(
    row: dict[str, str],
    *,
    image_root: Path,
    spec: CsvSpec,
    bbox_buffer: float,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    rel = row.get(spec.path_col, "")
    if not rel:
        return None
    img_path = (image_root / rel).expanduser()
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Optional: crop using AffectNet-provided bbox if available
    x = _safe_int(row, spec.face_x_col)
    y = _safe_int(row, spec.face_y_col)
    w = _safe_int(row, spec.face_w_col)
    h = _safe_int(row, spec.face_h_col)
    h_frame, w_frame = img.shape[:2]
    if None not in (x, y, w, h) and w is not None and h is not None and w > 1 and h > 1:
        px, py, pw, ph = expand_face_bbox(x, y, w, h, w_frame, h_frame, bbox_buffer)
        face = img[py : py + ph, px : px + pw]
        if face.size == 0:
            return None
        meta = {"img_path": str(img_path), "crop_mode": "csv_bbox", "bbox": (px, py, pw, ph)}
        return face, meta

    # Fallback: use whole image (still works but less precise)
    meta = {"img_path": str(img_path), "crop_mode": "full_image", "bbox": None}
    return img, meta


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0.0)
    m = float(cam.max()) if cam.size else 0.0
    if m <= 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam / m).astype(np.float32)


def _cam_energy_by_bands(cam01: np.ndarray) -> dict[str, float]:
    """Return % energy in upper/middle/lower bands (sum to 100)."""
    h, w = cam01.shape[:2]
    if h <= 0 or w <= 0:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}

    # Proxy regions on 224x224 face crop:
    # - upper: eyes + eyebrows (top 40%)
    # - middle: nose / cheeks (next 30%)
    # - lower: mouth + jaw (bottom 30%)
    a0 = int(0.40 * h)
    a1 = int(0.70 * h)
    upper = cam01[:a0, :].sum()
    middle = cam01[a0:a1, :].sum()
    lower = cam01[a1:, :].sum()
    total = float(upper + middle + lower)
    if total <= 1e-12:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}
    return {
        "upper": float(100.0 * upper / total),
        "middle": float(100.0 * middle / total),
        "lower": float(100.0 * lower / total),
    }


def _overlay_cam_on_bgr(face_bgr: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    cam_resized = cv2.resize(cam01, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(255 * cam_resized)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(face_bgr, 1.0 - alpha, heat, alpha, 0.0)
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_panel(
    out_path: Path,
    *,
    face_bgr: np.ndarray,
    overlay_v: np.ndarray,
    overlay_a: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    # 3-panel: original | valence | arousal
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    ov_v = cv2.cvtColor(overlay_v, cv2.COLOR_BGR2RGB)
    ov_a = cv2.cvtColor(overlay_a, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(face_rgb)
    ax[0].set_title("Input crop")
    ax[1].imshow(ov_v)
    ax[1].set_title("Grad-CAM (valence)")
    ax[2].imshow(ov_a)
    ax[2].set_title("Grad-CAM (arousal)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_aggregate_bar(out_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    vals = {"upper": [], "middle": [], "lower": []}
    aros = {"upper": [], "middle": [], "lower": []}
    for r in rows:
        for k in vals:
            vals[k].append(float(r[f"val_cam_{k}_pct"]))
            aros[k].append(float(r[f"aro_cam_{k}_pct"]))

    labels = ["upper (eyes/eyebrows)", "middle (nose/cheeks)", "lower (mouth/jaw)"]
    v_means = [np.mean(vals["upper"]), np.mean(vals["middle"]), np.mean(vals["lower"])]
    a_means = [np.mean(aros["upper"]), np.mean(aros["middle"]), np.mean(aros["lower"])]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(x - width / 2, v_means, width, label="Valence", color="#4C78A8")
    ax.bar(x + width / 2, a_means, width, label="Arousal", color="#F58518")
    ax.set_ylabel("Mean Grad-CAM energy (%)")
    ax.set_xticks(x, labels, rotation=10, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Grad-CAM region energy (100-image sample)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM analysis for GlowMind VA model.")
    ap.add_argument("--csv", required=True, help="Dataset CSV path (AffectNet-like).")
    ap.add_argument("--image-root", required=True, help="Root folder for image paths in CSV.")
    ap.add_argument("--outdir", default="gradcam_out", help="Output directory.")
    ap.add_argument("--n", type=int, default=100, help="Number of samples to analyze.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling RNG seed.")
    ap.add_argument(
        "--skip-invalid-gt",
        action="store_true",
        help="Skip rows with missing/invalid ground-truth (AffectNet often uses -2).",
    )
    ap.add_argument("--weights", default=None, help="Model weights path (default: MODEL_WEIGHTS or Settings default).")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (default: env DEVICE or cpu).")
    ap.add_argument("--bbox-buffer", type=float, default=0.1, help="Face bbox padding (AffectNet-style).")
    ap.add_argument("--path-col", default="subDirectory_filePath", help="CSV column for image path.")
    ap.add_argument("--valence-col", default="valence", help="CSV column for valence GT.")
    ap.add_argument("--arousal-col", default="arousal", help="CSV column for arousal GT.")
    ap.add_argument("--face-x-col", default="face_x", help="CSV column for face bbox x.")
    ap.add_argument("--face-y-col", default="face_y", help="CSV column for face bbox y.")
    ap.add_argument("--face-w-col", default="face_width", help="CSV column for face bbox width.")
    ap.add_argument("--face-h-col", default="face_height", help="CSV column for face bbox height.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    image_root = Path(args.image_root)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)
    _ensure_dir(outdir / "panels")

    spec = CsvSpec(
        path_col=args.path_col,
        valence_col=args.valence_col,
        arousal_col=args.arousal_col,
        face_x_col=args.face_x_col,
        face_y_col=args.face_y_col,
        face_w_col=args.face_w_col,
        face_h_col=args.face_h_col,
    )

    rows = _read_csv_rows(csv_path)
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: max(args.n * 10, args.n)]  # oversample for missing/bad images

    device = args.device or os.environ.get("DEVICE", "cpu")
    weights_path = args.weights or os.environ.get("MODEL_WEIGHTS", "resnet50_va_finetune.pth")

    model = build_va_resnet(device)
    load_model_weights(model, weights_path, device)
    model.eval()

    # Grad-CAM on last conv block
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = face_transform()

    results: list[dict[str, Any]] = []
    analyzed = 0
    for i, row in enumerate(rows):
        if analyzed >= args.n:
            break

        gt_v = _safe_float(row, spec.valence_col)
        gt_a = _safe_float(row, spec.arousal_col)
        if args.skip_invalid_gt:
            # AffectNet uses -2 for missing labels in some exports.
            if gt_v is None or gt_a is None or gt_v <= -1.5 or gt_a <= -1.5:
                continue

        loaded = _load_face_from_row(row, image_root=image_root, spec=spec, bbox_buffer=args.bbox_buffer)
        if loaded is None:
            continue
        face_bgr, meta = loaded
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = forward_va(model, inp)
        pred_v = float(out[0, 0].item())
        pred_a = float(out[0, 1].item())

        cam_v = cam(input_tensor=inp, targets=[VAIndexTarget(0)])[0]
        cam_a = cam(input_tensor=inp, targets=[VAIndexTarget(1)])[0]
        cam_v01 = _normalize_cam(cam_v)
        cam_a01 = _normalize_cam(cam_a)

        energy_v = _cam_energy_by_bands(cam_v01)
        energy_a = _cam_energy_by_bands(cam_a01)

        overlay_v = _overlay_cam_on_bgr(face_bgr, cam_v01)
        overlay_a = _overlay_cam_on_bgr(face_bgr, cam_a01)

        panel_path = outdir / "panels" / f"sample_{analyzed:03d}.png"
        title = f"Sample {analyzed:03d}"
        subtitle = (
            f"pred(v,a)=({pred_v:+.2f},{pred_a:+.2f})"
            + (f"  gt(v,a)=({gt_v:+.2f},{gt_a:+.2f})" if gt_v is not None and gt_a is not None else "")
        )
        _save_panel(panel_path, face_bgr=face_bgr, overlay_v=overlay_v, overlay_a=overlay_a, title=title, subtitle=subtitle)

        results.append(
            {
                "idx": analyzed,
                "img_path": meta["img_path"],
                "crop_mode": meta["crop_mode"],
                "pred_valence": pred_v,
                "pred_arousal": pred_a,
                "gt_valence": gt_v,
                "gt_arousal": gt_a,
                "val_cam_upper_pct": energy_v["upper"],
                "val_cam_middle_pct": energy_v["middle"],
                "val_cam_lower_pct": energy_v["lower"],
                "aro_cam_upper_pct": energy_a["upper"],
                "aro_cam_middle_pct": energy_a["middle"],
                "aro_cam_lower_pct": energy_a["lower"],
                "panel_path": str(panel_path),
            }
        )
        analyzed += 1

    if not results:
        raise SystemExit(
            "No samples analyzed. Check CSV column names, image_root, and that images exist on disk."
        )

    # Write per-image CSV
    out_csv = outdir / "gradcam_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Aggregate plot
    out_plot = outdir / "gradcam_region_energy.png"
    _plot_aggregate_bar(out_plot, results)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_plot}")
    print(f"Panels: {outdir / 'panels'}")


if __name__ == "__main__":
    main()

