"""
Smile Detector (MobileNetV3 + MediaPipe + FastAPI)
==================================================
Binary classifier for **smile vs. not_smile** using transfer learning, with face-detection-based cropping and a FastAPI endpoint.

Stack
- PyTorch + torchvision (MobileNetV3-Small)
- MediaPipe Face Detection for robust face crop
- FastAPI for /predict (single-image inference)

Install
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install mediapipe fastapi uvicorn[standard] pillow scikit-learn numpy

Data layout (ImageFolder)
    data/
      train/
        smile/
          img001.jpg
          ...
        not_smile/
          img101.jpg
          ...
      val/
        smile/
        not_smile/

Train
    python app_smile.py --train --data_dir ./data --epochs 20 --batch_size 64 --lr 3e-4

Run API (after training)
    uvicorn app_smile:app --host 0.0.0.0 --port 8003 --reload

Predict
    curl -F "file=@face.jpg" http://localhost:8003/predict

Artifacts
- models/smile_mobilenetv3.pt   # PyTorch weights + class names
- models/last_report.json       # metrics & confusion matrix
- models/smile_mobilenetv3.onnx # optional ONNX export

Notes
- Works for any number of classes (extend by adding folders). Default assumes 2 classes: ['not_smile','smile'] (alphabetical).
- If no face is detected, a center-crop fallback is used.
"""
from __future__ import annotations
import os
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix

# -------------
# Face detector
# -------------
class FaceCropper:
    """MediaPipe-based face cropper usable as a torchvision transform.
    Falls back to center crop if no face is detected.
    """
    def __init__(self, target_size: int = 224, margin: float = 0.2):
        self.target_size = target_size
        self.margin = margin
        self._detector = None  # lazy init to avoid import overhead in dataloader workers

    def _ensure_detector(self):
        if self._detector is None:
            import mediapipe as mp
            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            self._ensure_detector()
            rgb = np.array(img.convert('RGB'))
            h, w, _ = rgb.shape
            res = self._detector.process(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
            if res.detections:
                # take the most confident detection
                det = max(res.detections, key=lambda d: d.score[0])
                bbox = det.location_data.relative_bounding_box
                x, y, bw, bh = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                # convert to absolute
                x1 = max(0, int((x - self.margin * bw) * w))
                y1 = max(0, int((y - self.margin * bh) * h))
                x2 = min(w, int((x + (1 + self.margin) * bw) * w))
                y2 = min(h, int((y + (1 + self.margin) * bh) * h))
                if x2 > x1 and y2 > y1:
                    face = Image.fromarray(rgb[y1:y2, x1:x2])
                else:
                    face = img
            else:
                face = img
        except Exception:
            face = img
        # letterbox-pad to square then resize
        face = ImageOps.pad(face, (self.target_size, self.target_size), method=Image.BILINEAR, color=(0, 0, 0), centering=(0.5, 0.5))
        return face

# ---------------------
# Paths & configuration
# ---------------------
MODEL_DIR = os.path.join(os.getcwd(), "models")
PT_PATH = os.path.join(MODEL_DIR, "smile_mobilenetv3.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "smile_mobilenetv3.onnx")
REPORT_PATH = os.path.join(MODEL_DIR, "last_report.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------
# Dataloaders
# ---------------

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    face_crop = FaceCropper(target_size=img_size)
    train_tf = transforms.Compose([
        face_crop,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        face_crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 2, img_size: int = 224) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_tf, val_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_tf)

    class_names = train_ds.classes  # alphabetical

    # Weighted sampling for imbalance
    targets = np.array([y for _, y in train_ds.samples])
    class_sample_count = np.bincount(targets, minlength=len(class_names))
    class_weights = 1.0 / np.maximum(class_sample_count, 1)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names

# ---------------
# Model
# ---------------

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model

# ---------------
# Train / eval
# ---------------
@dataclass
class TrainConfig:
    data_dir: str
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 64
    num_workers: int = 2
    img_size: int = 224
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    acc = correct / max(total, 1)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    return acc, y_true, y_pred


def train(cfg: TrainConfig) -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.img_size)

    model = build_model(num_classes=len(class_names))
    model = model.to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    best_state = None
    patience, wait = 5, 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        val_acc, y_true, y_pred = evaluate(model, val_loader, cfg.device)
        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    torch.save({'state_dict': model.state_dict(), 'class_names': class_names}, PT_PATH)

    # Metrics
    acc, y_true, y_pred = evaluate(model, val_loader, cfg.device)
    cm = confusion_matrix(y_true, y_pred).tolist() if y_true.size else []
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True) if y_true.size else {}

    out = {'val_accuracy': float(acc), 'classes': class_names, 'confusion_matrix': cm, 'report': report, 'pt_path': PT_PATH}
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ONNX export (optional)
    try:
        dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
        model.eval()
        torch.onnx.export(model, dummy, ONNX_PATH, input_names=['input'], output_names=['logits'], opset_version=12)
        out['onnx_path'] = ONNX_PATH
    except Exception as e:
        out['onnx_export'] = f'skipped: {e}'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out

# ---------------
# Inference utils
# ---------------
_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_infer_model: Optional[nn.Module] = None
_infer_classes: Optional[List[str]] = None
_infer_tf = transforms.Compose([
    FaceCropper(target_size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_model() -> None:
    global _infer_model, _infer_classes
    if _infer_model is not None:
        return
    if not os.path.exists(PT_PATH):
        raise FileNotFoundError("Trained model not found. Train first.")
    ckpt = torch.load(PT_PATH, map_location=_device)
    _infer_classes = ckpt.get('class_names', ['not_smile', 'smile'])
    model = build_model(num_classes=len(_infer_classes), pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(_device)
    model.eval()
    _infer_model = model


def predict_image(pil_img: Image.Image) -> Dict[str, Any]:
    load_model()
    x = _infer_tf(pil_img.convert('RGB')).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _infer_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return {
        'pred_class': _infer_classes[idx],
        'probabilities': {cls: float(p) for cls, p in zip(_infer_classes, probs)}
    }

# ---------------
# FastAPI
# ---------------
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Smile Detector API", version="1.0.0")

@app.get('/health')
def health() -> Dict[str, str]:
    status = 'ready' if os.path.exists(PT_PATH) else 'no-model'
    return {'status': status}

@app.post('/predict')
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes))
        result = predict_image(pil)
        return {'ok': True, 'result': result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ---------------
# CLI
# ---------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Smile Detector Trainer')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    if args.train:
        cfg = TrainConfig(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, img_size=args.img_size)
        train(cfg)
    else:
        print('Nothing to do. Use --train to start training or run the FastAPI app via uvicorn.')
