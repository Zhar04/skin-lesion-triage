"""
Архитектура нейросети и загрузка картинок.

LesionNet — одна модель для двух режимов:
  - только фото:      картинка -> CNN (EfficientNet/ResNet) -> вектор 128 -> вероятность
  - фото + клиника:   картинка -> CNN -> вектор 128 ┐
                      клиника  -> MLP -> вектор 32  ┴ склеиваем (concat) -> вероятность
Второй режим и есть feature-level fusion из proposal: признаки двух источников
объединяются ДО финального классификатора, и вся сеть учится целиком (end-to-end).
"""
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class LesionNet(nn.Module):
    def __init__(self, arch="efficientnet_b0", n_clinical=0, pretrained=True, dropout=0.3):
        super().__init__()
        self.n_clinical = n_clinical
        # num_classes=0 -> timm отдаёт вектор признаков вместо 1000 классов ImageNet
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.num_features, 128), nn.ReLU(), nn.Dropout(dropout))
        fused = 128
        if n_clinical:
            self.clin_proj = nn.Sequential(nn.Linear(n_clinical, 32), nn.ReLU(), nn.Dropout(dropout))
            fused += 32
        self.head = nn.Linear(fused, 1)  # один выход = логит "злокачественное"

    def forward(self, image, clinical=None):
        f = self.img_proj(self.backbone(image))
        if self.n_clinical:
            f = torch.cat([f, self.clin_proj(clinical)], dim=1)
        return self.head(f).squeeze(1)


class ImageOnlyWrapper(nn.Module):
    """Для Grad-CAM: библиотеке нужна модель вида f(image) -> (N, классы).
    Клинические признаки фиксируем внутри обёртки."""
    def __init__(self, model, clinical=None):
        super().__init__()
        self.model = model
        self.clinical = clinical

    def forward(self, x):
        clin = None if self.clinical is None else self.clinical.expand(x.shape[0], -1)
        return self.model(x, clin).unsqueeze(1)


def gradcam_target_layer(model: LesionNet):
    """Последний свёрточный слой — по нему Grad-CAM строит карту внимания."""
    bb = model.backbone
    if hasattr(bb, "conv_head"):      # EfficientNet
        return bb.conv_head
    if hasattr(bb, "layer4"):         # ResNet
        return bb.layer4[-1]
    raise ValueError("Укажите слой для Grad-CAM для этой архитектуры")


def build_transforms(img_size=224, train=False):
    if train:
        # Аугментация: каждую эпоху сеть видит немного другую версию фото
        # -> меньше переобучения на маленьком датасете
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LesionDataset(Dataset):
    def __init__(self, img_dir, img_files, labels, clinical=None, transform=None):
        self.img_dir = Path(img_dir)
        self.img_files = list(img_files)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.clinical = None if clinical is None else np.asarray(clinical, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.img_files[i]).convert("RGB")
        x = self.transform(img)
        clin = torch.zeros(0) if self.clinical is None else torch.from_numpy(self.clinical[i])
        return x, clin, torch.tensor(self.labels[i])


def load_run(run_dir, device="cpu"):
    """Загружает обученную модель из папки outputs/models/<run>/ (для Grad-CAM и демо)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = LesionNet(cfg["arch"], n_clinical=cfg["n_clinical"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device).eval()
    prep = None
    if cfg["n_clinical"]:
        import joblib
        prep = joblib.load(run_dir / "preprocessor.joblib")
    return model, cfg, prep
