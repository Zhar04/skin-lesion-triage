"""
Этап 6.9. Grad-CAM: на какую часть фото "смотрит" нейросеть при решении.

Как работает: берём градиенты выхода модели по последнему свёрточному слою ->
понимаем, какие области карты признаков сильнее всего повлияли на ответ ->
накладываем тепловую карту на фото (красное = важно).

Зачем: проверить, что модель смотрит на само поражение, а не на фон, линейку, волосы
или маркер (это тоже shortcut learning). Хорошие и плохие примеры — в отчёт и на слайд.

Запуск (после evaluate.py, локально на CPU ~1 минута):
  python src/gradcam.py --run fusion_efficientnet_b0__seed42
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from config import DATA_DIR, IMG_SUBDIR, OUT_DIR
from features import feature_columns, load_metadata
from models import ImageOnlyWrapper, build_transforms, gradcam_target_layer, load_run


def compute_cam(model, cfg, prep, pil_img, meta_row=None):
    """Возвращает (вероятность, наложенная тепловая карта как np.uint8 RGB)."""
    size = cfg["img_size"]
    x = build_transforms(size)(pil_img).unsqueeze(0)
    clin = None
    if prep is not None:
        clin = torch.tensor(prep.transform(meta_row[feature_columns(cfg["clinical"])]), dtype=torch.float32)
    wrapper = ImageOnlyWrapper(model, clin).eval()
    with torch.no_grad():
        prob = torch.sigmoid(wrapper(x)).item()
    with GradCAM(model=wrapper, target_layers=[gradcam_target_layer(model)]) as cam:
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
    rgb = np.asarray(pil_img.convert("RGB").resize((size, size)), dtype=np.float32) / 255
    return prob, show_cam_on_image(rgb, heat, use_rgb=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="папка в outputs/models, например fusion_efficientnet_b0__seed42")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--per-group", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)

    model, cfg, prep = load_run(out / "models" / args.run)
    name, seed = args.run.split("__seed")
    preds = pd.read_csv(out / "preds" / f"{name}__seed{seed}.csv")
    thr_file = out / "models" / "thresholds.json"
    thr = json.loads(thr_file.read_text())[name]["low"] if thr_file.exists() else 0.5
    meta = load_metadata(Path(args.data_dir) / "metadata_split.csv").set_index("img_id")

    t = preds[preds["split"] == "test"].copy()
    t["pred"] = (t["prob"] >= thr).astype(int)
    k = args.per_group
    groups = {
        "Верно: рак (уверенно)": t[(t.label == 1) & (t.pred == 1)].nlargest(k, "prob"),
        "Пропущенный рак (FN)": t[(t.label == 1) & (t.pred == 0)].nsmallest(k, "prob"),
        "Ложная тревога (FP)": t[(t.label == 0) & (t.pred == 1)].nlargest(k, "prob"),
        "Верно: доброкач.": t[(t.label == 0) & (t.pred == 0)].nsmallest(k, "prob"),
    }
    groups = {g: d for g, d in groups.items() if len(d)}

    fig, axes = plt.subplots(len(groups), 2 * k, figsize=(2 * k * 1.9, len(groups) * 2.4))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (title, d) in enumerate(groups.items()):
        axes[r, 0].set_title(title, loc="left", fontsize=10, fontweight="bold", pad=18)
        for i, (_, row) in enumerate(d.iterrows()):
            img = Image.open(Path(args.data_dir) / IMG_SUBDIR / row["img_id"].replace(".png", ".jpg"))
            prob, overlay = compute_cam(model, cfg, prep, img, meta.loc[[row["img_id"]]])
            axes[r, 2 * i].imshow(img.resize((cfg["img_size"],) * 2))
            axes[r, 2 * i + 1].imshow(overlay)
            axes[r, 2 * i + 1].set_title(f"{row['diagnostic']} p={prob:.2f}", fontsize=8)
    fig.suptitle(f"Grad-CAM: {args.run} (порог {thr:.2f})")
    fig.tight_layout()
    path = out / "figures" / f"gradcam_{args.run}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"Сохранено: {path}")


if __name__ == "__main__":
    main()
