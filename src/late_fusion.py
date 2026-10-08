"""
Этап 5.1. Late fusion: взвешенное среднее вероятностей двух готовых моделей
  p = w * p_фото + (1 - w) * p_клиника
Вес w подбираем по val ROC-AUC. Это простейший способ объединить модальности —
полезная точка сравнения для end-to-end fusion (train_image.py --clinical safe).

Запуск (после того как есть предсказания image_* и clinical_logreg):
  python src/late_fusion.py --image-model image_efficientnet_b0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--image-model", default="image_efficientnet_b0")
    ap.add_argument("--clinical-model", default="clinical_logreg")
    args = ap.parse_args()
    preds = Path(args.out) / "preds"

    clin_files = sorted(preds.glob(f"{args.clinical_model}__seed*.csv"))
    img_files = sorted(preds.glob(f"{args.image_model}__seed*.csv"))
    if not clin_files or not img_files:
        raise SystemExit("Нет файлов предсказаний: сначала обучите модели по фото и по клинике")
    clin = pd.read_csv(clin_files[0])[["img_id", "prob"]].rename(columns={"prob": "p_clin"})

    for f in img_files:
        seed = f.stem.split("__seed")[1]
        df = pd.read_csv(f).merge(clin, on="img_id")
        val = df["split"] == "val"
        grid = np.round(np.arange(0, 1.01, 0.1), 1)
        aucs = [roc_auc_score(df.loc[val, "label"], w * df.loc[val, "prob"] + (1 - w) * df.loc[val, "p_clin"])
                for w in grid]
        w = grid[int(np.argmax(aucs))]
        df["prob"] = w * df["prob"] + (1 - w) * df["p_clin"]
        name = f"late_fusion_{args.image_model.removeprefix('image_')}"
        df.drop(columns="p_clin").to_csv(preds / f"{name}__seed{seed}.csv", index=False)
        print(f"seed {seed}: вес фото w = {w}, val AUC = {max(aucs):.3f} -> {name}")


if __name__ == "__main__":
    main()
