"""Вспомогательные функции: сохранение предсказаний в едином формате."""
from pathlib import Path

import pandas as pd

KEEP_COLS = ["img_id", "patient_id", "split", "label", "diagnostic",
             "fitspatrick", "gender", "age", "region"]


def save_preds(meta: pd.DataFrame, prob, model_name: str, seed: int, out_dir: Path):
    """
    Каждая модель сохраняет вероятность "злокачественное" для каждого снимка val и test
    в файл outputs/preds/<model>__seed<k>.csv. evaluate.py потом читает все такие файлы
    и сравнивает модели на одних и тех же снимках.
    """
    out = meta[[c for c in KEEP_COLS if c in meta]].copy()
    out["prob"] = prob
    out = out[out["split"].isin(["val", "test"])]
    path = Path(out_dir) / "preds" / f"{model_name}__seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
