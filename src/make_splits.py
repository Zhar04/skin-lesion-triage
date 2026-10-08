"""
Шаг 2. Деление на train / val / test ПО ПАЦИЕНТАМ (70 / 15 / 15).

Зачем:
  - у одного пациента бывает до 10 снимков, у одного поражения — несколько фото.
    Если фото одного пациента попадут и в train, и в test, модель "узнает" пациента,
    и метрика на тесте будет завышена (data leakage). Поэтому группируем по patient_id;
  - стратификация по diagnostic: в каждой части примерно одинаковая доля всех 6 классов
    (важно для редкой меланомы — всего 52 снимка);
  - фиксированный seed: деление одно и то же у всех моделей и при каждом запуске,
    поэтому сравнение моделей честное и воспроизводимое.

Запуск (после prepare_images.py):
  python src/make_splits.py
Результат: data/processed/metadata_split.csv с колонками split и label.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MALIGNANT = {"BCC", "SCC", "MEL"}
SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    args = p.parse_args()
    data = Path(args.data)

    meta = pd.read_csv(data / "metadata.csv")
    meta["label"] = meta["diagnostic"].isin(MALIGNANT).astype(int)  # 1 = злокачественное

    # 20 стратифицированных групповых фолдов по 5%: 3 фолда -> test, 3 -> val, 14 -> train
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=meta.index)
    for k, (_, idx) in enumerate(sgkf.split(meta, meta["diagnostic"], groups=meta["patient_id"])):
        fold.iloc[idx] = k
    meta["split"] = "train"
    meta.loc[fold.isin([0, 1, 2]), "split"] = "test"
    meta.loc[fold.isin([3, 4, 5]), "split"] = "val"

    # Проверка: ни один пациент не встречается в двух частях
    n_splits_per_patient = meta.groupby("patient_id")["split"].nunique()
    assert (n_splits_per_patient == 1).all(), "Утечка: пациент в нескольких частях!"

    print("Размеры частей (снимки / пациенты):")
    print(meta.groupby("split").agg(images=("img_id", "size"), patients=("patient_id", "nunique")))
    print("\nДиагнозы по частям:")
    print(pd.crosstab(meta["diagnostic"], meta["split"]))
    print("\nДоля злокачественных:")
    print(meta.groupby("split")["label"].mean().round(3))

    out = data / "metadata_split.csv"
    meta.to_csv(out, index=False)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
