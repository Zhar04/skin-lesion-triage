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


def remove_duplicates(meta, data):
    """
    Побайтно одинаковые картинки под разными img_id (найдено 15 пар, все внутри одного пациента).
    Часто у копий РАЗНЫЕ диагнозы и регионы тела -> ошибка разметки, верную метку не узнать.
      - метки конфликтуют -> удаляем все копии (шумная метка хуже, чем её отсутствие);
      - метки совпадают   -> оставляем одну копию (иначе фото считается дважды).
    Удалённые строки сохраняем в removed_duplicates.csv (для отчёта и EDA).
    """
    if "md5" not in meta or meta["md5"].isna().all():
        print("Нет колонки md5 — запустите prepare_images.py без --limit")
        return meta
    dup = meta[meta["md5"].duplicated(keep=False) & meta["md5"].notna()]
    conflict = dup.groupby("md5")["diagnostic"].transform("nunique") > 1
    drop_conflict = dup[conflict].index
    drop_extra = dup[~conflict & dup["md5"].duplicated(keep="first")].index
    removed = meta.loc[drop_conflict.union(drop_extra)].copy()
    removed["reason"] = ["conflicting_labels" if i in drop_conflict else "exact_copy" for i in removed.index]
    removed.to_csv(data / "removed_duplicates.csv", index=False)
    print(f"Дубликаты картинок: групп {dup['md5'].nunique()}, "
          f"удалено с конфликтом меток {len(drop_conflict)}, лишних копий {len(drop_extra)}")
    return meta.drop(removed.index).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    args = p.parse_args()
    data = Path(args.data)

    meta = pd.read_csv(data / "metadata.csv")
    meta = remove_duplicates(meta, data)
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
