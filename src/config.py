"""
Общие настройки проекта: пути, seed, списки признаков.
Все скрипты берут их отсюда, чтобы не было расхождений между моделями.
"""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"            # предсказания, модели, графики, таблицы
IMG_SUBDIR = "images_256"

SEED = 42
SEEDS = [42, 43, 44]                  # 3 запуска для mean ± std
MALIGNANT = {"BCC", "SCC", "MEL"}
TARGET_SENSITIVITY = 0.90             # порог выбираем так, чтобы на val ловить >= 90% раков

# --- Клинические признаки -------------------------------------------------
# "Безопасные" признаки: заполнены у ВСЕХ снимков, поэтому пропуск не выдаёт диагноз.
SAFE_NUM = ["age"]
SAFE_CAT = ["region"]
SAFE_BOOL = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]

# Полный набор: пропущен у 804 снимков (все доброкачественные!) -> используем только
# на подвыборке без пропусков (эксперимент "complete-case").
FULL_NUM = SAFE_NUM + ["fitspatrick", "diameter_1", "diameter_2"]
FULL_CAT = SAFE_CAT + ["gender", "background_father", "background_mother"]
FULL_BOOL = SAFE_BOOL + ["smoke", "drink", "pesticide", "skin_cancer_history",
                         "cancer_history", "has_piped_water", "has_sewage_system"]

# Никогда не подаём в модель: идентификаторы, ответ и утечка (biopsed=False -> всегда доброкач.)
NEVER_USE = ["patient_id", "lesion_id", "img_id", "img_file", "md5",
             "diagnostic", "label", "split", "biopsed"]


def set_seed(seed: int = SEED):
    """Фиксируем все генераторы случайных чисел -> воспроизводимые результаты."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
