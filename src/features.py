"""
Предобработка клинических признаков (общая для всех моделей и для демо).

Главное правило: все статистики (медиана, мода, среднее для масштабирования)
считаются ТОЛЬКО на train. Pipeline из sklearn делает это автоматически:
.fit() на train, .transform() на val/test.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FULL_BOOL, FULL_CAT, FULL_NUM, SAFE_BOOL, SAFE_CAT, SAFE_NUM

FEATURE_SETS = {
    "safe": (SAFE_NUM, SAFE_CAT, SAFE_BOOL),
    "full": (FULL_NUM, FULL_CAT, FULL_BOOL),
}


def load_metadata(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим значения к единому виду."""
    df = df.copy()
    # Булевы признаки: "True"/"False"/"UNK"/пусто -> 1.0 / 0.0 / NaN.
    # UNK у grew/changed в 88% случаев = рак (артефакт сбора данных), поэтому НЕ делаем
    # из него отдельную категорию, а считаем пропуском и заполняем модой из train.
    for c in FULL_BOOL:
        if c in df:
            df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    # Опечатка в данных: BRASIL и BRAZIL — одна страна
    for c in ["background_father", "background_mother"]:
        if c in df:
            df[c] = df[c].replace({"BRASIL": "BRAZIL"})
    return df


def build_preprocessor(feature_set: str = "safe") -> ColumnTransformer:
    num, cat, boolean = FEATURE_SETS[feature_set]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("bool", SimpleImputer(strategy="most_frequent"), boolean),
    ])


def feature_columns(feature_set: str = "safe"):
    num, cat, boolean = FEATURE_SETS[feature_set]
    return num + cat + boolean


def complete_case_mask(df: pd.DataFrame) -> np.ndarray:
    """Строки, где заполнены полные клинические данные (нет 804 'пустых' анкет)."""
    return df["smoke"].notna().to_numpy()
