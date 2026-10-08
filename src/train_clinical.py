"""
Этап 3. Модели только по клиническим данным (baseline) + эксперимент с утечкой.

Модели (все сохраняют предсказания в outputs/preds/):
  dummy                    — всегда предсказывает долю рака в train (нижняя граница, AUC = 0.5)
  clinical_logreg          — логистическая регрессия на "безопасных" признаках
  clinical_xgb             — градиентный бустинг на тех же признаках (3 seed)
  leaky_naive_logreg       — "как сделал бы новичок": все признаки + biopsed + пропуски как
                             отдельная категория. Демонстрирует shortcut learning (для слайда!)
  cc_safe_logreg / cc_full_logreg / cc_full_xgb
                           — только снимки с полными анкетами (complete-case): помогают ли
                             курение, тип кожи, диаметр и т.д., если убрать утечку через пропуски

Гиперпараметры подбираются по val ROC-AUC; test здесь не используется для выбора.

Запуск (локально, ~1 минута):
  python src/train_clinical.py
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from config import DATA_DIR, FULL_BOOL, FULL_CAT, FULL_NUM, OUT_DIR, SEED, SEEDS, set_seed
from features import build_preprocessor, complete_case_mask, feature_columns, load_metadata
from utils import save_preds

warnings.filterwarnings("ignore", category=UserWarning)

LOGREG_GRID = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]
XGB_GRID = [{"max_depth": d, "n_estimators": n, "learning_rate": 0.05}
            for d in [2, 3, 4] for n in [100, 300]]


def make_logreg(C, preprocessor):
    return Pipeline([("prep", preprocessor),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def make_xgb(params, preprocessor, seed):
    return Pipeline([("prep", preprocessor),
                     ("clf", XGBClassifier(**params, subsample=0.8, colsample_bytree=0.8,
                                           eval_metric="logloss", random_state=seed))])


def tune(name, factory, grid, X_tr, y_tr, X_val, y_val, log):
    """Перебираем параметры, выбираем лучшие по ROC-AUC на val."""
    best = None
    for params in grid:
        model = factory(params).fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        log.append({"model": name, **params, "val_auc": round(auc, 4)})
        if best is None or auc > best[0]:
            best = (auc, params)
    print(f"  {name}: лучшие параметры {best[1]}, val AUC = {best[0]:.3f}")
    return best[1]


def naive_leaky_preprocessor():
    """Так делать НЕЛЬЗЯ: пропуски кодируются явно, biopsed подаётся как признак."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), FULL_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         FULL_CAT),
        ("bool", SimpleImputer(strategy="constant", fill_value=-1), FULL_BOOL + ["biopsed_num"]),
    ])


def run_group(prefix, df, feature_set, out_dir, tuning_log, with_xgb=True):
    """Обучает LogReg (+XGB) на заданном наборе строк и признаков, сохраняет предсказания."""
    cols = feature_columns(feature_set)
    tr, va = df["split"] == "train", df["split"] == "val"
    X, y = df[cols], df["label"].to_numpy()

    p = tune(f"{prefix}_logreg", lambda g: make_logreg(g["C"], build_preprocessor(feature_set)),
             LOGREG_GRID, X[tr], y[tr], X[va], y[va], tuning_log)
    logreg = make_logreg(p["C"], build_preprocessor(feature_set)).fit(X[tr], y[tr])
    save_preds(df, logreg.predict_proba(X)[:, 1], f"{prefix}_logreg", SEED, out_dir)

    xgb_params = None
    if with_xgb:
        xgb_params = tune(f"{prefix}_xgb", lambda g: make_xgb(g, build_preprocessor(feature_set), SEED),
                          XGB_GRID, X[tr], y[tr], X[va], y[va], tuning_log)
        for s in SEEDS:
            xgb = make_xgb(xgb_params, build_preprocessor(feature_set), s).fit(X[tr], y[tr])
            save_preds(df, xgb.predict_proba(X)[:, 1], f"{prefix}_xgb", s, out_dir)
    return logreg, xgb_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "metadata_split.csv"))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    set_seed(SEED)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.data)
    tr, va = df["split"] == "train", df["split"] == "val"
    y = df["label"].to_numpy()
    tuning_log = []

    # 1) Dummy: вероятность = доля рака в train
    save_preds(df, np.full(len(df), y[tr].mean()), "dummy", SEED, out)
    print("dummy: сохранено")

    # 2) Основной эксперимент: безопасные признаки, все снимки
    print("Основной эксперимент (безопасные признаки, все снимки):")
    logreg, _ = run_group("clinical", df, "safe", out, tuning_log)
    joblib.dump(logreg, out / "models" / "clinical_logreg.joblib")  # пригодится для late fusion в демо

    # Коэффициенты логрегрессии: какой признак повышает/понижает риск
    names = logreg.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": logreg.named_steps["clf"].coef_[0]})
    coefs.sort_values("coef", key=abs, ascending=False).to_csv(out / "tables" / "clinical_logreg_coefs.csv",
                                                                index=False)
    # Permutation importance на val: насколько падает AUC, если "перемешать" признак
    cols = feature_columns("safe")
    pi = permutation_importance(logreg, df.loc[va, cols], y[va], scoring="roc_auc",
                                n_repeats=20, random_state=SEED)
    pd.DataFrame({"feature": cols, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}) \
        .sort_values("importance_mean", ascending=False) \
        .to_csv(out / "tables" / "clinical_permutation_importance.csv", index=False)

    # 3) Наивная модель с утечкой (для демонстрации shortcut learning)
    df_leak = df.copy()
    df_leak["biopsed_num"] = df_leak["biopsed"].astype(float)
    leak_cols = FULL_NUM + FULL_CAT + FULL_BOOL + ["biopsed_num"]
    leaky = Pipeline([("prep", naive_leaky_preprocessor()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    leaky.fit(df_leak.loc[tr, leak_cols], y[tr])
    prob = leaky.predict_proba(df_leak[leak_cols])[:, 1]
    print(f"  leaky_naive_logreg: val AUC = {roc_auc_score(y[va], prob[va]):.3f}  <- подозрительно высоко!")
    save_preds(df, prob, "leaky_naive_logreg", SEED, out)

    # 4) Complete-case: только полные анкеты
    print("Complete-case (только снимки с полными анкетами):")
    cc = df[complete_case_mask(df)].reset_index(drop=True)
    print(f"  снимков: {len(cc)} (из них рак: {cc['label'].mean():.0%})")
    run_group("cc_safe", cc, "safe", out, tuning_log, with_xgb=False)
    run_group("cc_full", cc, "full", out, tuning_log)

    pd.DataFrame(tuning_log).to_csv(out / "tables" / "clinical_tuning.csv", index=False)
    print(f"\nГотово. Предсказания: {out / 'preds'}, таблицы: {out / 'tables'}")


if __name__ == "__main__":
    main()
