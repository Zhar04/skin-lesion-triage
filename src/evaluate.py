"""
Этап 6. Итоговая оценка ВСЕХ моделей на test (запускать в самом конце, когда модели готовы).

Что делает:
  1. Для каждой модели и seed выбирает порог по VAL так, чтобы чувствительность >= 90%
     (режим триажа: лучше лишний раз отправить к врачу, чем пропустить рак).
  2. С этим порогом считает метрики на TEST: ROC-AUC, PR-AUC, sensitivity, specificity, F1,
     balanced accuracy. По seed: mean ± std.
  3. 95% доверительные интервалы bootstrap (по пациентам) для усреднённой по seed модели.
  4. Графики: ROC-кривые всех моделей, confusion matrix, learning curves.
  5. Для главной модели (--focus): метрики по подгруппам (тип кожи, пол, возраст),
     по каждому диагнозу, списки ошибок и галерея пропущенных раков (error analysis).
  6. Сохраняет пороги для демо в outputs/models/thresholds.json.

Запуск:
  python src/evaluate.py                       # главная модель выберется по val AUC
  python src/evaluate.py --focus fusion_efficientnet_b0
Результаты: outputs/tables/*.csv|*.md, outputs/figures/*.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

from config import DATA_DIR, IMG_SUBDIR, OUT_DIR, SEED, TARGET_SENSITIVITY

# Порядок и "человеческие" названия моделей для таблиц и графиков
DISPLAY = {
    "dummy": "Dummy (доля класса)",
    "clinical_logreg": "Клиника: LogReg",
    "clinical_xgb": "Клиника: XGBoost",
    "image_efficientnet_b0_frozen": "Фото: EfficientNet-B0 (заморож.)",
    "image_efficientnet_b0": "Фото: EfficientNet-B0 (fine-tune)",
    "image_resnet50": "Фото: ResNet50 (fine-tune)",
    "late_fusion_efficientnet_b0": "Late fusion (фото + клиника)",
    "fusion_efficientnet_b0": "Feature fusion (фото + клиника)",
    "fusion_resnet50": "Feature fusion ResNet50",
    "leaky_naive_logreg": "⚠ Наивная модель с утечкой",
    "cc_safe_logreg": "[полные анкеты] LogReg, безопасные",
    "cc_full_logreg": "[полные анкеты] LogReg, все признаки",
    "cc_full_xgb": "[полные анкеты] XGBoost, все признаки",
}
NOT_FOR_FOCUS = ("dummy", "leaky", "cc_")


def threshold_for_sensitivity(y, p, target=TARGET_SENSITIVITY):
    """Самый высокий порог, при котором на val ловим >= target доли раков."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(min(thr[ok[0]], 1.0))


def threshold_for_specificity(y, p, target=0.90):
    """Порог "высокого риска": на val не более 10% доброкачественных выше него."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= 1 - target)[0]
    return float(min(thr[ok[-1]], 1.0))


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    return {
        "roc_auc": roc_auc_score(y, p) if two else np.nan,
        "pr_auc": average_precision_score(y, p) if two else np.nan,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred) if two else np.nan,
    }


def bootstrap_ci(df, thr, n=1000, seed=SEED):
    """95% CI: пересэмплируем ПАЦИЕНТОВ (их снимки зависимы), считаем AUC и sensitivity."""
    rng = np.random.default_rng(seed)
    groups = {pid: idx for pid, idx in df.groupby("patient_id").indices.items()}
    pids = np.array(list(groups))
    aucs, sens = [], []
    y, p = df["label"].to_numpy(), df["prob"].to_numpy()
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.choice(pids, len(pids))])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        pos = y[idx] == 1
        sens.append((p[idx][pos] >= thr).mean())
    q = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    return q(aucs), q(sens)


def load_preds(pred_dir):
    models = {}
    for f in sorted(Path(pred_dir).glob("*__seed*.csv")):
        name, seed = f.stem.split("__seed")
        models.setdefault(name, {})[int(seed)] = pd.read_csv(f)
    return models


def ensemble(seed_frames):
    """Усреднение вероятностей по seed -> одна 'итоговая' модель для графиков и CI."""
    base = next(iter(seed_frames.values())).drop(columns="prob").copy()
    probs = pd.concat([d.set_index("img_id")["prob"] for d in seed_frames.values()], axis=1).mean(axis=1)
    base["prob"] = base["img_id"].map(probs)
    return base


def plot_confusion(df, thr, title, path):
    cm = confusion_matrix(df["label"], (df["prob"] >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["доброкач.", "злокач."]); ax.set_yticks([0, 1], ["доброкач.", "злокач."])
    ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def subgroup_table(df, thr):
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], [0, 40, 60, 75, 120], labels=["<40", "40-59", "60-74", "75+"])
    df["fitzpatrick"] = df["fitspatrick"].map(lambda v: "нет данных" if pd.isna(v) else f"тип {int(v)}")
    df["gender"] = df["gender"].fillna("нет данных")
    rows = []
    for col in ["fitzpatrick", "gender", "age_group"]:
        for val, g in df.groupby(col, observed=True):
            m = metrics(g["label"].to_numpy(), g["prob"].to_numpy(), thr)
            rows.append({"группа": col, "значение": val, "n": len(g), "n_рак": int(g["label"].sum()),
                         "roc_auc": m["roc_auc"], "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                         "надёжность": "мало данных" if len(g) < 30 else ""})
    return pd.DataFrame(rows)


def per_diagnosis(df, thr):
    """Для раков — доля пойманных (sensitivity), для доброкачественных — доля ложных тревог."""
    out = df.assign(pred=(df["prob"] >= thr).astype(int)).groupby("diagnostic").agg(
        n=("pred", "size"), predicted_malignant=("pred", "mean"), mean_prob=("prob", "mean")).reset_index()
    out["что значит"] = np.where(out["diagnostic"].isin(["BCC", "SCC", "MEL"]),
                                 "доля пойманных раков", "доля ложных тревог")
    return out


def error_gallery(errors, img_dir, title, path, n=12):
    errors = errors.head(n)
    if errors.empty:
        return
    cols = 4
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), errors.iterrows()):
        f = Path(img_dir) / r["img_id"].replace(".png", ".jpg")
        if f.exists():
            ax.imshow(Image.open(f))
        ax.set_title(f"{r['diagnostic']} p={r['prob']:.2f}\n{r['region']}, {int(r['age'])} лет", fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--focus", default=None, help="главная модель для подробного анализа")
    args = ap.parse_args()
    out = Path(args.out)
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)

    models = load_preds(out / "preds")
    order = [m for m in DISPLAY if m in models] + sorted(m for m in models if m not in DISPLAY)

    # --- 1-3. Таблица результатов ---------------------------------------------------
    rows, ens, thresholds = [], {}, {}
    for name in order:
        per_seed = []
        for seed, d in models[name].items():
            val, test = d[d["split"] == "val"], d[d["split"] == "test"]
            thr = threshold_for_sensitivity(val["label"], val["prob"])
            per_seed.append({"val_auc": roc_auc_score(val["label"], val["prob"]), "threshold": thr,
                             **metrics(test["label"].to_numpy(), test["prob"].to_numpy(), thr)})
        per_seed = pd.DataFrame(per_seed)
        e = ensemble(models[name]); ens[name] = e
        ev, et = e[e["split"] == "val"], e[e["split"] == "test"]
        thr_e = threshold_for_sensitivity(ev["label"], ev["prob"])
        thresholds[name] = {"low": thr_e, "high": threshold_for_specificity(ev["label"], ev["prob"])}
        ci_auc, ci_sens = bootstrap_ci(et, thr_e)
        row = {"model": name, "название": DISPLAY.get(name, name), "seeds": len(per_seed), "n_test": len(et)}
        for col in ["val_auc", "roc_auc", "pr_auc", "sensitivity", "specificity", "f1", "balanced_acc"]:
            m, s = per_seed[col].mean(), per_seed[col].std()
            row[col] = f"{m:.3f}" + (f" ± {s:.3f}" if len(per_seed) > 1 else "")
        row["roc_auc_95CI"], row["sens_95CI"] = ci_auc, ci_sens
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(tables / "results_test.csv", index=False)
    (tables / "results_test.md").write_text(res.drop(columns="model").to_markdown(index=False), encoding="utf-8")
    print(res[["название", "seeds", "val_auc", "roc_auc", "sensitivity", "specificity", "roc_auc_95CI"]]
          .to_string(index=False))
    (out / "models").mkdir(exist_ok=True)
    (out / "models" / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    # --- 4. ROC-кривые основных моделей (без cc_ — у них другой test) --------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in order:
        if name.startswith("cc_"):
            continue
        t = ens[name][ens[name]["split"] == "test"]
        fpr, tpr, _ = roc_curve(t["label"], t["prob"])
        ax.plot(fpr, tpr, lw=2 if "fusion" in name else 1.3, ls="--" if name.startswith(("dummy", "leaky")) else "-",
                label=f"{DISPLAY.get(name, name)} (AUC {roc_auc_score(t['label'], t['prob']):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("1 − specificity (ложные тревоги)"); ax.set_ylabel("Sensitivity (пойманные раки)")
    ax.set_title("ROC-кривые на test"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "roc_test.png", dpi=150); plt.close(fig)

    # Learning curves для всех нейросетей
    for hist in sorted((out / "models").glob("*__seed*/history.csv")):
        h = pd.read_csv(hist)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
        a1.plot(h["epoch"], h["train_loss"], label="train"); a1.plot(h["epoch"], h["val_loss"], label="val")
        a1.set_title("Loss"); a1.legend()
        a2.plot(h["epoch"], h["train_auc"], label="train"); a2.plot(h["epoch"], h["val_auc"], label="val")
        a2.set_title("ROC-AUC"); a2.legend()
        fig.suptitle(hist.parent.name); fig.tight_layout()
        fig.savefig(figs / f"learning_curves_{hist.parent.name}.png", dpi=130); plt.close(fig)

    # --- 5. Подробный анализ главной модели -------------------------------------------
    candidates = [m for m in order if not m.startswith(NOT_FOR_FOCUS)]
    focus = args.focus or max(candidates, key=lambda m: roc_auc_score(
        ens[m].query("split == 'val'")["label"], ens[m].query("split == 'val'")["prob"]))
    print(f"\nГлавная модель для анализа: {focus}")
    t = ens[focus][ens[focus]["split"] == "test"]
    thr = thresholds[focus]["low"]
    plot_confusion(t, thr, f"{DISPLAY.get(focus, focus)}\nпорог {thr:.2f} (sens≥90% на val)",
                   figs / f"cm_{focus}.png")
    best_clin = "clinical_logreg" if "clinical_logreg" in ens else None
    if best_clin:
        tc = ens[best_clin][ens[best_clin]["split"] == "test"]
        plot_confusion(tc, thresholds[best_clin]["low"], DISPLAY[best_clin], figs / f"cm_{best_clin}.png")

    subgroup_table(t, thr).to_csv(tables / f"subgroups_{focus}.csv", index=False)
    per_diagnosis(t, thr).to_csv(tables / f"per_diagnosis_{focus}.csv", index=False)
    t = t.assign(pred=(t["prob"] >= thr).astype(int))
    fn = t[(t["label"] == 1) & (t["pred"] == 0)].sort_values("prob")
    fp = t[(t["label"] == 0) & (t["pred"] == 1)].sort_values("prob", ascending=False)
    pd.concat([fn.assign(error="false_negative"), fp.assign(error="false_positive")]) \
        .to_csv(tables / f"errors_{focus}.csv", index=False)
    img_dir = Path(args.data_dir) / IMG_SUBDIR
    error_gallery(fn, img_dir, "Пропущенные раки (false negatives)", figs / f"false_negatives_{focus}.png")
    error_gallery(fp, img_dir, "Ложные тревоги (false positives)", figs / f"false_positives_{focus}.png")
    print(f"Пропущено раков: {len(fn)}, ложных тревог: {len(fp)}")
    print(f"Таблицы: {tables}\nГрафики: {figs}")


if __name__ == "__main__":
    main()
