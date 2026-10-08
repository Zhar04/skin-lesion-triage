"""
Этапы 4–5. Обучение нейросети по фото (и мультимодальной: фото + клиника).

Примеры (в Colab на GPU; на CPU тоже работает, но медленно):
  # 4.2 Замороженная сеть: учим только "голову" (быстрый baseline)
  python src/train_image.py --freeze
  # 4.3 Fine-tuning всей сети, 3 seed
  python src/train_image.py --seed 42
  # 5.2 Мультимодальная модель (feature-level fusion)
  python src/train_image.py --clinical safe --seed 42
  # 4.5 Другая архитектура
  python src/train_image.py --arch resnet50 --seed 42
  # Быстрая проверка, что всё запускается (200 картинок, 1 эпоха)
  python src/train_image.py --max-images 200 --epochs 1

Что сохраняется в outputs/models/<run>__seed<k>/:
  best.pt              веса лучшей эпохи (по val ROC-AUC)
  history.csv          loss и AUC по эпохам -> learning curves
  config.json          настройки запуска (воспроизводимость)
  preprocessor.joblib  предобработка клиники (только для fusion)
и предсказания в outputs/preds/<run>__seed<k>.csv
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import DATA_DIR, IMG_SUBDIR, OUT_DIR, set_seed
from features import build_preprocessor, feature_columns, load_metadata
from models import LesionDataset, LesionNet, build_transforms
from utils import save_preds


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    """Один проход по данным. Если optimizer задан — обучение, иначе только предсказание."""
    train = optimizer is not None
    model.train(train)
    if train and getattr(model, "frozen", False):
        model.backbone.eval()  # у замороженной сети BatchNorm не обновляем
    losses, probs, labels = [], [], []
    for x, clin, y in loader:
        x, y = x.to(device), y.to(device)
        clin = clin.to(device) if clin.numel() else None
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x, clin)
            loss = criterion(logits.float(), y)
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item() * len(y))
        probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
        labels.append(y.cpu().numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    return sum(losses) / len(labels), roc_auc_score(labels, probs), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="папка с images_256/ и metadata_split.csv")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--clinical", default="none", choices=["none", "safe"], help="safe = мультимодальная модель")
    ap.add_argument("--freeze", action="store_true", help="заморозить CNN, учить только голову")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6, help="early stopping: эпох без улучшения val AUC")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate для CNN (голова учится в 10 раз быстрее)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None, help="только для отладки")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind = "fusion" if args.clinical != "none" else "image"
    run = f"{kind}_{args.arch}{'_frozen' if args.freeze else ''}"
    run_dir = Path(args.out) / "models" / f"{run}__seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Запуск {run} seed={args.seed} на {device}")

    df = load_metadata(Path(args.data_dir) / "metadata_split.csv")
    if args.max_images:
        df = pd.concat([g.sample(min(len(g), args.max_images // 3), random_state=0)
                        for _, g in df.groupby("split")]).reset_index(drop=True)
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()

    # Клинические признаки: предобработка обучается ТОЛЬКО на train
    clinical, n_clin = None, 0
    if args.clinical != "none":
        cols = feature_columns(args.clinical)
        prep = build_preprocessor(args.clinical).fit(df.loc[tr, cols])
        clinical = prep.transform(df[cols]).astype(np.float32)
        n_clin = clinical.shape[1]
        joblib.dump(prep, run_dir / "preprocessor.joblib")

    img_dir = Path(args.data_dir) / IMG_SUBDIR
    def make_loader(mask, train):
        ds = LesionDataset(img_dir, df.loc[mask, "img_file"], df.loc[mask, "label"],
                           None if clinical is None else clinical[mask],
                           build_transforms(args.img_size, train=train))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                          pin_memory=device.type == "cuda", drop_last=train and len(ds) > args.batch_size)
    train_loader, val_loader = make_loader(tr, True), make_loader(va, False)
    all_loader = make_loader(np.ones(len(df), bool), False)

    model = LesionNet(args.arch, n_clinical=n_clin).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    if args.freeze:
        model.frozen = True
        for p in model.backbone.parameters():
            p.requires_grad = False
        param_groups = [{"params": head_params, "lr": args.lr * 10}]
    else:
        # Дифференцированный learning rate: предобученную CNN меняем осторожно, новую голову — быстрее
        param_groups = [{"params": model.backbone.parameters(), "lr": args.lr},
                        {"params": head_params, "lr": args.lr * 10}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # pos_weight: балансировка классов (у нас почти 50/50, но так корректно в общем случае)
    y_tr = df.loc[tr, "label"].to_numpy()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), device=device))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config = {**vars(args), "run": run, "n_clinical": n_clin, "device": str(device)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    history, best_auc, bad_epochs = [], -1, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_auc, _ = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_auc": tr_auc,
                        "val_loss": va_loss, "val_auc": va_auc})
        mark = ""
        if va_auc > best_auc:
            best_auc, bad_epochs, mark = va_auc, 0, "  <- лучшая"
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_epochs += 1
        print(f"эпоха {epoch:2d} | train loss {tr_loss:.3f} AUC {tr_auc:.3f} | "
              f"val loss {va_loss:.3f} AUC {va_auc:.3f} | {time.time() - t0:.0f}с{mark}")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if bad_epochs >= args.patience:
            print(f"Early stopping: {args.patience} эпох без улучшения")
            break

    # Предсказания лучшей модели для val и test (test здесь только сохраняем, не смотрим!)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    _, _, probs = run_epoch(model, all_loader, device, criterion)
    save_preds(df, probs, run, args.seed, args.out)
    print(f"Лучший val AUC = {best_auc:.3f}. Сохранено в {run_dir}")


if __name__ == "__main__":
    main()
