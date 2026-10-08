"""
Шаг 1. Подготовка изображений PAD-UFES-20.

Что делает:
  - читает картинки прямо из трёх zip-архивов (распаковывать вручную не нужно);
  - уменьшает каждую до SIZE x SIZE (короткая сторона -> SIZE, затем центральная обрезка);
  - сохраняет в JPG (вместо ~3.6 ГБ PNG получается ~100-200 МБ);
  - проверяет, что для каждой строки metadata.csv есть картинка;
  - упаковывает результат в один zip для загрузки в Google Drive / Colab.

Зачем: модели всё равно работают с картинками 224x224, а огромные PNG только
тормозят загрузку и не влезают в память Colab. Делаем это ОДИН раз.

Запуск (из папки skin-lesion-project):
  python src/prepare_images.py --raw "../zr7vgbcyr2-1"
Быстрая проверка на 20 картинках:
  python src/prepare_images.py --raw "../zr7vgbcyr2-1" --limit 20
"""
import argparse
import io
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def resize_and_save(args):
    name, data, out_dir, size = args
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    # ImageOps.fit = масштабирование + центральная обрезка до квадрата
    img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
    out_path = Path(out_dir) / (Path(name).stem + ".jpg")
    img.save(out_path, "JPEG", quality=95)
    return Path(name).name


def iter_zip_images(zip_paths, limit=None):
    count = 0
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.lower().endswith(".png"):
                    continue
                yield name, z.read(name)
                count += 1
                if limit and count >= limit:
                    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="папка zr7vgbcyr2-1 (с metadata.csv и images/)")
    p.add_argument("--out", default="data/processed", help="куда сохранить результат")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="обработать только N картинок (для теста)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    img_dir = out / f"images_{args.size}"
    img_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted((raw / "images").glob("imgs_part_*.zip"))
    if not zip_paths:
        raise SystemExit(f"Не найдены архивы imgs_part_*.zip в {raw / 'images'}")
    print("Архивы:", [z.name for z in zip_paths])

    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        jobs = ((n, d, str(img_dir), args.size) for n, d in iter_zip_images(zip_paths, args.limit))
        for i, name in enumerate(ex.map(resize_and_save, jobs), 1):
            done.append(name)
            if i % 100 == 0:
                print(f"  обработано {i}")
    print(f"Готово: {len(done)} изображений -> {img_dir}")

    # Сверка с метаданными: у каждой строки должна быть картинка
    meta = pd.read_csv(raw / "metadata.csv")
    meta["img_file"] = meta["img_id"].str.replace(".png", ".jpg", regex=False)
    have = {f.name for f in img_dir.glob("*.jpg")}
    missing = meta.loc[~meta["img_file"].isin(have), "img_id"]
    if args.limit is None and len(missing):
        print(f"ВНИМАНИЕ: нет картинок для {len(missing)} строк metadata, например: {list(missing[:5])}")
    else:
        print(f"Сверка с metadata.csv: строк {len(meta)}, картинок {len(have)}")
    meta.to_csv(out / "metadata.csv", index=False)

    if args.limit is None:
        archive = shutil.make_archive(str(out / f"pad_ufes_{args.size}"), "zip", root_dir=out,
                                      base_dir=img_dir.name)
        print("Архив для Colab:", archive)


if __name__ == "__main__":
    main()
